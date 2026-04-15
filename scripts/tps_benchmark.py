#!/usr/bin/env python3
"""
TPS Benchmark — GB10 Blackwell Superchip
==========================================
Measures tokens-per-second throughput for all locally cached models.
Uses the same Docker/vLLM pattern as VLLMEngine.

Each model is tested across:
  - 3 output lengths  : 64 / 256 / 512 tokens
  - 3 batch sizes     : 1 / 8 / 32 concurrent requests
  = 9 benchmarks per model

Reports:
  - Decode TPS  (output tokens / wall-clock time, all requests concurrent)
  - Requests/s
  - Mean latency per request

Output:
  - Console table
  - data/benchmarks/tps_YYYYMMDD_HHMMSS.json

Usage:
    python scripts/tps_benchmark.py                   # all models
    python scripts/tps_benchmark.py --model qwen2.5-7b
    python scripts/tps_benchmark.py --skip nemotron-30b olmo-32b
    python scripts/tps_benchmark.py --batch-sizes 1 8 --output-tokens 64 256
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import urllib.request
import urllib.error

from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/tps_benchmark_{_ts}.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalogue  (only models with actual weights in HF cache)
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name":             "qwen2.5-7b",
        "path":             "/home/neo/.cache/huggingface/hub/qwen2.5-7b",
        "size_gb":          15,
        "trust_remote_code": False,
    },
    {
        "name":             "ministral-14b",
        "path":             "/home/neo/.cache/huggingface/hub/ministral-14b",
        "size_gb":          28,
        "trust_remote_code": False,
    },
    {
        "name":             "gemma3-27b",
        "path":             "/home/neo/.cache/huggingface/hub/gemma3-27b",
        "size_gb":          52,
        "trust_remote_code": False,
    },
    {
        "name":             "nemotron-30b",
        "path":             "/home/neo/.cache/huggingface/hub/nemotron-30b",
        "size_gb":          59,
        "trust_remote_code": True,
    },
    {
        "name":             "olmo-32b",
        "path":             "/home/neo/.cache/huggingface/hub/olmo-32b-instruct",
        "size_gb":          61,
        "trust_remote_code": False,
    },
]

# ---------------------------------------------------------------------------
# Docker / vLLM constants
# ---------------------------------------------------------------------------
VLLM_IMAGE          = "nvcr.io/nvidia/vllm:26.01-py3"
HF_CACHE_HOST       = "/home/neo/.cache/huggingface/hub"
HF_CACHE_CONTAINER  = "/home/neo/.cache/huggingface/hub"
HF_TOKEN_PATH       = "/root/.cache/huggingface/token"
BASE_PORT           = 8100           # avoid collision with eval scripts on 8000
GPU_MEM_UTIL        = 0.85           # higher than eval (0.5) — TPS test, no jury batching
MAX_MODEL_LEN       = 4096
SERVER_READY_TIMEOUT = 1800
SERVER_READY_POLL   = 5

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_TOKENS = [64, 256, 512]
DEFAULT_BATCH_SIZES   = [1, 8, 32]
WARMUP_REQUESTS       = 3
WARMUP_TOKENS         = 64

# ---------------------------------------------------------------------------
# Prompts (calibrated to produce ~50 / ~200 / ~500 input tokens)
# ---------------------------------------------------------------------------
SHORT_PROMPT = (
    "Explain the mechanism of action of beta-blockers in treating hypertension "
    "in two concise sentences."
)

MEDIUM_PROMPT = (
    "A 58-year-old male presents with chest pain radiating to the left arm, "
    "diaphoresis, and shortness of breath. He has a history of type 2 diabetes "
    "and hypertension. ECG shows ST-elevation in leads II, III, and aVF. "
    "Troponin I is elevated. Describe the immediate management priorities, "
    "the pathophysiology underlying his presentation, and the expected "
    "complications that the clinical team should anticipate over the next 48 hours."
)

LONG_PROMPT = (
    "Patient case:\n"
    "A 72-year-old woman with a 40 pack-year smoking history, COPD (GOLD stage III), "
    "type 2 diabetes mellitus (HbA1c 8.4%), hypertension, and chronic kidney disease "
    "(eGFR 35 mL/min/1.73m²) presents to the emergency department with worsening "
    "dyspnea over 3 days, productive cough with yellow-green sputum, fever (38.9°C), "
    "and confusion. On examination: RR 28/min, SpO2 84% on room air, HR 112 bpm, "
    "BP 88/54 mmHg. Chest X-ray shows right lower lobe consolidation with bilateral "
    "pleural effusions. Labs: WBC 18.4×10⁹/L, CRP 287 mg/L, procalcitonin 4.2 ng/mL, "
    "creatinine 2.8 mg/dL (baseline 1.9), BNP 890 pg/mL.\n\n"
    "Questions:\n"
    "1. What is the most likely diagnosis and what organisms should be covered?\n"
    "2. How does her CKD affect antibiotic selection and dosing?\n"
    "3. What is the PORT/PSI score and what does it imply for disposition?\n"
    "4. How should you manage her hypotension in the context of COPD and heart failure?\n"
    "5. What non-infectious conditions are contributing to her presentation and how "
    "should they be addressed simultaneously?\n"
    "Provide a structured management plan."
)

PROMPTS_BY_LENGTH = {
    "short":  SHORT_PROMPT,
    "medium": MEDIUM_PROMPT,
    "long":   LONG_PROMPT,
}

OUTPUT_TOKEN_TO_PROMPT = {
    64:  "short",
    128: "short",
    256: "medium",
    512: "long",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    output_tokens:  int
    batch_size:     int
    elapsed_s:      float
    total_out_toks: int     # sum of actual completion_tokens across all requests
    total_in_toks:  int     # sum of prompt_tokens
    decode_tps:     float   # total_out_toks / elapsed_s
    requests_per_s: float
    mean_latency_s: float
    per_req_latencies: List[float] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ModelResult:
    name:       str
    size_gb:    int
    startup_s:  float
    runs:       List[RunResult] = field(default_factory=list)
    skipped:    bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------
def _read_hf_token() -> str:
    candidates = [
        HF_TOKEN_PATH,
        os.path.expanduser("~/.cache/huggingface/token"),
        os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
    ]
    for path in candidates:
        if not path:
            continue
        # If it looks like an actual token value (not a path), return it directly
        if not path.startswith("/"):
            return path
        try:
            with open(path) as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            continue
    return ""


def _wait_for_server(url: str, timeout: int = SERVER_READY_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(SERVER_READY_POLL)
    return False


def _stop_container(name: str):
    for cmd in [["docker", "stop", name], ["docker", "rm", "-f", name]]:
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception:
            pass


def start_model(model: dict, port: int) -> Optional[OpenAI]:
    """Start a vLLM Docker container for `model`, return OpenAI client or None."""
    name = model["name"]
    path = model["path"]
    container_name = f"vllm-tps-{name.replace('/', '-')}-{port}"
    base_url = f"http://localhost:{port}/v1"

    hf_token = _read_hf_token()

    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--runtime", "nvidia",
        "--network", "host",
        "--ipc", "host",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e", "HF_HOME=/home/neo/.cache/huggingface",
        "-e", "HUGGINGFACE_HUB_CACHE=/home/neo/.cache/huggingface/hub",
    ]
    if hf_token:
        cmd += ["-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}"]
    cmd += [
        "-v", f"{HF_CACHE_HOST}:{HF_CACHE_CONTAINER}",
        VLLM_IMAGE,
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  path,
        "--served-model-name",      name,
        "--port",                   str(port),
        "--download-dir",           HF_CACHE_CONTAINER,
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--tensor-parallel-size",   "1",
        "--max-model-len",          str(MAX_MODEL_LEN),
        "--enforce-eager",          # skip CUDA graph compilation
    ]
    if model.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]

    logger.info(f"[{name}] Starting container '{container_name}' on port {port}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"[{name}] Container ID: {result.stdout.strip()[:12]}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{name}] docker run failed: {e.stderr}")
        return None, container_name

    logger.info(f"[{name}] Waiting for server (up to {SERVER_READY_TIMEOUT}s)...")
    if not _wait_for_server(base_url):
        try:
            logs = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True, text=True, timeout=10
            )
            logger.error(f"[{name}] Container logs:\n{logs.stdout[-3000:]}")
        except Exception:
            pass
        _stop_container(container_name)
        return None, container_name

    client = OpenAI(base_url=base_url, api_key="none")
    available = [m.id for m in client.models.list().data]
    logger.info(f"[{name}] Server ready. Models: {available}")
    return client, container_name


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_single_benchmark(
    client: OpenAI,
    model_name: str,
    prompt: str,
    batch_size: int,
    output_tokens: int,
) -> RunResult:
    """
    Fire `batch_size` requests concurrently, all using `prompt` and
    generating up to `output_tokens` tokens. Measure wall-clock time and
    collect usage stats.
    """

    results = []
    errors = []

    def _single_request(idx: int):
        t0 = time.perf_counter()
        try:
            resp = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=output_tokens,
                temperature=0.0,
                stream=False,
            )
            elapsed = time.perf_counter() - t0
            usage = resp.usage
            return {
                "latency": elapsed,
                "completion_tokens": usage.completion_tokens if usage else output_tokens,
                "prompt_tokens":     usage.prompt_tokens     if usage else 0,
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return {"latency": elapsed, "error": str(e),
                    "completion_tokens": 0, "prompt_tokens": 0}

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(_single_request, i) for i in range(batch_size)]
        for fut in as_completed(futures):
            results.append(fut.result())

    wall_elapsed = time.perf_counter() - wall_start

    # Aggregate
    total_out = sum(r["completion_tokens"] for r in results)
    total_in  = sum(r["prompt_tokens"]     for r in results)
    latencies = [r["latency"] for r in results]
    errs = [r["error"] for r in results if "error" in r]

    decode_tps    = total_out / wall_elapsed if wall_elapsed > 0 else 0.0
    req_per_s     = batch_size / wall_elapsed if wall_elapsed > 0 else 0.0
    mean_lat      = sum(latencies) / len(latencies) if latencies else 0.0

    run = RunResult(
        output_tokens=output_tokens,
        batch_size=batch_size,
        elapsed_s=round(wall_elapsed, 3),
        total_out_toks=total_out,
        total_in_toks=total_in,
        decode_tps=round(decode_tps, 1),
        requests_per_s=round(req_per_s, 3),
        mean_latency_s=round(mean_lat, 3),
        per_req_latencies=[round(l, 3) for l in latencies],
        error="; ".join(errs) if errs else None,
    )

    status = f"✓" if not errs else f"⚠ {len(errs)}/{batch_size} errors"
    logger.info(
        f"  out={output_tokens:3d}tok  bs={batch_size:2d}  "
        f"elapsed={wall_elapsed:.2f}s  "
        f"decode_tps={decode_tps:6.1f}  req/s={req_per_s:.3f}  "
        f"mean_lat={mean_lat:.2f}s  {status}"
    )
    return run


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------
def benchmark_model(
    model: dict,
    port: int,
    output_tokens_list: List[int],
    batch_sizes: List[int],
) -> ModelResult:
    name = model["name"]
    logger.info("=" * 65)
    logger.info(f"MODEL: {name}  ({model['size_gb']} GB)")
    logger.info("=" * 65)

    result = ModelResult(name=name, size_gb=model["size_gb"], startup_s=0.0)

    t_startup = time.time()
    client, container_name = start_model(model, port)
    startup_s = time.time() - t_startup

    if client is None:
        result.skipped = True
        result.skip_reason = "Container failed to start"
        logger.error(f"[{name}] Skipping — container failed to start")
        return result

    result.startup_s = round(startup_s, 1)
    logger.info(f"[{name}] Startup: {startup_s:.1f}s")

    try:
        # ---- Warmup -------------------------------------------------------
        logger.info(f"[{name}] Warming up ({WARMUP_REQUESTS} requests)...")
        for _ in range(WARMUP_REQUESTS):
            try:
                client.completions.create(
                    model=name,
                    prompt=SHORT_PROMPT,
                    max_tokens=WARMUP_TOKENS,
                    temperature=0.0,
                )
            except Exception as e:
                logger.warning(f"[{name}] Warmup request failed: {e}")

        # ---- Benchmark runs -----------------------------------------------
        for out_tok in output_tokens_list:
            prompt_key = OUTPUT_TOKEN_TO_PROMPT.get(out_tok, "medium")
            prompt = PROMPTS_BY_LENGTH[prompt_key]
            for bs in batch_sizes:
                logger.info(f"[{name}] Benchmarking  out={out_tok}  batch={bs}")
                run = run_single_benchmark(
                    client=client,
                    model_name=name,
                    prompt=prompt,
                    batch_size=bs,
                    output_tokens=out_tok,
                )
                result.runs.append(run)

    finally:
        logger.info(f"[{name}] Stopping container '{container_name}'...")
        _stop_container(container_name)
        logger.info(f"[{name}] Done.")

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(all_results: List[ModelResult], output_tokens_list, batch_sizes):
    print()
    print("=" * 90)
    print("  TPS BENCHMARK RESULTS — GB10 Blackwell Superchip")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # Per-model summary table
    hdr = f"{'Model':<18} {'Size':>5} {'Start':>7} "
    for ot in output_tokens_list:
        for bs in batch_sizes:
            hdr += f" {ot}t/bs{bs:>2}"
    print(hdr)
    print("-" * 90)

    for r in all_results:
        if r.skipped:
            print(f"{'  ' + r.name:<18} {r.size_gb:>4}G  SKIPPED: {r.skip_reason}")
            continue
        row = f"  {r.name:<16} {r.size_gb:>4}G {r.startup_s:>6.0f}s"
        for ot in output_tokens_list:
            for bs in batch_sizes:
                run = next(
                    (x for x in r.runs if x.output_tokens == ot and x.batch_size == bs),
                    None
                )
                if run and not run.error:
                    row += f" {run.decode_tps:>7.0f}"
                elif run and run.error:
                    row += f"   ERR  "
                else:
                    row += f"     - "
        print(row)

    print("-" * 90)
    print(f"  Columns = decode TPS (output tokens/s, all requests concurrent)")
    print(f"  Rows sorted by model size ascending")

    # Detailed breakdown per model
    print()
    print("=" * 90)
    print("  DETAILED BREAKDOWN (decode_tps | mean_latency_s | elapsed_s)")
    print("=" * 90)
    for r in all_results:
        if r.skipped:
            continue
        print(f"\n  {r.name}  ({r.size_gb}GB, startup {r.startup_s:.0f}s)")
        print(f"  {'output_tok':>10} {'batch':>5} {'decode_tps':>11} {'req/s':>8} "
              f"{'mean_lat':>9} {'elapsed':>8} {'out_toks':>9} {'errors'}")
        print("  " + "-" * 70)
        for run in r.runs:
            err_flag = f"  [{run.error[:40]}]" if run.error else ""
            print(
                f"  {run.output_tokens:>10} {run.batch_size:>5} "
                f"{run.decode_tps:>11.1f} {run.requests_per_s:>8.3f} "
                f"{run.mean_latency_s:>9.3f} {run.elapsed_s:>8.3f} "
                f"{run.total_out_toks:>9}{err_flag}"
            )

    print()
    print("=" * 90)
    # Peak TPS per model
    print("  PEAK DECODE TPS PER MODEL")
    print("  " + "-" * 40)
    for r in sorted(all_results, key=lambda x: x.size_gb):
        if r.skipped or not r.runs:
            continue
        valid = [x for x in r.runs if not x.error]
        if valid:
            peak = max(valid, key=lambda x: x.decode_tps)
            print(
                f"  {r.name:<18} {peak.decode_tps:>7.0f} tok/s  "
                f"(out={peak.output_tokens}tok, bs={peak.batch_size})"
            )
    print("=" * 90)
    print()


def save_results(all_results: List[ModelResult], output_dir: Path, ts: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"tps_{ts}.json"
    payload = {
        "timestamp": ts,
        "hardware":  "GB10 Blackwell Superchip (96 GB unified LPDDR5X)",
        "vllm_image": VLLM_IMAGE,
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "max_model_len": MAX_MODEL_LEN,
        "results": [
            {
                "name":       r.name,
                "size_gb":    r.size_gb,
                "startup_s":  r.startup_s,
                "skipped":    r.skipped,
                "skip_reason": r.skip_reason,
                "runs": [asdict(run) for run in r.runs],
            }
            for r in all_results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TPS benchmark for all cached models")
    parser.add_argument(
        "--model", default=None,
        help="Benchmark only this model name (e.g. qwen2.5-7b)"
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Model names to skip"
    )
    parser.add_argument(
        "--output-tokens", nargs="+", type=int, default=DEFAULT_OUTPUT_TOKENS,
        metavar="N", help="Output token counts to test (default: 64 256 512)"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES,
        metavar="N", help="Batch sizes to test (default: 1 8 32)"
    )
    parser.add_argument(
        "--output-dir", default="data/benchmarks",
        help="Directory to write JSON results"
    )
    parser.add_argument(
        "--port", type=int, default=BASE_PORT,
        help=f"Base port for vLLM containers (default: {BASE_PORT})"
    )
    args = parser.parse_args()

    output_tokens_list = sorted(args.output_tokens)
    batch_sizes        = sorted(args.batch_sizes)
    output_dir         = Path(args.output_dir)
    ts                 = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Select models
    models = MODELS
    if args.model:
        models = [m for m in MODELS if m["name"] == args.model]
        if not models:
            print(f"ERROR: unknown model '{args.model}'. Available: {[m['name'] for m in MODELS]}")
            return 1
    if args.skip:
        models = [m for m in models if m["name"] not in args.skip]

    logger.info("=" * 65)
    logger.info("TPS BENCHMARK — GB10 Blackwell")
    logger.info("=" * 65)
    logger.info(f"Models        : {[m['name'] for m in models]}")
    logger.info(f"Output tokens : {output_tokens_list}")
    logger.info(f"Batch sizes   : {batch_sizes}")
    logger.info(f"GPU mem util  : {GPU_MEM_UTIL}")
    logger.info(f"Max model len : {MAX_MODEL_LEN}")
    logger.info(f"vLLM image    : {VLLM_IMAGE}")

    # Clean up any leftover benchmark containers from a previous failed run
    try:
        prune = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=vllm-tps-", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        stale = prune.stdout.strip().splitlines()
        if stale:
            logger.info(f"Removing stale containers: {stale}")
            for c in stale:
                _stop_container(c)
    except Exception:
        pass

    all_results: List[ModelResult] = []
    port = args.port

    for model in models:
        result = benchmark_model(
            model=model,
            port=port,
            output_tokens_list=output_tokens_list,
            batch_sizes=batch_sizes,
        )
        all_results.append(result)
        port += 1  # Each model gets a fresh port

    print_report(all_results, output_tokens_list, batch_sizes)
    saved = save_results(all_results, output_dir, ts)
    print(f"  Results saved to: {saved}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
