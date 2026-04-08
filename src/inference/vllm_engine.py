"""vLLM inference engine for No-Harm evaluation framework
   Manages vLLM Docker containers on demand — one container per model,
   started on load_model() and stopped on unload_model().
"""
import logging
import subprocess
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override via constructor or environment
# ---------------------------------------------------------------------------
VLLM_BASE_URL        = "http://localhost:{port}/v1"
VLLM_IMAGE           = "nvcr.io/nvidia/vllm:26.01-py3"
HF_CACHE_HOST        = "/home/neo/.cache/huggingface/hub"
HF_CACHE_CONTAINER   = "/home/neo/.cache/huggingface/hub"
HF_TOKEN_PATH        = "/root/.cache/huggingface/token"
SERVER_READY_TIMEOUT = 600   # seconds to wait for server to come up
SERVER_READY_POLL    = 5     # seconds between health-check polls
BASE_PORT            = 8000  # first port; each concurrent model gets base+n


def _read_hf_token() -> str:
    try:
        with open(HF_TOKEN_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _wait_for_server(url: str, timeout: int = SERVER_READY_TIMEOUT,
                     poll: int = SERVER_READY_POLL) -> bool:
    """Poll GET {url}/models until HTTP 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        logger.info(f"[VLLMEngine] Waiting for server at {url} ...")
        time.sleep(poll)
    return False


class _ModelState:
    """Bookkeeping for a single running model container."""
    def __init__(self, container_name: str, port: int, served_name: str):
        self.container_name = container_name
        self.port           = port
        self.served_name    = served_name
        self.base_url       = VLLM_BASE_URL.format(port=port)
        self.client         = OpenAI(base_url=self.base_url, api_key="none")


class VLLMEngine:
    """
    vLLM inference engine that manages Docker containers on demand.

    Each call to load_model() spins up a dedicated vLLM Docker container
    for that model on its own port.  unload_model() stops and removes it.
    Only one model needs to be in GPU memory at a time, enabling sequential
    evaluation across many large models on a single machine.

    Usage:
        engine = VLLMEngine()
        engine.load_model("ministral-14b", "/path/to/ministral-14b")
        responses = engine.generate_batch("ministral-14b", prompts)
        engine.unload_model("ministral-14b")

        engine.load_model("gemma3-27b", "/path/to/gemma3-27b")
        responses = engine.generate_batch("gemma3-27b", prompts)
        engine.unload_model("gemma3-27b")
    """

    def __init__(
        self,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        enable_cuda_graph: bool = False,  # enforce_eager — skips CUDA graph compilation (saves 10–35 min per model)
        vllm_image: str = VLLM_IMAGE,
        hf_cache_host: str = HF_CACHE_HOST,
        hf_token_path: str = HF_TOKEN_PATH,
        base_port: int = BASE_PORT,
    ):
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size   = tensor_parallel_size
        self.enable_cuda_graph      = enable_cuda_graph
        self.vllm_image             = vllm_image
        self.hf_cache_host          = hf_cache_host
        self.hf_token_path          = hf_token_path
        self.base_port              = base_port
        self._next_port             = base_port

        # model_name -> _ModelState
        self._states: Dict[str, _ModelState] = {}

        logger.info("[VLLMEngine] Initialized (Docker-managed, on-demand loading)")

    # ------------------------------------------------------------------
    # Port allocation
    # ------------------------------------------------------------------
    def _alloc_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_model(
        self,
        model_name: str,
        hf_model_path: str,
        max_model_len: Optional[int] = None,
        **vllm_kwargs,
    ) -> None:
        """
        Start a vLLM Docker container serving `hf_model_path` and register it
        under `model_name`.  Blocks until the server is healthy.

        Args:
            model_name:    Internal name used by generate_batch / unload_model.
            hf_model_path: HuggingFace model ID or local path
                           (e.g. "/home/neo/.cache/huggingface/hub/ministral-14b").
                           Passed to vLLM's --model flag.
            max_model_len: Optional --max-model-len override.
            **vllm_kwargs: Extra key=value flags forwarded to the vLLM server
                           as --key value (e.g. trust_remote_code=True).
                           tensor_parallel_size, gpu_memory_utilization, and
                           max_model_len are handled explicitly and stripped
                           from kwargs to avoid duplicate flags.
        """
        if model_name in self._states:
            logger.warning(f"[VLLMEngine] '{model_name}' already loaded — skipping")
            return

        port           = self._alloc_port()
        container_name = f"vllm-{model_name.replace('/', '-').replace(':', '-')}-{port}"
        base_url       = VLLM_BASE_URL.format(port=port)

        try:
            hf_token = _read_hf_token()
        except Exception:
            hf_token = ""

        # ---- Build docker run command ----------------------------------------
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
            "-v", f"{self.hf_cache_host}:{HF_CACHE_CONTAINER}",
            self.vllm_image,
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  hf_model_path,
            "--served-model-name",      model_name,
            "--port",                   str(port),
            "--download-dir",           HF_CACHE_CONTAINER,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size",   str(self.tensor_parallel_size),
        ]
        if max_model_len:
            cmd += ["--max-model-len", str(max_model_len)]
        if not self.enable_cuda_graph:
            cmd += ["--enforce-eager"]

        # Strip keys already handled explicitly to avoid duplicate flags
        vllm_kwargs.pop("tensor_parallel_size", None)
        vllm_kwargs.pop("gpu_memory_utilization", None)
        vllm_kwargs.pop("max_model_len", None)

        for k, v in vllm_kwargs.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]

        # ---- Launch container ------------------------------------------------
        logger.info(
            f"[VLLMEngine] Starting container '{container_name}' "
            f"for model '{model_name}' on port {port} ..."
        )
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[VLLMEngine] Container ID: {result.stdout.strip()[:12]}")
        except subprocess.CalledProcessError as e:
            logger.error(f"[VLLMEngine] docker run failed:\n{e.stderr}")
            raise RuntimeError(f"Failed to start container for '{model_name}'") from e

        # ---- Wait until ready ------------------------------------------------
        logger.info(
            f"[VLLMEngine] Waiting for '{model_name}' server "
            f"(up to {SERVER_READY_TIMEOUT}s) ..."
        )
        if not _wait_for_server(base_url):
            # Capture container logs before teardown for diagnosis
            try:
                log_result = subprocess.run(
                    ["docker", "logs", container_name],
                    capture_output=True, text=True, timeout=10
                )
                logger.error(
                    f"[VLLMEngine] Container logs for '{container_name}':\n"
                    f"{log_result.stdout[-3000:]}\n{log_result.stderr[-1000:]}"
                )
            except Exception as log_err:
                logger.warning(f"[VLLMEngine] Could not capture container logs: {log_err}")
            self._stop_container(container_name)
            raise RuntimeError(
                f"vLLM server for '{model_name}' did not become ready "
                f"within {SERVER_READY_TIMEOUT}s"
            )

        # ---- Verify the served model name ------------------------------------
        client    = OpenAI(base_url=base_url, api_key="none")
        available = [m.id for m in client.models.list().data]
        logger.info(f"[VLLMEngine] Server reports models: {available}")

        served_name = model_name if model_name in available else (
            available[0] if available else None
        )
        if served_name is None:
            self._stop_container(container_name)
            raise RuntimeError(f"No models found on server for '{model_name}'")

        self._states[model_name] = _ModelState(container_name, port, served_name)
        logger.info(
            f"[VLLMEngine] '{model_name}' ready -> served as '{served_name}' "
            f"on port {port}"
        )

    def generate_batch(
        self,
        model_name: str,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts concurrently.
        All prompts are sent to vLLM simultaneously so the server can
        batch them on the GPU, giving 10-20x speedup over sequential requests.

        Args:
            model_name:  Internal model name (must be registered via load_model).
            prompts:     List of prompt strings.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.
            top_p:       Nucleus sampling parameter.

        Returns:
            List of generated response strings (empty string on per-prompt error).
        """
        if model_name not in self._states:
            raise ValueError(
                f"Model '{model_name}' not loaded. Call load_model() first."
            )

        state        = self._states[model_name]
        is_ministral = "ministral" in model_name.lower()

        logger.info(
            f"[VLLMEngine] Generating {len(prompts)} responses "
            f"with '{state.served_name}' "
            f"(temp={temperature}, max_tokens={max_tokens})"
        )

        def _single_request(args):
            i, prompt = args
            messages = (
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": prompt},
                ]
                if is_ministral
                else [{"role": "user", "content": prompt}]
            )
            max_attempts = 4  # 1 primary + 3 retries
            backoff = [1, 2, 4]
            for attempt in range(max_attempts):
                try:
                    completion = state.client.chat.completions.create(
                        model=state.served_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    text = completion.choices[0].message.content or ""
                    if text.strip():
                        return i, text
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"[VLLMEngine] Empty response for prompt #{i} "
                            f"(attempt {attempt + 1}/{max_attempts}), retrying..."
                        )
                        time.sleep(backoff[attempt])
                    else:
                        logger.warning(
                            f"[VLLMEngine] Empty response for prompt #{i} "
                            f"after {max_attempts} attempts — giving up"
                        )
                        return i, ""
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"[VLLMEngine] Request failed for prompt #{i} "
                            f"(attempt {attempt + 1}/{max_attempts}): {e}, retrying..."
                        )
                        time.sleep(backoff[attempt])
                    else:
                        logger.error(
                            f"[VLLMEngine] Request failed for prompt #{i} "
                            f"after {max_attempts} attempts: {e}"
                        )
                        return i, ""
            return i, ""  # unreachable, satisfies linter

        # Send all prompts concurrently — vLLM batches them server-side
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = {executor.submit(_single_request, (i, p)): i
                       for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                i, text = future.result()
                results[i] = text

        responses = results

        empty_count = sum(1 for r in responses if not r.strip())
        if empty_count:
            logger.warning(
                f"[VLLMEngine] {empty_count}/{len(responses)} responses are empty"
            )
        logger.info(f"[VLLMEngine] Generated {len(responses)} responses")
        return responses

    def unload_model(self, model_name: str) -> None:
        """Stop and remove the Docker container for `model_name`."""
        if model_name not in self._states:
            logger.warning(
                f"[VLLMEngine] '{model_name}' not registered — nothing to unload"
            )
            return

        container_name = self._states[model_name].container_name
        self._stop_container(container_name)
        del self._states[model_name]
        logger.info(f"[VLLMEngine] Unloaded '{model_name}'")

    def get_loaded_models(self) -> List[str]:
        """Return list of currently loaded model names."""
        return list(self._states.keys())

    def __del__(self):
        """Best-effort cleanup of any still-running containers on GC."""
        for model_name in list(self._states.keys()):
            try:
                self.unload_model(model_name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stop_container(self, container_name: str) -> None:
        logger.info(f"[VLLMEngine] Stopping container '{container_name}' ...")
        for action in ("stop", "rm"):
            try:
                subprocess.run(
                    ["docker", action, container_name],
                    check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"[VLLMEngine] docker {action} '{container_name}' "
                    f"failed (may already be gone): {e.stderr.strip()}"
                )
