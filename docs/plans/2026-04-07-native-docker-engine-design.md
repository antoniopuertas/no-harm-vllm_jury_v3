# Native vs Docker Engine Selection Design

**Date:** 2026-04-07  
**Status:** Approved

---

## Problem

The GB10 Blackwell commit (`8dc0bed`) replaced the original native vLLM engine (which used `from vllm import LLM, SamplingParams` in-process) with a Docker-managed engine. As a result, H100 evaluations — which run vLLM natively — would incorrectly attempt to spin up Docker containers.

---

## Goal

Restore the native vLLM engine for H100 use, keep the Docker engine for GB10, and wire the selection automatically through the existing `--gpu` flag in the bash scripts.

---

## Design

### New file: `src/inference/vllm_engine_native.py`

Restored verbatim from git commit `7a7d3ca` (the last commit before GB10 replaced it). Uses `from vllm import LLM, SamplingParams` — loads models in-process, no Docker involved. Class name: `NativeVLLMEngine` (renamed from `VLLMEngine` to avoid import conflicts).

### `src/inference/vllm_engine.py` — unchanged

Docker engine stays exactly as-is.

### `scripts/run_full_vllm_evaluation.py`

Add `--engine` argument:
```python
parser.add_argument(
    "--engine",
    choices=["native", "docker"],
    default="docker",
    help="Inference engine: 'native' for H100 (vLLM in-process), 'docker' for GB10"
)
```

Engine instantiation becomes:
```python
if args.engine == "native":
    from src.inference.vllm_engine_native import NativeVLLMEngine
    engine = NativeVLLMEngine(
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1
    )
else:
    from src.inference.vllm_engine import VLLMEngine
    engine = VLLMEngine(
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1
    )
```

### Bash scripts (`run_harm_v2_sequential.sh`, `run_harm_v2_1000.sh`, `run_medqa_medmcqa.sh`)

In the Python invocation, append `--engine` based on `$GPU_LABEL`:
```bash
if [[ "$GPU_LABEL" == "GB10" ]]; then
    ENGINE_FLAG="--engine docker"
else
    ENGINE_FLAG="--engine native"
fi
```
Then pass `$ENGINE_FLAG` to the Python call.

---

## Data Flow

```
bash --gpu H100  →  python --engine native  →  vllm_engine_native.py (LLM in-process)
bash --gpu GB10  →  python --engine docker  →  vllm_engine.py (Docker containers)
```

---

## Out of Scope

- `ModelManager`, evaluation logic, output paths, configs — all unchanged.
- `run_full_vllm_evaluation_v3.py` — not modified (uses same VLLMEngine import but is not part of the harm_dimensions_v2 pipeline).
