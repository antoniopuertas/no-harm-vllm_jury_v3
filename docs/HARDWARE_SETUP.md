# Hardware Setup

Jury v3.0 requires significant VRAM to load its 5 jury models. This page explains the hardware requirements, VRAM budgets, and how to configure parallel model loading.

## VRAM Requirements

| Model | VRAM | Tensor Parallel |
|-------|------|-----------------|
| ministral-14b | 28 GB | 1 GPU |
| qwen2.5-coder-7b | 15 GB | 1 GPU |
| gemma3-27b | 54 GB | 1 or 2 GPUs |
| nemotron-30b | 60 GB | 1 or 2 GPUs |
| olmo-32b | 64 GB | 1 or 2 GPUs |
| **Total (all 5)** | **221 GB** | Cannot load all simultaneously |

The full jury cannot be loaded into VRAM at the same time. Models are loaded and unloaded sequentially during evaluation, or loaded in parallel rotation waves (see below).

## Recommended Hardware

- **Minimum**: 1× NVIDIA H100 (80 GB) — sequential model loading, ~4–5 hours for 1000 samples
- **Recommended**: 2× NVIDIA H100 NVL (94 GB each, 188 GB total) — parallel rotation waves, ~2–2.5 hours for 1000 samples

Because models are never all loaded simultaneously, 188 GB is sufficient — the dual-GPU configuration loads at most 2–3 models in parallel (up to ~131 GB), then rotates to the next batch.

## Single GPU Configuration (`config/vllm_jury_config.yaml`)

Models load one at a time. Each model uses `tensor_parallel_size: 1`. Evaluation is sequential.

```bash
python scripts/run_full_vllm_evaluation.py \
    --dataset medmcqa \
    --instances 1000 \
    --config config/vllm_jury_config.yaml \
    > logs/medmcqa_$(date +%Y%m%d_%H%M%S).log 2>&1
```

## Dual GPU Configuration (`config/vllm_jury_config_dual_gpu.yaml`)

Large models (gemma3-27b, nemotron-30b, olmo-32b) use `tensor_parallel_size: 2`, splitting layers across both GPUs via NVLink:

```
GPU 0: first half of model layers
GPU 1: second half of model layers
```

Layers are split evenly across both GPUs. This enables ~2× throughput for large models and doubles the viable batch size.

| Model | Single GPU batch | Dual GPU batch | Speedup |
|-------|-----------------|----------------|---------|
| ministral-14b | 64 | 64 | — |
| qwen2.5-coder-7b | 64 | 64 | — |
| gemma3-27b | 32 | 64 | ~2× |
| nemotron-30b | 24 | 48 | ~2× |
| olmo-32b | 24 | 48 | ~2× |

## Running with Dual GPU

All commands below assume you are running from the project root directory.

```bash
# Recommended: use the helper script
bash scripts/run_1000_dual_gpu.sh medmcqa

# Or manually
nohup python scripts/run_full_vllm_evaluation.py \
    --dataset medmcqa \
    --instances 1000 \
    --config config/vllm_jury_config_dual_gpu.yaml \
    > logs/medmcqa_dual_gpu_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Monitoring GPU Usage

```bash
# Verify both GPUs are active during dual-GPU evaluation
watch -n 1 nvidia-smi
# Expected: GPU 0 ~85% utilization, GPU 1 ~85% utilization
```

## Fallback to Single GPU

If dual-GPU causes issues (e.g. NVLink errors, OOM), use the original single-GPU config:

```bash
nohup python scripts/run_full_vllm_evaluation.py \
    --dataset medmcqa \
    --instances 1000 \
    --config config/vllm_jury_config.yaml \
    > logs/medmcqa_single_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
