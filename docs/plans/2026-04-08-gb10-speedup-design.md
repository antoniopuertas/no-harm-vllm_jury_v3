# GB10 Docker Evaluation Speedup — Design

**Date:** 2026-04-08  
**Scope:** GB10 (Docker vLLM engine) only — H100 native path untouched  
**Source analysis:** `docs/analysis/gb10_vs_h100_comparison.md`

---

## Problem Summary

GB10 evaluations take ~46–48 hours per dataset (3 datasets = ~137 hours total).
Two root causes drive this:

1. **gemma3-27b container never starts.** The vLLM Docker container is launched but its HTTP server never becomes healthy within the 600s timeout. The pipeline silently swallows the load failure, proceeds to score 1000 samples with a model that is not loaded, and records 0.0 for all dimensions — wasting 600s of dead wait per dataset and corrupting jury results.

2. **Sequential per-sample scoring.** `score_with_jury_batch()` processes samples one at a time: each iteration calls `engine.generate_batch()` with 7 prompts, then waits for parse + retry before starting the next sample. vLLM sits idle between 7-prompt micro-batches. With 1000 samples × 5 jurors, this means 5000 separate round-trips to the vLLM server per dataset.

---

## Design

### Section 1 — Fail-Fast on Juror Load Failure

**File:** `scripts/run_full_vllm_evaluation.py`

In `evaluate_dataset_full()`, check the return value of `manager.load_jury_member()`. On failure:
- Log a CRITICAL-level error (not silent)
- Fill the juror's result slot with `score=0.0, justification="Juror load failed"` + a `load_failed=True` flag per entry
- `continue` to the next juror — do not score with an unloaded model
- Record failed jurors in run metadata for downstream filtering

```python
ok = manager.load_jury_member(jury_member)
if not ok:
    logger.critical(
        f"[{jury_member}] Container failed to start — skipping this juror. "
        "Results will be missing this jury member's scores."
    )
    jury_results[jury_idx] = [
        {
            "instance_id": inst.get("id", f"instance_{i}"),
            "scores": {
                dim: {"score": 0.0, "justification": "Juror load failed"}
                for dim in all_dimensions
            },
            "load_failed": True,
        }
        for i, inst in enumerate(instances)
    ]
    continue
```

**Impact:** Eliminates 600s dead wait × 3 datasets = 30 minutes recovered. Stops silent result corruption.  
**H100 effect:** None — model loading succeeds on H100, the `if not ok` branch never fires.

---

### Section 2 — Cross-Sample Batch Scoring

**Files:** `src/evaluation/multi_dim_jury_v2.py`, `scripts/run_full_vllm_evaluation.py`

#### 2a. New method in `MultiDimensionalJuryScorer`

Add `score_samples_batch()` — takes N `(question, response)` pairs, builds N×7 prompts, sends them in a single `generate_batch()` call, parses results per sample, and retries only failed dimensions.

Existing `score_all_dimensions()` and `score_response_batch()` are **not modified** — they remain the default path for `scoring_batch_size=1`.

```python
def score_samples_batch(
    self,
    model_name: str,
    samples: List[Tuple[str, str]],   # (question, response) pairs
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> List[Optional[Dict[str, DimensionScore]]]:
    """
    Score multiple (question, response) pairs in a single generate_batch call.
    Sends len(samples) × 7 prompts at once; parses per-sample results after.
    Retries only dims that fail extraction, using existing score_dimension_with_retry().
    """
    n = len(samples)
    max_tokens = {
        'olmo-32b': 512, 'nemotron-30b': 1024
    }.get(model_name, 512)

    # Build flat list: [s0_d0, s0_d1, ..., s0_d6, s1_d0, ..., sN_d6]
    prompts = []
    for question, response in samples:
        for dim_key in self.dimensions:
            prompts.append(self.generate_scoring_prompt(question, response, dim_key))

    try:
        raw_responses = self.engine.generate_batch(
            model_name=model_name,
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"[{model_name}] Batch generation failed: {e}")
        return [None] * n

    results = []
    n_dims = len(self.dimensions)

    for i, (question, response) in enumerate(samples):
        offset = i * n_dims
        dim_scores = {}
        failed_dims = []

        for j, dim_key in enumerate(self.dimensions):
            raw = raw_responses[offset + j]
            score = self.extract_dimension_score(raw, model_name, dim_key)
            if score:
                dim_scores[dim_key] = score
            else:
                failed_dims.append(dim_key)

        # Retry failed dims individually (existing logic, unchanged)
        for dim_key in failed_dims:
            retry_score = self.score_dimension_with_retry(
                model_name=model_name,
                question=question,
                response=response,
                dimension=dim_key,
            )
            if retry_score:
                dim_scores[dim_key] = retry_score

        results.append(dim_scores if dim_scores else None)

    return results
```

#### 2b. Dispatch in `score_with_jury_batch()`

Add a `batch_size: int = 1` parameter. When `batch_size > 1`, dispatch to `score_samples_batch()` in chunks. When `batch_size == 1`, keep the current per-sample path unchanged.

```python
def score_with_jury_batch(
    engine, jury_member, instances, responses,
    batch_size: int = 1,
    progress_callback=None
):
    scorer = MultiDimensionalJuryScorer(engine)
    results = []

    if batch_size > 1:
        for i in range(0, len(instances), batch_size):
            chunk_inst = instances[i:i + batch_size]
            chunk_resp = responses[i:i + batch_size]
            samples = [
                (inst.get("question", ""), resp)
                for inst, resp in zip(chunk_inst, chunk_resp)
            ]
            batch_scores = scorer.score_samples_batch(jury_member, samples)
            for k, (instance, scores) in enumerate(zip(chunk_inst, batch_scores)):
                idx = i + k
                if scores is None:
                    results.append({
                        "instance_id": instance.get("id", f"instance_{idx}"),
                        "scores": {
                            dim: {"score": 0.0, "justification": "Parsing failed"}
                            for dim in scorer.dimensions
                        }
                    })
                else:
                    results.append({
                        "instance_id": instance.get("id", f"instance_{idx}"),
                        "scores": {
                            dim: {"score": ds.score, "justification": ds.justification}
                            for dim, ds in scores.items()
                        }
                    })
            if progress_callback:
                progress_callback(min(i + batch_size, len(instances)), len(instances))
    else:
        # existing per-sample path (unchanged)
        ...

    return results
```

#### 2c. Gating in `main()`

`scoring_batch_size` is set to `10` only for `--engine docker`. H100 native always gets `1`.

```python
scoring_batch_size = 10 if args.engine == "docker" else 1
```

Pass `scoring_batch_size` through `evaluate_dataset_full()` → `score_with_jury_batch()`.

**Impact:** 1000 `generate_batch` calls → 100 per juror. GPU stays busy continuously across samples.  
**H100 effect:** `scoring_batch_size=1` → existing per-sample path, no change.

---

### Section 3 — gemma3-27b Diagnosis Protocol

This is a one-time diagnostic step, not a permanent code change (except the log capture).

#### 3a. Capture container logs before teardown

In `VLLMEngine.load_model()`, before calling `_stop_container()` on timeout, capture and log the container's internal output:

```python
# In load_model(), after _wait_for_server() returns False:
try:
    log_result = subprocess.run(
        ["docker", "logs", container_name],
        capture_output=True, text=True, timeout=10
    )
    logger.error(
        f"[VLLMEngine] Container logs for '{container_name}':\n"
        f"{log_result.stdout[-3000:]}\n{log_result.stderr[-1000:]}"
    )
except Exception:
    pass
self._stop_container(container_name)
```

#### 3b. Isolated test run

After the pipeline fixes are deployed, trigger a 1-sample run with only the docker engine to surface the gemma3-27b failure cleanly:

```bash
CUDA_VISIBLE_DEVICES=0 /home/puertao/.conda/envs/vllm-gemma/bin/python \
  scripts/run_full_vllm_evaluation.py --engine docker --dataset medqa --num_samples 1
```

The CRITICAL log + captured container output will identify the root cause (OOM, CUDA mismatch, missing `--dtype`, bad model path, etc.).

#### 3c. Fix location

The fix goes in `config/vllm_jury_config.yaml` — add flags to gemma3-27b's `vllm_config` entry (e.g., `dtype: float16`, `max_model_len`, `quantization`). No Python code change needed.

---

## Files Changed

| File | Change |
|---|---|
| `scripts/run_full_vllm_evaluation.py` | Fail-fast on load failure; `scoring_batch_size` param; gate on `--engine docker` |
| `src/evaluation/multi_dim_jury_v2.py` | Add `score_samples_batch()` method |
| `src/inference/vllm_engine.py` | Add container log capture before teardown |
| `config/vllm_jury_config.yaml` | (Post-diagnosis) fix gemma3-27b vllm_config |

**Not changed:** `src/inference/vllm_engine_native.py`, H100 evaluation path, aggregation, parsing, tests.

---

## Expected Impact

| Improvement | Estimated Saving |
|---|---|
| Fail-fast (no 600s wait) | ~30 min / full run |
| Cross-sample batching (×10) | ~2–3× scoring throughput for 4 working jurors |
| gemma3-27b fix (post-diagnosis) | Restores 20% of jury signal; eliminates 0.0 contamination |
| **Combined (excl. gemma3 fix)** | ~46h → ~15–20h per dataset |

---

## Constraints

- H100 native path (`--engine native`) is not modified in behavior
- All existing tests remain valid
- Retry logic in `score_dimension_with_retry()` is not modified
- `score_all_dimensions()` and `score_response_batch()` are not modified
