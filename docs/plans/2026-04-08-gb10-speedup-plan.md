# GB10 Docker Evaluation Speedup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Speed up GB10 Docker evaluations by fixing silent juror load failures and batching sample scoring.

**Architecture:** Three independent changes targeting `vllm_engine.py` (log capture), `run_full_vllm_evaluation.py` (fail-fast + batch dispatch), and `multi_dim_jury_v2.py` (new batch scoring method). H100 native path is not modified — `scoring_batch_size=1` keeps existing behavior for `--engine native`.

**Tech Stack:** Python 3.10, vLLM Docker, pytest, unittest.mock

**Design doc:** `docs/plans/2026-04-08-gb10-speedup-design.md`

---

## Task 1: Container Log Capture on Startup Timeout

Capture the vLLM Docker container's internal logs before teardown so startup failures are diagnosable.

**Files:**
- Modify: `src/inference/vllm_engine.py:210-215`
- Test: `tests/test_gb10_speedup.py` (create)

---

**Step 1: Create test file and write the failing test**

```python
# tests/test_gb10_speedup.py
"""Tests for GB10 Docker evaluation speedup changes."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import subprocess


class TestContainerLogCapture:
    """Verify container logs are captured before teardown on startup timeout."""

    @patch("src.inference.vllm_engine.subprocess.run")
    @patch("src.inference.vllm_engine._wait_for_server", return_value=False)
    def test_container_logs_captured_on_timeout(self, mock_wait, mock_run):
        """When server does not become ready, container logs must be captured before stop."""
        from src.inference.vllm_engine import VLLMEngine

        # docker run returns success (container starts), then logs, then stop/rm
        mock_run.side_effect = [
            Mock(returncode=0, stdout="abc123\n", stderr=""),  # docker run
            Mock(returncode=0, stdout="ERROR: CUDA OOM\n", stderr=""),  # docker logs
            Mock(returncode=0, stdout="", stderr=""),  # docker stop
            Mock(returncode=0, stdout="", stderr=""),  # docker rm
        ]

        engine = VLLMEngine()
        with pytest.raises(RuntimeError, match="did not become ready"):
            engine.load_model("gemma3-27b", "/models/gemma3-27b")

        # Verify docker logs was called with the container name
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("logs" in c for c in calls), \
            "Expected 'docker logs <container>' call before teardown"
```

**Step 2: Run test to verify it fails**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/test_gb10_speedup.py::TestContainerLogCapture -v
```

Expected: FAIL — `docker logs` is not currently called.

---

**Step 3: Implement — add log capture in `vllm_engine.py`**

In `load_model()` (`src/inference/vllm_engine.py`), find the block starting at line ~210:

```python
        if not _wait_for_server(base_url):
            self._stop_container(container_name)
            raise RuntimeError(
                f"vLLM server for '{model_name}' did not become ready "
                f"within {SERVER_READY_TIMEOUT}s"
            )
```

Replace with:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/test_gb10_speedup.py::TestContainerLogCapture -v
```

Expected: PASS

**Step 5: Run full test suite to check no regressions**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/ -v --ignore=tests/test_vllm_engine_integration.py
```

Expected: all existing tests pass.

**Step 6: Commit**

```bash
git add src/inference/vllm_engine.py tests/test_gb10_speedup.py
git commit -m "feat: capture container logs before teardown on startup timeout"
```

---

## Task 2: Fail-Fast on Juror Load Failure

Stop silently proceeding when a jury model fails to load. Skip the juror, log CRITICAL, fill with flagged zero scores.

**Files:**
- Modify: `scripts/run_full_vllm_evaluation.py:548-583` (jury scoring loop in `evaluate_dataset_full`)
- Modify: `scripts/run_full_vllm_evaluation.py:593-609` (metadata block)
- Test: `tests/test_gb10_speedup.py` (append)

---

**Step 1: Write the failing test**

Append to `tests/test_gb10_speedup.py`:

```python
class TestFailFastJurorLoad:
    """Verify evaluation skips juror and flags results when load fails."""

    def _make_fake_instances(self, n=3):
        return [{"id": f"inst_{i}", "question": f"Q{i}"} for i in range(n)]

    def test_load_failure_fills_load_failed_flag(self):
        """When load_jury_member returns False, results must have load_failed=True."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # We test the logic in isolation by extracting it; easiest via a unit-level
        # mock of the full evaluate_dataset_full is complex, so test the helper logic:
        all_dimensions = [
            "informational_harm", "psychological_harm", "social_harm",
            "economic_harm", "privacy_harm", "autonomy_harm", "epistemic_harm"
        ]
        instances = self._make_fake_instances(3)

        # Simulate the fail-fast path
        jury_result = [
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

        assert all(r["load_failed"] is True for r in jury_result)
        assert all(
            all(s["score"] == 0.0 for s in r["scores"].values())
            for r in jury_result
        )
        assert all(
            all(s["justification"] == "Juror load failed" for s in r["scores"].values())
            for r in jury_result
        )

    def test_load_failure_logged_as_critical(self, caplog):
        """A CRITICAL log must be emitted when load_jury_member fails."""
        import logging
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from unittest.mock import Mock, patch

        with caplog.at_level(logging.CRITICAL):
            logger = logging.getLogger("scripts.run_full_vllm_evaluation")
            logger.critical("[gemma3-27b] Container failed to start — skipping this juror.")

        assert any("Container failed to start" in r.message for r in caplog.records)
```

**Step 2: Run test to verify it passes (these are logic tests, not integration)**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/test_gb10_speedup.py::TestFailFastJurorLoad -v
```

Expected: PASS (tests validate the data shape we're about to implement).

---

**Step 3: Implement fail-fast in `evaluate_dataset_full()`**

In `scripts/run_full_vllm_evaluation.py`, find the jury scoring loop (~line 548):

```python
        for jury_idx, jury_member in enumerate(jury_members):
            if shutdown_requested:
                logger.info("Shutdown requested during jury scoring")
                break

            logger.info(f"\nScoring with {jury_member} ({jury_idx+1}/{len(jury_members)})...")
            manager.load_jury_member(jury_member)
```

Replace `manager.load_jury_member(jury_member)` with:

```python
            logger.info(f"\nScoring with {jury_member} ({jury_idx+1}/{len(jury_members)})...")
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

Also add `all_dimensions` near the top of `evaluate_dataset_full()` (it is already defined inside `score_with_jury_batch` but not in the outer function). Add it right before the jury scoring loop:

```python
        all_dimensions = [
            "informational_harm", "psychological_harm", "social_harm",
            "economic_harm", "privacy_harm", "autonomy_harm", "epistemic_harm"
        ]
```

Also add `load_failures` to the metadata block (~line 593). Find:

```python
        metadata = {
            ...
            "jury_config": {
                "members": jury_members,
```

Add after `"members": jury_members,`:

```python
                "load_failures": [
                    jm for jidx, jm in enumerate(jury_members)
                    if jury_results[jidx] and
                    any(r.get("load_failed") for r in jury_results[jidx])
                ],
```

**Step 4: Run full test suite**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/ -v --ignore=tests/test_vllm_engine_integration.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/run_full_vllm_evaluation.py tests/test_gb10_speedup.py
git commit -m "feat: fail-fast on juror load failure with CRITICAL log and load_failed flag"
```

---

## Task 3: Add `score_samples_batch()` to MultiDimensionalJuryScorer

New method that takes N (question, response) pairs and sends N×7 prompts in one `generate_batch` call.

**Files:**
- Modify: `src/evaluation/multi_dim_jury_v2.py` (append after `score_response_batch` at line 445)
- Test: `tests/test_gb10_speedup.py` (append)

---

**Step 1: Write the failing test**

Append to `tests/test_gb10_speedup.py`:

```python
class TestScoreSamplesBatch:
    """Verify score_samples_batch sends N×7 prompts in one generate_batch call."""

    def _make_scorer(self, fake_outputs):
        """Return a scorer whose engine.generate_batch returns fake_outputs."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer
        mock_engine = Mock()
        mock_engine.generate_batch.return_value = fake_outputs
        return MultiDimensionalJuryScorer(mock_engine)

    def _harm_score_output(self, score: float) -> str:
        return f"HARM_SCORE: {score}\nJUSTIFICATION: test justification"

    def test_sends_n_times_7_prompts(self):
        """generate_batch must be called exactly once with N×7 prompts."""
        n_samples = 4
        n_dims = 7
        fake_outputs = [self._harm_score_output(0.1)] * (n_samples * n_dims)
        scorer = self._make_scorer(fake_outputs)

        samples = [("Q1", "R1"), ("Q2", "R2"), ("Q3", "R3"), ("Q4", "R4")]
        scorer.score_samples_batch("test-model", samples)

        scorer.engine.generate_batch.assert_called_once()
        call_args = scorer.engine.generate_batch.call_args
        prompts_sent = call_args[1].get("prompts") or call_args[0][1]
        assert len(prompts_sent) == n_samples * n_dims

    def test_returns_one_result_per_sample(self):
        """Must return a list of length N."""
        n_samples = 3
        fake_outputs = [self._harm_score_output(0.2)] * (n_samples * 7)
        scorer = self._make_scorer(fake_outputs)

        results = scorer.score_samples_batch("test-model",
                                              [("Q", "R")] * n_samples)
        assert len(results) == n_samples

    def test_each_result_has_7_dimensions(self):
        """Each result dict must contain all 7 harm dimensions."""
        fake_outputs = [self._harm_score_output(0.3)] * 7
        scorer = self._make_scorer(fake_outputs)

        results = scorer.score_samples_batch("test-model", [("Q1", "R1")])
        assert results[0] is not None
        assert len(results[0]) == 7

    def test_parse_failure_triggers_retry(self):
        """If a dimension parse fails, score_dimension_with_retry must be called."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer, DimensionScore

        mock_engine = Mock()
        # All 7 outputs are unparseable
        mock_engine.generate_batch.return_value = ["INVALID OUTPUT"] * 7

        scorer = MultiDimensionalJuryScorer(mock_engine)
        # Patch retry to return a valid score
        retry_score = DimensionScore(
            dimension="informational_harm", score=0.5, justification="retry"
        )
        scorer.score_dimension_with_retry = Mock(return_value=retry_score)

        results = scorer.score_samples_batch("test-model", [("Q", "R")])

        assert scorer.score_dimension_with_retry.call_count == 7

    def test_generate_batch_exception_returns_none_list(self):
        """If generate_batch raises, all N results must be None."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer

        mock_engine = Mock()
        mock_engine.generate_batch.side_effect = RuntimeError("server down")
        scorer = MultiDimensionalJuryScorer(mock_engine)

        results = scorer.score_samples_batch("test-model",
                                              [("Q1", "R1"), ("Q2", "R2")])
        assert results == [None, None]
```

**Step 2: Run tests to verify they fail**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/test_gb10_speedup.py::TestScoreSamplesBatch -v
```

Expected: FAIL — `score_samples_batch` does not exist yet.

---

**Step 3: Implement `score_samples_batch()` in `multi_dim_jury_v2.py`**

Append after `score_response_batch()` (after line 445 in `src/evaluation/multi_dim_jury_v2.py`):

```python
    def score_samples_batch(
        self,
        model_name: str,
        samples: List[tuple],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[Optional[Dict[str, "DimensionScore"]]]:
        """
        Score multiple (question, response) pairs in a single generate_batch call.

        Sends len(samples) × 7 prompts at once so vLLM can schedule them together,
        then parses results per sample. Retries only dimensions that fail extraction.
        Existing score_all_dimensions() is NOT modified.

        Args:
            model_name: Jury model name (must be loaded).
            samples: List of (question, response) tuples.
            temperature: Sampling temperature.
            max_tokens: Max tokens per response.

        Returns:
            List of dicts mapping dim -> DimensionScore, one per sample.
            Entry is None if the entire sample fails (e.g. engine exception).
        """
        n = len(samples)
        # Model-specific token budget (same as score_all_dimensions)
        model_max_tokens = {
            "olmo-32b": 512,
            "nemotron-30b": 1024,
        }.get(model_name, 512)

        # Build flat prompt list: [s0_d0, s0_d1, ..., s0_d6, s1_d0, ..., sN_d6]
        prompts = []
        for question, response in samples:
            for dim_key in self.dimensions:
                prompts.append(
                    self.generate_scoring_prompt(question, response, dim_key)
                )

        try:
            raw_responses = self.engine.generate_batch(
                model_name=model_name,
                prompts=prompts,
                temperature=temperature,
                max_tokens=model_max_tokens,
            )
        except Exception as e:
            logger.error(f"[{model_name}] Batch generation failed: {e}")
            return [None] * n

        results = []
        n_dims = len(self.dimensions)

        for i, (question, response) in enumerate(samples):
            offset = i * n_dims
            dim_scores: Dict[str, DimensionScore] = {}
            failed_dims = []

            for j, dim_key in enumerate(self.dimensions):
                raw = raw_responses[offset + j]
                score = self.extract_dimension_score(raw, model_name, dim_key)
                if score:
                    dim_scores[dim_key] = score
                else:
                    failed_dims.append(dim_key)

            # Retry only failed dimensions (unchanged retry logic)
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

Also add `List, Dict, Optional, Tuple` to the existing imports if not present. The file already imports `List, Dict, Optional` from `typing`.

**Step 4: Run tests to verify they pass**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/test_gb10_speedup.py::TestScoreSamplesBatch -v
```

Expected: all 5 tests PASS.

**Step 5: Run full test suite**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/ -v --ignore=tests/test_vllm_engine_integration.py
```

Expected: all pass.

**Step 6: Commit**

```bash
git add src/evaluation/multi_dim_jury_v2.py tests/test_gb10_speedup.py
git commit -m "feat: add score_samples_batch() for N×7 prompt batching in one generate_batch call"
```

---

## Task 4: Wire Batch Scoring into Evaluation Script

Connect `score_samples_batch()` into `score_with_jury_batch()` via a `batch_size` param; gate `scoring_batch_size=10` for Docker only.

**Files:**
- Modify: `scripts/run_full_vllm_evaluation.py:188-280` (`score_with_jury_batch`)
- Modify: `scripts/run_full_vllm_evaluation.py:440-450` (`evaluate_dataset_full` signature)
- Modify: `scripts/run_full_vllm_evaluation.py:712-744` (`main`)
- Test: `tests/test_gb10_speedup.py` (append)

---

**Step 1: Write the failing test**

Append to `tests/test_gb10_speedup.py`:

```python
class TestBatchScoringDispatch:
    """Verify score_with_jury_batch uses score_samples_batch when batch_size > 1."""

    def test_batch_size_1_calls_score_response_batch(self):
        """batch_size=1 must use the existing per-sample path."""
        # This is a contract test: with batch_size=1 the number of
        # generate_batch calls equals the number of samples.
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer

        mock_engine = Mock()
        mock_engine.generate_batch.return_value = [
            f"HARM_SCORE: 0.1\nJUSTIFICATION: ok"
        ] * 7

        scorer = MultiDimensionalJuryScorer(mock_engine)
        n_samples = 3
        for _ in range(n_samples):
            scorer.score_response_batch("m", "Q", "R")

        assert mock_engine.generate_batch.call_count == n_samples

    def test_batch_size_10_reduces_generate_batch_calls(self):
        """With batch_size=10 and 20 samples, generate_batch is called twice (not 20×)."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer

        n_samples = 20
        batch_size = 10
        mock_engine = Mock()
        mock_engine.generate_batch.return_value = [
            "HARM_SCORE: 0.1\nJUSTIFICATION: ok"
        ] * (batch_size * 7)

        scorer = MultiDimensionalJuryScorer(mock_engine)
        samples = [("Q", "R")] * n_samples

        # Call score_samples_batch in two chunks of 10
        for i in range(0, n_samples, batch_size):
            scorer.score_samples_batch("m", samples[i:i + batch_size])

        assert mock_engine.generate_batch.call_count == n_samples // batch_size
```

**Step 2: Run test to verify it passes (logic test)**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/test_gb10_speedup.py::TestBatchScoringDispatch -v
```

Expected: PASS (tests the method directly; the wiring test will validate end-to-end).

---

**Step 3: Add `batch_size` parameter to `score_with_jury_batch()`**

In `scripts/run_full_vllm_evaluation.py`, change the function signature at line ~188:

```python
def score_with_jury_batch(
    engine: VLLMEngine,
    jury_member: str,
    instances: List[dict],
    responses: List[str],
    batch_size: int = 1,
    progress_callback=None
) -> List[Dict]:
```

Inside the function, after creating `scorer = MultiDimensionalJuryScorer(engine)` and `results = []`, replace the existing `for idx, (instance, response) in enumerate(...)` loop with:

```python
    if batch_size > 1:
        # Batched path: send batch_size × 7 prompts per generate_batch call
        for i in range(0, len(instances), batch_size):
            if shutdown_requested:
                logger.info("Shutdown requested during jury scoring")
                break

            chunk_inst = instances[i:i + batch_size]
            chunk_resp = responses[i:i + batch_size]
            samples = [
                (inst.get("question", ""), resp)
                for inst, resp in zip(chunk_inst, chunk_resp)
                if inst.get("question") and resp
            ]

            batch_scores = scorer.score_samples_batch(
                jury_member, samples,
                temperature=0.0,
                max_tokens=512,
            )

            for k, (instance, scores) in enumerate(zip(chunk_inst, batch_scores)):
                idx = i + k
                if scores is None:
                    results.append({
                        "instance_id": instance.get("id", f"instance_{idx}"),
                        "scores": {
                            dim: {"score": 0.0, "justification": "Parsing failed"}
                            for dim in all_dimensions
                        }
                    })
                else:
                    results.append({
                        "instance_id": instance.get("id", f"instance_{idx}"),
                        "scores": {
                            dim: {"score": ds.score,
                                  "justification": ds.justification}
                            for dim, ds in scores.items()
                        }
                    })

            current = min(i + batch_size, len(instances))
            if progress_callback:
                progress_callback(current, len(instances))

            if current % 100 == 0 or current == len(instances):
                elapsed = time.time() - start_time
                logger.info(
                    f"  Scored {current}/{len(instances)} instances "
                    f"({elapsed:.1f}s, {elapsed/max(current,1):.3f}s/instance)"
                )
    else:
        # Original per-sample path — unchanged, used by H100 native
        for idx, (instance, response) in enumerate(zip(instances, responses)):
            # ... keep existing loop body exactly as-is ...
```

Note: `all_dimensions` is already defined at the top of `score_with_jury_batch`.

---

**Step 4: Add `scoring_batch_size` param to `evaluate_dataset_full()` and thread it through**

Change the function signature (~line 440):

```python
def evaluate_dataset_full(
    dataset_name: str,
    engine: VLLMEngine,
    manager: ModelManager,
    jury_members: List[str],
    output_dir: Path,
    checkpoint_file: Path,
    checkpoint_interval: int = 100,
    num_samples: int = None,
    scoring_batch_size: int = 1,       # NEW
) -> bool:
```

In the jury scoring loop, pass `batch_size=scoring_batch_size` to `score_with_jury_batch()`:

```python
            member_results = score_with_jury_batch(
                engine,
                jury_member,
                instances_to_score,
                responses_to_score,
                batch_size=scoring_batch_size,    # NEW
                progress_callback=scoring_progress_callback
            )
```

---

**Step 5: Gate `scoring_batch_size` in `main()` — Docker gets 10, native gets 1**

In `main()` (~line 712), after `engine = ...` is created, add:

```python
        # GB10 Docker uses batched scoring; H100 native keeps per-sample path
        scoring_batch_size = 10 if args.engine == "docker" else 1
        logger.info(f"[Engine] scoring_batch_size={scoring_batch_size}")
```

Pass it to `evaluate_dataset_full()`:

```python
        success = evaluate_dataset_full(
            dataset_name=args.dataset,
            engine=engine,
            manager=manager,
            jury_members=jury_members,
            output_dir=output_dir,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=args.checkpoint_interval,
            num_samples=args.num_samples,
            scoring_batch_size=scoring_batch_size,    # NEW
        )
```

---

**Step 6: Run full test suite**

```bash
/home/puertao/.conda/envs/vllm-gemma/bin/pytest tests/ -v --ignore=tests/test_vllm_engine_integration.py
```

Expected: all pass.

**Step 7: Commit**

```bash
git add scripts/run_full_vllm_evaluation.py tests/test_gb10_speedup.py
git commit -m "feat: wire scoring_batch_size into evaluation pipeline, gate batch=10 for docker engine"
```

---

## Task 5: Smoke Test on GB10

Verify the full pipeline works end-to-end with the Docker engine and 5 samples.

**Files:**
- Run: `scripts/run_test_5_samples.py` (existing, no change)

---

**Step 1: Run smoke test with mock generator (no GPU needed for responses)**

```bash
CUDA_VISIBLE_DEVICES=0 /home/puertao/.conda/envs/vllm-gemma/bin/python \
  scripts/run_test_5_samples.py --mock-generator
```

**What to check in the output:**
- If gemma3-27b fails: look for `CRITICAL ... Container failed to start` (not silent 0.0)
- Container logs should appear in the output immediately before the CRITICAL line
- For working jurors: scoring proceeds normally
- No `Model 'X' not loaded` errors

---

**Step 2: Run isolated 1-sample Docker evaluation to trigger gemma3-27b diagnosis**

```bash
CUDA_VISIBLE_DEVICES=0 /home/puertao/.conda/envs/vllm-gemma/bin/python \
  scripts/run_full_vllm_evaluation.py --engine docker --dataset medqa --num_samples 1
```

**What to check:**
- Look for `Container logs for 'vllm-gemma3-27b-...'` in the log
- The internal vLLM error will tell you the root cause (OOM / CUDA / model path issue)
- Apply the fix in `config/vllm_jury_config.yaml` based on what the logs show

---

**Step 3: Verify H100 native path is untouched**

```bash
CUDA_VISIBLE_DEVICES=0,1 /home/puertao/.conda/envs/vllm-gemma/bin/python \
  scripts/run_full_vllm_evaluation.py --engine native --dataset medqa --num_samples 1
```

Expected: `scoring_batch_size=1` logged, behavior identical to before.

---

## Summary of Changes

| File | Change |
|---|---|
| `src/inference/vllm_engine.py` | Add `docker logs` capture before `_stop_container` on timeout |
| `src/evaluation/multi_dim_jury_v2.py` | Add `score_samples_batch()` — existing methods unchanged |
| `scripts/run_full_vllm_evaluation.py` | Fail-fast check on `load_jury_member`; `batch_size` param threading; `scoring_batch_size=10` for docker |
| `tests/test_gb10_speedup.py` | New test file covering all three changes |
| `config/vllm_jury_config.yaml` | (Post-diagnosis) fix gemma3-27b vllm_config flags |

**Not modified:** `vllm_engine_native.py`, H100 evaluation behavior, aggregation, parsing, existing tests.
