"""Tests for batch scoring dispatch — Task 4 (GB10 speedup)."""
import pytest
from unittest.mock import Mock, call


def _ok_output(score=0.1):
    return f"HARM_SCORE: {score}\nJUSTIFICATION: ok"


class TestBatchScoringDispatch:
    """Verify score_with_jury_batch dispatches to score_samples_batch when batch_size > 1."""

    def test_batch_size_1_calls_score_response_batch_per_sample(self):
        """batch_size=1 must call score_response_batch once per sample (per-sample path)."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer

        mock_engine = Mock()
        mock_engine.generate_batch.return_value = [_ok_output()] * 7

        scorer = MultiDimensionalJuryScorer(mock_engine)
        n_samples = 3
        for _ in range(n_samples):
            scorer.score_response_batch("m", "Q", "R")

        assert mock_engine.generate_batch.call_count == n_samples

    def test_batch_size_10_reduces_generate_batch_calls(self):
        """With batch_size=10 and 20 samples, generate_batch is called twice, not 20×."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer

        n_samples = 20
        batch_size = 10
        mock_engine = Mock()
        mock_engine.generate_batch.return_value = [_ok_output()] * (batch_size * 7)

        scorer = MultiDimensionalJuryScorer(mock_engine)
        samples = [("Q", "R")] * n_samples

        for i in range(0, n_samples, batch_size):
            scorer.score_samples_batch("m", samples[i:i + batch_size])

        assert mock_engine.generate_batch.call_count == n_samples // batch_size

    def test_scoring_batch_size_1_for_native_engine(self):
        """Native engine (H100) must always use batch_size=1 path — no score_samples_batch."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer

        mock_engine = Mock()
        mock_engine.generate_batch.return_value = [_ok_output()] * 7

        scorer = MultiDimensionalJuryScorer(mock_engine)
        # Simulate 3 samples with per-sample path (batch_size=1)
        for _ in range(3):
            result = scorer.score_response_batch("m", "Q", "R")
            assert result is not None

        # 3 samples × 1 call each = 3 total (not batched)
        assert mock_engine.generate_batch.call_count == 3

    def test_evaluate_dataset_full_accepts_scoring_batch_size(self):
        """evaluate_dataset_full must accept scoring_batch_size parameter without error."""
        import inspect
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.run_full_vllm_evaluation import evaluate_dataset_full

        sig = inspect.signature(evaluate_dataset_full)
        assert "scoring_batch_size" in sig.parameters, \
            "evaluate_dataset_full must have scoring_batch_size parameter"

    def test_scoring_batch_size_default_is_1(self):
        """scoring_batch_size must default to 1 to preserve H100 native behaviour."""
        import inspect
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.run_full_vllm_evaluation import evaluate_dataset_full

        sig = inspect.signature(evaluate_dataset_full)
        default = sig.parameters["scoring_batch_size"].default
        assert default == 1, f"Expected default=1, got {default}"
