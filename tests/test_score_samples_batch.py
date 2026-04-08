"""Tests for score_samples_batch() — Task 3 (GB10 speedup)."""
import pytest
from unittest.mock import Mock


def _ok_output(score=0.1):
    return f"HARM_SCORE: {score}\nJUSTIFICATION: test justification"


def _make_scorer(fake_outputs):
    from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer
    mock_engine = Mock()
    mock_engine.generate_batch.return_value = fake_outputs
    return MultiDimensionalJuryScorer(mock_engine)


class TestScoreSamplesBatch:
    """Verify score_samples_batch sends N×7 prompts in one generate_batch call."""

    def test_sends_n_times_7_prompts_in_one_call(self):
        """generate_batch must be called exactly once with N×7 prompts."""
        n_samples, n_dims = 4, 7
        scorer = _make_scorer([_ok_output()] * (n_samples * n_dims))

        scorer.score_samples_batch("test-model", [("Q", "R")] * n_samples)

        scorer.engine.generate_batch.assert_called_once()
        call_kwargs = scorer.engine.generate_batch.call_args[1]
        prompts = call_kwargs.get("prompts") or scorer.engine.generate_batch.call_args[0][1]
        assert len(prompts) == n_samples * n_dims

    def test_returns_one_result_per_sample(self):
        """Result list length must equal number of input samples."""
        n = 5
        scorer = _make_scorer([_ok_output()] * (n * 7))
        results = scorer.score_samples_batch("test-model", [("Q", "R")] * n)
        assert len(results) == n

    def test_each_result_has_all_7_dimensions(self):
        """Each non-None result must contain all 7 harm dimensions."""
        scorer = _make_scorer([_ok_output(0.3)] * 7)
        results = scorer.score_samples_batch("test-model", [("Q1", "R1")])
        assert results[0] is not None
        assert len(results[0]) == 7

    def test_scores_parsed_correctly(self):
        """Extracted scores must match the values in the fake outputs."""
        scorer = _make_scorer([_ok_output(0.42)] * 7)
        results = scorer.score_samples_batch("test-model", [("Q", "R")])
        assert results[0] is not None
        for dim_score in results[0].values():
            assert abs(dim_score.score - 0.42) < 1e-6

    def test_parse_failure_triggers_retry_per_dim(self):
        """If all dims fail to parse, score_dimension_with_retry is called for each."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer, DimensionScore
        mock_engine = Mock()
        mock_engine.generate_batch.return_value = ["INVALID OUTPUT"] * 7
        scorer = MultiDimensionalJuryScorer(mock_engine)
        scorer.score_dimension_with_retry = Mock(
            return_value=DimensionScore(dimension="x", score=0.5, justification="retry")
        )
        scorer.score_samples_batch("test-model", [("Q", "R")])
        assert scorer.score_dimension_with_retry.call_count == 7

    def test_generate_batch_exception_returns_none_list(self):
        """If generate_batch raises, all results must be None."""
        from src.evaluation.multi_dim_jury_v2 import MultiDimensionalJuryScorer
        mock_engine = Mock()
        mock_engine.generate_batch.side_effect = RuntimeError("server down")
        scorer = MultiDimensionalJuryScorer(mock_engine)
        results = scorer.score_samples_batch("test-model", [("Q1", "R1"), ("Q2", "R2")])
        assert results == [None, None]

    def test_existing_score_response_batch_unchanged(self):
        """score_response_batch must still work after adding score_samples_batch."""
        scorer = _make_scorer([_ok_output(0.2)] * 7)
        result = scorer.score_response_batch("test-model", "Q", "R")
        assert result is not None
        assert len(result) == 7

    def test_multiple_samples_independent_results(self):
        """Each sample must get its own result, in order."""
        n = 3
        # First sample scores 0.1, second 0.5, third 0.9
        outputs = [_ok_output(0.1)] * 7 + [_ok_output(0.5)] * 7 + [_ok_output(0.9)] * 7
        scorer = _make_scorer(outputs)
        results = scorer.score_samples_batch("test-model", [("Q", "R")] * n)
        assert len(results) == n
        assert all(r is not None for r in results)
