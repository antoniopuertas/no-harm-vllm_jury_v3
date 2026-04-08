"""Tests for run_full_vllm_evaluation.py helper functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Load the script as a module (it lives in scripts/, not src/)
_spec = importlib.util.spec_from_file_location(
    "run_eval",
    Path(__file__).parent.parent / "scripts" / "run_full_vllm_evaluation.py"
)
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


class TestEmptyResponseHandling:
    """score_with_jury_batch must use score=0.0 and set valid=False for empty responses."""

    def test_empty_response_scores_zero_not_half(self):
        """Empty response should produce score=0.0, not 0.5."""
        mock_engine = Mock()
        instances = [{"id": "test_0000", "question": "What is diabetes?"}]
        responses = [""]  # empty response

        results = run_eval.score_with_jury_batch(
            engine=mock_engine,
            jury_member="ministral-14b",
            instances=instances,
            responses=responses,
            batch_size=1,
        )

        for dim_data in results[0]["scores"].values():
            assert dim_data["score"] == 0.0, (
                f"Expected 0.0 for empty response, got {dim_data['score']}"
            )

    def test_empty_response_sets_valid_false(self):
        """Empty response result must carry valid=False."""
        mock_engine = Mock()
        instances = [{"id": "test_0000", "question": "What is diabetes?"}]
        responses = [""]

        results = run_eval.score_with_jury_batch(
            engine=mock_engine,
            jury_member="ministral-14b",
            instances=instances,
            responses=responses,
            batch_size=1,
        )

        assert results[0].get("valid") is False, (
            f"Expected valid=False for empty response, got: {results[0].get('valid')}"
        )

    def test_valid_response_sets_valid_true(self):
        """Normal response result must carry valid=True."""
        from src.evaluation.multi_dim_jury_v2 import DimensionScore
        mock_engine = Mock()
        mock_scorer = Mock()
        mock_scorer.score_response_batch.return_value = {
            dim: DimensionScore(dimension=dim, score=0.1, justification="ok")
            for dim in [
                "informational_harm", "psychological_harm", "social_harm",
                "economic_harm", "privacy_harm", "autonomy_harm", "epistemic_harm"
            ]
        }

        instances = [{"id": "test_0001", "question": "What is hypertension?"}]
        responses = ["Hypertension is high blood pressure."]

        with patch.object(run_eval, "MultiDimensionalJuryScorer", return_value=mock_scorer):
            results = run_eval.score_with_jury_batch(
                engine=mock_engine,
                jury_member="ministral-14b",
                instances=instances,
                responses=responses,
                batch_size=1,
            )

        assert results[0].get("valid") is True, (
            f"Expected valid=True for non-empty response, got: {results[0].get('valid')}"
        )
