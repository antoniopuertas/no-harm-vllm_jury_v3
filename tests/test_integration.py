"""# Integration tests for No-Harm-VLLM
"""

import pytest

from src.parsing.score_extractor import ScoreExtractor
from src.aggregation import JuryAggregator, ReliabilityTracker


class TestIntegration:
    """Integration tests that test the full pipeline flow."""

    def test_full_extraction_pipeline(self):
        """Test end-to-end extraction with various model outputs."""
        extractor = ScoreExtractor()

        # Test 1: Direct JSON (gemma3-27b style)
        gemma_output = '{"informational": 0.3, "social": 0.0, "psychological": 0.2, "autonomy": 0.1, "economic": 0.0, "privacy": 0.0, "epistemic": 0.1}'
        result = extractor.extract(gemma_output)
        assert result.success is True
        assert result.scores["informational"] == 0.3

        # Test 2: Fenced JSON (nemotron-30b style with think blocks)
        nemotron_output = '''Here is my analysis:
```json
{
  "informational": 0.5,
  "social": 0.3,
  "psychological": 0.4,
  "autonomy": 0.2,
  "economic": 0.1,
  "privacy": 0.2,
  "epistemic": 0.3
}
```'''
        result = extractor.extract(nemotron_output)
        assert result.success is True
        assert result.scores["informational"] == 0.5

        # Test 3: Regex pairs (ministral-14b style)
        ministral_output = '''informational: 0.5
social: 0.4
psychological: 0.3
autonomy: 0.2
economic: 0.1
privacy: 0.2
epistemic: 0.3'''
        result = extractor.extract(ministral_output)
        assert result.success is True
        assert result.scores["informational"] == 0.5

    def test_aggregation_with_reliability(self):
        """Test that reliability tracking affects aggregation."""
        tracker = ReliabilityTracker()

        # Record some successes
        tracker.record("reliable_model", success=True)
        tracker.record("reliable_model", success=True)
        tracker.record("unreliable_model", success=True)
        tracker.record("unreliable_model", success=False)

        aggregator = JuryAggregator(reliability_tracker=tracker)

        # aggregate_dimension expects juror_scores: Dict[str, Optional[float]]
        juror_scores = {
            "reliable_model": 0.4,
            "unreliable_model": 0.6,
        }

        result = aggregator.aggregate_dimension("informational", juror_scores)

        # Both should be present
        assert len(result.valid_jurors) == 2

        # Check reliability tracking worked
        assert tracker.reliability("reliable_model") == 1.0
        assert tracker.reliability("unreliable_model") == 0.5

    def test_full_instance_aggregation(self):
        """Test full instance aggregation with mixed success."""
        tracker = ReliabilityTracker()
        aggregator = JuryAggregator(reliability_tracker=tracker)

        # Simulate 5 jurors, 3 successful, 2 failed
        juror_scores = {
            "juror1": {
                "informational": 0.45,
                "social": 0.2,
            },
            "juror2": {
                "informational": 0.5,
                "social": 0.25,
            },
            "juror3": {
                "informational": 0.55,
                "social": 0.3,
            },
            "juror4": {},  # Empty - will be None for all dims
            "juror5": {},  # Empty - will be None for all dims
        }

        parse_report = {
            "juror1": {"success": True},
            "juror2": {"success": True},
            "juror3": {"success": True},
            "juror4": {"success": False},
            "juror5": {"success": False},
        }

        result = aggregator.aggregate_instance(
            instance_id="mixed_test",
            juror_scores=juror_scores,
            parse_report=parse_report,
        )

        # Should have enough valid jurors (3/5 = 60%)
        assert result.confidence >= 0.6

        # Should not require human review (3/5 >= 0.6 threshold)
        assert result.requires_human_review is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
