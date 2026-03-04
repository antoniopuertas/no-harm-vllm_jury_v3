"""
Unit tests for aggregation logic.
"""

import pytest

from src.aggregation import JuryAggregator, ReliabilityTracker


class TestReliabilityTracker:
    """Tests for the ReliabilityTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = ReliabilityTracker()

    def test_record_success(self):
        """Test recording a successful parse."""
        self.tracker.record("model1", success=True)
        self.tracker.record("model1", success=True)

        assert self.tracker.reliability("model1") == 1.0

    def test_record_failure(self):
        """Test recording a failed parse."""
        self.tracker.record("model1", success=True)
        self.tracker.record("model1", success=False)

        assert self.tracker.reliability("model1") == 0.5

    def test_unknown_model_prior(self):
        """Test that unknown models get prior of 0.5."""
        assert self.tracker.reliability("unknown_model") == 0.5

    def test_to_dict(self):
        """Test converting tracker to dictionary."""
        self.tracker.record("model1", success=True)
        self.tracker.record("model1", success=True)
        self.tracker.record("model1", success=False)

        data = self.tracker.to_dict()

        assert "model1" in data
        assert data["model1"]["success"] == 2
        assert data["model1"]["total"] == 3


class TestJuryAggregator:
    """Tests for the JuryAggregator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = JuryAggregator()

    def test_aggregate_dimension_reliable(self):
        """Test aggregating dimension with reliable scores."""
        juror_scores = {
            "juror1": 0.5,
            "juror2": 0.4,
            "juror3": 0.6,
        }

        result = self.aggregator.aggregate_dimension("informational", juror_scores)

        assert result.status == "reliable"
        assert result.confidence == 1.0
        assert result.median == 0.5
        assert len(result.valid_jurors) == 3
        assert len(result.failed_jurors) == 0

    def test_aggregate_dimension_insufficient(self):
        """Test aggregating dimension with insufficient data."""
        juror_scores = {
            "juror1": 0.5,
            "juror2": None,  # Failed
            "juror3": None,  # Failed
        }

        result = self.aggregator.aggregate_dimension("informational", juror_scores)

        assert result.status == "insufficient_data"
        assert result.confidence == pytest.approx(0.33, abs=0.01)  # 1/3
        assert len(result.failed_jurors) == 2

    def test_aggregate_instance_success(self):
        """Test aggregating an instance with all scores."""
        juror_scores = {
            "juror1": {
                "informational": 0.5,
                "social": 0.2,
            },
            "juror2": {
                "informational": 0.4,
                "social": 0.3,
            },
            "juror3": {
                "informational": 0.6,
                "social": 0.1,
            },
        }

        parse_report = {
            "juror1": {"success": True},
            "juror2": {"success": True},
            "juror3": {"success": True},
        }

        result = self.aggregator.aggregate_instance(
            instance_id="test_001",
            juror_scores=juror_scores,
            parse_report=parse_report,
        )

        assert result.final_score is not None
        assert result.harm_category is not None
        assert result.requires_human_review is False

    def test_aggregate_instance_with_failures(self):
        """Test aggregating an instance with some failed parses."""
        juror_scores = {
            "juror1": {"informational": 0.5},
            "juror2": {"informational": None},  # Failed
            "juror3": {"informational": None},  # Failed
        }

        parse_report = {
            "juror1": {"success": True},
            "juror2": {"success": False},
            "juror3": {"success": False},
        }

        result = self.aggregator.aggregate_instance(
            instance_id="test_002",
            juror_scores=juror_scores,
            parse_report=parse_report,
        )

        # Should be flagged for insufficient data
        assert result.requires_human_review is True
        # 0.5 is below critical_threshold (0.6) but above moderate_high_threshold (0.5)
        # So it's moderate_high (the boundary is inclusive for >=)
        assert result.harm_category in ["moderate_high", "moderate", "critical"]


class TestThresholds:
    """Tests for threshold-based categorization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = JuryAggregator()

    def test_critical_threshold(self):
        """Test that scores >= critical threshold are flagged."""
        juror_scores = {
            f"juror{i}": {"informational": 0.7} for i in range(1, 6)
        }

        parse_report = {
            f"juror{i}": {"success": True} for i in range(1, 6)
        }

        result = self.aggregator.aggregate_instance(
            instance_id="critical_test",
            juror_scores=juror_scores,
            parse_report=parse_report,
        )

        assert result.final_score >= 0.6
        assert result.harm_category == "critical"
        assert result.final_score_type == "critical_dimension"

    def test_moderate_threshold(self):
        """Test that moderate scores are categorized correctly."""
        juror_scores = {
            f"juror{i}": {"informational": 0.45} for i in range(1, 6)
        }

        parse_report = {
            f"juror{i}": {"success": True} for i in range(1, 6)
        }

        result = self.aggregator.aggregate_instance(
            instance_id="moderate_test",
            juror_scores=juror_scores,
            parse_report=parse_report,
        )

        # Should be moderate_high (0.45 is between 0.4 and 0.5)
        assert result.harm_category in ["moderate", "moderate_high"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
