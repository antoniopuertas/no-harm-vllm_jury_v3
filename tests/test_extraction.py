"""
Unit tests for ScoreExtractor.
"""

import pytest

from src.parsing.score_extractor import ScoreExtractor, ExtractionResult, ExtractionStrategy


class TestScoreExtractor:
    """Tests for the ScoreExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ScoreExtractor()

    def test_extract_direct_json_success(self):
        """Test extracting scores from direct JSON output."""
        raw_output = '{"informational": 0.3, "social": 0.1, "psychological": 0.2, "autonomy": 0.1, "economic": 0.0, "privacy": 0.0, "epistemic": 0.2}'
        result = self.extractor.extract(raw_output)

        assert result.success is True
        assert result.strategy == "DIRECT_JSON"
        assert result.scores is not None
        assert len(result.scores) == 7
        assert result.scores["informational"] == 0.3

    def test_extract_fenced_json_success(self):
        """Test extracting scores from fenced JSON block."""
        raw_output = '''Here is my analysis:

```json
{
    "informational": 0.5,
    "social": 0.2,
    "psychological": 0.3,
    "autonomy": 0.1,
    "economic": 0.0,
    "privacy": 0.1,
    "epistemic": 0.2
}
```

I hope this helps!'''
        result = self.extractor.extract(raw_output)

        assert result.success is True
        assert result.scores is not None
        assert len(result.scores) == 7

    def test_extract_regex_pairs_success(self):
        """Test extracting scores from free-form text with regex."""
        raw_output = '''informational: 0.5
social: 0.2
psychological: 0.3
autonomy: 0.1
economic: 0.0
privacy: 0.1
epistemic: 0.2'''
        result = self.extractor.extract(raw_output)

        assert result.success is True
        assert result.scores is not None
        assert len(result.scores) == 7

    def test_extract_line_by_line_success(self):
        """Test extracting scores from line-by-line format."""
        raw_output = '''analysis:
- informational: 0.5
- social: 0.2
- psychological: 0.3
- autonomy: 0.1
- economic: 0.0
- privacy: 0.1
- epistemic: 0.2'''
        result = self.extractor.extract(raw_output)

        # This should work if exactly 7 floats found
        assert result.success is True
        assert result.scores is not None
        assert len(result.scores) == 7

    def test_extract_failure_returns_none(self):
        """Test that failure returns None scores, not 0.5."""
        raw_output = "This is just random text with no scores in it."

        result = self.extractor.extract(raw_output)

        assert result.success is False
        assert result.scores is None
        assert result.strategy == ExtractionStrategy.NONE

    def test_extract_validates_score_range(self):
        """Test that scores outside [0,1] range are rejected."""
        raw_output = '{"informational": 1.5, "social": 0.2, "psychological": 0.3, "autonomy": 0.1, "economic": 0.0, "privacy": 0.1, "epistemic": 0.2}'
        result = self.extractor.extract(raw_output)

        # 1.5 is out of range, so validation should fail
        assert result.success is False
        assert result.scores is None

    def test_extract_validates_all_dimensions(self):
        """Test that missing dimensions cause failure."""
        raw_output = '{"informational": 0.5, "social": 0.2}'  # Only 2 dimensions
        result = self.extractor.extract(raw_output)

        assert result.success is False
        assert result.scores is None

    def test_extract_normalizes_keys(self):
        """Test that dimension keys are normalized."""
        raw_output = '{"Informational_Harm": 0.5, "Social Harm": 0.2, "Psychological": 0.3, "Autonomy": 0.1, "Economic": 0.0, "Privacy": 0.1, "Epistemic": 0.2}'
        result = self.extractor.extract(raw_output)

        assert result.success is True
        assert "informational" in result.scores
        assert "social" in result.scores


class TestRetryLogic:
    """Tests for the retry logic."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.parsing.retry_logic import RetryLogic

        self.retry_logic = RetryLogic(max_retries=2)

    def test_needs_retry_when_false(self):
        """Test that needs_retry returns False on success."""
        assert self.retry_logic.needs_retry(success=True) is False

    def test_needs_retry_allows_retry(self):
        """Test that needs_retry allows retry on failure."""
        assert self.retry_logic.needs_retry(success=False) is True

    def test_max_retries_limited(self):
        """Test that max retries is enforced."""
        self.retry_logic.history = [{"attempt": 1, "success": False}]

        assert self.retry_logic.needs_retry(success=False) is True

        self.retry_logic.history = [{"attempt": 1, "success": False}, {"attempt": 2, "success": False}]

        assert self.retry_logic.needs_retry(success=False) is False

    def test_retry_prompts_available(self):
        """Test that retry prompts are available."""
        prompt1 = self.retry_logic.get_next_prompt(
            question="Test question",
            response="Test response",
            attempt=1
        )
        assert "0 to 10" in prompt1

        prompt2 = self.retry_logic.get_next_prompt(
            question="Test question",
            response="Test response",
            attempt=2
        )
        assert "LOW" in prompt2 or "MEDIUM" in prompt2 or "HIGH" in prompt2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
