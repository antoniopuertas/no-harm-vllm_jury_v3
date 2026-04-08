"""Tests for hybrid MultiDimensionalJuryScorer combining v2.3 + v3.0"""
import pytest
from unittest.mock import Mock, MagicMock
from src.evaluation.multi_dim_jury import MultiDimensionalJuryScorer
from src.parsing.model_profiles import clean_output


class TestMultiDimensionalJuryScorerInit:
    """Test initialization of hybrid scorer"""

    def test_init_with_vllm_engine(self):
        """Should initialize with VLLMEngine and v3.0 components"""
        # Arrange
        mock_engine = Mock()

        # Act
        scorer = MultiDimensionalJuryScorer(mock_engine)

        # Assert
        assert scorer.engine == mock_engine
        assert hasattr(scorer, 'extractor')
        assert hasattr(scorer, 'retry_logic')
        assert hasattr(scorer, 'reliability_tracker')
        assert hasattr(scorer, 'dimensions')


class TestPromptGeneration:
    """Test prompt generation for dimension scoring"""

    def test_generate_scoring_prompt_format(self):
        """Should generate v2.3 style prompt with dimension info"""
        # Arrange
        mock_engine = Mock()
        scorer = MultiDimensionalJuryScorer(mock_engine)
        question = "What causes diabetes?"
        response = "High blood sugar levels"
        dimension = "informational_harm"

        # Act
        prompt = scorer.generate_scoring_prompt(question, response, dimension)

        # Assert
        assert "medical safety expert" in prompt.lower()
        assert question in prompt
        assert response in prompt
        assert "HARM_SCORE:" in prompt
        assert "JUSTIFICATION:" in prompt
        assert "0.0" in prompt and "1.0" in prompt  # Score range


class TestScoreExtraction:
    """Test score extraction using v3.0 parsing"""

    def test_extract_dimension_score_success(self):
        """Should extract score using v3.0 ScoreExtractor"""
        # Arrange
        mock_engine = Mock()
        scorer = MultiDimensionalJuryScorer(mock_engine)
        raw_output = "HARM_SCORE: 0.3\nJUSTIFICATION: Minor concern about accuracy"
        model_name = "test-model"
        dimension = "informational"

        # Act
        result = scorer.extract_dimension_score(raw_output, model_name, dimension)

        # Assert
        assert result is not None
        assert result.dimension == dimension
        assert 0.0 <= result.score <= 1.0
        assert result.justification != ""

    def test_extract_dimension_score_with_thinking_tags(self):
        """Should strip thinking tags before extraction"""
        # Arrange
        mock_engine = Mock()
        scorer = MultiDimensionalJuryScorer(mock_engine)
        raw_output = "<think>Let me analyze...</think>HARM_SCORE: 0.5\nJUSTIFICATION: Moderate concern"
        model_name = "olmo-32b"  # Has thinking mode
        dimension = "informational"

        # Act
        result = scorer.extract_dimension_score(raw_output, model_name, dimension)

        # Assert
        assert result is not None
        assert "<think>" not in result.justification


class TestScoreAllDimensions:
    """Test scoring across all 7 dimensions"""

    def test_score_all_dimensions_success(self):
        """Should score all 7 dimensions for a single jury member"""
        # Arrange
        mock_engine = Mock()
        mock_engine.generate_batch = Mock(return_value=[
            "HARM_SCORE: 0.2\nJUSTIFICATION: Good",
        ] * 7)  # Mock response for each dimension

        scorer = MultiDimensionalJuryScorer(mock_engine)
        model_name = "test-model"
        question = "What causes fever?"
        response = "Infection or inflammation"

        # Act
        result = scorer.score_all_dimensions(model_name, question, response)

        # Assert
        assert result is not None
        assert result.jury_member == model_name
        assert len(result.dimension_scores) == 7
        assert all(0.0 <= ds.score <= 1.0 for ds in result.dimension_scores.values())


class TestRetryLogic:
    """Test retry logic when parsing fails"""

    def test_retry_on_parse_failure(self):
        """Should retry with simplified prompt when parse fails"""
        # Arrange
        mock_engine = Mock()
        # First call fails to parse, second succeeds
        mock_engine.generate_batch = Mock(side_effect=[
            ["unparseable garbage"],  # Primary fails
            ["3"],  # Retry 1 succeeds (0-10 scale)
        ])

        scorer = MultiDimensionalJuryScorer(mock_engine)
        model_name = "test-model"
        question = "Test question"
        response = "Test response"
        dimension = "informational_harm"

        # Act
        result = scorer.score_dimension_with_retry(
            model_name, question, response, dimension, max_retries=2
        )

        # Assert
        assert result is not None
        assert result.dimension == dimension
        assert 0.0 <= result.score <= 1.0


class TestDimensionScoreIsRetryField:
    """DimensionScore must carry an is_retry flag."""

    def test_dimension_score_has_is_retry_field(self):
        from src.evaluation.multi_dim_jury_v2 import DimensionScore
        score = DimensionScore(
            dimension="informational_harm",
            score=0.3,
            justification="ok"
        )
        assert hasattr(score, "is_retry")
        assert score.is_retry is False  # default

    def test_dimension_score_is_retry_true_when_set(self):
        from src.evaluation.multi_dim_jury_v2 import DimensionScore
        score = DimensionScore(
            dimension="informational_harm",
            score=0.3,
            justification="Retry 2: HIGH",
            is_retry=True
        )
        assert score.is_retry is True
