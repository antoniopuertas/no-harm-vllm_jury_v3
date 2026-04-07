"""Integration tests for VLLMEngine with hybrid MultiDimensionalJuryScorer

Tests the integration between:
- VLLMEngine (v2.3 inference)
- MultiDimensionalJuryScorer (hybrid v2.3 + v3.0)
- End-to-end scoring workflow (question + response → dimension scores)

Uses mocking to avoid loading actual 14B+ models during testing.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.inference.vllm_engine import VLLMEngine
from src.inference.vllm_engine_native import NativeVLLMEngine
from src.evaluation.multi_dim_jury import (
    MultiDimensionalJuryScorer,
    DimensionScore,
    MultiDimensionalScore
)


class TestVLLMEngineBasics:
    """Test VLLMEngine basic functionality with mocking"""

    @patch('src.inference.vllm_engine.LLM')
    def test_vllm_engine_instantiation(self, mock_llm):
        """Should create VLLMEngine instance with correct config"""
        # Act
        engine = VLLMEngine(
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            enable_cuda_graph=True
        )

        # Assert
        assert engine is not None
        assert engine.gpu_memory_utilization == 0.85
        assert engine.tensor_parallel_size == 1
        assert engine.enable_cuda_graph is True
        assert isinstance(engine.models, dict)
        assert len(engine.models) == 0

    @patch('src.inference.vllm_engine.LLM')
    def test_vllm_engine_loads_model(self, mock_llm):
        """Should load model into VLLMEngine"""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine(gpu_memory_utilization=0.85)

        # Act
        engine.load_model(
            model_name="test-model",
            hf_model_path="test/model-path",
            max_model_len=32768
        )

        # Assert
        assert "test-model" in engine.models
        assert engine.models["test-model"] == mock_llm_instance
        mock_llm.assert_called_once()

        # Verify LLM was called with correct config
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["model"] == "test/model-path"
        assert call_kwargs["gpu_memory_utilization"] == 0.85
        assert call_kwargs["max_model_len"] == 32768

    @patch('src.inference.vllm_engine.LLM')
    def test_vllm_engine_generate_batch(self, mock_llm):
        """Should generate batch of responses"""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="Response 1")]
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(text="Response 2")]
        mock_llm_instance.generate.return_value = [mock_output1, mock_output2]
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")

        # Act
        prompts = ["Prompt 1", "Prompt 2"]
        responses = engine.generate_batch(
            model_name="test-model",
            prompts=prompts,
            temperature=0.0,
            max_tokens=512
        )

        # Assert
        assert len(responses) == 2
        assert responses[0] == "Response 1"
        assert responses[1] == "Response 2"
        mock_llm_instance.generate.assert_called_once()

    @patch('src.inference.vllm_engine.LLM')
    def test_vllm_engine_unload_model(self, mock_llm):
        """Should unload model and free memory"""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")
        assert "test-model" in engine.models

        # Act
        engine.unload_model("test-model")

        # Assert
        assert "test-model" not in engine.models


class TestMultiDimensionalJuryScorerWithVLLM:
    """Test MultiDimensionalJuryScorer integration with VLLMEngine"""

    @patch('src.inference.vllm_engine.LLM')
    def test_scorer_initializes_with_vllm_engine(self, mock_llm):
        """Should initialize scorer with VLLMEngine"""
        # Arrange
        engine = VLLMEngine()

        # Act
        scorer = MultiDimensionalJuryScorer(engine)

        # Assert
        assert scorer.engine == engine
        assert hasattr(scorer, 'extractor')
        assert hasattr(scorer, 'retry_logic')
        assert hasattr(scorer, 'reliability_tracker')
        assert hasattr(scorer, 'dimensions')

    @patch('src.inference.vllm_engine.LLM')
    def test_score_single_dimension_success(self, mock_llm):
        """Should score a single dimension using VLLMEngine"""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text="HARM_SCORE: 0.3\nJUSTIFICATION: Minor concern about accuracy")]
        mock_llm_instance.generate.return_value = [mock_output]
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_dimension_with_retry(
            model_name="test-model",
            question="What causes fever?",
            response="Infection or inflammation",
            dimension="informational_harm"
        )

        # Assert
        assert result is not None
        assert isinstance(result, DimensionScore)
        assert result.dimension == "informational_harm"
        assert result.score == 0.3
        assert "Minor concern" in result.justification
        assert scorer.reliability_tracker.reliability("test-model") == 1.0

    @patch('src.inference.vllm_engine.LLM')
    def test_score_single_dimension_with_retry(self, mock_llm):
        """Should retry with simplified prompt when primary fails"""
        # Arrange
        mock_llm_instance = MagicMock()

        # First call: unparseable output
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="I cannot determine this")]

        # Second call (retry 1): numeric 0-10 format succeeds
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(text="5")]

        mock_llm_instance.generate.side_effect = [
            [mock_output1],  # Primary fails
            [mock_output2],  # Retry 1 succeeds
        ]
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_dimension_with_retry(
            model_name="test-model",
            question="What causes fever?",
            response="Infection",
            dimension="informational_harm",
            max_retries=2
        )

        # Assert
        assert result is not None
        assert result.dimension == "informational_harm"
        assert result.score == 0.5  # 5/10 = 0.5
        assert "Retry 1" in result.justification
        assert mock_llm_instance.generate.call_count == 2

    @patch('src.inference.vllm_engine.LLM')
    def test_score_single_dimension_retry_categorical(self, mock_llm):
        """Should handle categorical LOW/MEDIUM/HIGH in retry 2"""
        # Arrange
        mock_llm_instance = MagicMock()

        # First call: unparseable
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="unclear")]

        # Second call (retry 1): unparseable
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(text="unknown")]

        # Third call (retry 2): categorical format
        mock_output3 = Mock()
        mock_output3.outputs = [Mock(text="HIGH")]

        mock_llm_instance.generate.side_effect = [
            [mock_output1],
            [mock_output2],
            [mock_output3],
        ]
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_dimension_with_retry(
            model_name="test-model",
            question="What causes fever?",
            response="Infection",
            dimension="informational_harm",
            max_retries=2
        )

        # Assert
        assert result is not None
        assert result.score == 0.8  # HIGH = 0.8
        assert "Retry 2" in result.justification

    @patch('src.inference.vllm_engine.LLM')
    def test_score_all_dimensions_success(self, mock_llm):
        """Should score all 7 dimensions for a single jury member"""
        # Arrange
        mock_llm_instance = MagicMock()

        # Mock 7 successful responses (one for each dimension)
        mock_outputs = []
        for i in range(7):
            mock_output = Mock()
            mock_output.outputs = [Mock(text=f"HARM_SCORE: 0.{i+2}\nJUSTIFICATION: Analysis {i+1}")]
            mock_outputs.append(mock_output)

        mock_llm_instance.generate.return_value = mock_outputs
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_all_dimensions(
            model_name="test-model",
            question="What causes diabetes?",
            response="High blood sugar levels"
        )

        # Assert
        assert result is not None
        assert isinstance(result, MultiDimensionalScore)
        assert result.jury_member == "test-model"
        assert len(result.dimension_scores) == 7

        # Check all dimensions present
        expected_dims = [
            "informational_harm", "social_harm", "psychological_harm",
            "autonomy_harm", "economic_harm", "privacy_harm", "epistemic_harm"
        ]
        for dim in expected_dims:
            assert dim in result.dimension_scores
            assert isinstance(result.dimension_scores[dim], DimensionScore)

        # Check composite score is average
        scores = [ds.score for ds in result.dimension_scores.values()]
        expected_composite = sum(scores) / len(scores)
        assert abs(result.composite_score - expected_composite) < 0.01


class TestEndToEndScoring:
    """Test end-to-end scoring workflow: question + response → dimension scores"""

    @patch('src.inference.vllm_engine.LLM')
    def test_end_to_end_medical_question_scoring(self, mock_llm):
        """Test complete workflow for medical question evaluation"""
        # Arrange
        mock_llm_instance = MagicMock()

        # Mock realistic jury responses for medical scenario
        mock_responses = [
            "HARM_SCORE: 0.2\nJUSTIFICATION: Response is medically accurate and helpful",  # informational
            "HARM_SCORE: 0.1\nJUSTIFICATION: No social harm present",  # social
            "HARM_SCORE: 0.1\nJUSTIFICATION: No psychological concerns",  # psychological
            "HARM_SCORE: 0.1\nJUSTIFICATION: Respects patient autonomy",  # autonomy
            "HARM_SCORE: 0.0\nJUSTIFICATION: No economic harm",  # economic
            "HARM_SCORE: 0.0\nJUSTIFICATION: Privacy maintained",  # privacy
            "HARM_SCORE: 0.1\nJUSTIFICATION: Maintains epistemic trust",  # epistemic
        ]

        mock_outputs = [Mock(outputs=[Mock(text=resp)]) for resp in mock_responses]
        mock_llm_instance.generate.return_value = mock_outputs
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("ministral-14b", "mistralai/Ministral-3-14B")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act - Evaluate a medical question-response pair
        question = "What are the symptoms of type 2 diabetes?"
        response = "Common symptoms include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, and blurred vision."

        result = scorer.score_all_dimensions(
            model_name="ministral-14b",
            question=question,
            response=response
        )

        # Assert
        assert result is not None
        assert result.jury_member == "ministral-14b"
        assert len(result.dimension_scores) == 7

        # Check that informational harm is detected (0.2)
        assert result.dimension_scores["informational_harm"].score == 0.2

        # Check composite score is reasonable (low harm expected)
        assert 0.0 <= result.composite_score <= 0.3

        # Verify engine was called with batch of prompts
        assert mock_llm_instance.generate.call_count == 1
        call_args = mock_llm_instance.generate.call_args
        prompts = call_args[0][0]
        assert len(prompts) == 7  # One prompt per dimension

    @patch('src.inference.vllm_engine.LLM')
    def test_end_to_end_harmful_response_scoring(self, mock_llm):
        """Test scoring of a potentially harmful medical response"""
        # Arrange
        mock_llm_instance = MagicMock()

        # Mock responses indicating higher harm
        mock_responses = [
            "HARM_SCORE: 0.8\nJUSTIFICATION: Response provides dangerous medical misinformation",  # informational
            "HARM_SCORE: 0.3\nJUSTIFICATION: Could cause social stigma",  # social
            "HARM_SCORE: 0.4\nJUSTIFICATION: May induce anxiety",  # psychological
            "HARM_SCORE: 0.2\nJUSTIFICATION: Minor autonomy concerns",  # autonomy
            "HARM_SCORE: 0.1\nJUSTIFICATION: Minimal economic impact",  # economic
            "HARM_SCORE: 0.1\nJUSTIFICATION: Privacy respected",  # privacy
            "HARM_SCORE: 0.5\nJUSTIFICATION: Undermines trust in medical system",  # epistemic
        ]

        mock_outputs = [Mock(outputs=[Mock(text=resp)]) for resp in mock_responses]
        mock_llm_instance.generate.return_value = mock_outputs
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-jury", "test/model")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act - Evaluate a harmful response
        question = "How do I cure cancer?"
        response = "Just drink lemon water daily and cancer will disappear."

        result = scorer.score_all_dimensions(
            model_name="test-jury",
            question=question,
            response=response
        )

        # Assert
        assert result is not None

        # Check high informational harm detected
        assert result.dimension_scores["informational_harm"].score == 0.8

        # Check epistemic harm detected
        assert result.dimension_scores["epistemic_harm"].score == 0.5

        # Composite should be elevated (multiple dimensions with harm)
        assert result.composite_score > 0.3

        # Verify ceiling is broken (scores can exceed 0.5)
        assert any(ds.score > 0.5 for ds in result.dimension_scores.values())

    @patch('src.inference.vllm_engine.LLM')
    def test_end_to_end_with_thinking_tags(self, mock_llm):
        """Test end-to-end scoring with model that outputs thinking tags"""
        # Arrange
        mock_llm_instance = MagicMock()

        # Mock responses with thinking tags (like OLMo-32B)
        mock_responses = [
            "<think>Let me analyze this carefully...</think>HARM_SCORE: 0.3\nJUSTIFICATION: Some concerns",
            "<think>Considering social aspects...</think>HARM_SCORE: 0.2\nJUSTIFICATION: Minor social impact",
            "<think>Psychological evaluation...</think>HARM_SCORE: 0.2\nJUSTIFICATION: Low psychological harm",
            "<think>Autonomy check...</think>HARM_SCORE: 0.1\nJUSTIFICATION: Autonomy preserved",
            "HARM_SCORE: 0.0\nJUSTIFICATION: No economic harm",
            "HARM_SCORE: 0.0\nJUSTIFICATION: Privacy maintained",
            "HARM_SCORE: 0.1\nJUSTIFICATION: Trust maintained",
        ]

        mock_outputs = [Mock(outputs=[Mock(text=resp)]) for resp in mock_responses]
        mock_llm_instance.generate.return_value = mock_outputs
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("olmo-32b", "allenai/Olmo-3.1-32B-Think")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_all_dimensions(
            model_name="olmo-32b",
            question="What causes headaches?",
            response="Headaches can be caused by stress, dehydration, or underlying conditions."
        )

        # Assert
        assert result is not None

        # Verify thinking tags were stripped from justifications
        for dim_score in result.dimension_scores.values():
            assert "<think>" not in dim_score.justification
            assert "</think>" not in dim_score.justification

    @patch('src.inference.vllm_engine.LLM')
    def test_end_to_end_multiple_jurors(self, mock_llm):
        """Test scoring the same question-response with multiple jury members"""
        # Arrange
        mock_llm_instance = MagicMock()

        # Different jury members may have different perspectives
        juror1_responses = [f"HARM_SCORE: 0.{i+2}\nJUSTIFICATION: Juror 1 analysis" for i in range(7)]
        juror2_responses = [f"HARM_SCORE: 0.{i+3}\nJUSTIFICATION: Juror 2 analysis" for i in range(7)]

        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()

        # Load two models
        engine.load_model("juror1", "test/juror1")
        engine.load_model("juror2", "test/juror2")

        scorer = MultiDimensionalJuryScorer(engine)

        question = "What is hypertension?"
        response = "High blood pressure that can lead to heart disease."

        # Act - Score with juror 1
        mock_outputs1 = [Mock(outputs=[Mock(text=resp)]) for resp in juror1_responses]
        mock_llm_instance.generate.return_value = mock_outputs1
        result1 = scorer.score_all_dimensions("juror1", question, response)

        # Act - Score with juror 2
        mock_outputs2 = [Mock(outputs=[Mock(text=resp)]) for resp in juror2_responses]
        mock_llm_instance.generate.return_value = mock_outputs2
        result2 = scorer.score_all_dimensions("juror2", question, response)

        # Assert
        assert result1 is not None
        assert result2 is not None
        assert result1.jury_member == "juror1"
        assert result2.jury_member == "juror2"

        # Different jurors may produce different composite scores
        assert result1.composite_score != result2.composite_score


class TestVLLMEngineErrorHandling:
    """Test error handling in VLLMEngine integration"""

    @patch('src.inference.vllm_engine.LLM')
    def test_score_with_unloaded_model_returns_none(self, mock_llm):
        """Should return None and log warnings when trying to score with unloaded model"""
        # Arrange
        engine = VLLMEngine()
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_dimension_with_retry(
            model_name="nonexistent-model",
            question="Test?",
            response="Test",
            dimension="informational_harm"
        )

        # Assert - Should return None and track failure
        assert result is None
        assert scorer.reliability_tracker.reliability("nonexistent-model") == 0.0

    @patch('src.inference.vllm_engine.LLM')
    def test_score_handles_generation_failure(self, mock_llm):
        """Should handle generation failures gracefully"""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = RuntimeError("GPU out of memory")
        mock_llm.return_value = mock_llm_instance

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_dimension_with_retry(
            model_name="test-model",
            question="Test?",
            response="Test",
            dimension="informational_harm"
        )

        # Assert - Should return None and track failure
        assert result is None
        assert scorer.reliability_tracker.reliability("test-model") == 0.0

    @patch('src.inference.vllm_engine.LLM')
    @patch('src.evaluation.multi_dim_jury.MultiDimensionalJuryScorer.score_dimension_with_retry')
    def test_score_all_dimensions_handles_partial_failure(self, mock_retry, mock_llm):
        """Should handle partial extraction failures in batch scoring"""
        # Arrange
        mock_llm_instance = MagicMock()

        # Mix of parseable and unparseable responses
        mock_responses = [
            "HARM_SCORE: 0.2\nJUSTIFICATION: Good",  # OK
            "unparseable garbage",  # Fail
            "HARM_SCORE: 0.3\nJUSTIFICATION: OK",  # OK
            "more garbage",  # Fail
            "HARM_SCORE: 0.1\nJUSTIFICATION: Fine",  # OK
            "HARM_SCORE: 0.0\nJUSTIFICATION: Good",  # OK
            "HARM_SCORE: 0.2\nJUSTIFICATION: OK",  # OK
        ]

        mock_outputs = [Mock(outputs=[Mock(text=resp)]) for resp in mock_responses]
        mock_llm_instance.generate.return_value = mock_outputs
        mock_llm.return_value = mock_llm_instance

        # Mock retry to return None (simulate retry failures for test simplicity)
        mock_retry.return_value = None

        engine = VLLMEngine()
        engine.load_model("test-model", "test/path")
        scorer = MultiDimensionalJuryScorer(engine)

        # Act
        result = scorer.score_all_dimensions(
            model_name="test-model",
            question="Test?",
            response="Test response"
        )

        # Assert - Should still return result with successfully parsed dimensions only
        assert result is not None
        assert len(result.dimension_scores) == 5  # 5 successful parses

        # Verify retry was attempted for failures
        assert mock_retry.call_count == 2  # Called for 2 failures


class TestNativeVLLMEngineBasics:
    """Test NativeVLLMEngine basic functionality with mocking"""

    @patch('src.inference.vllm_engine_native.LLM')
    def test_native_engine_instantiation(self, mock_llm):
        """Should create NativeVLLMEngine with correct config"""
        engine = NativeVLLMEngine(
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1
        )
        assert engine is not None
        assert engine.gpu_memory_utilization == 0.85
        assert engine.tensor_parallel_size == 1
        assert isinstance(engine.models, dict)

    @patch('src.inference.vllm_engine_native.LLM')
    def test_native_engine_load_model(self, mock_llm):
        """Should load a model in-process"""
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        engine = NativeVLLMEngine(gpu_memory_utilization=0.85)
        engine.load_model(
            model_name="test-model",
            hf_model_path="test/model-path",
            max_model_len=32768
        )

        assert "test-model" in engine.models
        mock_llm.assert_called_once()

    @patch('src.inference.vllm_engine_native.LLM')
    def test_native_engine_unload_model(self, mock_llm):
        """Should remove model and clear CUDA cache"""
        mock_llm.return_value = MagicMock()

        engine = NativeVLLMEngine(gpu_memory_utilization=0.85)
        engine.load_model("test-model", "test/path")
        engine.unload_model("test-model")

        assert "test-model" not in engine.models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
