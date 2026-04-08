"""Tests for container log capture on startup timeout — Task 1 (GB10 speedup)."""
import pytest
from unittest.mock import Mock, patch


class TestContainerLogCapture:
    """Verify container logs are captured before teardown on startup timeout."""

    @patch("src.inference.vllm_engine.subprocess.run")
    @patch("src.inference.vllm_engine._wait_for_server", return_value=False)
    def test_container_logs_captured_on_timeout(self, mock_wait, mock_run):
        """When server does not become ready, docker logs must be called before stop."""
        from src.inference.vllm_engine import VLLMEngine

        # docker run → success; docker logs → captured; docker stop → ok; docker rm → ok
        mock_run.side_effect = [
            Mock(returncode=0, stdout="abc123\n", stderr=""),
            Mock(returncode=0, stdout="ERROR: CUDA OOM\n", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
        ]

        engine = VLLMEngine()
        with pytest.raises(RuntimeError, match="did not become ready"):
            engine.load_model("gemma3-27b", "/models/gemma3-27b")

        calls = [str(c) for c in mock_run.call_args_list]
        assert any("logs" in c for c in calls), \
            "Expected 'docker logs <container>' call before teardown"

    @patch("src.inference.vllm_engine.subprocess.run")
    @patch("src.inference.vllm_engine._wait_for_server", return_value=False)
    def test_logs_captured_before_stop(self, mock_wait, mock_run):
        """docker logs must appear in call order BEFORE docker stop."""
        from src.inference.vllm_engine import VLLMEngine

        mock_run.side_effect = [
            Mock(returncode=0, stdout="cid\n", stderr=""),
            Mock(returncode=0, stdout="some log\n", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
        ]

        engine = VLLMEngine()
        with pytest.raises(RuntimeError):
            engine.load_model("test-model", "/models/test")

        call_args = [list(c.args[0]) for c in mock_run.call_args_list]
        commands = [args[1] if len(args) > 1 else args[0] for args in call_args]
        logs_idx = next((i for i, c in enumerate(commands) if "logs" in c), None)
        stop_idx = next((i for i, c in enumerate(commands) if "stop" in c), None)
        assert logs_idx is not None, "docker logs not called"
        assert stop_idx is not None, "docker stop not called"
        assert logs_idx < stop_idx, "docker logs must come before docker stop"
