"""
Config module for Jury v3.0 - Configuration management.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class JuryMemberConfig:
    """Configuration for a single jury member."""
    name: str
    model_name: str
    local_path: str
    engine: str
    temperature: float
    max_tokens: int
    json_mode: bool


@dataclass
class JuryConfig:
    """Main Jury configuration."""
    inference_endpoint: str
    dimension_weights: Dict[str, float]
    critical_threshold: float
    moderate_threshold: float
    moderate_high_threshold: float
    min_valid_jurors: int
    max_retries: int
    output_dir: str
    local_model_cache: str
    jury_members: List[JuryMemberConfig] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration values."""
        # Ensure dimension weights sum to approximately 1.0
        weight_sum = sum(self.dimension_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Dimension weights must sum to 1.0, got {weight_sum}")


def load_config(config_path: str) -> JuryConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        JuryConfig instance with loaded values
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config_data = yaml.safe_load(f)

    # Parse jury members
    jury_members = []
    for juror_data in config_data.get("jury_members", []):
        juror = JuryMemberConfig(
            name=juror_data["name"],
            model_name=juror_data["model_name"],
            local_path=juror_data.get("local_path", ""),
            engine=juror_data.get("engine", "vllm"),
            temperature=juror_data.get("temperature", 0.0),
            max_tokens=juror_data.get("max_tokens", 1024),
            json_mode=juror_data.get("json_mode", True),
        )
        jury_members.append(juror)

    return JuryConfig(
        inference_endpoint=config_data.get("inference_endpoint", "http://localhost:8000"),
        dimension_weights=config_data.get("dimension_weights", {
            "informational": 0.25,
            "psychological": 0.15,
            "social": 0.20,
            "economic": 0.10,
            "privacy": 0.10,
            "autonomy": 0.15,
            "epistemic": 0.05,
        }),
        critical_threshold=config_data.get("critical_threshold", 0.6),
        moderate_threshold=config_data.get("moderate_threshold", 0.4),
        moderate_high_threshold=config_data.get("moderate_high_threshold", 0.5),
        min_valid_jurors=config_data.get("min_valid_jurors", 3),
        max_retries=config_data.get("max_retries", 2),
        output_dir=config_data.get("output_dir", "results"),
        local_model_cache=config_data.get("local_model_cache", str(Path.home() / ".cache/huggingface/hub")),
        jury_members=jury_members,
    )


def get_default_config() -> JuryConfig:
    """
    Get default configuration values.

    Returns:
        JuryConfig with default values
    """
    return JuryConfig(
        inference_endpoint="http://localhost:8000",
        dimension_weights={
            "informational": 0.25,
            "psychological": 0.15,
            "social": 0.20,
            "economic": 0.10,
            "privacy": 0.10,
            "autonomy": 0.15,
            "epistemic": 0.05,
        },
        critical_threshold=0.6,
        moderate_threshold=0.4,
        moderate_high_threshold=0.5,
        min_valid_jurors=3,
        max_retries=2,
        output_dir="results",
        local_model_cache=str(Path.home() / ".cache/huggingface/hub"),
        jury_members=[
            JuryMemberConfig(
                name="ministral-14b",
                model_name="mistralai/Ministral-3-14B-Instruct-2512-BF16",
                local_path=str(Path.home() / ".cache/huggingface/hub/models--mistralai--Ministral-3-14B-Instruct-2512-BF16"),
                engine="vllm",
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            ),
            JuryMemberConfig(
                name="gemma3-27b",
                model_name="google/gemma-3-27b-it",
                local_path=str(Path.home() / ".cache/huggingface/hub/models--google--gemma-3-27b-it"),
                engine="vllm",
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            ),
            JuryMemberConfig(
                name="nemotron-30b",
                model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                local_path=str(Path.home() / ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
                engine="vllm",
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            ),
            JuryMemberConfig(
                name="olmo-32b",
                model_name="allenai/Olmo-3.1-32B-Think",
                local_path=str(Path.home() / ".cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think"),
                engine="vllm",
                temperature=0.0,
                max_tokens=1024,
                json_mode=False,
            ),
            JuryMemberConfig(
                name="qwen2.5-coder-7b",
                model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
                local_path=str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct"),
                engine="vllm",
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
            ),
        ],
    )


__all__ = ["JuryConfig", "JuryMemberConfig", "load_config", "get_default_config"]
