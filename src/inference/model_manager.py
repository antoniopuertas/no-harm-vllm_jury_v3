"""Model manager for handling jury member lifecycle"""
import logging
import yaml
from typing import Dict, Optional, List
from pathlib import Path
import torch

from .vllm_engine import VLLMEngine

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage model loading/unloading with memory tracking"""

    def __init__(
        self,
        vllm_engine: VLLMEngine,
        max_memory_gb: int,
        config_path: str = "config/vllm_jury_config.yaml"
    ):
        """
        Initialize model manager

        Args:
            vllm_engine: VLLMEngine instance
            max_memory_gb: Maximum GPU memory in GB
            config_path: Path to jury config YAML
        """
        self.engine = vllm_engine
        self.max_memory_gb = max_memory_gb
        self.config_path = config_path
        self.config = self._load_config()
        self.current_model: Optional[str] = None

        logger.info(
            f"[ModelManager] Initialized with max_memory={max_memory_gb}GB"
        )

    def _load_config(self) -> dict:
        """Load jury configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"[ModelManager] Loaded config from {self.config_path}")

            # Process jury members to use local_path if available
            for member in config.get('jury_members', []):
                # If local_path is specified, use it for hf_model
                if 'local_path' in member and Path(member['local_path']).exists():
                    logger.info(f"Using local path for {member['name']}: {member['local_path']}")
                    member['hf_model'] = member['local_path']

            return config
        except Exception as e:
            logger.error(f"[ModelManager] Failed to load config: {e}")
            raise

    def get_jury_member_config(self, jury_member_name: str) -> dict:
        """
        Get configuration for a jury member

        Args:
            jury_member_name: Name of jury member

        Returns:
            Configuration dict

        Raises:
            ValueError: If jury member not found
        """
        for member in self.config.get("jury_members", []):
            if member["name"] == jury_member_name:
                return member

        raise ValueError(f"Jury member '{jury_member_name}' not found in config")

    def load_jury_member(self, jury_member_name: str) -> bool:
        """
        Load jury member from config

        Args:
            jury_member_name: Name of jury member to load

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If jury member not found in config
        """
        # Get jury member config
        member_config = self.get_jury_member_config(jury_member_name)

        # Check if another model is loaded
        if self.current_model:
            logger.warning(
                f"[ModelManager] Model {self.current_model} already loaded, "
                "unloading first"
            )
            self.unload_current_model()

        # Load model
        try:
            hf_model = member_config["hf_model"]
            vllm_config = member_config.get("vllm_config", {})

            self.engine.load_model(
                model_name=jury_member_name,
                hf_model_path=hf_model,
                **vllm_config
            )

            self.current_model = jury_member_name
            logger.info(f"[ModelManager] Successfully loaded {jury_member_name}")
            return True

        except Exception as e:
            logger.error(f"[ModelManager] Failed to load {jury_member_name}: {e}")
            return False

    def unload_current_model(self) -> None:
        """Unload currently loaded model"""
        if not self.current_model:
            logger.warning("[ModelManager] No model currently loaded")
            return

        self.engine.unload_model(self.current_model)
        self.current_model = None

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage

        Returns:
            Dict with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - allocated
        }

    def get_all_jury_members(self) -> List[str]:
        """
        Get list of all jury member names from config

        Returns:
            List of jury member names
        """
        return [member["name"] for member in self.config.get("jury_members", [])]
