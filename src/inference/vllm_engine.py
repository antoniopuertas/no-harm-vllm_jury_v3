"""vLLM inference engine for No-Harm evaluation framework"""
import logging
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
import torch

logger = logging.getLogger(__name__)


class VLLMEngine:
    """vLLM inference engine with batch processing support"""

    def __init__(
        self,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        enable_cuda_graph: bool = True
    ):
        """
        Initialize vLLM engine

        Args:
            gpu_memory_utilization: Fraction of GPU memory to use (0-1)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enable_cuda_graph: Enable CUDA graph optimization
        """
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_cuda_graph = enable_cuda_graph
        self.models: Dict[str, LLM] = {}

        logger.info(
            f"[VLLMEngine] Initialized with gpu_memory={gpu_memory_utilization}, "
            f"tensor_parallel={tensor_parallel_size}"
        )

    def load_model(
        self,
        model_name: str,
        hf_model_path: str,
        max_model_len: Optional[int] = None,
        **vllm_kwargs
    ) -> None:
        """
        Load model from HuggingFace into vLLM

        Args:
            model_name: Internal name for the model
            hf_model_path: HuggingFace model path (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
            max_model_len: Maximum sequence length (optional)
            **vllm_kwargs: Additional vLLM arguments
        """
        if model_name in self.models:
            logger.warning(f"[VLLMEngine] Model {model_name} already loaded")
            return

        logger.info(f"[VLLMEngine] Loading {model_name} from {hf_model_path}")

        # Build vLLM config
        vllm_config = {
            "model": hf_model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "disable_log_stats": True,
        }

        if max_model_len:
            vllm_config["max_model_len"] = max_model_len

        vllm_config.update(vllm_kwargs)

        # Load model
        try:
            llm = LLM(**vllm_config)
            self.models[model_name] = llm
            logger.info(f"[VLLMEngine] Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"[VLLMEngine] Failed to load {model_name}: {e}")
            raise

    def generate_batch(
        self,
        model_name: str,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for batch of prompts

        Args:
            model_name: Name of loaded model
            prompts: List of prompt strings
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional sampling parameters

        Returns:
            List of generated response strings
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        llm = self.models[model_name]
        logger.info(
            f"[VLLMEngine] Generating {len(prompts)} responses with {model_name}"
        )

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        # Generate batch
        try:
            # Apply chat template for instruct models
            # Get the tokenizer from the LLM
            tokenizer = llm.get_tokenizer()

            # Convert prompts to chat format using the model's chat template
            formatted_prompts = []
            for prompt in prompts:
                # For Ministral: Use chat template with explicit system message override
                # This prevents the default "Le Chat" system prompt
                is_ministral = "ministral" in model_name.lower()

                if is_ministral:
                    # Override system prompt for Ministral to prevent "Le Chat" conflict
                    chat_messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    chat_messages = [{"role": "user", "content": prompt}]

                # Apply chat template to convert to proper string format
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted_prompt)
                except Exception as e:
                    # Fallback to raw prompt if chat template fails
                    logger.warning(f"[VLLMEngine] Chat template failed, using raw prompt: {e}")
                    formatted_prompts.append(prompt)

            outputs = llm.generate(formatted_prompts, sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            logger.info(f"[VLLMEngine] Generated {len(responses)} responses")

            # Debug: Check if responses are empty
            empty_count = sum(1 for r in responses if not r.strip())
            if empty_count > 0:
                logger.warning(f"[VLLMEngine] {empty_count}/{len(responses)} responses are empty!")
                # Log details of first empty response
                for i, (output, resp) in enumerate(zip(outputs, responses)):
                    if not resp.strip():
                        logger.warning(f"[VLLMEngine] Empty response #{i}: prompt_len={len(prompts[i])}, "
                                     f"tokens_generated={len(output.outputs[0].token_ids)}, "
                                     f"finish_reason={output.outputs[0].finish_reason}")
                        logger.info(f"[VLLMEngine] Prompt sample: {prompts[i][:300]}...")
                        break

            return responses
        except Exception as e:
            logger.error(f"[VLLMEngine] Generation failed: {e}")
            raise

    def unload_model(self, model_name: str) -> None:
        """
        Unload model and free GPU memory

        Args:
            model_name: Name of model to unload
        """
        if model_name not in self.models:
            logger.warning(f"[VLLMEngine] Model {model_name} not loaded")
            return

        logger.info(f"[VLLMEngine] Unloading {model_name}")
        del self.models[model_name]

        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"[VLLMEngine] Successfully unloaded {model_name}")

    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded models

        Returns:
            List of model names
        """
        return list(self.models.keys())
