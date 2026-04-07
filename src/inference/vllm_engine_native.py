"""vLLM native inference engine for No-Harm evaluation framework.
   Loads models in-process using the vLLM Python library (no Docker).
   Use this engine on H100 servers where vLLM is installed natively.
"""
import logging
import gc
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
import torch

logger = logging.getLogger(__name__)


class NativeVLLMEngine:
    """vLLM inference engine with batch processing support (native, no Docker)"""

    def __init__(
        self,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        enable_cuda_graph: bool = True
    ):
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_cuda_graph = enable_cuda_graph
        self.models: Dict[str, LLM] = {}

        logger.info(
            f"[NativeVLLMEngine] Initialized with gpu_memory={gpu_memory_utilization}, "
            f"tensor_parallel={tensor_parallel_size}"
        )

    def load_model(
        self,
        model_name: str,
        hf_model_path: str,
        max_model_len: Optional[int] = None,
        **vllm_kwargs
    ) -> None:
        if model_name in self.models:
            logger.warning(f"[NativeVLLMEngine] Model {model_name} already loaded")
            return

        logger.info(f"[NativeVLLMEngine] Loading {model_name} from {hf_model_path}")

        vllm_config = {
            "model": hf_model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "disable_log_stats": True,
        }

        if max_model_len:
            vllm_config["max_model_len"] = max_model_len

        vllm_config.update(vllm_kwargs)

        try:
            llm = LLM(**vllm_config)
            self.models[model_name] = llm
            logger.info(f"[NativeVLLMEngine] Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"[NativeVLLMEngine] Failed to load {model_name}: {e}")
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
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        llm = self.models[model_name]
        logger.info(
            f"[NativeVLLMEngine] Generating {len(prompts)} responses with {model_name}"
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        try:
            tokenizer = llm.get_tokenizer()
            formatted_prompts = []
            for prompt in prompts:
                is_ministral = "ministral" in model_name.lower()
                if is_ministral:
                    chat_messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    chat_messages = [{"role": "user", "content": prompt}]

                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted_prompt)
                except Exception as e:
                    logger.warning(f"[NativeVLLMEngine] Chat template failed, using raw prompt: {e}")
                    formatted_prompts.append(prompt)

            outputs = llm.generate(formatted_prompts, sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            logger.info(f"[NativeVLLMEngine] Generated {len(responses)} responses")

            empty_count = sum(1 for r in responses if not r.strip())
            if empty_count > 0:
                logger.warning(f"[NativeVLLMEngine] {empty_count}/{len(responses)} responses are empty!")

            return responses
        except Exception as e:
            logger.error(f"[NativeVLLMEngine] Generation failed: {e}")
            raise

    def unload_model(self, model_name: str) -> None:
        if model_name not in self.models:
            logger.warning(f"[NativeVLLMEngine] Model {model_name} not loaded")
            return

        logger.info(f"[NativeVLLMEngine] Unloading {model_name}")
        del self.models[model_name]
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[NativeVLLMEngine] Successfully unloaded {model_name}")

    def get_loaded_models(self) -> List[str]:
        return list(self.models.keys())
