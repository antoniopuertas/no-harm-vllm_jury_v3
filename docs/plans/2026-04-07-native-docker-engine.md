# Native vs Docker Engine Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the original native vLLM engine for H100 and wire `--gpu H100|GB10` in the bash scripts to automatically select the correct engine.

**Architecture:** Create `src/inference/vllm_engine_native.py` (restored from git `7a7d3ca`) as `NativeVLLMEngine`. Add `--engine native|docker` to `run_full_vllm_evaluation.py`. Bash scripts derive `--engine` from `$GPU_LABEL` automatically.

**Tech Stack:** Python, vLLM (`from vllm import LLM, SamplingParams`), Bash

---

### Task 1: Create `src/inference/vllm_engine_native.py`

**Files:**
- Create: `src/inference/vllm_engine_native.py`
- Test: `tests/test_vllm_engine_integration.py` (update import)

**Step 1: Write the failing test**

Add to `tests/test_vllm_engine_integration.py` at the top of the file, after the existing imports:

```python
from src.inference.vllm_engine_native import NativeVLLMEngine
```

And add a new test class:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3
python -m pytest tests/test_vllm_engine_integration.py::TestNativeVLLMEngineBasics -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError: No module named 'src.inference.vllm_engine_native'`

**Step 3: Create `src/inference/vllm_engine_native.py`**

This is restored verbatim from git commit `7a7d3ca`, with the class renamed from `VLLMEngine` to `NativeVLLMEngine`:

```python
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
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3
python -m pytest tests/test_vllm_engine_integration.py::TestNativeVLLMEngineBasics -v 2>&1 | tail -10
```
Expected: 3 tests PASSED

**Step 5: Commit**

```bash
git add src/inference/vllm_engine_native.py tests/test_vllm_engine_integration.py
git commit -m "feat: add NativeVLLMEngine for H100 in-process vLLM inference"
```

---

### Task 2: Add `--engine` argument to `run_full_vllm_evaluation.py`

**Files:**
- Modify: `scripts/run_full_vllm_evaluation.py:649-680` (argparse block)
- Modify: `scripts/run_full_vllm_evaluation.py:705-715` (engine instantiation)

**Step 1: Add `--engine` to the argparse block**

In `scripts/run_full_vllm_evaluation.py`, find the last `parser.add_argument` call (around line 674) and add after it:

```python
    parser.add_argument(
        "--engine",
        choices=["native", "docker"],
        default="docker",
        help="Inference engine: 'native' for H100 (vLLM in-process), 'docker' for GB10"
    )
```

**Step 2: Update engine instantiation**

Find this block (around line 705):
```python
    # Load model configs and initialize engine
    try:
        engine = VLLMEngine(
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1
        )
```

Replace with:
```python
    # Load model configs and initialize engine
    try:
        if args.engine == "native":
            from src.inference.vllm_engine_native import NativeVLLMEngine
            engine = NativeVLLMEngine(
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1
            )
            logger.info("[Engine] Using native vLLM engine (H100)")
        else:
            engine = VLLMEngine(
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1
            )
            logger.info("[Engine] Using Docker vLLM engine (GB10)")
```

**Step 3: Verify the argument is accepted**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3
python scripts/run_full_vllm_evaluation.py --help | grep engine
```
Expected output includes: `--engine {native,docker}`

**Step 4: Commit**

```bash
git add scripts/run_full_vllm_evaluation.py
git commit -m "feat: add --engine native|docker flag to run_full_vllm_evaluation.py"
```

---

### Task 3: Update bash scripts to pass `--engine` based on `$GPU_LABEL`

**Files:**
- Modify: `scripts/run_harm_v2_sequential.sh`
- Modify: `scripts/run_harm_v2_1000.sh`
- Modify: `scripts/run_medqa_medmcqa.sh`

The pattern is identical in all three scripts. After the `OUTPUT_DIR` is set and before the Python call, add:

```bash
if [[ "$GPU_LABEL" == "GB10" ]]; then
    ENGINE_FLAG="--engine docker"
else
    ENGINE_FLAG="--engine native"
fi
```

Then in each Python invocation, append `$ENGINE_FLAG`. Find the `"$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py"` call in each script and add `$ENGINE_FLAG` to it.

**`run_harm_v2_sequential.sh` — find:**
```bash
    "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$dataset" \
        --num_samples 1000 \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 100 \
```
Replace with:
```bash
    "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$dataset" \
        --num_samples 1000 \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 100 \
        $ENGINE_FLAG \
```

**`run_harm_v2_1000.sh` — find:**
```bash
        "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
            --dataset "$dataset" \
            --num_samples 1000 \
            --output_dir "$OUTPUT_DIR" \
            --config "$CONFIG" \
            --checkpoint_interval 100 \
```
Replace with:
```bash
        "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
            --dataset "$dataset" \
            --num_samples 1000 \
            --output_dir "$OUTPUT_DIR" \
            --config "$CONFIG" \
            --checkpoint_interval 100 \
            $ENGINE_FLAG \
```

**`run_medqa_medmcqa.sh` — find:**
```bash
    "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$dataset" \
        --num_samples 1000 \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 100 \
```
Replace with:
```bash
    "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$dataset" \
        --num_samples 1000 \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 100 \
        $ENGINE_FLAG \
```

**Verify each script passes the right engine flag:**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3
bash scripts/run_harm_v2_sequential.sh --gpu H100 2>&1 | grep -i "engine\|Engine" | head -3
```
Expected: `[Engine] Using native vLLM engine (H100)` (or the script exits with dataset error before that — either is fine, just confirm `--engine native` is being passed)

Simpler check — grep the modified scripts:
```bash
grep "ENGINE_FLAG\|engine" scripts/run_harm_v2_sequential.sh scripts/run_harm_v2_1000.sh scripts/run_medqa_medmcqa.sh
```
Expected: Each script shows `ENGINE_FLAG` set and passed to Python.

**Commit:**
```bash
git add scripts/run_harm_v2_sequential.sh scripts/run_harm_v2_1000.sh scripts/run_medqa_medmcqa.sh
git commit -m "feat: pass --engine flag to evaluation script based on --gpu selection"
```

---

### Task 4: Push to GitHub

**Step 1: Verify commits**

```bash
git log --oneline origin/master..HEAD
```
Expected: 3 new commits (Tasks 1–3) plus the design doc commit.

**Step 2: Push**

```bash
git push origin master
```

**Step 3: Verify**

```bash
git log --oneline -5
```
Expected: HEAD matches origin/master.
