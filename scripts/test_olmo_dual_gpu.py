#!/usr/bin/env python3
"""
Test script to verify OLMo-32B loads and runs with tensor_parallel_size=2

This will quickly test if OLMo can:
1. Load successfully with 2 GPUs
2. Initialize NCCL without hanging
3. Generate a test response

If this succeeds, you can use hybrid config safely.
If it hangs like Gemma3, stick to single GPU.
"""

import sys
import signal
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Test timed out!")

# Set 3-minute timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(180)

print("="*60)
print("OLMo-32B Dual-GPU Test")
print("="*60)
print()

try:
    print("Step 1: Loading OLMo-32B with tensor_parallel_size=2...")
    print("This is where Gemma3 hangs. If OLMo works, you'll see progress.")
    print()

    start_time = time.time()

    # Load with 2 GPUs
    llm = LLM(
        model="/nfs/staging/puertao/noharm/huggingface_cache/olmo-32b-think",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        disable_log_stats=True
    )

    load_time = time.time() - start_time
    print(f"✓ Model loaded successfully in {load_time:.1f}s")
    print()

    print("Step 2: Testing generation (5 samples)...")
    start_time = time.time()

    # Test prompts
    prompts = [
        "Answer this medical question: What is hypertension?",
        "Answer this medical question: What causes diabetes?",
        "Answer this medical question: What is pneumonia?",
        "Answer this medical question: What is anemia?",
        "Answer this medical question: What is asthma?"
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100
    )

    outputs = llm.generate(prompts, sampling_params)

    gen_time = time.time() - start_time
    print(f"✓ Generated {len(outputs)} responses in {gen_time:.1f}s")
    print(f"  Average: {gen_time/len(outputs):.2f}s per sample")
    print()

    # Show one sample response
    print("Sample output (first response):")
    print("-" * 60)
    print(outputs[0].outputs[0].text[:200] + "...")
    print("-" * 60)
    print()

    # Cancel timeout
    signal.alarm(0)

    print("="*60)
    print("✓ TEST PASSED!")
    print("="*60)
    print()
    print("OLMo-32B works successfully with 2 GPUs.")
    print("You can safely use config/vllm_jury_config_hybrid_gpu.yaml")
    print()
    print("Expected speedup for OLMo: ~2x")
    print("This will make the overall evaluation ~20-30% faster")
    print()

    sys.exit(0)

except TimeoutException:
    print()
    print("="*60)
    print("✗ TEST FAILED - TIMEOUT")
    print("="*60)
    print()
    print("OLMo-32B hung during loading/generation with 2 GPUs.")
    print("This is the same issue as Gemma3.")
    print()
    print("RECOMMENDATION: Use single-GPU config only")
    print()
    sys.exit(1)

except Exception as e:
    signal.alarm(0)
    print()
    print("="*60)
    print("✗ TEST FAILED - ERROR")
    print("="*60)
    print()
    print(f"Error: {e}")
    print()
    print("RECOMMENDATION: Use single-GPU config only")
    print()
    sys.exit(1)
