#!/bin/bash
#
# Run evaluations with DUAL GPU configuration (2x faster)
#
# This script uses tensor parallelism to split large models across both H100 GPUs
# Expected speedup: ~2x for OLMo, Nemotron, and Gemma models
#

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset>"
    echo ""
    echo "Datasets: pubmedqa, medqa, medmcqa"
    echo ""
    echo "Examples:"
    echo "  $0 medmcqa"
    echo "  $0 pubmedqa"
    exit 1
fi

DATASET=$1
BASE_OUTPUT_DIR="/home/puertao/llm/no-harm-vllm_jury_v3/data/results/vllm"
CONFIG="/home/puertao/llm/no-harm-vllm_jury_v3/config/vllm_jury_config_dual_gpu.yaml"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET}_1000_dual_gpu"

echo "=========================================="
echo "Dual-GPU Evaluation (2x Faster)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Instances: 1000"
echo "Config: Dual-GPU (tensor_parallel_size=2)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "GPU Configuration:"
echo "  - OLMo-32B: 2 GPUs, batch_size=48"
echo "  - Nemotron-30B: 2 GPUs, batch_size=48"
echo "  - Gemma3-27B: 2 GPUs, batch_size=64"
echo "=========================================="
echo ""

# Run with dual-GPU config
nohup python scripts/run_full_vllm_evaluation.py \
    --dataset "$DATASET" \
    --instances 1000 \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --checkpoint_interval 100 \
    > logs/${DATASET}_dual_gpu_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "Started evaluation in background (PID: $PID)"
echo ""
echo "Monitor with:"
echo "  tail -f logs/${DATASET}_dual_gpu_*.log"
echo ""
echo "Check status:"
echo "  ps aux | grep run_full_vllm_evaluation"
