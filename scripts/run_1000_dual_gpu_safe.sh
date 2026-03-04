#!/bin/bash
#
# Run evaluations with DUAL GPU configuration with safer NCCL settings
#
# This version includes environment variables to avoid NCCL initialization hangs
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
echo "Dual-GPU Evaluation (SAFE MODE)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Instances: 1000"
echo "Config: Dual-GPU (tensor_parallel_size=2)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "NCCL Settings (to avoid hangs):"
echo "  - NCCL_DEBUG=WARN"
echo "  - NCCL_TIMEOUT=1800"
echo "  - NCCL_IB_DISABLE=0"
echo "  - VLLM_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/libnccl.so"
echo ""
echo "GPU Configuration:"
echo "  - OLMo-32B: 2 GPUs, batch_size=48"
echo "  - Nemotron-30B: 2 GPUs, batch_size=48"
echo "  - Gemma3-27B: 2 GPUs, batch_size=64"
echo "=========================================="
echo ""

# Set NCCL environment variables to avoid hangs
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker,lo
export CUDA_VISIBLE_DEVICES=0,1

# Optional: specify NCCL library path if needed
# export VLLM_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/libnccl.so

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
echo "Watch GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check status:"
echo "  ps aux | grep run_full_vllm_evaluation"
