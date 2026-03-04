#!/bin/bash
#
# Complete restart: Stop stuck evaluation and restart with safe settings
#
# Usage: bash scripts/restart_evaluation.sh medmcqa
#

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "Example: $0 medmcqa"
    exit 1
fi

DATASET=$1

echo "=========================================="
echo "Complete Restart Procedure"
echo "=========================================="
echo ""

# Step 1: Force stop
echo "Step 1: Stopping all evaluation processes..."
bash scripts/force_stop_evaluation.sh

# Step 2: Wait for cleanup
echo ""
echo "Step 2: Waiting for GPU memory cleanup..."
sleep 5

# Step 3: Check GPU status
echo ""
echo "Step 3: Checking GPU status..."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Step 4: Restart
echo ""
echo "Step 4: Restarting evaluation with safe dual-GPU settings..."
sleep 2
bash scripts/run_1000_dual_gpu_safe.sh "$DATASET"

echo ""
echo "=========================================="
echo "Restart complete!"
echo "=========================================="
