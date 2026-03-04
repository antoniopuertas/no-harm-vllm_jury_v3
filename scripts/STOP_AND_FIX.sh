#!/bin/bash
#
# Stop current hanging evaluation and show working alternatives
#

echo "=========================================="
echo "STOPPING HUNG EVALUATION"
echo "=========================================="
echo ""

# Get PIDs
MAIN_PID=$(ps aux | grep "python scripts/run_full_vllm_evaluation.py" | grep -v grep | awk '{print $2}')
VLLM_PIDS=$(ps aux | grep "VLLM::EngineCore" | grep -v grep | awk '{print $2}')

if [ -n "$MAIN_PID" ]; then
    echo "Killing main process: $MAIN_PID"
    kill -9 $MAIN_PID 2>/dev/null
fi

if [ -n "$VLLM_PIDS" ]; then
    echo "Killing vLLM processes: $VLLM_PIDS"
    for pid in $VLLM_PIDS; do
        kill -9 $pid 2>/dev/null
    done
fi

sleep 3

echo ""
echo "Checking if stopped..."
REMAINING=$(ps aux | grep -E "(run_full_vllm_evaluation|VLLM::EngineCore)" | grep -v grep)

if [ -z "$REMAINING" ]; then
    echo "✓ All processes stopped"
else
    echo "⚠ Some processes still running, may need manual cleanup"
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "=========================================="
echo "WORKING SOLUTIONS"
echo "=========================================="
echo ""
echo "OPTION 1: Single GPU (RELIABLE, works immediately)"
echo "---------------------------------------------------------"
echo "nohup python scripts/run_full_vllm_evaluation.py \\"
echo "    --dataset medmcqa \\"
echo "    --instances 1000 \\"
echo "    > logs/medmcqa_single_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo ""
echo "Time: ~50-60 minutes for 1000 instances"
echo ""
echo ""
echo "OPTION 2: Hybrid GPU (Only OLMo uses 2 GPUs, others single)"
echo "---------------------------------------------------------"
echo "nohup python scripts/run_full_vllm_evaluation.py \\"
echo "    --dataset medmcqa \\"
echo "    --instances 1000 \\"
echo "    --config config/vllm_jury_config_hybrid_gpu.yaml \\"
echo "    > logs/medmcqa_hybrid_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo ""
echo "Time: ~40-45 minutes (OLMo is 2x faster, others normal)"
echo ""
echo ""
echo "RECOMMENDED: Option 1 (Single GPU)"
echo "It's faster to run 1000 samples on single GPU than"
echo "waste time debugging tensor parallelism issues."
echo ""
echo "=========================================="
