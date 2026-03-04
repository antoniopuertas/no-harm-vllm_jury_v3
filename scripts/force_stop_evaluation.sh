#!/bin/bash
#
# Force stop stuck evaluation and clean up all processes
#
# Use this when the evaluation is stuck/hung (especially with multi-GPU)
#

echo "=========================================="
echo "Force Stop Evaluation"
echo "=========================================="
echo ""

# Find all related processes
MAIN_PID=$(ps aux | grep "python scripts/run_full_vllm_evaluation.py" | grep -v grep | awk '{print $2}')
VLLM_PIDS=$(ps aux | grep "VLLM::EngineCore" | grep -v grep | awk '{print $2}')
RAY_PIDS=$(ps aux | grep "ray::" | grep -v grep | awk '{print $2}')

if [ -z "$MAIN_PID" ] && [ -z "$VLLM_PIDS" ]; then
    echo "No evaluation processes found."
    exit 0
fi

echo "Found processes to stop:"
if [ -n "$MAIN_PID" ]; then
    echo "  Main process: $MAIN_PID"
fi
if [ -n "$VLLM_PIDS" ]; then
    echo "  vLLM processes: $VLLM_PIDS"
fi
if [ -n "$RAY_PIDS" ]; then
    echo "  Ray processes: $RAY_PIDS"
fi
echo ""

# Try graceful shutdown first
if [ -n "$MAIN_PID" ]; then
    echo "Attempting graceful shutdown (SIGINT)..."
    kill -SIGINT $MAIN_PID 2>/dev/null
    sleep 3
fi

# Check if still running
STILL_RUNNING=$(ps aux | grep -E "(run_full_vllm_evaluation|VLLM::EngineCore)" | grep -v grep)

if [ -n "$STILL_RUNNING" ]; then
    echo "Processes still running. Using SIGTERM..."

    # Kill vLLM processes
    if [ -n "$VLLM_PIDS" ]; then
        for pid in $VLLM_PIDS; do
            kill -SIGTERM $pid 2>/dev/null
        done
    fi

    # Kill main process
    if [ -n "$MAIN_PID" ]; then
        kill -SIGTERM $MAIN_PID 2>/dev/null
    fi

    sleep 3
fi

# Check again
STILL_RUNNING=$(ps aux | grep -E "(run_full_vllm_evaluation|VLLM::EngineCore)" | grep -v grep)

if [ -n "$STILL_RUNNING" ]; then
    echo "Processes still stuck. Using SIGKILL (force)..."

    # Force kill vLLM processes
    if [ -n "$VLLM_PIDS" ]; then
        for pid in $VLLM_PIDS; do
            kill -9 $pid 2>/dev/null
        done
    fi

    # Force kill main process
    if [ -n "$MAIN_PID" ]; then
        kill -9 $MAIN_PID 2>/dev/null
    fi

    sleep 2
fi

# Final check
REMAINING=$(ps aux | grep -E "(run_full_vllm_evaluation|VLLM::EngineCore)" | grep -v grep)

if [ -z "$REMAINING" ]; then
    echo ""
    echo "✓ All processes stopped successfully"
    echo ""
    echo "Checking GPU status..."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo ""
    echo "You can now restart the evaluation."
else
    echo ""
    echo "⚠ Some processes may still be running:"
    echo "$REMAINING"
    echo ""
    echo "You may need to manually kill them or wait for timeout."
fi

echo ""
echo "=========================================="
