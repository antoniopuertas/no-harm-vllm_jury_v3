#!/bin/bash
#
# Gracefully stop running evaluation and preserve checkpoint
#
# This sends SIGINT to the running evaluation process, which will:
# 1. Complete the current batch
# 2. Save a checkpoint
# 3. Exit cleanly
#

echo "=========================================="
echo "Stopping Running Evaluation"
echo "=========================================="

# Find running evaluation process
PID=$(ps aux | grep "python scripts/run_full_vllm_evaluation.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "No running evaluation found."
    exit 0
fi

echo "Found running evaluation (PID: $PID)"
echo ""
echo "Process details:"
ps aux | grep $PID | grep -v grep
echo ""
echo "This will:"
echo "  1. Send SIGINT signal (graceful shutdown)"
echo "  2. Allow current batch to complete"
echo "  3. Save checkpoint before exit"
echo ""
read -p "Proceed? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Sending SIGINT to PID $PID..."
    kill -SIGINT $PID
    echo ""
    echo "Signal sent. The process will:"
    echo "  - Complete current batch"
    echo "  - Save checkpoint"
    echo "  - Exit cleanly"
    echo ""
    echo "Monitor with: tail -f logs/medmcqa_*.log"
    echo "Or check if stopped: ps aux | grep run_full_vllm_evaluation"
else
    echo "Cancelled."
fi
