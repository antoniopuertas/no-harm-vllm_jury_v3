#!/bin/bash
#
# Launch Full Dataset Evaluations Overnight
#
# This script launches full dataset evaluations for all 3 medical LLM datasets
# using vLLM. It runs evaluations sequentially (to manage GPU memory) and
# provides monitoring capabilities.
#
# Usage:
#   # Launch all 3 evaluations in background
#   ./launch_full_evaluations.sh
#
#   # Launch with custom output directory
#   ./launch_full_evaluations.sh --output_dir /custom/path
#
#   # Launch and immediately start monitoring
#   ./launch_full_evaluations.sh --monitor
#
#   # Show help
#   ./launch_full_evaluations.sh --help
#
# Notes:
# - Evaluations run sequentially due to GPU memory constraints (190GB total)
# - Each evaluation loads/unloads 5 models (Ministral-14B, Gemma3-27B, Nemotron-30B, OLMo-32B, Qwen2.5-7B)
# - Checkpointing enabled for resumability
# - Logs saved to logs/full_eval_TIMESTAMP.log
#
# Expected duration (estimated):
# - PubMedQA (1000 samples): ~2-3 hours
# - MedQA (1273 samples): ~3-4 hours
# - MedMCQA (4183 samples): ~10-14 hours
# - Total: ~15-21 hours (overnight should suffice)
#
# Author: Evaluation Framework v2.3
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
SCRIPTS_DIR="${SCRIPTS_DIR:-$SCRIPT_DIR}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/data/results/vllm/full_runs}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/logs}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/vllm_jury_config.yaml}"

# Parse arguments
OUTPUT_DIR=""
MONITOR_MODE=false
DATASET_FILTER=""
NUM_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --monitor|-m)
            MONITOR_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output_dir DIR     Output directory for results (default: $RESULTS_DIR)"
            echo "  --dataset NAME       Launch specific dataset only (pubmedqa, medqa, medmcqa)"
            echo "  --num_samples N      Number of samples to evaluate (default: full dataset)"
            echo "  --monitor            Start monitoring after launch"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                  # Launch all evaluations (full)"
            echo "  $0 --monitor                        # Launch and start monitoring"
            echo "  $0 --dataset pubmedqa               # Launch only PubMedQA (full)"
            echo "  $0 --dataset pubmedqa --num_samples 10  # Test with 10 samples"
            echo "  $0 --output_dir /custom/path        # Use custom output directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set output directory
if [[ -n "$OUTPUT_DIR" ]]; then
    RESULTS_DIR="$OUTPUT_DIR"
fi

# Ensure directories exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Create unique run ID based on timestamp
RUN_ID="full_eval_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "=============================================="
echo "Full Dataset Evaluation Launcher"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Output Directory: $RUN_DIR"
echo "Config: $CONFIG_PATH"
echo "Timestamp: $(date)"
echo "=============================================="

# Check GPU availability
echo ""
echo "Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU info not available"
else
    echo "WARNING: nvidia-smi not found - GPU processing may not be available"
fi

# Get list of datasets to evaluate
if [[ -n "$DATASET_FILTER" ]]; then
    DATASETS=("$DATASET_FILTER")
else
    DATASETS=("pubmedqa" "medqa" "medmcqa")
fi

echo ""
echo "Datasets to evaluate: ${DATASETS[*]}"
echo ""

# Function to launch evaluation
launch_evaluation() {
    local dataset=$1
    local log_file="$LOGS_DIR/full_eval_${dataset}_$(date +%Y%m%d_%H%M%S).log"

    echo "=============================================="
    echo "Launching evaluation: $dataset"
    echo "Log file: $log_file"
    echo "=============================================="

    # Build command with optional num_samples
    local cmd="nohup python3 $SCRIPTS_DIR/run_full_vllm_evaluation.py"
    cmd="$cmd --dataset $dataset"
    cmd="$cmd --output_dir $RUN_DIR"
    cmd="$cmd --config $CONFIG_PATH"
    cmd="$cmd --checkpoint_interval 50"
    if [[ -n "$NUM_SAMPLES" ]]; then
        cmd="$cmd --num_samples $NUM_SAMPLES"
        echo "Sample limit: $NUM_SAMPLES"
    fi

    # Launch in background with nohup to survive SSH disconnection
    eval "$cmd > '$log_file' 2>&1 &"

    local pid=$!
    echo "Launched with PID: $pid"
    echo "Run: $cmd"
    echo ""

    # Save process info
    echo "$pid|$dataset|$log_file|$(date -Iseconds)" >> "$RUN_DIR/processes.txt"

    return $pid
}

# Launch evaluations sequentially
echo "Starting evaluations..."
echo ""

for dataset in "${DATASETS[@]}"; do
    launch_evaluation "$dataset"

    # Wait a moment between launches to let processes stabilize
    sleep 2
done

echo ""
echo "=============================================="
echo "All evaluations launched!"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Output: $RUN_DIR"
echo ""
echo "Process info saved to: $RUN_DIR/processes.txt"
echo "Logs: $LOGS_DIR/full_eval_*.log"
echo ""

# Display quick status
echo "Quick Status:"
echo "-------------"
cat "$RUN_DIR/processes.txt" 2>/dev/null || echo "No processes file yet"

echo ""
echo "To monitor progress:"
echo "  $SCRIPTS_DIR/monitor_evaluations.sh"
echo ""
echo "To watch in real-time:"
echo "  $SCRIPTS_DIR/monitor_evaluations.sh --watch"
echo ""

# Start monitoring if requested
if [[ "$MONITOR_MODE" == "true" ]]; then
    echo "Starting monitor..."
    echo "Press Ctrl+C to exit monitor (evaluations will continue running)"
    echo ""
    sleep 2
    $SCRIPTS_DIR/monitor_evaluations.sh --watch
fi

echo ""
echo "Evaluation launched successfully!"
echo "Check logs and progress regularly."
