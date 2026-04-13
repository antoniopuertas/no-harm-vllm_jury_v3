#!/bin/bash
#
# Background simulation on GB10 using 4-juror config (no gemma3-27b).
# Use this while gemma3-27b download is in progress.
#
# Runs a smoke test (5 samples, mock generator) or a real eval run.
#
# Usage:
#   # Quick smoke test in background (mock responses, ~10 min):
#   bash scripts/run_bg_sim_no_gemma.sh --smoke
#
#   # Real eval, 50 samples, medmcqa (real inference, ~3 h):
#   bash scripts/run_bg_sim_no_gemma.sh --eval --dataset medmcqa --samples 50
#
#   # Real eval, 100 samples, pubmedqa:
#   bash scripts/run_bg_sim_no_gemma.sh --eval --dataset pubmedqa --samples 100
#
# Monitor:
#   tail -f logs/bg_sim_no_gemma_<timestamp>.log
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python)"
LOGS_DIR="$REPO_ROOT/logs"
CONFIG="$REPO_ROOT/config/vllm_jury_config_gb10_no_gemma.yaml"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGS_DIR/bg_sim_no_gemma_${TIMESTAMP}.log"

MODE=""
DATASET="pubmedqa"
SAMPLES=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            MODE="smoke"
            shift
            ;;
        --eval)
            MODE="eval"
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --smoke | --eval [--dataset pubmedqa|medqa|medmcqa] [--samples N]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: specify --smoke or --eval"
    echo "Usage: $0 --smoke | --eval [--dataset pubmedqa|medqa|medmcqa] [--samples N]"
    exit 1
fi

mkdir -p "$LOGS_DIR"

if [[ "$MODE" == "smoke" ]]; then
    echo "Launching smoke test (4-juror, mock generator) in background..."
    echo "Log: $LOG"
    nohup "$PYTHON" "$SCRIPT_DIR/run_test_5_samples.py" \
        --mock-generator \
        --config "$CONFIG" \
        > "$LOG" 2>&1 &
    PID=$!
    echo "PID: $PID"
    echo ""
    echo "Monitor: tail -f $LOG"
    echo "Stop:    kill $PID"
else
    OUTPUT_DIR="$REPO_ROOT/data/results/vllm/bg_sim_no_gemma"
    mkdir -p "$OUTPUT_DIR"
    echo "Launching eval (4-juror, $SAMPLES samples, $DATASET) in background..."
    echo "Output: $OUTPUT_DIR"
    echo "Log:    $LOG"
    nohup "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$DATASET" \
        --num_samples "$SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 10 \
        --engine docker \
        > "$LOG" 2>&1 &
    PID=$!
    echo "PID: $PID"
    echo ""
    echo "Monitor: tail -f $LOG"
    echo "Stop:    kill $PID"
fi
