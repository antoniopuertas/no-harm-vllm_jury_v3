#!/bin/bash
#
# Run all 3 dataset evaluations with harm_dimensions_v2, 1000 samples each
#
# Usage:
#   bash scripts/run_harm_v2_1000.sh
#   bash scripts/run_harm_v2_1000.sh --dataset pubmedqa   # single dataset
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2"
CONFIG="$REPO_ROOT/config/vllm_jury_config.yaml"
PYTHON="/home/puertao/.conda/envs/vllm-gemma/bin/python"
LOGS_DIR="$REPO_ROOT/logs"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# Parse optional --dataset filter
DATASET_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset pubmedqa|medqa|medmcqa]"
            exit 1
            ;;
    esac
done

if [[ -n "$DATASET_FILTER" ]]; then
    DATASETS=("$DATASET_FILTER")
else
    DATASETS=("pubmedqa" "medqa" "medmcqa")
fi

echo "=============================================="
echo "harm_dimensions_v2 — 1000-sample evaluation"
echo "=============================================="
echo "Datasets : ${DATASETS[*]}"
echo "Samples  : 1000 each"
echo "Output   : $OUTPUT_DIR"
echo "Config   : $CONFIG"
echo "=============================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    LOG="$LOGS_DIR/harm_v2_${dataset}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching $dataset -> $LOG"

    nohup "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$dataset" \
        --num_samples 1000 \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 100 \
        > "$LOG" 2>&1 &

    echo "  PID $! | log: $LOG"
    echo ""
done

echo "All evaluations launched."
echo "Monitor with: tail -f $LOGS_DIR/harm_v2_*.log"
