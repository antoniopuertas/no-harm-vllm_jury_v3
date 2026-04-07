#!/bin/bash
#
# Run all 3 dataset evaluations with harm_dimensions_v2, 1000 samples each
#
# Usage:
#   bash scripts/run_harm_v2_1000.sh --gpu H100
#   bash scripts/run_harm_v2_1000.sh --gpu H100 --dataset pubmedqa
#   bash scripts/run_harm_v2_1000.sh --gpu GB10
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/puertao/.conda/envs/vllm-gemma/bin/python"
LOGS_DIR="$REPO_ROOT/logs"
GPU_LABEL=""
DATASET_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_LABEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --gpu <H100|GB10> [--dataset pubmedqa|medqa|medmcqa]"
            exit 1
            ;;
    esac
done

if [[ -z "$GPU_LABEL" ]]; then
    echo "ERROR: --gpu flag is required. Use --gpu H100 or --gpu GB10"
    exit 1
fi

if [[ "$GPU_LABEL" == "GB10" ]]; then
    CONFIG="$REPO_ROOT/config/vllm_jury_config_gb10.yaml"
else
    CONFIG="$REPO_ROOT/config/vllm_jury_config.yaml"
fi

OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/$GPU_LABEL"

if [[ "$GPU_LABEL" == "GB10" ]]; then
    ENGINE_FLAG="--engine docker"
else
    ENGINE_FLAG="--engine native"
fi

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

if [[ -n "$DATASET_FILTER" ]]; then
    DATASETS=("$DATASET_FILTER")
else
    DATASETS=("pubmedqa" "medqa" "medmcqa")
fi

echo "=============================================="
echo "harm_dimensions_v2 — 1000-sample evaluation [$GPU_LABEL]"
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
        $ENGINE_FLAG \
        > "$LOG" 2>&1 &

    echo "  PID $! | log: $LOG"
    echo ""
done

echo "All evaluations launched."
echo "Monitor with: tail -f $LOGS_DIR/harm_v2_*.log"
