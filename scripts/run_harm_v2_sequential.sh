#!/bin/bash
#
# Run harm_dimensions_v2 evaluations sequentially — 1000 samples per dataset
# Datasets run one after another to avoid GPU OOM.
#
# Usage:
#   bash scripts/run_harm_v2_sequential.sh
#
# Monitor:
#   tail -f logs/harm_v2_sequential.log
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python)"
GPU_LABEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_LABEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --gpu <H100|GB10>"
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
LOGS_DIR="$REPO_ROOT/logs"
MAIN_LOG="$LOGS_DIR/harm_v2_sequential_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

DATASETS=("pubmedqa" "medqa" "medmcqa")

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "================================================"
log "harm_dimensions_v2 — sequential evaluation [$GPU_LABEL]"
log "Datasets : ${DATASETS[*]}"
log "Samples  : 1000 each"
log "Output   : $OUTPUT_DIR"
log "Log      : $MAIN_LOG"
log "================================================"

TOTAL_START=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    log ""
    log ">>> Starting: $dataset"
    DATASET_START=$(date +%s)

    "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$dataset" \
        --num_samples 1000 \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 100 \
        2>&1 | tee -a "$MAIN_LOG"

    DATASET_END=$(date +%s)
    ELAPSED=$(( (DATASET_END - DATASET_START) / 60 ))
    log "<<< Finished: $dataset (${ELAPSED}m)"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

log ""
log "================================================"
log "All 3 evaluations complete (${TOTAL_ELAPSED}m total)"
log "Results: $OUTPUT_DIR"
log "================================================"
