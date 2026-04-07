#!/bin/bash
#
# Run medqa and medmcqa evaluations sequentially (pubmedqa already complete).
# Cleans up any orphaned vLLM Docker containers before starting.
#
# Usage:
#   bash scripts/run_medqa_medmcqa.sh --gpu H100
#   bash scripts/run_medqa_medmcqa.sh --gpu GB10
#
# Monitor:
#   tail -f logs/harm_v2_sequential_<timestamp>.log
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python)"
LOGS_DIR="$REPO_ROOT/logs"
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
MAIN_LOG="$LOGS_DIR/harm_v2_sequential_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

DATASETS=("medqa" "medmcqa")

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "================================================"
log "harm_dimensions_v2 — medqa + medmcqa evaluation [$GPU_LABEL]"
log "Datasets : ${DATASETS[*]}"
log "Samples  : 1000 each"
log "Output   : $OUTPUT_DIR"
log "Log      : $MAIN_LOG"
log "================================================"

# --- Clean up orphaned vLLM containers ---
log ""
log "Checking for orphaned vLLM containers..."
VLLM_CONTAINERS=$(docker ps -a --filter "name=vllm-" --format "{{.Names}}" 2>/dev/null)
if [[ -n "$VLLM_CONTAINERS" ]]; then
    log "Found containers to remove: $VLLM_CONTAINERS"
    docker rm -f $VLLM_CONTAINERS 2>&1 | tee -a "$MAIN_LOG"
    log "Orphaned containers removed."
else
    log "No orphaned containers found."
fi

# --- Run datasets sequentially ---
TOTAL_START=$(date +%s)
FAILED=()

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

    EXIT_CODE=${PIPESTATUS[0]}
    DATASET_END=$(date +%s)
    ELAPSED=$(( (DATASET_END - DATASET_START) / 60 ))

    if [[ $EXIT_CODE -ne 0 ]]; then
        log "<<< FAILED: $dataset (${ELAPSED}m, exit code $EXIT_CODE)"
        FAILED+=("$dataset")
    else
        log "<<< Finished: $dataset (${ELAPSED}m)"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

log ""
log "================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    log "All evaluations complete (${TOTAL_ELAPSED}m total)"
else
    log "Completed with failures (${TOTAL_ELAPSED}m total)"
    log "Failed datasets: ${FAILED[*]}"
fi
log "Results: $OUTPUT_DIR"
log "================================================"
