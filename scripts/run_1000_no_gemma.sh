#!/bin/bash
#
# Full 1000-sample evaluation on GB10 — 4-juror config (no gemma3-27b).
# Runs all 3 datasets sequentially.
#
# Usage:
#   nohup bash scripts/run_1000_no_gemma.sh > logs/no_gemma_1000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Single dataset:
#   nohup bash scripts/run_1000_no_gemma.sh --dataset medmcqa > logs/no_gemma_1000_medmcqa.log 2>&1 &
#
# Monitor:
#   tail -f logs/no_gemma_1000_*.log
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python)"
CONFIG="$REPO_ROOT/config/vllm_jury_config_gb10_no_gemma.yaml"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/GB10_no_gemma"
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

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "================================================"
log "No-Harm-VLLM — 4-juror run (no gemma3-27b)"
log "Datasets : ${DATASETS[*]}"
log "Samples  : 1000 each"
log "Config   : $CONFIG"
log "Output   : $OUTPUT_DIR"
log "================================================"

# Clean up orphaned vLLM containers
log "Checking for orphaned vLLM containers..."
VLLM_CONTAINERS=$(docker ps -a --filter "name=vllm-" --format "{{.Names}}" 2>/dev/null)
if [[ -n "$VLLM_CONTAINERS" ]]; then
    log "Removing: $VLLM_CONTAINERS"
    docker rm -f $VLLM_CONTAINERS 2>/dev/null
    log "Removed."
else
    log "None found."
fi

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
        --engine docker

    EXIT_CODE=$?
    ELAPSED=$(( ($(date +%s) - DATASET_START) / 60 ))

    if [[ $EXIT_CODE -ne 0 ]]; then
        log "<<< FAILED: $dataset (${ELAPSED}m, exit=$EXIT_CODE)"
        FAILED+=("$dataset")
    else
        log "<<< Done: $dataset (${ELAPSED}m)"
    fi
done

TOTAL_ELAPSED=$(( ($(date +%s) - TOTAL_START) / 60 ))

log ""
log "================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    log "All complete (${TOTAL_ELAPSED}m total)"
else
    log "Done with failures (${TOTAL_ELAPSED}m total) — failed: ${FAILED[*]}"
fi
log "Results: $OUTPUT_DIR"
log "================================================"
