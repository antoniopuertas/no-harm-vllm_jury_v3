#!/bin/bash
#
# Full 1000-sample evaluation on GB10 — 5-juror config (with gemma3-27b).
# Uses the fixed qwen2.5-coder-7b strip_patterns (retry storm resolved).
# Runs all 3 datasets sequentially.
#
# Usage:
#   nohup bash scripts/run_1000_gb10.sh > logs/gb10_5juror_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Single dataset:
#   nohup bash scripts/run_1000_gb10.sh --dataset medqa > logs/gb10_5juror_medqa.log 2>&1 &
#
# Monitor:
#   tail -f logs/gb10_5juror_*.log
#
# Note:
#   Use this for a fresh 5-juror run from scratch.
#   For merging gemma onto the existing 4-juror results, use run_gemma_merge.sh instead.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python)"
CONFIG="$REPO_ROOT/config/vllm_jury_config_gb10.yaml"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/GB10_5juror_fresh"
DATASET_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset pubmedqa|medqa|medmcqa] [--output_dir DIR]"
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
log "No-Harm-VLLM — 5-juror run (full, with gemma3-27b)"
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
