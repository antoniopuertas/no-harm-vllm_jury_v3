#!/bin/bash
#
# Background launcher: gemma3-27b scoring only + 5-juror merge
#
# This script runs ONLY gemma3-27b scoring against the already-completed
# 4-juror GB10_no_gemma results, then re-aggregates to 5-juror final scores.
#
# Prerequisites:
#   - GB10_no_gemma run is complete (all 3 datasets, jury_details.json present)
#   - gemma3-27b model is downloaded to /home/neo/.cache/huggingface/hub/gemma3-27b
#
# Usage:
#   # All 3 datasets (background):
#   nohup bash scripts/run_gemma_merge.sh > logs/gemma_merge_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Single dataset (background):
#   nohup bash scripts/run_gemma_merge.sh --dataset medqa > logs/gemma_merge_medqa.log 2>&1 &
#
# Monitor:
#   tail -f logs/gemma_merge_*.log
#
# Output:
#   data/results/vllm/harm_dimensions_v2/GB10_5juror/
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python)"
CONFIG="$REPO_ROOT/config/vllm_jury_config_gb10.yaml"
SOURCE_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/GB10_no_gemma"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/GB10_5juror"
DATASET_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_ARG="--dataset $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset pubmedqa|medqa|medmcqa]"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "================================================"
log "Gemma-Only Scoring + 5-Juror Merge"
log "Source : $SOURCE_DIR"
log "Output : $OUTPUT_DIR"
log "Config : $CONFIG"
log "Dataset: ${DATASET_ARG:-all 3}"
log "================================================"

# Verify source data exists
for ds in pubmedqa medqa medmcqa; do
    JD="$SOURCE_DIR/${ds}_full_results/jury_details.json"
    if [[ ! -f "$JD" ]]; then
        log "WARNING: Source file missing: $JD"
    else
        COUNT=$(python3 -c "import json; d=json.load(open('$JD')); print(len(d))" 2>/dev/null)
        log "  $ds: $COUNT instances found"
    fi
done

# Clean up orphaned vLLM containers
log "Checking for orphaned vLLM containers..."
VLLM_CONTAINERS=$(docker ps -a --filter "name=vllm-" --format "{{.Names}}" 2>/dev/null)
if [[ -n "$VLLM_CONTAINERS" ]]; then
    log "Removing: $VLLM_CONTAINERS"
    docker rm -f $VLLM_CONTAINERS 2>/dev/null
fi

START=$(date +%s)

"$PYTHON" "$SCRIPT_DIR/run_gemma_scoring_only.py" \
    $DATASET_ARG \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --batch_size 10

EXIT_CODE=$?
ELAPSED=$(( ($(date +%s) - START) / 60 ))

log "================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    log "Done in ${ELAPSED}m — results: $OUTPUT_DIR"
else
    log "FAILED after ${ELAPSED}m (exit=$EXIT_CODE)"
fi
log "================================================"

exit $EXIT_CODE
