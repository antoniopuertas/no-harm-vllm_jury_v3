#!/bin/bash
#
# Replace qwen scores with retry-storm-fixed run and re-aggregate.
#
# Prerequisites:
#   - pubmedqa: GB10_5juror_fresh/ must be complete (all 5 jurors done)
#   - medqa/medmcqa: GB10_5juror/ must be complete (from gemma merge run)
#
# Usage:
#   # All 3 datasets (background):
#   nohup bash scripts/run_qwen_fix.sh > logs/qwen_fix_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Single dataset:
#   nohup bash scripts/run_qwen_fix.sh --dataset medqa > logs/qwen_fix_medqa.log 2>&1 &
#
# Output: data/results/vllm/harm_dimensions_v2/GB10_final/
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python)"
CONFIG="$REPO_ROOT/config/vllm_jury_config_gb10.yaml"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/GB10_final"
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
log "Qwen Fix — replace retry-storm scores + re-aggregate"
log "Output : $OUTPUT_DIR"
log "Dataset: ${DATASET_ARG:-all 3}"
log "================================================"

# Verify source data
for ds in pubmedqa medqa medmcqa; do
    if [[ "$ds" == "pubmedqa" ]]; then
        SRC="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/GB10_5juror_fresh"
    else
        SRC="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/GB10_5juror"
    fi
    JD="$SRC/${ds}_full_results/jury_details.json"
    if [[ -f "$JD" ]]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('$JD'))))" 2>/dev/null)
        log "  $ds: $COUNT instances found in $(basename $SRC)"
    else
        log "  WARNING: $ds source missing: $JD"
    fi
done

# Clean up orphaned vLLM containers
VLLM_CONTAINERS=$(docker ps -a --filter "name=vllm-" --format "{{.Names}}" 2>/dev/null)
if [[ -n "$VLLM_CONTAINERS" ]]; then
    log "Removing orphaned containers: $VLLM_CONTAINERS"
    docker rm -f $VLLM_CONTAINERS 2>/dev/null
fi

START=$(date +%s)

"$PYTHON" "$SCRIPT_DIR/run_qwen_scoring_only.py" \
    $DATASET_ARG \
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
