#!/bin/bash
#
# Full harm_dimensions_v2 evaluation on H100 — all 3 datasets, 1000 samples each.
# Runs sequentially to avoid GPU OOM. Uses native vLLM (no Docker).
# Each dataset runs under nohup so SSH disconnects don't kill the job.
#
# Usage:
#   bash scripts/run_full_h100_evaluation.sh
#   bash scripts/run_full_h100_evaluation.sh --dataset pubmedqa   # single dataset
#
# Recommended launch (detach from terminal entirely):
#   nohup bash scripts/run_full_h100_evaluation.sh > logs/h100_launch.log 2>&1 &
#   echo "PID: $!"
#
# Monitor:
#   tail -f logs/h100_full_eval_<timestamp>.log          # main summary log
#   tail -f logs/h100_pubmedqa_<timestamp>.log           # per-dataset log
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/puertao/.conda/envs/vllm-gemma/bin/python"
CONFIG="$REPO_ROOT/config/vllm_jury_config.yaml"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2/H100"
LOGS_DIR="$REPO_ROOT/logs"
MAIN_LOG="$LOGS_DIR/h100_full_eval_$(date +%Y%m%d_%H%M%S).log"

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

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "================================================"
log "harm_dimensions_v2 — H100 full evaluation"
log "Engine   : native vLLM (no Docker)"
log "Datasets : ${DATASETS[*]}"
log "Samples  : 1000 each"
log "Output   : $OUTPUT_DIR"
log "Config   : $CONFIG"
log "Log      : $MAIN_LOG"
log "================================================"

TOTAL_START=$(date +%s)
FAILED=()

for dataset in "${DATASETS[@]}"; do
    log ""
    log ">>> Starting: $dataset"
    DATASET_START=$(date +%s)
    DATASET_LOG="$LOGS_DIR/h100_${dataset}_$(date +%Y%m%d_%H%M%S).log"
    log "    Log: $DATASET_LOG"

    nohup "$PYTHON" "$SCRIPT_DIR/run_full_vllm_evaluation.py" \
        --dataset "$dataset" \
        --num_samples 1000 \
        --output_dir "$OUTPUT_DIR" \
        --config "$CONFIG" \
        --checkpoint_interval 100 \
        --engine native \
        > "$DATASET_LOG" 2>&1 &

    DATASET_PID=$!
    log "    PID: $DATASET_PID"

    wait $DATASET_PID
    EXIT_CODE=$?

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
log "Results : $OUTPUT_DIR"
log "================================================"
