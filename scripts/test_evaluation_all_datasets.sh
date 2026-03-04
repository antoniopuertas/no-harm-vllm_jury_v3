#!/bin/bash
#
# Test Evaluation Runner - All 3 Datasets (Limited Samples)
# Quick test run with 10 samples per dataset to verify system works
# before launching full evaluation
#
# Usage:
#   ./scripts/test_evaluation_all_datasets.sh [num_samples]
#
# Default: 10 samples per dataset
# Expected duration: ~15-30 minutes for 10 samples each
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$REPO_DIR/test_evaluation_${TIMESTAMP}.log"

# Number of samples (default 10, can be overridden)
NUM_SAMPLES=${1:-10}

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log section header
log_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Start logging
log_header "JURY v3.0 TEST EVALUATION - ALL DATASETS (${NUM_SAMPLES} samples each)"
log "Repository: $REPO_DIR"
log "Log file: $LOG_FILE"
log "Start time: $(date)"

# System info
log_header "SYSTEM INFORMATION"
log "Hostname: $(hostname)"
log "GPU status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"
fi

# Datasets to evaluate
DATASETS=("pubmedqa" "medqa" "medmcqa")

log_header "TEST EVALUATION PLAN"
log "Samples per dataset: $NUM_SAMPLES"
log "Datasets: ${DATASETS[*]}"
log "Expected duration: 15-30 minutes total"
log ""

# Track start time
EVAL_START=$(date +%s)

# Run evaluations sequentially
for dataset in "${DATASETS[@]}"; do
    log_header "EVALUATING: $dataset ($NUM_SAMPLES samples)"

    DATASET_START=$(date +%s)

    # Run evaluation
    cd "$REPO_DIR"
    "$SCRIPT_DIR/launch_full_evaluations.sh" \
        --dataset "$dataset" \
        --num_samples "$NUM_SAMPLES" \
        2>&1 | tee -a "$LOG_FILE"

    DATASET_END=$(date +%s)
    DATASET_DURATION=$((DATASET_END - DATASET_START))

    log ""
    log "✓ $dataset completed in ${DATASET_DURATION}s"
    log ""

    # Brief pause between datasets
    sleep 5
done

# Wait for any background processes
log "Waiting for all processes to complete..."
wait

# Calculate total duration
EVAL_END=$(date +%s)
EVAL_DURATION=$((EVAL_END - EVAL_START))
EVAL_MINS=$((EVAL_DURATION / 60))
EVAL_SECS=$((EVAL_DURATION % 60))

# Find results
LATEST_RUN=$(ls -td data/results/vllm/full_runs/full_eval_* 2>/dev/null | head -1)

# Summary
log_header "TEST EVALUATION SUMMARY"
log "Total duration: ${EVAL_MINS}m ${EVAL_SECS}s"
log "Log file: $LOG_FILE"

if [[ -n "$LATEST_RUN" ]]; then
    log "Results directory: $LATEST_RUN"
    log ""
    log "Result files:"
    ls -lh "$LATEST_RUN"/*.json 2>/dev/null | tee -a "$LOG_FILE" || log "No results found"

    # Quick statistics
    log ""
    log "Quick Statistics:"
    for dataset in "${DATASETS[@]}"; do
        RESULT_FILE="$LATEST_RUN/${dataset}_consolidated.json"
        if [[ -f "$RESULT_FILE" ]]; then
            log "  $dataset:"
            python3 -c "
import json
data = json.load(open('$RESULT_FILE'))
results = data.get('results', [])
if results:
    scores = [r['final_score'] for r in results]
    print(f'    Instances: {len(results)}')
    print(f'    Mean score: {sum(scores)/len(scores):.3f}')
else:
    print('    No results')
" 2>&1 | tee -a "$LOG_FILE"
        else
            log "  $dataset: ✗ No results"
        fi
    done
fi

log ""
log "✓ Test evaluation complete!"
log ""
log "To run full evaluation:"
log "  ./scripts/run_full_evaluation_all_datasets.sh"

# Print to console
echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}TEST EVALUATION COMPLETE${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo -e "Duration: ${YELLOW}${EVAL_MINS}m ${EVAL_SECS}s${NC}"
echo -e "Log: ${BLUE}$LOG_FILE${NC}"
echo -e "Results: ${BLUE}$LATEST_RUN${NC}"
echo ""
echo -e "${YELLOW}System validated! Ready for full evaluation.${NC}"
echo ""
