#!/bin/bash
#
# Full Evaluation Runner - All 3 Datasets
# Runs complete evaluation on PubMedQA, MedQA, and MedMCQA
# Saves comprehensive log to repository
#
# Usage:
#   ./scripts/run_full_evaluation_all_datasets.sh
#
# Expected duration: ~15-21 hours for full datasets
# Output: Timestamped log file in the repository root
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$REPO_DIR/full_evaluation_${TIMESTAMP}.log"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
log_header "No-Harm-VLLM FULL EVALUATION - ALL DATASETS"
log "Repository: $REPO_DIR"
log "Log file: $LOG_FILE"
log "Start time: $(date)"

# System info
log_header "SYSTEM INFORMATION"
log "Hostname: $(hostname)"
log "User: $(whoami)"
log "Python: $(python3 --version)"
log "Working directory: $(pwd)"

# GPU info
log_header "GPU INFORMATION"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"
else
    log "WARNING: nvidia-smi not found"
fi

# Environment info
log_header "ENVIRONMENT"
log "Conda environment: ${CONDA_DEFAULT_ENV:-none}"
log "Virtual environment: ${VIRTUAL_ENV:-none}"

# Configuration check
log_header "CONFIGURATION CHECK"
if [[ -f "$REPO_DIR/config/vllm_jury_config.yaml" ]]; then
    log "✓ Config file found: config/vllm_jury_config.yaml"
    log "Jury members configured:"
    grep "name:" "$REPO_DIR/config/vllm_jury_config.yaml" | head -5 | tee -a "$LOG_FILE"
else
    log "✗ ERROR: Config file not found!"
    exit 1
fi

# Datasets to evaluate
DATASETS=("pubmedqa" "medqa" "medmcqa")
EXPECTED_DURATIONS=("2-3 hours" "3-4 hours" "10-14 hours")

log_header "EVALUATION PLAN"
log "Datasets to evaluate: ${DATASETS[*]}"
log ""
for i in "${!DATASETS[@]}"; do
    log "  ${DATASETS[$i]}: Expected duration ${EXPECTED_DURATIONS[$i]}"
done
log ""
log "Total expected duration: 15-21 hours"
log "Evaluation will run sequentially (GPU memory management)"

# Confirmation
echo ""
echo -e "${YELLOW}WARNING: This will run evaluations for approximately 15-21 hours.${NC}"
echo -e "${YELLOW}Press Ctrl+C within 10 seconds to cancel...${NC}"
echo ""
sleep 10

log_header "STARTING EVALUATION"

# Track start time
EVAL_START=$(date +%s)

# Run evaluation using the launcher script
log "Launching evaluations with launch_full_evaluations.sh..."
log "Command: $SCRIPT_DIR/launch_full_evaluations.sh"
log ""

# Run the launcher and capture all output
cd "$REPO_DIR"
"$SCRIPT_DIR/launch_full_evaluations.sh" 2>&1 | tee -a "$LOG_FILE"

# Wait for evaluations to complete
log_header "MONITORING EVALUATION PROGRESS"
log "Evaluations running in background..."
log "Use scripts/monitor_evaluations.sh to check progress"
log ""

# Get the most recent run directory
LATEST_RUN=$(ls -td data/results/vllm/full_runs/full_eval_* 2>/dev/null | head -1)

if [[ -n "$LATEST_RUN" ]]; then
    log "Run directory: $LATEST_RUN"
    PROCESSES_FILE="$LATEST_RUN/processes.txt"

    # Wait for all processes to complete
    if [[ -f "$PROCESSES_FILE" ]]; then
        log "Waiting for evaluation processes to complete..."
        log "Process info:"
        cat "$PROCESSES_FILE" | tee -a "$LOG_FILE"

        # Extract PIDs
        PIDS=$(cut -d'|' -f1 "$PROCESSES_FILE")

        # Wait for all PIDs
        for pid in $PIDS; do
            if ps -p "$pid" > /dev/null 2>&1; then
                log "Waiting for PID $pid..."
                wait "$pid" 2>/dev/null || true
                log "PID $pid completed"
            fi
        done
    fi

    # Check completion
    log_header "EVALUATION COMPLETE - CHECKING RESULTS"

    # List result files
    log "Result files in $LATEST_RUN:"
    ls -lh "$LATEST_RUN"/*.json 2>/dev/null | tee -a "$LOG_FILE" || log "No JSON result files found"

    # Summary statistics
    for dataset in "${DATASETS[@]}"; do
        RESULT_FILE="$LATEST_RUN/${dataset}_consolidated.json"
        if [[ -f "$RESULT_FILE" ]]; then
            log ""
            log "=== $dataset Results ==="
            python3 -c "
import json
import sys
try:
    data = json.load(open('$RESULT_FILE'))
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    print(f'  Instances evaluated: {len(results)}')
    print(f'  Duration: {metadata.get(\"duration_seconds\", 0):.1f}s')
    if results:
        scores = [r['final_score'] for r in results]
        print(f'  Mean score: {sum(scores)/len(scores):.3f}')
        print(f'  Median score: {sorted(scores)[len(scores)//2]:.3f}')
except Exception as e:
    print(f'  Error reading results: {e}')
" 2>&1 | tee -a "$LOG_FILE"
        else
            log "  ✗ Result file not found: $RESULT_FILE"
        fi
    done
else
    log "WARNING: Could not find run directory"
fi

# Calculate total duration
EVAL_END=$(date +%s)
EVAL_DURATION=$((EVAL_END - EVAL_START))
EVAL_HOURS=$((EVAL_DURATION / 3600))
EVAL_MINS=$(((EVAL_DURATION % 3600) / 60))

# Final summary
log_header "EVALUATION SUMMARY"
log "Start time: $(date -d @$EVAL_START '+%Y-%m-%d %H:%M:%S')"
log "End time: $(date -d @$EVAL_END '+%Y-%m-%d %H:%M:%S')"
log "Total duration: ${EVAL_HOURS}h ${EVAL_MINS}m"
log ""
log "Log file saved to: $LOG_FILE"
log "Results directory: $LATEST_RUN"
log ""
log "✓ Evaluation complete!"

# Print final message to console
echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}EVALUATION COMPLETE${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo -e "Log file: ${BLUE}$LOG_FILE${NC}"
echo -e "Results: ${BLUE}$LATEST_RUN${NC}"
echo -e "Duration: ${YELLOW}${EVAL_HOURS}h ${EVAL_MINS}m${NC}"
echo ""
