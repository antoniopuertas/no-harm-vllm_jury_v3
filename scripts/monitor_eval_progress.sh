#!/bin/bash
#
# Timed progress monitor for the medqa + medmcqa evaluation run.
#
# Fires a status report at 3 estimated milestones:
#   1. ~16:25 Mar 30  — medqa Phase 1 (response generation) done
#   2. ~17:50 Mar 31  — medqa jury scoring done
#   3. ~19:30 Apr  1  — medmcqa jury scoring done (full run complete)
#
# Run in background:
#   nohup bash scripts/monitor_eval_progress.sh > logs/monitor_eval_progress.log 2>&1 &
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/harm_dimensions_v2"
LOGS_DIR="$REPO_ROOT/logs"
MONITOR_LOG="$LOGS_DIR/monitor_eval_progress.log"

# Milestone timestamps (epoch seconds) — Rome (UTC+2) converted to UTC
# Mar 30 16:25 Rome = 14:25 UTC
MILESTONE_1=$(date -d "2026-03-30 14:25:00 UTC" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "2026-03-30 14:25:00" +%s)
# Mar 31 17:50 Rome = 15:50 UTC
MILESTONE_2=$(date -d "2026-03-31 15:50:00 UTC" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "2026-03-31 15:50:00" +%s)
# Apr  1 19:30 Rome = 17:30 UTC
MILESTONE_3=$(date -d "2026-04-01 17:30:00 UTC" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "2026-04-01 17:30:00" +%s)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

report() {
    local label="$1"
    log ""
    log "======================================================"
    log "CHECKPOINT: $label"
    log "======================================================"

    # --- Active process ---
    PROC=$(ps aux | grep "run_full_vllm_evaluation.py" | grep -v grep | awk '{print $2, $11, $12, $13}')
    if [[ -n "$PROC" ]]; then
        log "Process running: $PROC"
    else
        log "Process: NOT RUNNING"
    fi

    # --- Docker containers ---
    CONTAINERS=$(docker ps --filter "name=vllm-" --format "{{.Names}}: {{.Status}}" 2>/dev/null)
    if [[ -n "$CONTAINERS" ]]; then
        log "Docker: $CONTAINERS"
    else
        log "Docker: no vllm containers running"
    fi

    # --- Latest log progress ---
    LATEST_LOG=$(ls -t "$LOGS_DIR"/harm_v2_sequential_*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        log "Log: $LATEST_LOG"
        LAST_PROGRESS=$(grep -E "Response generation|Scoring progress|Phase [0-9]|Finished|FAILED|complete" "$LATEST_LOG" | tail -5)
        log "Recent progress:"
        echo "$LAST_PROGRESS" | while IFS= read -r line; do log "  $line"; done
        LAST_ERROR=$(grep "ERROR" "$LATEST_LOG" | tail -3)
        if [[ -n "$LAST_ERROR" ]]; then
            log "Recent errors:"
            echo "$LAST_ERROR" | while IFS= read -r line; do log "  $line"; done
        fi
    else
        log "No sequential log found."
    fi

    # --- Result files ---
    for dataset in medqa medmcqa; do
        RESULTS="$OUTPUT_DIR/${dataset}_full_results/results.json"
        if [[ -f "$RESULTS" ]]; then
            COUNT=$(python3 -c "import json; d=json.load(open('$RESULTS')); print(len(d))" 2>/dev/null)
            log "Results $dataset: $COUNT instances in results.json"
        else
            log "Results $dataset: not yet written"
        fi
        CHECKPOINT="$OUTPUT_DIR/.checkpoint_${dataset}.json"
        if [[ -f "$CHECKPOINT" ]]; then
            CKPT_INFO=$(python3 -c "
import json
d = json.load(open('$CHECKPOINT'))
scored = sum(1 for v in d.values() if v.get('jury_scores'))
print(f'checkpoint: {scored} instances scored')
" 2>/dev/null)
            log "  $CKPT_INFO"
        fi
    done

    log "======================================================"
    log ""
}

sleep_until() {
    local target=$1
    local label=$2
    local now
    now=$(date +%s)
    local delta=$(( target - now ))
    if [[ $delta -le 0 ]]; then
        log "Milestone '$label' is in the past — reporting now."
    else
        local h=$(( delta / 3600 ))
        local m=$(( (delta % 3600) / 60 ))
        log "Sleeping ${h}h ${m}m until milestone: $label"
        sleep $delta
    fi
}

mkdir -p "$LOGS_DIR"
log "Monitor started. Milestones:"
log "  1. $(date -d @$MILESTONE_1 '+%Y-%m-%d %H:%M UTC' 2>/dev/null) — medqa Phase 1 done"
log "  2. $(date -d @$MILESTONE_2 '+%Y-%m-%d %H:%M UTC' 2>/dev/null) — medqa jury scoring done"
log "  3. $(date -d @$MILESTONE_3 '+%Y-%m-%d %H:%M UTC' 2>/dev/null) — medmcqa complete"
log ""

sleep_until $MILESTONE_1 "medqa Phase 1 done"
report "medqa Phase 1 — response generation should be complete"

sleep_until $MILESTONE_2 "medqa jury scoring done"
report "medqa — jury scoring should be complete"

sleep_until $MILESTONE_3 "medmcqa complete"
report "medmcqa — full run should be complete"

log "All milestone checks done."
