#!/bin/bash
#
# Monitor Full Evaluation Progress
#
# This script monitors the progress of full dataset evaluations running in background.
# It provides real-time updates on:
# - Running processes
# - Checkpoint files showing current sample progress
# - Estimated completion time
# - GPU memory usage
#
# Usage:
#   # Monitor all running evaluations
#   ./monitor_evaluations.sh
#
#   # Monitor specific dataset
#   ./monitor_evaluations.sh pubmedqa
#
#   # Continuous monitoring (updates every 30 seconds)
#   ./monitor_evaluations.sh --watch
#
#   # Monitor with JSON output for external tools
#   ./monitor_evaluations.sh --json
#
# Author: Evaluation Framework v2.3
#

set -e

# Configuration
RESULTS_DIR="${RESULTS_DIR:-/home/puertao/llm/no-harm-vllm_jury_v3/data/results/vllm/full_runs}"
LOGS_DIR="${LOGS_DIR:-/home/puertao/llm/no-harm-vllm_jury_v3/logs}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/home/puertao/llm/no-harm-vllm_jury_v3/scripts}"
CHECKPOINT_DIR="${RESULTS_DIR}"

# Dataset configurations
declare -A DATASET_CONFIGS=(
    ["pubmedqa"]="1000"
    ["medqa"]="1273"
    ["medmcqa"]="4183"
)

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Parse arguments
WATCH_MODE=false
JSON_OUTPUT=false
SPECIFIC_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --watch|-w)
            WATCH_MODE=true
            shift
            ;;
        --json|-j)
            JSON_OUTPUT=true
            shift
            ;;
        *)
            SPECIFIC_DATASET="$1"
            shift
            ;;
    esac
done

# Function to get GPU memory usage
get_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

# Function to get process info
get_process_info() {
    local dataset=$1
    local pid=$(pgrep -f "run_full_vllm_evaluation.*--dataset ${dataset}" 2>/dev/null | head -1)
    if [[ -n "$pid" ]]; then
        local cmd=$(ps -p "$pid" -o args --no-headers 2>/dev/null | head -c 100)
        local mem=$(ps -p "$pid" -o %mem --no-headers 2>/dev/null || echo "N/A")
        local cpu=$(ps -p "$pid" -o %cpu --no-headers 2>/dev/null || echo "N/A")
        local elapsed=$(ps -p "$pid" -o etime --no-headers 2>/dev/null || echo "N/A")
        echo "${pid}|${cmd}|${mem}|${cpu}|${elapsed}"
    else
        echo "N/A"
    fi
}

# Function to get checkpoint progress
get_checkpoint_progress() {
    local dataset=$1
    local checkpoint_file="${CHECKPOINT_DIR}/.checkpoint_${dataset}.json"

    if [[ -f "$checkpoint_file" ]]; then
        local current_sample=$(grep -o '"current_sample": [0-9]*' "$checkpoint_file" 2>/dev/null | grep -o '[0-9]*' || echo "0")
        local total_samples=$(grep -o '"total_samples": [0-9]*' "$checkpoint_file" 2>/dev/null | grep -o '[0-9]*' || echo "${DATASET_CONFIGS[$dataset]}")

        if [[ -n "$current_sample" && -n "$total_samples" && "$total_samples" -gt 0 ]]; then
            local percentage=$((current_sample * 100 / total_samples))
            echo "${current_sample}/${total_samples} (${percentage}%)"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

# Function to get log file info
get_log_info() {
    local dataset=$1
    local log_files=$(ls -t "${LOGS_DIR}/full_eval_*.log" 2>/dev/null | grep -v ".gz" | head -5)

    if [[ -n "$log_files" ]]; then
        for log_file in $log_files; do
            if grep -q "dataset.*${dataset}" "$log_file" 2>/dev/null; then
                local last_line=$(tail -1 "$log_file" 2>/dev/null)
                local lines=$(wc -l < "$log_file" 2>/dev/null || echo "0")
                echo "${lines} lines | Last: ${last_line}"
                return
            fi
        done
    fi
    echo "No log file found"
}

# Function to calculate estimated completion time
get_estimated_completion() {
    local dataset=$1
    local checkpoint_file="${CHECKPOINT_DIR}/.checkpoint_${dataset}.json"

    if [[ -f "$checkpoint_file" ]]; then
        local current_sample=$(grep -o '"current_sample": [0-9]*' "$checkpoint_file" 2>/dev/null | grep -o '[0-9]*' || echo "0")
        local total_samples="${DATASET_CONFIGS[$dataset]}"

        if [[ -n "$current_sample" && "$current_sample" -gt 0 ]]; then
            local pid=$(pgrep -f "run_full_vllm_evaluation.*--dataset ${dataset}" 2>/dev/null | head -1)
            if [[ -n "$pid" ]]; then
                local elapsed=$(ps -p "$pid" -o etime --no-headers 2>/dev/null || echo "00:00")
                # Convert elapsed to seconds (simplified)
                local elapsed_secs=0

                if echo "$elapsed" | grep -q ":"; then
                    local hours=$(echo "$elapsed" | cut -d: -f1)
                    local mins=$(echo "$elapsed" | cut -d: -f2)
                    local secs=$(echo "$elapsed" | cut -d: -f3)
                    elapsed_secs=$((hours * 3600 + mins * 60 + secs))
                fi

                if [[ "$elapsed_secs" -gt 0 && "$current_sample" -gt 0 ]]; then
                    local samples_per_sec=$(echo "scale=2; $current_sample / $elapsed_secs" | bc 2>/dev/null || echo "1")
                    local remaining=$((total_samples - current_sample))
                    local remaining_secs=$(echo "scale=0; $remaining / $samples_per_sec" | bc 2>/dev/null || echo "$remaining")

                    # Convert to human readable
                    local hours=$((remaining_secs / 3600))
                    local mins=$(( (remaining_secs % 3600) / 60 ))
                    local secs=$((remaining_secs % 60))

                    echo "${hours}h ${mins}m ${secs}s remaining"
                else
                    echo "Calculating..."
                fi
            fi
        fi
    fi
    echo "N/A"
}

# Function to display status in table format
display_status() {
    local datasets=("pubmedqa" "medqa" "medmcqa")

    if [[ -n "$SPECIFIC_DATASET" ]]; then
        datasets=("$SPECIFIC_DATASET")
    fi

    echo ""
    echo -e "${BOLD}Full Evaluation Status${NC}"
    echo -e "${BOLD}====================${NC}"
    echo ""

    for dataset in "${datasets[@]}"; do
        local config="${DATASET_CONFIGS[$dataset]}"
        local progress=$(get_checkpoint_progress "$dataset")
        local proc_info=$(get_process_info "$dataset")
        local log_info=$(get_log_info "$dataset")
        local estimated=$(get_estimated_completion "$dataset")
        local gpu_mem=$(get_gpu_memory)

        echo -e "${CYAN}$dataset${NC} (${config} samples)"
        echo "  Progress: ${progress}"
        echo "  Status: $([ -n "$(pgrep -f "run_full_vllm_evaluation.*--dataset ${dataset}" 2>/dev/null | head -1)" ] && echo -e "${GREEN}Running${NC}" || echo -e "${YELLOW}Not running${NC}")"

        if [[ "$proc_info" != "N/A" ]]; then
            local pid=$(echo "$proc_info" | cut -d'|' -f1)
            local elapsed=$(echo "$proc_info" | cut -d'|' -f5)
            echo "  PID: $pid | Elapsed: $elapsed"
        fi

        if [[ "$gpu_mem" != "N/A" ]]; then
            local mem_used=$(echo "$gpu_mem" | cut -d',' -f1 | tr -d ' ')
            local mem_total=$(echo "$gpu_mem" | cut -d',' -f2 | tr -d ' ')
            echo "  GPU Memory: ${mem_used}MB / ${mem_total}MB"
        fi

        echo "  Estimated: $estimated"
        echo ""
    done

    # Summary
    echo -e "${BOLD}Summary${NC}"
    echo -e "${BOLD}-------${NC}"
    local running_count=0
    for dataset in "${datasets[@]}"; do
        if [[ -n "$(pgrep -f "run_full_vllm_evaluation.*--dataset ${dataset}" 2>/dev/null | head -1)" ]]; then
            ((running_count++))
        fi
    done
    echo "Running evaluations: $running_count / ${#datasets[@]}"
}

# Function to display JSON output
display_json_status() {
    local result='{"timestamp": "'$(date -Iseconds)'", "evaluations": {'

    local first=true
    for dataset in "pubmedqa" "medqa" "medmcqa"; do
        if [[ "$first" != "true" ]]; then
            result+=','
        fi
        first=false

        local config="${DATASET_CONFIGS[$dataset]}"
        local progress=$(get_checkpoint_progress "$dataset")
        local pid=$(pgrep -f "run_full_vllm_evaluation.*--dataset ${dataset}" 2>/dev/null | head -1)
        local is_running="false"
        [[ -n "$pid" ]] && is_running="true"

        result+='"'"$dataset"'"'
        result+=': {'
        result+='"dataset_size": '"$config"','
        result+='"progress": '"$progress"','
        result+='"running": '"$is_running"','
        result+='"pid": '"${pid:-null}"','
        result+='"gpu_memory": '"$(get_gpu_memory | head -1)"''
        result+='}'
    done

    result+='}}'
    echo "$result"
}

# Main execution
if [[ "$JSON_OUTPUT" == "true" ]]; then
    display_json_status
else
    if [[ "$WATCH_MODE" == "true" ]]; then
        while true; do
            clear
            display_status
            echo ""
            echo -e "${YELLOW}Press Ctrl+C to exit watch mode${NC}"
            sleep 30
        done
    else
        display_status
    fi
fi

echo ""
