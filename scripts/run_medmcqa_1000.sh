#!/bin/bash
#
# Run MedMCQA evaluation on 1000 samples only
#
# Usage:
#   bash scripts/run_medmcqa_1000.sh
#

set -e

echo "=========================================="
echo "MedMCQA Evaluation - 1000 Samples"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$REPO_ROOT/data/results/vllm/medmcqa_1000"
CONFIG="$REPO_ROOT/config/vllm_jury_config.yaml"

echo "Output directory: $OUTPUT_DIR"
echo "Starting evaluation..."
echo ""

python scripts/run_full_vllm_evaluation.py \
    --dataset medmcqa \
    --instances 1000 \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --checkpoint_interval 100

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
