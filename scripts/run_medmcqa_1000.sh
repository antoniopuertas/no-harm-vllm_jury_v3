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

OUTPUT_DIR="/home/puertao/llm/no-harm-vllm_jury_v3/data/results/vllm/medmcqa_1000"
CONFIG="/home/puertao/llm/no-harm-vllm_jury_v3/config/vllm_jury_config.yaml"

echo "Output directory: $OUTPUT_DIR"
echo "Starting evaluation..."
echo ""

python scripts/run_full_vllm_evaluation.py \
    --dataset medmcqa \
    --num_samples 1000 \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --checkpoint_interval 100

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
