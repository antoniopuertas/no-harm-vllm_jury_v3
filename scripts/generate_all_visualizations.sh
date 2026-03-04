#!/bin/bash
# Master script to generate all visualizations for vLLM full run results

set -e  # Exit on error

echo "================================================================================"
echo "vLLM Full Runs Visualization Generator"
echo "================================================================================"
echo ""

# Check if we're in the correct directory
if [ ! -d "data/results/vllm/full_runs" ]; then
    echo "Error: Must run from repository root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check for required input files
echo "Checking input files..."
required_files=(
    "data/results/vllm/full_runs/medqa_consolidated.json"
    "data/results/vllm/full_runs/pubmedqa_consolidated.json"
    "data/results/vllm/full_runs/medmcqa_consolidated.json"
)

all_present=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ✗ Missing: $file"
        all_present=false
    else
        echo "  ✓ Found: $file"
    fi
done

if [ "$all_present" = false ]; then
    echo ""
    echo "Error: Missing required consolidated JSON files"
    exit 1
fi

echo ""
echo "All input files present!"
echo ""

# Check for optional jury_details files
echo "Checking jury detail files (for per-member analysis)..."
jury_detail_files=(
    "data/results/vllm/full_runs/medqa_full_results/jury_details.json"
    "data/results/vllm/full_runs/pubmedqa_full_results/jury_details.json"
    "data/results/vllm/full_runs/medmcqa_full_results/jury_details.json"
)

for file in "${jury_detail_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ⚠ Optional: $file (not found - jury agreement plots will be limited)"
    else
        echo "  ✓ Found: $file"
    fi
done

echo ""

# Check for required Python packages
echo "Checking Python dependencies..."
python3 -c "import matplotlib; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing required Python packages"
    echo "Install with: pip install matplotlib numpy"
    exit 1
fi
echo "  ✓ All Python dependencies available"
echo ""

# Run individual dataset visualizations
echo "================================================================================"
echo "Step 1/2: Generating Individual Dataset Visualizations"
echo "================================================================================"
echo ""

python3 scripts/visualize_individual_datasets_v2_3.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Individual visualizations failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Step 2/2: Generating Cross-Dataset Comparison Visualizations"
echo "================================================================================"
echo ""

python3 scripts/compare_v2_3_evaluations_with_viz.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Comparison visualizations failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ ALL VISUALIZATIONS COMPLETE"
echo "================================================================================"
echo ""

# Count generated files
individual_count=$(find data/results/vllm/full_runs_visualizations/ -type f 2>/dev/null | wc -l)
comparison_count=$(find data/results/vllm/full_runs_comparison_visualizations/ -type f 2>/dev/null | wc -l)

echo "Summary:"
echo "  Individual visualizations: $individual_count files"
echo "    → data/results/vllm/full_runs_visualizations/"
echo ""
echo "  Comparison visualizations: $comparison_count files"
echo "    → data/results/vllm/full_runs_comparison_visualizations/"
echo ""
echo "  Total: $((individual_count + comparison_count)) files generated"
echo ""
echo "View the visualizations and reports in the output directories above."
echo ""
