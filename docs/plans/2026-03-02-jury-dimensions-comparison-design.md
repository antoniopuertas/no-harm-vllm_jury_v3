# Design: Jury v3 Per-Dimension Model Comparison Visualizations

**Date:** 2026-03-02

## Goal

Create scatter plot visualizations comparing how two jury member models score each of the 7 harm dimensions on a per-instance basis, across all three datasets.

## Data Source

- `data/results/vllm/full_runs/{dataset}_full_results/jury_details.json`
- Contains per-instance scores from each jury member for all 7 dimensions
- 3 datasets: medqa (1273), pubmedqa, medmcqa

## Jury Members Available

`ministral-14b`, `gemma3-27b`, `nemotron-30b`, `olmo-32b`, `qwen2.5-coder-7b`

## Output

- **Folder**: `data/results/vllm/full_runs/Jury_V3_dimensions/`
- **3 PNG files**: `{dataset}_{model1}_vs_{model2}.png`

## Chart Layout (per dataset)

- Single figure, 2-row grid: 4 subplots top row + 3 subplots bottom row = 7 total
- Each subplot = one harm dimension
- **X-axis**: Model 2 score (0–1)
- **Y-axis**: Model 1 score (0–1)
- **Each point**: one instance
- **Diagonal line**: y=x reference (perfect agreement)
- **Jitter + alpha**: handle discrete score overlap
- **Color**: by harm_category (Low vs Critical from aggregated results.json)
- **Figure title**: `{Dataset} — {Model1} vs {Model2}`
- **Subplot title**: dimension name

## Script

- `scripts/visualize_jury_dimensions_comparison.py`
- CLI args: `--model1`, `--model2`, `--dataset` (default: all three)
- Hardcoded defaults at top of file for easy selection

## Approach

Option A: one figure per dataset with 7 scatter subplots in a 4+3 grid
