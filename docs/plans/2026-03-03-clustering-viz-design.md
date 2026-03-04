# Clustering Visualization Design
**Date:** 2026-03-03
**Script:** `scripts/visualize_jury_dimensions_clustering.py`
**Output:** `data/results/vllm/full_runs/Jury_V3_dimensions/`

## Goal
Show how each jury model behaves across the 7 harm dimensions — specifically how dimensions overlap (co-fire) or scatter (fire independently) — producing one figure per model.

## Figure A — Dimension Correlation Matrix
**Output files:** `{model}_dimension_correlations.png` (5 files)

- Layout: 1 row × 3 columns (medqa | pubmedqa | medmcqa)
- Each subplot: 7×7 Pearson correlation heatmap between harm dimensions
- Annotated with correlation values
- Diverging color scale (red = strong positive, blue = negative), shared vmin/vmax across subplots
- Computed over all instances
- Reveals which dimensions co-fire within each model's scoring behavior

## Figure B — Radar Chart per Harm Category
**Output files:** `{model}_radar_by_category.png` (5 files)

- Layout: 1 row × 3 columns (medqa | pubmedqa | medmcqa)
- Each subplot: radar/spider chart with 7 axes (one per harm dimension)
- Two filled polygons overlaid: Low (blue, semi-transparent) vs Critical (red, semi-transparent)
- Values = mean score per dimension per category
- Shared radial scale 0–1 across all subplots for cross-dataset comparability
- Legend shows instance counts per category
- Reveals the model's scoring "shape" and how well it separates harm categories across dimensions

## Models
- ministral-14b, gemma3-27b, nemotron-30b, olmo-32b, qwen2.5-coder-7b

## Datasets
- medqa (1273 instances), pubmedqa (1000), medmcqa (1000)

## Harm Dimensions
informational, psychological, social, economic, privacy, autonomy, epistemic

## Harm Categories
Low, Critical (Unknown instances skipped)
