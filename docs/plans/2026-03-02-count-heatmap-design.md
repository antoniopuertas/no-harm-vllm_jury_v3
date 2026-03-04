# Design: Replace Scatter Plots with Count Heatmaps

**Date:** 2026-03-02

## Goal

Replace the scatter plot visualization with count heatmaps that properly represent the discrete nature of jury scores.

## Problem

Jury scores are discrete (0.0–1.0 in 0.1 steps, or retry values 0.2/0.5/0.8), so scatter plots with jitter misrepresent the data. Count heatmaps make the discrete grid a feature: each cell shows how many instances both models assigned that exact score pair.

## Design

### Grid
- Fixed 11×11 grid: axes 0.0 → 1.0 in 0.1 steps
- X-axis = Model 2 score, Y-axis = Model 1 score
- Cell value = count of instances at (model2_score, model1_score)

### Visuals
- Colormap: `Blues` (white = 0, dark blue = high count)
- Colorbar per subplot
- Dashed diagonal (y=x) agreement reference line
- Two output variants per run:
  1. Color-only: `{dataset}_{m1}_vs_{m2}_heatmap.png`
  2. Annotated: `{dataset}_{m1}_vs_{m2}_heatmap_annotated.png` (count text in non-zero cells)

### Changes
- Replace `create_dimension_scatter` with `create_dimension_heatmap(data, dataset_name, model1, model2, output_dir, annotate: bool)`
- Call twice per dataset (annotate=False then annotate=True)
- No changes to `load_dataset`, `main()`, or CLI interface
- Old scatter PNGs not deleted
