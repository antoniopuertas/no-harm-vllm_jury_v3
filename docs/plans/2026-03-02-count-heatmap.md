# Count Heatmap Replacement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace scatter plots with count heatmaps in `visualize_jury_dimensions_comparison.py`, producing two output files per dataset (color-only and annotated).

**Architecture:** Add `create_dimension_heatmap(data, dataset_name, model1, model2, output_dir, annotate)` after the existing `create_dimension_scatter` function. Update `main()` to call it twice per dataset (annotate=False then annotate=True). The 11×11 count grid is built by rounding each score to the nearest 0.1 and indexing into a fixed matrix. `create_dimension_scatter` is kept but no longer called from main.

**Tech Stack:** Python 3.10, matplotlib (imshow, colorbar), numpy

---

### Task 1: Add `create_dimension_heatmap` function

**Files:**
- Modify: `scripts/visualize_jury_dimensions_comparison.py` (insert after line 217, after `create_dimension_scatter`)

**Step 1: Insert the function after `create_dimension_scatter`**

Insert this complete function between `create_dimension_scatter` and `main()` (after line 217):

```python
# Score bins: 0.0, 0.1, ..., 1.0  (11 ticks)
SCORE_TICKS = [round(i * 0.1, 1) for i in range(11)]


def create_dimension_heatmap(
    data: dict,
    dataset_name: str,
    model1: str,
    model2: str,
    output_dir: Path,
    annotate: bool = False,
):
    """
    Create a figure with 7 count-heatmap subplots (4 top row + 3 bottom row).
    X-axis = model2 score, Y-axis = model1 score.
    Cell color = number of instances at (model2_score, model1_score).
    """
    model1_scores = data["model1_scores"]
    model2_scores = data["model2_scores"]
    n = len(data["harm_categories"])

    n_bins = len(SCORE_TICKS)  # 11

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    for idx, dim in enumerate(DIMENSIONS):
        ax = axes_flat[idx]

        x_raw = np.array(model2_scores[dim])  # model2 → x-axis
        y_raw = np.array(model1_scores[dim])  # model1 → y-axis

        # Build 11×11 count matrix
        # Round to nearest 0.1, clip to [0.0, 1.0], convert to index 0–10
        x_idx = np.clip(np.round(x_raw * 10).astype(int), 0, 10)
        y_idx = np.clip(np.round(y_raw * 10).astype(int), 0, 10)

        count_matrix = np.zeros((n_bins, n_bins), dtype=int)
        for xi, yi in zip(x_idx, y_idx):
            count_matrix[yi, xi] += 1  # row = y (model1), col = x (model2)

        # Plot heatmap (origin='lower' puts score 0.0 at bottom)
        im = ax.imshow(
            count_matrix,
            origin="lower",
            aspect="equal",
            cmap="Blues",
            interpolation="nearest",
            vmin=0,
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Count", fontsize=7)

        # Diagonal reference line (perfect agreement: y_idx == x_idx)
        ax.plot([-0.5, n_bins - 0.5], [-0.5, n_bins - 0.5],
                "r--", linewidth=1.0, alpha=0.7)

        # Axis ticks at bin centers
        tick_positions = list(range(n_bins))
        tick_labels = [f"{t:.1f}" for t in SCORE_TICKS]
        ax.set_xticks(tick_positions[::2])           # every other tick to avoid clutter
        ax.set_xticklabels(tick_labels[::2], fontsize=6, rotation=45)
        ax.set_yticks(tick_positions[::2])
        ax.set_yticklabels(tick_labels[::2], fontsize=6)

        ax.set_xlabel(f"{model2}", fontsize=8)
        ax.set_ylabel(f"{model1}", fontsize=8)
        ax.set_title(DIMENSION_LABELS[dim], fontweight="bold")

        # Optional: annotate non-zero cells with count
        if annotate:
            max_count = count_matrix.max() or 1
            for yi in range(n_bins):
                for xi in range(n_bins):
                    cnt = count_matrix[yi, xi]
                    if cnt > 0:
                        # White text on dark cells, dark text on light cells
                        text_color = "white" if cnt > max_count * 0.6 else "black"
                        ax.text(
                            xi, yi, str(cnt),
                            ha="center", va="center",
                            fontsize=4.5, color=text_color, fontweight="bold",
                        )

    # Hide unused 8th subplot
    axes_flat[7].set_visible(False)

    variant = "annotated" if annotate else "heatmap"
    m1_short = model1.replace("/", "-")
    m2_short = model2.replace("/", "-")
    output_path = output_dir / f"{dataset_name}_{m1_short}_vs_{m2_short}_{variant}.png"

    fig.suptitle(
        f"{dataset_name.upper()} — {model1} vs {model2}\n"
        f"Per-Dimension Harm Score Counts ({n} instances)"
        + (" [annotated]" if annotate else ""),
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    return output_path
```

**Step 2: Verify syntax**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3 && python -c "
import ast, pathlib
src = pathlib.Path('scripts/visualize_jury_dimensions_comparison.py').read_text()
ast.parse(src)
funcs = [n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)]
print('Syntax OK, functions:', funcs)
"
```

Expected: `functions: ['load_dataset', 'create_dimension_scatter', 'create_dimension_heatmap', 'main']`

---

### Task 2: Update `main()` to call heatmap instead of scatter

**Files:**
- Modify: `scripts/visualize_jury_dimensions_comparison.py` (update the `create_dimension_scatter` call in `main()`)

**Step 1: Replace the scatter call in main() with two heatmap calls**

Find this line in `main()` (currently around line 266):
```python
        create_dimension_scatter(data, dataset_name, model1, model2, OUTPUT_DIR)
```

Replace it with:
```python
        create_dimension_heatmap(data, dataset_name, model1, model2, OUTPUT_DIR, annotate=False)
        create_dimension_heatmap(data, dataset_name, model1, model2, OUTPUT_DIR, annotate=True)
```

**Step 2: Update the module docstring** to reflect the new output format.

Find:
```python
"""
No-Harm-VLLM Per-Dimension Model Comparison Visualizations

Generates scatter plots comparing two jury member models across the 7 harm
dimensions, one figure per dataset (4+3 grid layout).

Usage:
    python scripts/visualize_jury_dimensions_comparison.py
    python scripts/visualize_jury_dimensions_comparison.py --model1 gemma3-27b --model2 olmo-32b
    python scripts/visualize_jury_dimensions_comparison.py --dataset medqa
"""
```

Replace with:
```python
"""
No-Harm-VLLM Per-Dimension Model Comparison Visualizations

Generates count heatmaps comparing two jury member models across the 7 harm
dimensions, one figure per dataset (4+3 grid layout). Produces two output files
per dataset: color-only heatmap and annotated heatmap with counts.

Usage:
    python scripts/visualize_jury_dimensions_comparison.py
    python scripts/visualize_jury_dimensions_comparison.py --model1 gemma3-27b --model2 olmo-32b
    python scripts/visualize_jury_dimensions_comparison.py --dataset medqa
"""
```

**Step 3: Verify syntax again**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3 && python -c "import ast; ast.parse(open('scripts/visualize_jury_dimensions_comparison.py').read()); print('OK')"
```

---

### Task 3: Run and verify output

**Step 1: Run with default models, single dataset first**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3 && python scripts/visualize_jury_dimensions_comparison.py --dataset medqa
```

Expected output:
```
Processing MEDQA...
  Loaded 1273 instances
  ✓ Saved: .../Jury_V3_dimensions/medqa_ministral-14b_vs_gemma3-27b_heatmap.png
  ✓ Saved: .../Jury_V3_dimensions/medqa_ministral-14b_vs_gemma3-27b_heatmap_annotated.png
```

**Step 2: Verify both files exist and have reasonable sizes**

```bash
ls -lh data/results/vllm/full_runs/Jury_V3_dimensions/*heatmap*.png
```

Expected: both files present, each at least 200 KB.

**Step 3: Run all 3 datasets**

```bash
python scripts/visualize_jury_dimensions_comparison.py
```

Expected: 6 heatmap PNGs total (2 per dataset × 3 datasets).

**Step 4: Final file listing**

```bash
ls -lh data/results/vllm/full_runs/Jury_V3_dimensions/
```

Expected: 4 old scatter PNGs + 6 new heatmap PNGs = 10 files total.

---

## Notes

- `create_dimension_scatter` is kept in the file but no longer called from `main()` — it can be used directly if needed
- `SCORE_TICKS` is a module-level constant so it can be reused
- `origin="lower"` on imshow means row 0 = score 0.0 at bottom (intuitive: higher score = higher on y-axis)
- Annotation font size 4.5 is intentionally small to fit in 11×11 cells at 300 DPI
- The diagonal reference line uses pixel coordinates (`-0.5` to `n_bins - 0.5`) matching imshow cell centers
