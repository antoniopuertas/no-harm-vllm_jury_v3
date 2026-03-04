# Jury v3 Dimension Comparison Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a script that generates per-dimension scatter plots comparing two jury member models across all 3 datasets, saving results in `Jury_V3_dimensions/`.

**Architecture:** A standalone Python script reads `jury_details.json` (per-model per-dimension scores) and `results.json` (harm_category for coloring) for each dataset, then produces one PNG per dataset containing a 4+3 grid of 7 scatter subplots (X = Model2 score, Y = Model1 score). Jitter + alpha handle the discrete score distribution (0.0–1.0 in 0.1 steps). Models are set via CLI args with hardcoded defaults at the top of the file.

**Tech Stack:** Python 3.10, matplotlib, numpy, json (stdlib), argparse

---

### Task 1: Create output directory and verify data paths

**Files:**
- Create: `data/results/vllm/full_runs/Jury_V3_dimensions/.gitkeep`

**Step 1: Create the output directory**

```bash
mkdir -p data/results/vllm/full_runs/Jury_V3_dimensions
touch data/results/vllm/full_runs/Jury_V3_dimensions/.gitkeep
```

**Step 2: Confirm the 3 jury_details.json files exist**

```bash
ls data/results/vllm/full_runs/medqa_full_results/jury_details.json
ls data/results/vllm/full_runs/pubmedqa_full_results/jury_details.json
ls data/results/vllm/full_runs/medmcqa_full_results/jury_details.json
```

Expected: all 3 files found with no errors.

**Step 3: Confirm the 3 results.json files exist**

```bash
ls data/results/vllm/full_runs/medqa_full_results/results.json
ls data/results/vllm/full_runs/pubmedqa_full_results/results.json
ls data/results/vllm/full_runs/medmcqa_full_results/results.json
```

Expected: all 3 files found.

---

### Task 2: Write the script skeleton with data loading

**Files:**
- Create: `scripts/visualize_jury_dimensions_comparison.py`

**Step 1: Create the script with imports, constants, and data loading**

Create `scripts/visualize_jury_dimensions_comparison.py` with this content:

```python
#!/usr/bin/env python3
"""
Jury v3 Per-Dimension Model Comparison Visualizations

Generates scatter plots comparing two jury member models across the 7 harm
dimensions, one figure per dataset (4+3 grid layout).

Usage:
    python scripts/visualize_jury_dimensions_comparison.py
    python scripts/visualize_jury_dimensions_comparison.py --model1 gemma3-27b --model2 olmo-32b
    python scripts/visualize_jury_dimensions_comparison.py --dataset medqa
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Defaults (change these to set your preferred model pair) ─────────────────
DEFAULT_MODEL1 = "ministral-14b"
DEFAULT_MODEL2 = "gemma3-27b"
# ─────────────────────────────────────────────────────────────────────────────

DIMENSIONS = [
    "informational_harm",
    "psychological_harm",
    "social_harm",
    "economic_harm",
    "privacy_harm",
    "autonomy_harm",
    "epistemic_harm",
]

DIMENSION_LABELS = {
    "informational_harm": "Informational",
    "psychological_harm": "Psychological",
    "social_harm": "Social",
    "economic_harm": "Economic",
    "privacy_harm": "Privacy",
    "autonomy_harm": "Autonomy",
    "epistemic_harm": "Epistemic",
}

DATASETS = {
    "medqa": "data/results/vllm/full_runs/medqa_full_results",
    "pubmedqa": "data/results/vllm/full_runs/pubmedqa_full_results",
    "medmcqa": "data/results/vllm/full_runs/medmcqa_full_results",
}

OUTPUT_DIR = Path("data/results/vllm/full_runs/Jury_V3_dimensions")

# Style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11


def load_dataset(dataset_dir: str, model1: str, model2: str):
    """
    Load jury_details.json and results.json for a dataset.

    Returns a dict:
      {
        "model1_scores": {dim: [float, ...]},
        "model2_scores": {dim: [float, ...]},
        "harm_categories": [str, ...]
      }
    All lists are aligned by instance index.
    """
    base = Path(dataset_dir)
    jury_path = base / "jury_details.json"
    results_path = base / "results.json"

    with open(jury_path) as f:
        jury_data = json.load(f)

    with open(results_path) as f:
        results_data = json.load(f)

    # Build lookup: instance_id -> harm_category
    category_lookup = {r["instance_id"]: r.get("harm_category", "Unknown")
                       for r in results_data}

    model1_scores = {dim: [] for dim in DIMENSIONS}
    model2_scores = {dim: [] for dim in DIMENSIONS}
    harm_categories = []

    for instance in jury_data:
        instance_id = instance["instance_id"]
        jury_scores = instance.get("jury_scores", {})

        if model1 not in jury_scores or model2 not in jury_scores:
            continue  # skip instances missing either model

        for dim in DIMENSIONS:
            score1 = jury_scores[model1].get(dim, {}).get("score", 0.0)
            score2 = jury_scores[model2].get(dim, {}).get("score", 0.0)
            model1_scores[dim].append(float(score1))
            model2_scores[dim].append(float(score2))

        harm_categories.append(category_lookup.get(instance_id, "Unknown"))

    return {
        "model1_scores": model1_scores,
        "model2_scores": model2_scores,
        "harm_categories": harm_categories,
    }
```

**Step 2: Verify the script loads without syntax errors**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3 && python -c "import scripts.visualize_jury_dimensions_comparison" 2>/dev/null || python scripts/visualize_jury_dimensions_comparison.py --help 2>&1 | head -5
```

Expected: argparse not yet defined — that's fine, we're building incrementally.

---

### Task 3: Add the plot creation function

**Files:**
- Modify: `scripts/visualize_jury_dimensions_comparison.py` (append after `load_dataset`)

**Step 1: Append the `create_dimension_scatter` function**

Add this function to the script after `load_dataset`:

```python
def create_dimension_scatter(data: dict, dataset_name: str, model1: str, model2: str, output_dir: Path):
    """
    Create a figure with 7 scatter subplots (4 top row + 3 bottom row).
    X-axis = model2 score, Y-axis = model1 score.
    Points are jittered and colored by harm_category.
    """
    harm_categories = data["harm_categories"]
    model1_scores = data["model1_scores"]
    model2_scores = data["model2_scores"]
    n = len(harm_categories)

    # Color mapping
    color_map = {
        "Low": "#2E86AB",
        "Critical": "#E63946",
        "Unknown": "#888888",
    }
    # Normalize any unexpected category values
    colors = [color_map.get(cat, "#888888") for cat in harm_categories]

    jitter_strength = 0.012  # small jitter relative to 0–1 scale

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    # Use only 7 of the 8 slots; hide the last one
    axes_flat = axes.flatten()

    for idx, dim in enumerate(DIMENSIONS):
        ax = axes_flat[idx]

        x = np.array(model2_scores[dim])
        y = np.array(model1_scores[dim])

        # Add jitter
        rng = np.random.default_rng(seed=42)
        x_jit = x + rng.uniform(-jitter_strength, jitter_strength, size=len(x))
        y_jit = y + rng.uniform(-jitter_strength, jitter_strength, size=len(y))

        # Plot by category so legend works
        for cat, color in color_map.items():
            mask = np.array([c == cat for c in harm_categories])
            if mask.any():
                ax.scatter(
                    x_jit[mask], y_jit[mask],
                    c=color, alpha=0.45, s=12, linewidths=0,
                    label=cat,
                )

        # Diagonal reference line (perfect agreement)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="y = x")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"{model2}", fontsize=8)
        ax.set_ylabel(f"{model1}", fontsize=8)
        ax.set_title(DIMENSION_LABELS[dim], fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # Hide unused 8th subplot
    axes_flat[7].set_visible(False)

    # Single legend for the figure
    handles = [
        mpatches.Patch(color=color_map["Low"], label=f"Low (n={sum(c == 'Low' for c in harm_categories)})"),
        mpatches.Patch(color=color_map["Critical"], label=f"Critical (n={sum(c == 'Critical' for c in harm_categories)})"),
        plt.Line2D([0], [0], linestyle="--", color="black", alpha=0.5, label="y = x (agreement)"),
    ]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(0.98, 0.05), fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"{dataset_name.upper()} — {model1} vs {model2}\n"
        f"Per-Dimension Harm Scores ({n} instances)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # Output filename
    m1_short = model1.replace("/", "-")
    m2_short = model2.replace("/", "-")
    output_path = output_dir / f"{dataset_name}_{m1_short}_vs_{m2_short}.png"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    return output_path
```

---

### Task 4: Add the main function and CLI

**Files:**
- Modify: `scripts/visualize_jury_dimensions_comparison.py` (append at the end)

**Step 1: Append the `main()` function and entry point**

```python
def main():
    parser = argparse.ArgumentParser(
        description="Compare two Jury v3 models across 7 harm dimensions"
    )
    parser.add_argument(
        "--model1", default=DEFAULT_MODEL1,
        help=f"First model (Y-axis). Default: {DEFAULT_MODEL1}"
    )
    parser.add_argument(
        "--model2", default=DEFAULT_MODEL2,
        help=f"Second model (X-axis). Default: {DEFAULT_MODEL2}"
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=["all"] + list(DATASETS.keys()),
        help="Dataset to visualize. Default: all"
    )
    args = parser.parse_args()

    model1 = args.model1
    model2 = args.model2

    if model1 == model2:
        print(f"Error: --model1 and --model2 must be different models.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets_to_run = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    print("=" * 70)
    print("JURY v3 PER-DIMENSION MODEL COMPARISON")
    print("=" * 70)
    print(f"  Model 1 (Y-axis): {model1}")
    print(f"  Model 2 (X-axis): {model2}")
    print(f"  Output:           {OUTPUT_DIR}")
    print()

    for dataset_name, dataset_dir in datasets_to_run.items():
        print(f"Processing {dataset_name.upper()}...")
        data = load_dataset(dataset_dir, model1, model2)
        n = len(data["harm_categories"])
        if n == 0:
            print(f"  ✗ No instances found for models '{model1}' / '{model2}' in {dataset_name}")
            continue
        print(f"  Loaded {n} instances")
        create_dimension_scatter(data, dataset_name, model1, model2, OUTPUT_DIR)

    print()
    print("=" * 70)
    print("✓ DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

### Task 5: Run the script with defaults and verify output

**Step 1: Run with default models**

```bash
cd /home/puertao/llm/no-harm-vllm_jury_v3 && python scripts/visualize_jury_dimensions_comparison.py
```

Expected output:
```
======================================================================
JURY v3 PER-DIMENSION MODEL COMPARISON
======================================================================
  Model 1 (Y-axis): ministral-14b
  Model 2 (X-axis): gemma3-27b
  Output:           data/results/vllm/full_runs/Jury_V3_dimensions

Processing MEDQA...
  Loaded 1273 instances
  ✓ Saved: data/results/vllm/full_runs/Jury_V3_dimensions/medqa_ministral-14b_vs_gemma3-27b.png
Processing PUBMEDQA...
  ...
Processing MEDMCQA...
  ...

======================================================================
✓ DONE
======================================================================
```

**Step 2: Verify the 3 PNG files exist**

```bash
ls -lh data/results/vllm/full_runs/Jury_V3_dimensions/*.png
```

Expected: 3 PNG files, each several hundred KB.

**Step 3: Verify script works with a different model pair**

```bash
python scripts/visualize_jury_dimensions_comparison.py --model1 nemotron-30b --model2 olmo-32b --dataset medqa
```

Expected: 1 PNG created for medqa with nemotron vs olmo.

**Step 4: Verify error handling for same model**

```bash
python scripts/visualize_jury_dimensions_comparison.py --model1 gemma3-27b --model2 gemma3-27b 2>&1
```

Expected: `Error: --model1 and --model2 must be different models.`

---

## Notes

- Scores in this dataset are discrete (multiples of 0.1), so jitter is essential for readability
- `harm_category` values in results.json are "Low" and "Critical" — any other value falls back to grey
- The script is designed to be run from the project root directory
- All 5 jury members are valid choices: `ministral-14b`, `gemma3-27b`, `nemotron-30b`, `olmo-32b`, `qwen2.5-coder-7b`
