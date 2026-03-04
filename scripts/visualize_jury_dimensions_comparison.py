#!/usr/bin/env python3
"""
Jury v3 Per-Dimension Model Comparison Visualizations

Generates count heatmaps comparing two jury member models across the 7 harm
dimensions, one figure per dataset (4+3 grid layout). Produces two output files
per dataset: color-only heatmap and annotated heatmap with counts.

Usage:
    python scripts/visualize_jury_dimensions_comparison.py
    python scripts/visualize_jury_dimensions_comparison.py --model1 gemma3-27b --model2 olmo-32b
    python scripts/visualize_jury_dimensions_comparison.py --dataset medqa
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

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
    "medqa":    _REPO_ROOT / "data/results/vllm/full_runs/medqa_full_results",
    "pubmedqa": _REPO_ROOT / "data/results/vllm/full_runs/pubmedqa_full_results",
    "medmcqa":  _REPO_ROOT / "data/results/vllm/full_runs/medmcqa_full_results",
}

OUTPUT_DIR = _REPO_ROOT / "data/results/vllm/full_runs/Jury_V3_dimensions"

# Style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11


def load_dataset(dataset_dir: Path, model1: str, model2: str):
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

        scores1 = {}
        scores2 = {}
        valid = True
        for dim in DIMENSIONS:
            raw1 = jury_scores[model1].get(dim, {}).get("score", 0.0)
            raw2 = jury_scores[model2].get(dim, {}).get("score", 0.0)
            try:
                scores1[dim] = float(raw1)
                scores2[dim] = float(raw2)
            except (TypeError, ValueError):
                valid = False
                break
        if valid:
            for dim in DIMENSIONS:
                model1_scores[dim].append(scores1[dim])
                model2_scores[dim].append(scores2[dim])
            harm_categories.append(category_lookup.get(instance_id, "Unknown"))

    return {
        "model1_scores": model1_scores,
        "model2_scores": model2_scores,
        "harm_categories": harm_categories,
    }


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

    jitter_strength = 0.012  # small jitter relative to 0–1 scale
    rng = np.random.default_rng(seed=42)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    # Use only 7 of the 8 slots; hide the last one
    axes_flat = axes.flatten()

    for idx, dim in enumerate(DIMENSIONS):
        ax = axes_flat[idx]

        x = np.array(model2_scores[dim])
        y = np.array(model1_scores[dim])

        # Add jitter
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
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)

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
    n_low = sum(c == "Low" for c in harm_categories)
    n_critical = sum(c == "Critical" for c in harm_categories)
    n_unknown = sum(c == "Unknown" for c in harm_categories)
    handles = [
        mpatches.Patch(color=color_map["Low"], label=f"Low (n={n_low})"),
        mpatches.Patch(color=color_map["Critical"], label=f"Critical (n={n_critical})"),
    ]
    if n_unknown > 0:
        handles.append(mpatches.Patch(color=color_map["Unknown"], label=f"Unknown (n={n_unknown})"))
    handles.append(plt.Line2D([0], [0], linestyle="--", color="black", alpha=0.5, label="y = x (agreement)"))
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

    # Pre-compute global vmax so all subplots share the same color scale
    global_vmax = 1  # minimum of 1 to avoid division by zero
    for dim in DIMENSIONS:
        x_raw = np.array(model2_scores[dim])
        y_raw = np.array(model1_scores[dim])
        x_idx = np.clip(np.round(x_raw * 10).astype(int), 0, 10)
        y_idx = np.clip(np.round(y_raw * 10).astype(int), 0, 10)
        tmp = np.zeros((n_bins, n_bins), dtype=int)
        for xi, yi in zip(x_idx, y_idx):
            tmp[yi, xi] += 1
        global_vmax = max(global_vmax, int(tmp.max()))

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
            vmax=global_vmax,
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
            for yi in range(n_bins):
                for xi in range(n_bins):
                    cnt = count_matrix[yi, xi]
                    if cnt > 0:
                        # White text on dark cells, dark text on light cells
                        text_color = "white" if (cnt / global_vmax) > 0.6 else "black"
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
        create_dimension_heatmap(data, dataset_name, model1, model2, OUTPUT_DIR, annotate=False)
        create_dimension_heatmap(data, dataset_name, model1, model2, OUTPUT_DIR, annotate=True)

    print()
    print("=" * 70)
    print("✓ DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
