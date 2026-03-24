#!/usr/bin/env python3
"""
Jury v3 Per-Model Dimension Clustering Visualizations

For each jury model produces two figures:
  A) 7x7 Pearson correlation matrix between harm dimensions (one subplot per dataset)
  B) Radar chart of mean scores per harm category (one subplot per dataset)

Usage:
    python scripts/visualize_jury_dimensions_clustering.py
    python scripts/visualize_jury_dimensions_clustering.py --model ministral-14b
"""

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

import matplotlib.pyplot as plt
import numpy as np

DIMENSIONS = [
    "informational_harm",
    "psychological_harm",
    "social_harm",
    "economic_harm",
    "privacy_harm",
    "autonomy_harm",
    "epistemic_harm",
]

DIM_LABELS = [
    "Informational",
    "Psychological",
    "Social",
    "Economic",
    "Privacy",
    "Autonomy",
    "Epistemic",
]

MODELS = [
    "ministral-14b",
    "gemma3-27b",
    "nemotron-30b",
    "olmo-32b",
    "qwen2.5-coder-7b",
]

DATASETS = {
    "medqa":    _REPO_ROOT / "data/results/vllm/full_runs/medqa_full_results",
    "pubmedqa": _REPO_ROOT / "data/results/vllm/full_runs/pubmedqa_full_results",
    "medmcqa":  _REPO_ROOT / "data/results/vllm/full_runs/medmcqa_full_results",
}

OUTPUT_DIR = _REPO_ROOT / "data/results/vllm/full_runs/Jury_V3_dimensions"

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 9


def load_dataset(dataset_dir: Path, model: str):
    """
    Load scores and harm categories for a single model from a dataset directory.

    Returns:
      {
        "scores": np.ndarray of shape (n_instances, 7),   # columns = DIMENSIONS
        "categories": list[str],                           # "Low" or "Critical"
      }
    Only instances with a valid score for all 7 dimensions are included.
    """
    jury_path = Path(dataset_dir) / "jury_details.json"
    results_path = Path(dataset_dir) / "results.json"

    with open(jury_path) as f:
        jury_data = json.load(f)
    with open(results_path) as f:
        results_data = json.load(f)

    cat_lookup = {r["instance_id"]: r.get("harm_category", "Unknown")
                  for r in results_data}

    rows = []
    categories = []
    for instance in jury_data:
        iid = instance["instance_id"]
        jury_scores = instance.get("jury_scores", {})
        if model not in jury_scores:
            continue
        model_scores = jury_scores[model]
        try:
            row = [float(model_scores.get(d, {}).get("score", 0.0)) for d in DIMENSIONS]
        except (TypeError, ValueError):
            continue
        rows.append(row)
        categories.append(cat_lookup.get(iid, "Unknown"))

    return {
        "scores": np.array(rows) if rows else np.empty((0, len(DIMENSIONS))),
        "categories": categories,
    }


def create_correlation_figure(model: str, output_dir: Path, datasets=None):
    """
    Figure A: 1×3 grid of 7×7 Pearson correlation heatmaps.
    One subplot per dataset. Shared diverging color scale [-1, 1].
    """
    _datasets = datasets if datasets is not None else DATASETS
    dataset_names = list(_datasets.keys())
    n_datasets = len(dataset_names)

    # Pre-load all datasets
    all_corr = []
    all_n = []
    for ds_name, ds_dir in _datasets.items():
        data = load_dataset(ds_dir, model)
        scores = data["scores"]
        n = len(scores)
        all_n.append(n)
        if n < 2:
            all_corr.append(np.eye(len(DIMENSIONS)))
        else:
            # np.corrcoef returns shape (7,7); handle NaN columns
            corr = np.corrcoef(scores.T)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            all_corr.append(corr)

    fig, axes = plt.subplots(1, n_datasets, figsize=(18, 6))

    for idx, (ax, ds_name, corr, n) in enumerate(
        zip(axes, dataset_names, all_corr, all_n)
    ):
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")

        # Annotate cells
        for yi in range(len(DIMENSIONS)):
            for xi in range(len(DIMENSIONS)):
                val = corr[yi, xi]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)

        tick_pos = list(range(len(DIMENSIONS)))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(DIM_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(DIM_LABELS, fontsize=7)
        ax.set_title(f"{ds_name.upper()}  (n={n})", fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    m_short = model.replace("/", "-")
    fig.suptitle(
        f"{model} — Harm Dimension Correlations",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = output_dir / f"{m_short}_dimension_correlations.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")
    return out


def create_radar_figure(model: str, output_dir: Path, datasets=None):
    """
    Figure B: 1×3 grid of radar/spider charts.
    Each subplot shows mean dimension scores for Low vs Critical instances.
    """
    _datasets = datasets if datasets is not None else DATASETS
    n_dims = len(DIMENSIONS)
    # Angles for each dimension, evenly spaced, closing the polygon
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    dataset_names = list(_datasets.keys())
    fig, axes = plt.subplots(
        1, len(dataset_names),
        figsize=(18, 6),
        subplot_kw={"projection": "polar"},
    )

    color_map = {"Low": "#2E86AB", "Critical": "#E63946"}

    for ax, ds_name, ds_dir in zip(axes, dataset_names, _datasets.values()):
        data = load_dataset(ds_dir, model)
        scores = data["scores"]
        categories = data["categories"]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIM_LABELS, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=5)

        for cat, color in color_map.items():
            mask = np.array([c == cat for c in categories])
            if not mask.any():
                continue
            means = scores[mask].mean(axis=0).tolist()
            means += means[:1]  # close polygon
            n_cat = int(mask.sum())
            ax.plot(angles, means, color=color, linewidth=1.5)
            ax.fill(angles, means, color=color, alpha=0.25,
                    label=f"{cat} (n={n_cat})")

        ax.set_title(f"{ds_name.upper()}", fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)

    m_short = model.replace("/", "-")
    fig.suptitle(
        f"{model} — Harm Dimension Radar by Category",
        fontsize=13, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    out = output_dir / f"{m_short}_radar_by_category.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Per-model dimension clustering visualizations (correlation + radar)"
    )
    parser.add_argument("--model", default=None,
                        help="Single model to run. Default: all models.")
    parser.add_argument(
        "--results-dir",
        default=str(_REPO_ROOT / "data/results/vllm/full_runs"),
        help="Base results directory (default: data/results/vllm/full_runs)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    datasets = {
        "medqa":    results_dir / "medqa_full_results",
        "pubmedqa": results_dir / "pubmedqa_full_results",
        "medmcqa":  results_dir / "medmcqa_full_results",
    }
    output_dir = results_dir / "Jury_V3_dimensions"
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = [args.model] if args.model else MODELS

    print("=" * 70)
    print("JURY v3 DIMENSION CLUSTERING VISUALIZATIONS")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print()

    for model in models_to_run:
        print(f"Processing {model}...")
        create_correlation_figure(model, output_dir, datasets=datasets)
        create_radar_figure(model, output_dir, datasets=datasets)

    print()
    print("=" * 70)
    print("✓ DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
