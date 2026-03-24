#!/usr/bin/env python3
"""
Model Behavior Visualization for harm_dimensions_v2 Jury Evaluation

Produces 6 figures focused on HOW each juror model scores, not just what
the aggregate harm levels are. Output goes to:
  <results-dir>/model_behavior_viz/

Figures:
  1. dimension_signatures.png   — per-juror ratio vs jury mean per dimension
  2. scoring_distributions.png  — discrete score value frequencies per juror
  3. juror_influence.png        — deciding-vote % and lone-outlier % per juror
  4. discriminative_ability.png — Critical/Low score ratio per juror
  5. pairwise_agreement.png     — pairwise Pearson correlation heatmap
  6. bias_direction.png         — (above median % - below median %) heatmap

Usage:
    python scripts/visualize_model_behavior_v2.py
    python scripts/visualize_model_behavior_v2.py --results-dir data/results/vllm/harm_dimensions_v2
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Constants ────────────────────────────────────────────────────────────────

JURORS = [
    "ministral-14b",
    "gemma3-27b",
    "nemotron-30b",
    "olmo-32b",
    "qwen2.5-coder-7b",
]

JUROR_LABELS = {
    "ministral-14b":    "ministral\n14b",
    "gemma3-27b":       "gemma3\n27b",
    "nemotron-30b":     "nemotron\n30b",
    "olmo-32b":         "olmo\n32b",
    "qwen2.5-coder-7b": "qwen2.5\ncoder-7b",
}

JUROR_COLORS = {
    "ministral-14b":    "#E63946",
    "gemma3-27b":       "#2A9D8F",
    "nemotron-30b":     "#E9C46A",
    "olmo-32b":         "#F4A261",
    "qwen2.5-coder-7b": "#457B9D",
}

DIMS = [
    "informational_harm",
    "social_harm",
    "psychological_harm",
    "autonomy_harm",
    "economic_harm",
    "privacy_harm",
    "epistemic_harm",
]

DIM_LABELS = {
    "informational_harm": "Informational",
    "social_harm":        "Social",
    "psychological_harm": "Psychological",
    "autonomy_harm":      "Autonomy",
    "economic_harm":      "Economic",
    "privacy_harm":       "Privacy",
    "epistemic_harm":     "Epistemic",
}

DATASETS = ["pubmedqa", "medqa", "medmcqa"]
DATASET_LABELS = {"pubmedqa": "PubMedQA", "medqa": "MedQA", "medmcqa": "MedMCQA"}

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11


# ── Data loading ─────────────────────────────────────────────────────────────

def _get_score(v):
    if isinstance(v, dict):
        return v.get("score", None)
    return v


def load_all(results_dir: Path):
    """
    Returns:
        details[ds]  — list of jury_detail dicts
        results[ds]  — dict of instance_id -> result dict
    """
    details, results = {}, {}
    for ds in DATASETS:
        ds_dir = results_dir / f"{ds}_full_results"
        with open(ds_dir / "jury_details.json") as f:
            details[ds] = json.load(f)
        with open(ds_dir / "results.json") as f:
            results[ds] = {r["instance_id"]: r for r in json.load(f)}
    return details, results


# ── Figure 1: Dimension Signatures ──────────────────────────────────────────

def fig_dimension_signatures(details, output_dir):
    """Grouped bar chart: juror_mean / jury_mean per dimension."""
    # Accumulate per-juror per-dim scores (combined across datasets)
    juror_dim = {j: {d: [] for d in DIMS} for j in JURORS}
    jury_dim  = {d: [] for d in DIMS}

    for ds in DATASETS:
        for item in details[ds]:
            for j in JURORS:
                jscores = item.get("jury_scores", {}).get(j, {})
                for d in DIMS:
                    s = _get_score(jscores.get(d))
                    if s is not None:
                        juror_dim[j][d].append(s)
                        jury_dim[d].append(s)

    jury_mean  = {d: np.mean(jury_dim[d]) for d in DIMS}
    ratios     = {j: [np.mean(juror_dim[j][d]) / jury_mean[d] if jury_mean[d] > 0 else 1.0
                      for d in DIMS] for j in JURORS}

    n_dims  = len(DIMS)
    n_j     = len(JURORS)
    x       = np.arange(n_dims)
    width   = 0.15
    offsets = np.linspace(-(n_j - 1) / 2, (n_j - 1) / 2, n_j) * width

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, j in enumerate(JURORS):
        bars = ax.bar(x + offsets[i], ratios[j], width,
                      label=j, color=JUROR_COLORS[j], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, ratio in zip(bars, ratios[j]):
            if abs(ratio - 1.0) >= 0.15:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{ratio:.2f}×", ha="center", va="bottom", fontsize=6.5,
                        color=JUROR_COLORS[j], fontweight="bold")

    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.6, label="Jury mean (1.0×)")
    ax.axhspan(0.85, 1.15, alpha=0.06, color="gray", label="±15% band")
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMS], rotation=20, ha="right")
    ax.set_ylabel("Juror mean / Jury mean (ratio)")
    ax.set_title("Juror Dimension Signatures\n(ratio > 1 = over-scores this dimension vs jury average)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, max(max(r) for r in ratios.values()) * 1.18)
    fig.tight_layout()
    path = output_dir / "dimension_signatures.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Figure 2: Scoring Distributions ─────────────────────────────────────────

def fig_scoring_distributions(details, output_dir):
    """Per-juror bar chart of discrete score values used."""
    fig, axes = plt.subplots(1, len(JURORS), figsize=(18, 5), sharey=False)

    for ax, j in zip(axes, JURORS):
        counter = Counter()
        for ds in DATASETS:
            for item in details[ds]:
                jscores = item.get("jury_scores", {}).get(j, {})
                for d in DIMS:
                    s = _get_score(jscores.get(d))
                    if s is not None:
                        counter[round(s, 2)] += 1

        total   = sum(counter.values())
        top     = sorted(counter.items(), key=lambda x: -x[1])[:12]
        vals    = [v for v, _ in top]
        pcts    = [100 * c / total for _, c in top]
        colors  = [JUROR_COLORS[j] if p > 5 else JUROR_COLORS[j] + "88" for p in pcts]

        bars = ax.bar([str(v) for v in vals], pcts, color=colors, edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

        unique = len(counter)
        entropy_val = -sum((c / total) * np.log2(c / total) for _, c in counter.items() if c > 0)
        ax.set_title(f"{j}\n{unique} unique values | H={entropy_val:.2f} bits", fontsize=8)
        ax.set_xlabel("Score value")
        ax.set_ylabel("Frequency (%)" if j == JURORS[0] else "")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("Juror Scoring Distributions\n(which discrete values each model actually uses)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "scoring_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Figure 3: Juror Influence ────────────────────────────────────────────────

def fig_juror_influence(details, results, output_dir):
    """Deciding-vote % and lone-outlier % per juror, per dataset + combined."""
    deciding    = {j: {ds: 0 for ds in DATASETS + ["all"]} for j in JURORS}
    lone        = {j: {ds: 0 for ds in DATASETS + ["all"]} for j in JURORS}
    totals      = {ds: 0 for ds in DATASETS}

    for ds in DATASETS:
        for item in details[ds]:
            totals[ds] += 1
            for d in DIMS:
                scores_this = {}
                for j in JURORS:
                    jscores = item.get("jury_scores", {}).get(j, {})
                    s = _get_score(jscores.get(d))
                    if s is not None:
                        scores_this[j] = s
                if len(scores_this) < 3:
                    continue
                med = float(np.median(list(scores_this.values())))
                for j, s in scores_this.items():
                    if abs(s - med) < 0.001:
                        deciding[j][ds]  += 1
                        deciding[j]["all"] += 1
                dists   = {j: abs(s - med) for j, s in scores_this.items()}
                max_d   = max(dists.values())
                if max_d > 0.05:
                    outs = [j for j, d2 in dists.items() if d2 >= max_d - 0.001]
                    if len(outs) == 1:
                        lone[outs[0]][ds]    += 1
                        lone[outs[0]]["all"] += 1

    total_all = sum(totals.values()) * len(DIMS)
    keys      = DATASETS + ["all"]
    key_labels = [DATASET_LABELS[k] if k in DATASET_LABELS else "Combined" for k in keys]
    tot_map   = {ds: totals[ds] * len(DIMS) for ds in DATASETS}
    tot_map["all"] = total_all

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    n_keys = len(keys)
    n_j    = len(JURORS)
    x      = np.arange(n_keys)
    w      = 0.15
    offs   = np.linspace(-(n_j - 1) / 2, (n_j - 1) / 2, n_j) * w

    for i, j in enumerate(JURORS):
        dec_pcts  = [100 * deciding[j][k] / tot_map[k] for k in keys]
        lone_pcts = [100 * lone[j][k]     / tot_map[k] for k in keys]
        ax1.bar(x + offs[i], dec_pcts,  w, label=j, color=JUROR_COLORS[j], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.bar(x + offs[i], lone_pcts, w, label=j, color=JUROR_COLORS[j], alpha=0.85, edgecolor="white", linewidth=0.5)

    for ax, title, ylabel in [
        (ax1, "Deciding Vote Frequency\n(score = jury median)", "% of dimension evaluations"),
        (ax2, "Lone Outlier Frequency\n(furthest from median, >0.05, unique)", "% of dimension evaluations"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(key_labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=1)

    fig.suptitle("Juror Influence on Final Score", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "juror_influence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Figure 4: Discriminative Ability ────────────────────────────────────────

def fig_discriminative_ability(details, results, output_dir):
    """Critical/Low mean score ratio per juror per dataset + mean."""
    ratios = {j: [] for j in JURORS}

    for ds in DATASETS:
        crit_ids = {iid for iid, r in results[ds].items() if r["harm_category"] == "Critical"}
        low_ids  = {iid for iid, r in results[ds].items() if r["harm_category"] == "Low"}

        juror_crit = {j: [] for j in JURORS}
        juror_low  = {j: [] for j in JURORS}

        for item in details[ds]:
            iid = item["instance_id"]
            for j in JURORS:
                jscores = item.get("jury_scores", {}).get(j, {})
                vals = [_get_score(jscores.get(d)) for d in DIMS]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                mean_s = np.mean(vals)
                if iid in crit_ids:
                    juror_crit[j].append(mean_s)
                elif iid in low_ids:
                    juror_low[j].append(mean_s)

        for j in JURORS:
            lm = np.mean(juror_low[j])  if juror_low[j]  else None
            cm = np.mean(juror_crit[j]) if juror_crit[j] else None
            ratio = cm / lm if (lm and lm > 0 and cm) else 0
            ratios[j].append(ratio)

    # add mean across datasets
    mean_ratios = {j: np.mean(ratios[j]) for j in JURORS}

    fig, ax = plt.subplots(figsize=(12, 6))
    ds_labels = [DATASET_LABELS[d] for d in DATASETS] + ["Mean"]
    n_keys = len(ds_labels)
    n_j    = len(JURORS)
    x      = np.arange(n_keys)
    w      = 0.15
    offs   = np.linspace(-(n_j - 1) / 2, (n_j - 1) / 2, n_j) * w

    for i, j in enumerate(JURORS):
        vals = ratios[j] + [mean_ratios[j]]
        bars = ax.bar(x + offs[i], vals, w, label=j,
                      color=JUROR_COLORS[j], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.1f}×", ha="center", va="bottom", fontsize=6.5,
                    color=JUROR_COLORS[j], fontweight="bold")

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Mean score ratio (Critical / Low)")
    ax.set_title("Juror Discriminative Ability\n(higher ratio = better separation of critical from benign cases)")
    ax.legend(fontsize=8, ncol=2)
    # Add separator before Mean column
    ax.axvline(n_keys - 1.5, color="gray", linewidth=1, linestyle=":", alpha=0.7)
    fig.tight_layout()
    path = output_dir / "discriminative_ability.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Figure 5: Pairwise Agreement ─────────────────────────────────────────────

def fig_pairwise_agreement(details, output_dir):
    """5×5 Pearson correlation heatmap of per-juror score vectors."""
    juror_vecs = {j: [] for j in JURORS}

    for ds in DATASETS:
        for item in details[ds]:
            for d in DIMS:
                for j in JURORS:
                    jscores = item.get("jury_scores", {}).get(j, {})
                    s = _get_score(jscores.get(d))
                    juror_vecs[j].append(s if s is not None else 0.0)

    corr = np.zeros((len(JURORS), len(JURORS)))
    for i, j1 in enumerate(JURORS):
        for k, j2 in enumerate(JURORS):
            v1 = np.array(juror_vecs[j1])
            v2 = np.array(juror_vecs[j2])
            corr[i, k] = np.corrcoef(v1, v2)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(corr, cmap=cmap, vmin=0.3, vmax=1.0)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    short_labels = [j.replace("-", "\n") for j in JURORS]
    ax.set_xticks(range(len(JURORS)))
    ax.set_yticks(range(len(JURORS)))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    for i in range(len(JURORS)):
        for k in range(len(JURORS)):
            color = "white" if corr[i, k] < 0.5 else "black"
            ax.text(k, i, f"{corr[i, k]:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Pairwise Juror Agreement\n(Pearson correlation of score vectors, all dims × all datasets)")
    fig.tight_layout()
    path = output_dir / "pairwise_agreement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Figure 6: Bias Direction Heatmap ────────────────────────────────────────

def fig_bias_direction(details, output_dir):
    """Heatmap of (above-median % - below-median %) per juror×dimension."""
    above = {j: {d: 0 for d in DIMS} for j in JURORS}
    below = {j: {d: 0 for d in DIMS} for j in JURORS}
    n_obs = {j: {d: 0 for d in DIMS} for j in JURORS}

    for ds in DATASETS:
        for item in details[ds]:
            for d in DIMS:
                scores_this = {}
                for j in JURORS:
                    jscores = item.get("jury_scores", {}).get(j, {})
                    s = _get_score(jscores.get(d))
                    if s is not None:
                        scores_this[j] = s
                if len(scores_this) < 3:
                    continue
                med = float(np.median(list(scores_this.values())))
                for j, s in scores_this.items():
                    n_obs[j][d] += 1
                    if s > med + 0.001:
                        above[j][d] += 1
                    elif s < med - 0.001:
                        below[j][d] += 1

    # Build matrix: rows = jurors, cols = dims
    matrix = np.array([
        [(above[j][d] - below[j][d]) / n_obs[j][d] * 100 if n_obs[j][d] > 0 else 0.0
         for d in DIMS]
        for j in JURORS
    ])

    fig, ax = plt.subplots(figsize=(11, 5))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    cmap = plt.cm.RdBu_r
    im   = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Above − Below median (%)\nred = over-scorer, blue = under-scorer")

    dim_labels = [DIM_LABELS[d] for d in DIMS]
    ax.set_xticks(range(len(DIMS)))
    ax.set_yticks(range(len(JURORS)))
    ax.set_xticklabels(dim_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(JURORS, fontsize=9)

    for i in range(len(JURORS)):
        for k in range(len(DIMS)):
            val = matrix[i, k]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(k, i, f"{val:+.0f}%", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_title("Juror Bias Direction per Dimension\n(% of cases above jury median  minus  % below jury median)")
    fig.tight_layout()
    path = output_dir / "bias_direction.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model behavior visualizations for harm_dimensions_v2")
    parser.add_argument(
        "--results-dir",
        default=str(_REPO_ROOT / "data/results/vllm/harm_dimensions_v2"),
        help="Base results directory",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = results_dir / "model_behavior_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MODEL BEHAVIOR VISUALIZATIONS — harm_dimensions_v2")
    print("=" * 70)
    print(f"  Input:  {results_dir}")
    print(f"  Output: {output_dir}\n")

    details, results = load_all(results_dir)

    print("Figure 1: Dimension Signatures")
    fig_dimension_signatures(details, output_dir)

    print("Figure 2: Scoring Distributions")
    fig_scoring_distributions(details, output_dir)

    print("Figure 3: Juror Influence")
    fig_juror_influence(details, results, output_dir)

    print("Figure 4: Discriminative Ability")
    fig_discriminative_ability(details, results, output_dir)

    print("Figure 5: Pairwise Agreement")
    fig_pairwise_agreement(details, output_dir)

    print("Figure 6: Bias Direction")
    fig_bias_direction(details, output_dir)

    print(f"\n{'=' * 70}")
    print(f"  ✓ DONE — {len(list(output_dir.glob('*.png')))} figures saved to {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
