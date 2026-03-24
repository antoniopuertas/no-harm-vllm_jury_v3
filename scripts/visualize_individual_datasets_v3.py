#!/usr/bin/env python3
"""
Individual Dataset Visualization and Analysis for vLLM Full Runs
Creates detailed visualizations and reports for each dataset
Uses *_consolidated.json files from data/results/vllm/full_runs/
"""

import argparse
import json
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import textwrap

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

DIMENSIONS = ['informational', 'psychological', 'social', 'economic', 'privacy', 'autonomy', 'epistemic']
DIMENSION_WEIGHTS = {
    'informational': 0.25,
    'social': 0.20,
    'psychological': 0.15,
    'autonomy': 0.15,
    'economic': 0.10,
    'privacy': 0.10,
    'epistemic': 0.05
}

DATASET_COLORS = {
    'medqa': '#2E86AB',
    'pubmedqa': '#A23B72',
    'medmcqa': '#F18F01'
}

DATASET_DESCRIPTIONS = {
    'medqa': {
        'title': 'MedQA - US Medical Licensing Exam',
        'characteristics': [
            'Complex clinical scenarios',
            'Patient case presentations',
            'Diagnostic reasoning required',
            'Treatment decision-making',
            'Multi-step problem solving'
        ],
        'notes': 'vLLM Jury v3 evaluation with 5 models'
    },
    'pubmedqa': {
        'title': 'PubMedQA - Biomedical Research Questions',
        'characteristics': [
            'Research methodology focus',
            'Evidence interpretation',
            'Scientific reasoning',
            'Abstract-based questions',
            'Yes/No/Maybe answer format'
        ],
        'notes': 'Higher epistemic scores due to research uncertainty'
    },
    'medmcqa': {
        'title': 'MedMCQA - Indian Medical Entrance Exam',
        'characteristics': [
            'Factual medical knowledge',
            'Anatomy and physiology',
            'Straightforward questions',
            'Clear correct answers',
            'Less clinical ambiguity'
        ],
        'notes': 'Lowest harm scores - factual knowledge focus'
    }
}


def calculate_dimension_scores(results):
    """Extract pre-aggregated dimension scores from consolidated results"""
    dimension_scores = defaultdict(list)
    composite_scores = []

    for result in results:
        scores = result['dimension_scores']
        for dim in DIMENSIONS:
            dim_key = f'{dim}_harm'
            if dim_key in scores:
                dimension_scores[dim].append(scores[dim_key])
        composite_scores.append(result['final_score'])

    return dimension_scores, composite_scores


def create_dataset_overview_plot(dataset_name, data, output_dir):
    """Create comprehensive overview plot for a dataset"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    dimension_scores, composite_scores = calculate_dimension_scores(data['results'])
    color = DATASET_COLORS[dataset_name]

    # 1. Composite Score Distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(composite_scores, bins=20, color=color, alpha=0.7, edgecolor='black')
    ax1.axvline(statistics.mean(composite_scores), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {statistics.mean(composite_scores):.3f}')
    ax1.axvline(statistics.median(composite_scores), color='green', linestyle='--',
                linewidth=2, label=f'Median: {statistics.median(composite_scores):.3f}')
    ax1.set_xlabel('Composite Harm Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Composite Score Distribution')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Dimension Bar Chart (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    dim_means = [statistics.mean(dimension_scores[dim]) for dim in DIMENSIONS]
    y_pos = np.arange(len(DIMENSIONS))
    bars = ax2.barh(y_pos, dim_means, color=color, alpha=0.7, edgecolor='black')

    for i, v in enumerate(dim_means):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, weight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([d.capitalize() for d in DIMENSIONS], fontsize=9)
    ax2.set_xlabel('Mean Harm Score')
    ax2.set_title('Dimension Scores')
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3)

    # 3. Harm Categories Pie Chart (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    low = sum(1 for s in composite_scores if s < 0.3)
    moderate = sum(1 for s in composite_scores if 0.3 <= s <= 0.5)
    high = sum(1 for s in composite_scores if s > 0.5)

    sizes = [low, moderate, high]
    labels = [f'Low (<0.3)\n{low} ({low/len(composite_scores)*100:.1f}%)',
              f'Moderate (0.3-0.5)\n{moderate} ({moderate/len(composite_scores)*100:.1f}%)',
              f'High (>0.5)\n{high} ({high/len(composite_scores)*100:.1f}%)']
    colors_pie = ['#90EE90', '#FFD700', '#FF6B6B']

    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='',
                                         startangle=90, textprops={'fontsize': 9})
    ax3.set_title('Harm Category Distribution')

    # 4. Box Plot of Dimensions (middle left, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    data_for_box = [dimension_scores[dim] for dim in DIMENSIONS]
    bp = ax4.boxplot(data_for_box, labels=[d.capitalize() for d in DIMENSIONS],
                     patch_artist=True, widths=0.6)

    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Harm Score')
    ax4.set_title('Dimension Score Distributions')
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(axis='y', alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    # 5. Statistics Table (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    jury_cfg = data['metadata']['jury_config']
    stats_text = f"""
DATASET STATISTICS

Total Samples: {len(data['results'])}

Composite Score:
  Mean:   {statistics.mean(composite_scores):.4f}
  Median: {statistics.median(composite_scores):.4f}
  Std:    {statistics.stdev(composite_scores):.4f}
  Min:    {min(composite_scores):.4f}
  Max:    {max(composite_scores):.4f}

Harm Categories:
  Low:      {low:3d} ({low/len(composite_scores)*100:5.1f}%)
  Moderate: {moderate:3d} ({moderate/len(composite_scores)*100:5.1f}%)
  High:     {high:3d} ({high/len(composite_scores)*100:5.1f}%)

Aggregation: {jury_cfg['aggregation_method']}
Timestamp: {data['metadata']['timestamp'][:10]}
    """

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 6. Dimension Correlation Matrix (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])

    n_dims = len(DIMENSIONS)
    corr_matrix = np.zeros((n_dims, n_dims))

    for i, dim1 in enumerate(DIMENSIONS):
        for j, dim2 in enumerate(DIMENSIONS):
            if len(dimension_scores[dim1]) > 1 and len(dimension_scores[dim2]) > 1:
                corr = np.corrcoef(dimension_scores[dim1], dimension_scores[dim2])[0, 1]
                corr_matrix[i, j] = corr

    im = ax6.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax6.set_xticks(np.arange(n_dims))
    ax6.set_yticks(np.arange(n_dims))
    ax6.set_xticklabels([d[:4].capitalize() for d in DIMENSIONS], fontsize=8, rotation=45)
    ax6.set_yticklabels([d[:4].capitalize() for d in DIMENSIONS], fontsize=8)
    ax6.set_title('Dimension Correlations', fontsize=10)

    for i in range(n_dims):
        for j in range(n_dims):
            if not np.isnan(corr_matrix[i, j]):
                ax6.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=7)

    # 7. Score Scatter Plot (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])

    info_scores = dimension_scores['informational']
    social_scores = dimension_scores['social']

    ax7.scatter(info_scores, social_scores, alpha=0.6, color=color, s=50, edgecolors='black')
    ax7.set_xlabel('Informational Harm (25% weight)')
    ax7.set_ylabel('Social Harm (20% weight)')
    ax7.set_title('Top 2 Weighted Dimensions')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-0.05, 1.05)
    ax7.set_ylim(-0.05, 1.05)
    ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    # 8. Dataset Characteristics (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    desc = DATASET_DESCRIPTIONS[dataset_name]
    char_text = f"{desc['title']}\n\n"
    char_text += "CHARACTERISTICS:\n"
    for i, char in enumerate(desc['characteristics'], 1):
        char_text += f"{i}. {char}\n"
    char_text += f"\nNOTE: {desc['notes']}"

    ax8.text(0.05, 0.95, char_text, transform=ax8.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    fig.suptitle(f'{DATASET_DESCRIPTIONS[dataset_name]["title"]} - Comprehensive Analysis\n' +
                 f'vLLM Full Run ({data["metadata"]["jury_config"]["aggregation_method"]} aggregation)',
                 fontsize=16, weight='bold', y=0.98)

    output_path = output_dir / f'{dataset_name}_comprehensive_analysis.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_sample_responses_visualization(dataset_name, data, jury_details, output_dir):
    """Create visualization showing sample high/low harm responses"""
    results = data['results']
    composite_scores = [r['final_score'] for r in results]

    # Build lookup from instance_id to jury_details entry
    details_by_id = {d['instance_id']: d for d in jury_details}

    sorted_indices = sorted(range(len(composite_scores)), key=lambda i: composite_scores[i])

    low_idx = sorted_indices[0]
    median_idx = sorted_indices[len(sorted_indices)//2]
    high_idx = sorted_indices[-1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    color = DATASET_COLORS[dataset_name]

    cases = [
        (low_idx, 'LOWEST HARM CASE', 'green'),
        (median_idx, 'MEDIAN HARM CASE', 'orange'),
        (high_idx, 'HIGHEST HARM CASE', 'red')
    ]

    for ax, (idx, title, title_color) in zip(axes, cases):
        result = results[idx]
        score = composite_scores[idx]
        instance_id = result['instance_id']

        # Get question/response from jury_details lookup
        detail = details_by_id.get(instance_id, {})
        question = detail.get('question', 'N/A')
        response = detail.get('response', 'N/A')

        # Get dimension scores from consolidated result
        case_dim_scores = {}
        for dim in DIMENSIONS:
            dim_key = f'{dim}_harm'
            if dim_key in result['dimension_scores']:
                case_dim_scores[dim] = result['dimension_scores'][dim_key]

        ax.axis('off')

        wrapped_q = textwrap.fill(f"Q: {question}", width=80)
        wrapped_r = textwrap.fill(f"A: {response[:300]}...", width=80)

        text_content = f"{title}\nComposite Score: {score:.3f}\n\n{wrapped_q}\n\n{wrapped_r}"

        ax.text(0.02, 0.98, text_content, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=title_color, alpha=0.2))

        ax_inset = ax.inset_axes([0.65, 0.05, 0.33, 0.4])
        dims = list(case_dim_scores.keys())
        scores_list = list(case_dim_scores.values())

        y_pos = np.arange(len(dims))
        ax_inset.barh(y_pos, scores_list, color=color, alpha=0.7, edgecolor='black')
        ax_inset.set_yticks(y_pos)
        ax_inset.set_yticklabels([d[:4].capitalize() for d in dims], fontsize=7)
        ax_inset.set_xlabel('Score', fontsize=8)
        ax_inset.set_xlim(0, 1.0)
        ax_inset.grid(axis='x', alpha=0.3)
        ax_inset.set_title('Dimension Scores', fontsize=9, weight='bold')

        for i, v in enumerate(scores_list):
            ax_inset.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7)

    fig.suptitle(f'{DATASET_DESCRIPTIONS[dataset_name]["title"]}\nSample Response Analysis (vLLM)',
                 fontsize=14, weight='bold', y=0.995)

    output_path = output_dir / f'{dataset_name}_sample_responses.png'
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def create_jury_agreement_analysis(dataset_name, data, jury_details, output_dir):
    """Analyze and visualize jury member agreement using per-member scores from jury_details"""
    results = data['results']
    jury_members = data['metadata']['jury_config']['members']

    # Build lookup from instance_id to jury_details entry
    details_by_id = {d['instance_id']: d for d in jury_details}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    color = DATASET_COLORS[dataset_name]

    # 1. Variance per instance across jury members for informational dimension (top left)
    ax1 = axes[0, 0]
    variances = []

    for result in results:
        instance_id = result['instance_id']
        detail = details_by_id.get(instance_id)
        if not detail:
            continue
        jury_scores = detail.get('jury_scores', {})
        info_scores = []
        for member in jury_members:
            if member in jury_scores:
                dim_entry = jury_scores[member].get('informational_harm', {})
                score = dim_entry.get('score') if isinstance(dim_entry, dict) else dim_entry
                if score is not None:
                    info_scores.append(score)
        if len(info_scores) > 1:
            variances.append(statistics.variance(info_scores))

    if variances:
        ax1.hist(variances, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax1.axvline(statistics.mean(variances), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {statistics.mean(variances):.3f}')
        ax1.set_xlabel('Jury Variance (Informational Dim.)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Jury Agreement Distribution')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Jury details not available', ha='center', va='center',
                 transform=ax1.transAxes)
        ax1.set_title('Jury Agreement Distribution')

    # 2. Mean scores per jury member (top right)
    ax2 = axes[0, 1]

    member_composites = defaultdict(list)
    for result in results:
        instance_id = result['instance_id']
        detail = details_by_id.get(instance_id)
        if not detail:
            continue
        jury_scores = detail.get('jury_scores', {})
        for member in jury_members:
            if member in jury_scores:
                member_dim_scores = jury_scores[member]
                composite = 0.0
                for dim in DIMENSIONS:
                    dim_entry = member_dim_scores.get(f'{dim}_harm', {})
                    score = dim_entry.get('score') if isinstance(dim_entry, dict) else dim_entry
                    if score is not None:
                        composite += score * DIMENSION_WEIGHTS[dim]
                member_composites[member].append(composite)

    members = [m for m in jury_members if m in member_composites]
    if members:
        means = [statistics.mean(member_composites[m]) for m in members]
        stds = [statistics.stdev(member_composites[m]) if len(member_composites[m]) > 1 else 0
                for m in members]

        x = np.arange(len(members))
        bars = ax2.bar(x, means, yerr=stds, capsize=5, color=color, alpha=0.7, edgecolor='black')

        ax2.set_ylabel('Mean Composite Score')
        ax2.set_xlabel('Jury Member')
        ax2.set_title('Average Scores by Jury Member')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.split('-')[0] for m in members], rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Jury details not available', ha='center', va='center',
                 transform=ax2.transAxes)
        ax2.set_title('Average Scores by Jury Member')

    # 3. Agreement categories (bottom left)
    ax3 = axes[1, 0]

    if variances:
        high_agreement = sum(1 for v in variances if v < 0.1)
        moderate_agreement = sum(1 for v in variances if 0.1 <= v < 0.3)
        low_agreement = sum(1 for v in variances if v >= 0.3)

        categories = ['High\nAgreement\n(var<0.1)', 'Moderate\nAgreement\n(0.1-0.3)',
                      'Low\nAgreement\n(var≥0.3)']
        counts = [high_agreement, moderate_agreement, low_agreement]
        colors_bar = ['#90EE90', '#FFD700', '#FF6B6B']

        bars = ax3.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of Instances')
        ax3.set_title('Jury Agreement Categories')
        ax3.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(variances)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, weight='bold')
    else:
        ax3.text(0.5, 0.5, 'Jury details not available', ha='center', va='center',
                 transform=ax3.transAxes)
        ax3.set_title('Jury Agreement Categories')

    # 4. Statistics text (bottom right)
    ax4 = axes[1, 1]
    ax4.axis('off')

    if variances:
        high_agreement = sum(1 for v in variances if v < 0.1)
        moderate_agreement = sum(1 for v in variances if 0.1 <= v < 0.3)
        low_agreement = sum(1 for v in variances if v >= 0.3)
        stats_text = f"""
JURY AGREEMENT ANALYSIS

Total Instances: {len(results)}
Jury Members: {len(jury_members)}

Variance Statistics:
  Mean:   {statistics.mean(variances):.4f}
  Median: {statistics.median(variances):.4f}
  Min:    {min(variances):.4f}
  Max:    {max(variances):.4f}

Agreement Categories:
  High:     {high_agreement:3d} ({high_agreement/len(variances)*100:5.1f}%)
  Moderate: {moderate_agreement:3d} ({moderate_agreement/len(variances)*100:5.1f}%)
  Low:      {low_agreement:3d} ({low_agreement/len(variances)*100:5.1f}%)

Jury Members:
"""
        for i, member in enumerate(jury_members, 1):
            stats_text += f"  {i}. {member}\n"
    else:
        stats_text = f"""
JURY AGREEMENT ANALYSIS

Total Instances: {len(results)}
Jury Members: {len(jury_members)}

(jury_details.json not available)

Jury Members:
"""
        for i, member in enumerate(jury_members, 1):
            stats_text += f"  {i}. {member}\n"

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle(f'{DATASET_DESCRIPTIONS[dataset_name]["title"]}\nJury Agreement Analysis (vLLM)',
                 fontsize=14, weight='bold', y=0.995)

    output_path = output_dir / f'{dataset_name}_jury_agreement.png'
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def generate_individual_report(dataset_name, data, output_dir):
    """Generate individual markdown report for dataset"""
    report_path = output_dir / f'{dataset_name}_detailed_report.md'

    dimension_scores, composite_scores = calculate_dimension_scores(data['results'])
    results = data['results']
    desc = DATASET_DESCRIPTIONS[dataset_name]
    jury_cfg = data['metadata']['jury_config']

    with open(report_path, 'w') as f:
        f.write(f"# {desc['title']}\n")
        f.write(f"**Detailed Analysis Report - vLLM Full Run**\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Dataset Overview
        f.write("## Dataset Overview\n\n")
        f.write(f"**Dataset:** {dataset_name.upper()}\n\n")
        f.write(f"**Total Samples:** {len(results)}\n\n")
        f.write(f"**Evaluation Date:** {data['metadata']['timestamp'][:10]}\n\n")
        f.write(f"**Aggregation Method:** {jury_cfg['aggregation_method']}\n\n")
        f.write(f"**Response Model:** {jury_cfg['response_model']}\n\n")
        f.write(f"**Critical Threshold:** {jury_cfg['critical_threshold']}\n\n")

        f.write("### Jury Members\n\n")
        for i, member in enumerate(jury_cfg['members'], 1):
            f.write(f"{i}. {member}\n")
        f.write("\n")

        f.write("### Characteristics\n\n")
        for char in desc['characteristics']:
            f.write(f"- {char}\n")
        f.write(f"\n**Note:** {desc['notes']}\n\n")

        f.write("---\n\n")

        # Statistical Summary
        f.write("## Statistical Summary\n\n")
        f.write("### Composite Harm Score\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean | {statistics.mean(composite_scores):.4f} |\n")
        f.write(f"| Median | {statistics.median(composite_scores):.4f} |\n")
        f.write(f"| Std Dev | {statistics.stdev(composite_scores):.4f} |\n")
        f.write(f"| Min | {min(composite_scores):.4f} |\n")
        f.write(f"| Max | {max(composite_scores):.4f} |\n\n")

        # Dimension Scores
        f.write("### Dimension Scores\n\n")
        f.write("| Dimension | Mean | Median | Std Dev | Weight |\n")
        f.write("|-----------|------|--------|---------|--------|\n")

        for dim in DIMENSIONS:
            scores = dimension_scores[dim]
            f.write(f"| **{dim.capitalize()}** | {statistics.mean(scores):.3f} | ")
            f.write(f"{statistics.median(scores):.3f} | {statistics.stdev(scores):.3f} | ")
            f.write(f"{DIMENSION_WEIGHTS[dim]:.2f} |\n")

        f.write("\n---\n\n")

        # Harm Distribution
        f.write("## Harm Distribution\n\n")
        low = sum(1 for s in composite_scores if s < 0.3)
        moderate = sum(1 for s in composite_scores if 0.3 <= s <= 0.5)
        high = sum(1 for s in composite_scores if s > 0.5)
        total = len(composite_scores)

        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        f.write(f"| Low (<0.3) | {low} | {low/total*100:.1f}% |\n")
        f.write(f"| Moderate (0.3-0.5) | {moderate} | {moderate/total*100:.1f}% |\n")
        f.write(f"| High (>0.5) | {high} | {high/total*100:.1f}% |\n\n")

        f.write("---\n\n")

        # Top/Bottom Cases
        f.write("## Notable Cases\n\n")

        sorted_indices = sorted(range(len(composite_scores)), key=lambda i: composite_scores[i])

        f.write("### Lowest Harm Cases (Top 3)\n\n")
        for i, idx in enumerate(sorted_indices[:3], 1):
            result = results[idx]
            score = composite_scores[idx]
            f.write(f"**{i}. Score: {score:.3f}** (instance: {result['instance_id']})\n\n")

        f.write("### Highest Harm Cases (Top 3)\n\n")
        for i, idx in enumerate(sorted_indices[-3:], 1):
            result = results[idx]
            score = composite_scores[idx]
            f.write(f"**{i}. Score: {score:.3f}** (instance: {result['instance_id']})\n\n")

        f.write("---\n\n")

        # Visualizations
        f.write("## Generated Visualizations\n\n")
        f.write("The following visualizations have been generated for this dataset:\n\n")
        f.write(f"1. `{dataset_name}_comprehensive_analysis.png` - Complete overview with 8 subplots\n")
        f.write(f"2. `{dataset_name}_sample_responses.png` - Sample high/medium/low harm cases\n")
        f.write(f"3. `{dataset_name}_jury_agreement.png` - Jury member agreement analysis\n\n")

        f.write("---\n\n")
        f.write(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"  ✓ Saved: {report_path.name}")


def main():
    """Main visualization pipeline"""
    print("=" * 80)
    print("INDIVIDUAL DATASET VISUALIZATION - vLLM FULL RUNS")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    parser = argparse.ArgumentParser(
        description="Individual dataset visualization for vLLM full runs"
    )
    parser.add_argument(
        "--results-dir",
        default="data/results/vllm/full_runs",
        help="Base results directory (default: data/results/vllm/full_runs)"
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    files = {
        'medqa': results_dir / "medqa_consolidated.json",
        'pubmedqa': results_dir / "pubmedqa_consolidated.json",
        'medmcqa': results_dir / "medmcqa_consolidated.json"
    }
    jury_detail_files = {
        'medqa': results_dir / "medqa_full_results/jury_details.json",
        'pubmedqa': results_dir / "pubmedqa_full_results/jury_details.json",
        'medmcqa': results_dir / "medmcqa_full_results/jury_details.json"
    }

    # Create output directory
    output_dir = results_dir / "Jury_v3" / "individual"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Process each dataset
    for dataset_name, file_path in files.items():
        print(f"\nProcessing {dataset_name.upper()}...")
        print("-" * 80)

        if not file_path.exists():
            print(f"  ✗ Error: {file_path} not found")
            continue

        with open(file_path) as f:
            data = json.load(f)

        print(f"  Loaded {len(data['results'])} samples")

        # Load jury details for per-member analysis and question/response text
        jury_details = []
        jury_detail_path = jury_detail_files[dataset_name]
        if jury_detail_path.exists():
            with open(jury_detail_path) as f:
                jury_details = json.load(f)
            print(f"  Loaded {len(jury_details)} jury detail entries")
        else:
            print(f"  Warning: jury_details.json not found, skipping per-member analysis")

        # Generate visualizations
        print("  Generating visualizations...")
        create_dataset_overview_plot(dataset_name, data, output_dir)
        create_sample_responses_visualization(dataset_name, data, jury_details, output_dir)
        create_jury_agreement_analysis(dataset_name, data, jury_details, output_dir)

        # Generate report
        print("  Generating report...")
        generate_individual_report(dataset_name, data, output_dir)

        print(f"  ✓ {dataset_name.upper()} complete")

    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nOutput location: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*')))}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
