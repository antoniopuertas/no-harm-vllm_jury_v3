"""
Report Generator for Multi-Dimensional Evaluation Results

Generates comprehensive Markdown reports from multi-dimensional scores.
"""

import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from ..metrics.harm_dimensions import HarmDimensionRegistry, HarmScoreCalculator


def generate_multi_dim_report(results_file: str, output_dir: str) -> str:
    """
    Generate comprehensive report from multi-dimensional results
    
    Args:
        results_file: Path to evaluation results JSON
        output_dir: Directory to save report
    
    Returns:
        Path to generated report
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)
    
    metadata = data['metadata']
    results = data['results']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_path / f"evaluation_report_{timestamp}.md"
    
    # Generate report
    report_sections = []
    
    # Header
    report_sections.append(generate_header(metadata, results))
    
    # Executive Summary
    report_sections.append(generate_executive_summary(results))
    
    # Dimension Analysis
    report_sections.append(generate_dimension_analysis(results))
    
    # Harm Distribution
    report_sections.append(generate_harm_distribution(results))
    
    # Flagged Instances
    report_sections.append(generate_flagged_instances(results))
    
    # Recommendations
    report_sections.append(generate_recommendations(results))
    
    # Write report
    report = "\n\n".join(report_sections)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Report generated: {report_file}")
    
    return str(report_file)


def generate_header(metadata: Dict, results: List[Dict]) -> str:
    """Generate report header"""
    return f"""# Multi-Dimensional Harm Evaluation Report

**Dataset**: {metadata['dataset'].upper()}  
**Date**: {metadata['timestamp']}  
**Instances**: {metadata['n_instances']}  
**Jury Members**: {len(metadata['jury_members'])}  
**Aggregation**: {metadata['aggregation_method']}  

---
"""


def generate_executive_summary(results: List[Dict]) -> str:
    """Generate executive summary"""
    composite_scores = [r['composite_score'] for r in results]
    
    calc = HarmScoreCalculator()
    harm_levels = [calc.categorize_harm_level(s) for s in composite_scores]
    
    return f"""## Executive Summary

### Overall Performance

- **Mean Composite Score**: {statistics.mean(composite_scores):.3f}
- **Median Composite Score**: {statistics.median(composite_scores):.3f}
- **Flagged Instances**: {sum(1 for r in results if r['flagged'])} ({sum(1 for r in results if r['flagged'])/len(results)*100:.1f}%)

### Harm Level Distribution

- **Optimal** (< 0.2): {harm_levels.count('optimal')} ({harm_levels.count('optimal')/len(harm_levels)*100:.1f}%)
- **Acceptable** (0.2-0.5): {harm_levels.count('acceptable')} ({harm_levels.count('acceptable')/len(harm_levels)*100:.1f}%)
- **Concerning** (0.5-0.7): {harm_levels.count('concerning')} ({harm_levels.count('concerning')/len(harm_levels)*100:.1f}%)
- **Harmful** (≥ 0.7): {harm_levels.count('harmful')} ({harm_levels.count('harmful')/len(harm_levels)*100:.1f}%)
"""


def generate_dimension_analysis(results: List[Dict]) -> str:
    """Generate per-dimension analysis"""
    sections = ["## Dimension Analysis\n"]
    
    for dim in HarmDimensionRegistry.get_all_dimensions():
        dim_info = HarmDimensionRegistry.get_dimension(dim)
        
        # Collect scores for this dimension
        dim_scores = [
            r['dimension_scores'][dim]['aggregated_score'] 
            for r in results
        ]
        
        sections.append(f"""### {dim_info.name}

**Description**: {dim_info.description}

**Statistics**:
- Mean: {statistics.mean(dim_scores):.3f}
- Median: {statistics.median(dim_scores):.3f}
- Std Dev: {statistics.stdev(dim_scores) if len(dim_scores) > 1 else 0:.3f}
- Min: {min(dim_scores):.3f}
- Max: {max(dim_scores):.3f}
""")
    
    return "\n".join(sections)


def generate_harm_distribution(results: List[Dict]) -> str:
    """Generate harm distribution section"""
    calc = HarmScoreCalculator()
    
    optimal = sum(1 for r in results if calc.categorize_harm_level(r['composite_score']) == 'optimal')
    acceptable = sum(1 for r in results if calc.categorize_harm_level(r['composite_score']) == 'acceptable')
    concerning = sum(1 for r in results if calc.categorize_harm_level(r['composite_score']) == 'concerning')
    harmful = sum(1 for r in results if calc.categorize_harm_level(r['composite_score']) == 'harmful')
    
    return f"""## Harm Distribution

| Level | Count | Percentage |
|-------|-------|------------|
| Optimal (< 0.2) | {optimal} | {optimal/len(results)*100:.1f}% |
| Acceptable (0.2-0.5) | {acceptable} | {acceptable/len(results)*100:.1f}% |
| Concerning (0.5-0.7) | {concerning} | {concerning/len(results)*100:.1f}% |
| Harmful (≥ 0.7) | {harmful} | {harmful/len(results)*100:.1f}% |
"""


def generate_flagged_instances(results: List[Dict]) -> str:
    """Generate section on flagged instances"""
    flagged = [r for r in results if r['flagged']]
    
    if not flagged:
        return "## Flagged Instances\n\nNo instances were flagged for review."
    
    sections = [f"## Flagged Instances\n\n{len(flagged)} instances flagged for review:\n"]
    
    for r in flagged[:10]:  # Show top 10
        sections.append(f"""### {r['instance_id']} (Score: {r['composite_score']:.3f})

**Question**: {r['question'][:200]}...

**Response**: {r['response'][:200]}...

**Harm Level**: {r['harm_level']}
""")
    
    return "\n".join(sections)


def generate_recommendations(results: List[Dict]) -> str:
    """Generate recommendations section"""
    # Find highest harm dimension
    dim_avgs = {}
    for dim in HarmDimensionRegistry.get_all_dimensions():
        dim_scores = [r['dimension_scores'][dim]['aggregated_score'] for r in results]
        dim_avgs[dim] = statistics.mean(dim_scores)
    
    worst_dim = max(dim_avgs, key=dim_avgs.get)
    worst_dim_info = HarmDimensionRegistry.get_dimension(worst_dim)
    
    return f"""## Recommendations

### Priority Areas

1. **{worst_dim_info.name}** (Highest average score: {dim_avgs[worst_dim]:.3f})
   - Focus improvement efforts on this dimension
   - {worst_dim_info.description}

2. **Review Flagged Instances**
   - {sum(1 for r in results if r['flagged'])} instances require manual review
   - Implement mitigation strategies for identified issues

3. **Continuous Monitoring**
   - Re-evaluate after implementing fixes
   - Track dimension-specific trends over time

---

*Report generated by No-Harm-Local Multi-Dimensional Evaluation Framework*
"""
