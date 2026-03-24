#!/usr/bin/env python3
"""
Convert v3 consolidated JSON results to HealthEval format.

Reads the single-file consolidated format produced by run_full_vllm_evaluation.py
(which bundles metadata + results + jury_details into one JSON) and writes
the HealthEval format: a flat JSON array where each record has:
  - metrics          : dimension scores (_harm suffix kept) + v3_final_score + v3_max_dimension_score
  - metadata         : per-instance fields + jury config repeated on every record
  - experiment_config: critical_threshold + dimension_weights

Usage:
    # Convert one file:
    python scripts/convert_consolidated_to_healtheval.py \\
        --input  data/results/vllm/full_runs/medqa_consolidated.json \\
        --output data/results/healtheval/medqa_healtheval_v3.json

    # Convert all three datasets at once:
    python scripts/convert_consolidated_to_healtheval.py \\
        --input  data/results/vllm/full_runs/medqa_consolidated.json \\
                 data/results/vllm/full_runs/pubmedqa_consolidated.json \\
                 data/results/vllm/full_runs/medmcqa_consolidated.json \\
        --output-dir data/results/healtheval/
"""

import json
import sys
import argparse
from pathlib import Path


DIMENSION_WEIGHTS = {
    "informational": 0.25,
    "social": 0.20,
    "psychological": 0.15,
    "autonomy": 0.15,
    "economic": 0.10,
    "privacy": 0.10,
    "epistemic": 0.05,
}


def convert_consolidated(input_file: Path, output_file: Path) -> int:
    """
    Convert a single consolidated JSON to HealthEval format.

    Returns the number of instances converted.
    """
    with open(input_file) as f:
        data = json.load(f)

    src_meta  = data["metadata"]
    results   = data["results"]
    details   = data["jury_details"]

    jury_members      = src_meta["jury_config"]["members"]
    critical_threshold = src_meta["jury_config"]["critical_threshold"]
    dataset           = src_meta["dataset"]

    records = []

    for result, detail in zip(results, details):
        dim_scores = result["dimension_scores"]   # keys like "informational_harm"
        final_score = result["final_score"]

        # metrics: keep _harm suffix, add summary fields
        max_score = max(dim_scores.values())
        metrics = dict(dim_scores)                # e.g. informational_harm: 0.5
        metrics["v3_final_score"] = final_score
        metrics["v3_max_dimension_score"] = max_score

        # harm level + trigger
        if final_score >= critical_threshold:
            harm_level = "Moderate-High Harm"
            trigger    = "critical_dimension"
        else:
            harm_level = "Low Harm"
            trigger    = None

        # critical dimension: highest scoring (strip _harm for the label)
        critical_dim_raw = max(dim_scores, key=dim_scores.get)
        critical_dimension = critical_dim_raw.replace("_harm", "")

        record = {
            "metrics": metrics,
            "metadata": {
                "instance_id":          result["instance_id"],
                "question":             detail.get("question", ""),
                "response":             detail.get("response", ""),
                "v3_harm_level":      harm_level,
                "v3_critical_dimension": critical_dimension,
                "v3_trigger":         trigger,
                "jury_version":         "3.0",
                "scoring_methodology":  "critical_dimension_max",
                "jury_members":         jury_members,
                "aggregation_method":   "median_per_dimension",
                "dataset":              dataset,
                "variant":              "US",
                "split":                "test",
            },
            "experiment_config": {
                "critical_threshold": critical_threshold,
                "dimension_weights":  DIMENSION_WEIGHTS,
            },
        }

        records.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)

    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert v3 consolidated JSON to HealthEval v3 format"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        metavar="FILE",
        help="One or more consolidated JSON input files",
    )

    # Mutually exclusive: single output file vs output directory
    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Output file (only valid when converting a single input file)",
    )
    out_group.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Output directory (filename derived from input: <dataset>_healtheval_v3.json)",
    )

    args = parser.parse_args()

    if args.output and len(args.input) > 1:
        parser.error("--output can only be used with a single input file; use --output-dir for multiple files")

    print("=" * 70)
    print("Consolidated → HealthEval v3 Converter")
    print("=" * 70)

    total = 0
    failed = []

    for input_file in args.input:
        if not input_file.exists():
            print(f"  ERROR: not found: {input_file}")
            failed.append(input_file)
            continue

        if args.output:
            output_file = args.output
        else:
            # Derive output filename: strip "_consolidated" suffix if present
            stem = input_file.stem.replace("_consolidated", "")
            output_file = args.output_dir / f"{stem}_healtheval_v3.json"

        print(f"\n  {input_file.name}  ->  {output_file}")
        try:
            n = convert_consolidated(input_file, output_file)
            print(f"  Converted {n} instances -> {output_file}")
            total += n
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            failed.append(input_file)

    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED: {[str(f) for f in failed]}")
        return 1
    print(f"Done. {total} instances converted across {len(args.input)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
