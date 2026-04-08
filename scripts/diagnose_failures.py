#!/usr/bin/env python3
"""
Diagnose empty-response failures in H100 jury_details.json.
Usage: python scripts/diagnose_failures.py [--dataset medqa] [--gpu H100]
"""
import json
import argparse
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data/results/vllm/harm_dimensions_v2"


def diagnose(gpu: str, dataset: str) -> dict:
    jury_file = RESULTS_DIR / gpu / f"{dataset}_full_results" / "jury_details.json"
    if not jury_file.exists():
        raise FileNotFoundError(f"Not found: {jury_file}")

    with open(jury_file) as f:
        entries = json.load(f)

    if not entries:
        raise ValueError(f"jury_details.json is empty: {jury_file}")

    empty_indices = [i for i, e in enumerate(entries) if not (e.get("response") or "").strip()]
    empty_ids     = [entries[i].get("instance_id", f"<no-id@{i}>") for i in empty_indices]

    all_q_lens   = [len(e.get("question", "")) for e in entries]
    empty_q_lens = [len(entries[i].get("question", "")) for i in empty_indices]

    report = {
        "gpu": gpu,
        "dataset": dataset,
        "total_entries": len(entries),
        "empty_count": len(empty_indices),
        "empty_pct": round(100 * len(empty_indices) / len(entries), 2),
        "empty_instance_ids": empty_ids,
        "empty_indices": empty_indices,
        "index_min": min(empty_indices) if empty_indices else None,
        "index_max": max(empty_indices) if empty_indices else None,
        "index_span": (max(empty_indices) - min(empty_indices) + 1) if empty_indices else 0,
        "clustered": (
            (max(empty_indices) - min(empty_indices) + 1) == len(empty_indices)
            if empty_indices else False
        ),
        "question_len_all_mean":   round(statistics.mean(all_q_lens), 1),
        "question_len_empty_mean": round(statistics.mean(empty_q_lens), 1) if empty_q_lens else None,
        "question_len_empty_max":  max(empty_q_lens) if empty_q_lens else None,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medqa")
    parser.add_argument("--gpu",     default="H100")
    parser.add_argument("--out",     default=None, help="Write JSON report to file")
    args = parser.parse_args()

    report = diagnose(args.gpu, args.dataset)

    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: {args.gpu} / {args.dataset}")
    print(f"{'='*60}")
    print(f"Total entries : {report['total_entries']}")
    print(f"Empty responses: {report['empty_count']} ({report['empty_pct']}%)")
    if report['empty_count']:
        print(f"Index range   : {report['index_min']} – {report['index_max']} "
              f"(span {report['index_span']})")
        print(f"Clustered     : {report['clustered']} (contiguous block of failures)")
        print(f"Q-len (all)   : {report['question_len_all_mean']}")
        print(f"Q-len (empty) : {report['question_len_empty_mean']} "
              f"(max {report['question_len_empty_max']})")
        print(f"\nEmpty IDs (first 20): {report['empty_instance_ids'][:20]}")
    print(f"{'='*60}\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.out}")

    return report


if __name__ == "__main__":
    main()
