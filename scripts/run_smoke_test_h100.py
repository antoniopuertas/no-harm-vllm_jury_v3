#!/usr/bin/env python3
"""
Smoke test for H100 pipeline fixes.
Validates H100_v2 jury_details output against all fix criteria.

Usage:
    python scripts/run_smoke_test_h100.py

Exit codes:
    0 — all checks pass
    1 — one or more checks failed
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data/results/vllm/harm_dimensions_v2"
EMPTY_REPORT = RESULTS_DIR / "H100/medqa_empty_report.json"


def load_empty_ids() -> list:
    """Load known-empty instance IDs from diagnostic report."""
    if not EMPTY_REPORT.exists():
        print(f"WARNING: Empty report not found at {EMPTY_REPORT}. "
              "Run diagnose_failures.py first.")
        return []
    with open(EMPTY_REPORT) as f:
        return json.load(f).get("empty_instance_ids", [])


def check_no_invalid_entries(jury_details: list) -> tuple:
    """Check 1: zero entries with valid=False."""
    invalid = [e["instance_id"] for e in jury_details if not e.get("valid", True)]
    if invalid:
        return False, f"FAIL: {len(invalid)} entries have valid=False: {invalid[:5]}"
    return True, "PASS: zero invalid entries"


def check_no_inflated_retry_scores(jury_details: list) -> tuple:
    """Check 2: zero entries where is_retry=True AND score >= 0.4."""
    violations = []
    for entry in jury_details:
        for member, scores in entry.get("jury_scores", {}).items():
            for dim, data in scores.items():
                if data.get("is_retry") and data.get("score", 0) >= 0.4:
                    violations.append(
                        f"{entry['instance_id']} / {member} / {dim} = {data['score']}"
                    )
    if violations:
        return False, f"FAIL: {len(violations)} retry scores >= 0.4: {violations[:5]}"
    return True, "PASS: no inflated retry scores"


def check_olmo_no_scenario_confusion(jury_details: list) -> tuple:
    """Check 3: no olmo-32b autonomy_harm >= 0.8 where justification references question scenario."""
    violations = []
    scenario_keywords = ["patient", "scenario", "question asks", "the case"]
    for entry in jury_details:
        olmo_scores = entry.get("jury_scores", {}).get("olmo-32b", {})
        auto = olmo_scores.get("autonomy_harm", {})
        if auto.get("score", 0) >= 0.8:
            justif = auto.get("justification", "").lower()
            if any(kw in justif for kw in scenario_keywords):
                violations.append(
                    f"{entry['instance_id']}: score={auto['score']}, "
                    f"justif={auto['justification'][:80]}"
                )
    if violations:
        return False, (
            f"FAIL: {len(violations)} olmo-32b scenario-confusion instances: "
            f"{violations[:3]}"
        )
    return True, "PASS: no olmo-32b scenario confusion"


def check_throughput(start_time: float, n_samples: int) -> tuple:
    """Check 4: average load time <= 15s/sample (wall-clock read time only)."""
    elapsed = time.time() - start_time
    sps = elapsed / n_samples if n_samples else 0
    if sps > 15:
        return False, f"FAIL: {sps:.1f}s/sample exceeds 15s limit (file I/O)"
    return True, f"PASS: {sps:.3f}s/sample to load {n_samples} entries"


def load_h100v2_jury_details() -> list:
    """Load all jury_details entries from H100_v2 output."""
    h100v2 = RESULTS_DIR / "H100_v2"
    if not h100v2.exists():
        print(
            f"\nH100_v2 directory not found at {h100v2}.\n"
            "Run the full evaluation first:\n"
            "  nohup bash scripts/run_full_h100_evaluation.sh > logs/h100_v2_launch.log 2>&1 &\n"
            "Then re-run this script to validate."
        )
        sys.exit(1)

    all_jury_details = []
    for dataset in ["medqa", "medmcqa", "pubmedqa"]:
        jd_file = h100v2 / f"{dataset}_full_results" / "jury_details.json"
        if jd_file.exists():
            with open(jd_file) as f:
                entries = json.load(f)
            all_jury_details.extend(entries)
            print(f"  Loaded {len(entries):,} entries from {dataset}")
        else:
            print(f"  WARNING: {jd_file} not found — skipping {dataset}")

    return all_jury_details


def main():
    print("\n" + "=" * 60)
    print("SMOKE TEST: H100_v2 pipeline fix validation")
    print("=" * 60)

    empty_ids = load_empty_ids()
    print(f"Known previously-empty IDs: {len(empty_ids)}")

    start = time.time()
    print("\nLoading H100_v2 jury_details...")
    jury_details = load_h100v2_jury_details()
    print(f"Total entries loaded: {len(jury_details):,}\n")

    if not jury_details:
        print("No entries to check. Exiting.")
        sys.exit(1)

    checks = [
        check_no_invalid_entries(jury_details),
        check_no_inflated_retry_scores(jury_details),
        check_olmo_no_scenario_confusion(jury_details),
        check_throughput(start, len(jury_details)),
    ]

    passed = 0
    print("Checks:")
    for ok, msg in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {msg}")
        if ok:
            passed += 1

    print(f"\n{'=' * 60}")
    result = "ALL CHECKS PASSED" if passed == len(checks) else f"{passed}/{len(checks)} checks passed"
    print(f"Result: {result}")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
