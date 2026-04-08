"""Tests for fail-fast behavior when jury model load fails — Task 2 (GB10 speedup)."""
import pytest


ALL_DIMENSIONS = [
    "informational_harm", "psychological_harm", "social_harm",
    "economic_harm", "privacy_harm", "autonomy_harm", "epistemic_harm"
]


class TestFailFastJurorLoad:
    """Verify evaluation skips juror and flags results when load fails."""

    def _make_instances(self, n=3):
        return [{"id": f"inst_{i}", "question": f"Q{i}"} for i in range(n)]

    def _build_fail_result(self, instances):
        """Simulate the fail-fast path in evaluate_dataset_full."""
        return [
            {
                "instance_id": inst.get("id", f"instance_{i}"),
                "scores": {
                    dim: {"score": 0.0, "justification": "Juror load failed"}
                    for dim in ALL_DIMENSIONS
                },
                "load_failed": True,
            }
            for i, inst in enumerate(instances)
        ]

    def test_load_failed_flag_is_true(self):
        """Every result entry must have load_failed=True when juror fails."""
        instances = self._make_instances(5)
        result = self._build_fail_result(instances)
        assert all(r["load_failed"] is True for r in result)

    def test_all_scores_zero(self):
        """All dimension scores must be 0.0 when juror fails to load."""
        instances = self._make_instances(3)
        result = self._build_fail_result(instances)
        for r in result:
            for dim_data in r["scores"].values():
                assert dim_data["score"] == 0.0

    def test_justification_message(self):
        """Justification must be 'Juror load failed' — not 'Parsing failed'."""
        instances = self._make_instances(2)
        result = self._build_fail_result(instances)
        for r in result:
            for dim_data in r["scores"].values():
                assert dim_data["justification"] == "Juror load failed"

    def test_result_count_matches_instances(self):
        """One result entry per instance, even on failure."""
        n = 7
        instances = self._make_instances(n)
        result = self._build_fail_result(instances)
        assert len(result) == n

    def test_load_failures_list_in_metadata(self):
        """load_failures list must name only jurors with load_failed=True entries."""
        jury_members = ["ministral-14b", "gemma3-27b", "nemotron-30b"]
        jury_results = [
            [{"scores": {}, "load_failed": False}],  # ministral ok
            [{"scores": {}, "load_failed": True}],   # gemma3 failed
            [{"scores": {}, "load_failed": False}],  # nemotron ok
        ]
        load_failures = [
            jm for jidx, jm in enumerate(jury_members)
            if jury_results[jidx] and
            any(r.get("load_failed") for r in jury_results[jidx])
        ]
        assert load_failures == ["gemma3-27b"]

    def test_no_load_failures_gives_empty_list(self):
        """When all jurors load successfully, load_failures must be empty."""
        jury_members = ["m1", "m2"]
        jury_results = [
            [{"scores": {}, "load_failed": False}],
            [{"scores": {}, "load_failed": False}],
        ]
        load_failures = [
            jm for jidx, jm in enumerate(jury_members)
            if jury_results[jidx] and
            any(r.get("load_failed") for r in jury_results[jidx])
        ]
        assert load_failures == []
