"""Filter persistence checks and missing-metadata fallback tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.hitl.queries import (
    BUCKETS_BY_PROFILE,
    filter_by_bucket,
    load_cluster_frame,
)
from src.hitl.status import diagnostics_sidebar_summary, load_diagnostics_safe
from src.matching.writer import _encode_run_id_path_segment
from src.resolution.writer import (
    CLUSTERS_FILENAME,
    RESOLUTION_DIAGNOSTICS_FILENAME,
    RESOLUTION_STAGE_DIRNAME,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_resolution_artifact(
    data_dir: Path,
    run_id: str,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    """Write one resolution JSON artifact to the per-run output directory."""
    dir_name = _encode_run_id_path_segment(run_id)
    resolution_dir = data_dir / "runs" / dir_name / RESOLUTION_STAGE_DIRNAME
    resolution_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolution_dir / filename
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _sample_diagnostics() -> dict[str, Any]:
    """Minimal diagnostics payload for sidebar tests."""
    return {
        "cluster_count": 12,
        "cluster_singleton_count": 4,
        "cluster_singleton_rate": 0.333333,
        "selected_routing_profile": "quick_low_hitl",
    }


def _build_filter_test_clusters() -> dict[str, Any]:
    """Small clusters.json payload for filter consistency checks."""
    return {
        "run_id": "filter_run",
        "cluster_count": 3,
        "clusters": [
            {
                "cluster_id": "c1",
                "cluster_size": 4,
                "confidence": 0.91,
                "min_edge_score": 0.85,
                "density": 1.0,
                "canonical_name": "A",
                "canonical_type": "ORG",
                "suspicious_merge": False,
                "routing_actions_by_profile": {
                    "balanced_hitl": "auto_merge",
                    "quick_low_hitl": "auto_merge",
                },
            },
            {
                "cluster_id": "c2",
                "cluster_size": 2,
                "confidence": 0.70,
                "min_edge_score": 0.60,
                "density": 1.0,
                "canonical_name": "B",
                "canonical_type": "PER",
                "suspicious_merge": True,
                "routing_actions_by_profile": {
                    "balanced_hitl": "review",
                    "quick_low_hitl": "defer",
                },
            },
            {
                "cluster_id": "c3",
                "cluster_size": 1,
                "confidence": 0.0,
                "min_edge_score": 0.0,
                "density": 1.0,
                "canonical_name": "C",
                "canonical_type": "LOC",
                "suspicious_merge": False,
                "routing_actions_by_profile": {
                    "balanced_hitl": "keep_separate",
                    "quick_low_hitl": "keep_separate",
                },
            },
        ],
    }


# ── load_diagnostics_safe ─────────────────────────────────────────────────────


def test_load_diagnostics_safe_returns_dict(tmp_path: Path) -> None:
    run_id = "diag_run"
    _write_resolution_artifact(
        tmp_path, run_id, RESOLUTION_DIAGNOSTICS_FILENAME, _sample_diagnostics()
    )
    result = load_diagnostics_safe(tmp_path, run_id)
    assert result is not None
    assert result["cluster_count"] == 12


def test_load_diagnostics_safe_returns_none_for_missing(tmp_path: Path) -> None:
    assert load_diagnostics_safe(tmp_path, "no_such_run") is None


def test_load_diagnostics_safe_returns_none_for_corrupt_json(
    tmp_path: Path,
) -> None:
    run_id = "corrupt_run"
    dir_name = _encode_run_id_path_segment(run_id)
    resolution_dir = tmp_path / "runs" / dir_name / RESOLUTION_STAGE_DIRNAME
    resolution_dir.mkdir(parents=True, exist_ok=True)
    (resolution_dir / RESOLUTION_DIAGNOSTICS_FILENAME).write_text(
        "not valid json{{{", encoding="utf-8"
    )
    assert load_diagnostics_safe(tmp_path, run_id) is None


# ── diagnostics_sidebar_summary ───────────────────────────────────────────────


def test_sidebar_summary_contains_expected_keys() -> None:
    summary = diagnostics_sidebar_summary(_sample_diagnostics())
    assert "Clusters" in summary
    assert "Singleton rate" in summary
    assert "Routing profile" in summary


def test_sidebar_summary_formats_singleton_rate_as_percent() -> None:
    summary = diagnostics_sidebar_summary(_sample_diagnostics())
    assert "%" in summary["Singleton rate"]


def test_sidebar_summary_handles_missing_fields() -> None:
    """Empty diagnostics dict should produce '?' placeholders, not crash."""
    summary = diagnostics_sidebar_summary({})
    assert summary["Clusters"] == "?"
    assert summary["Singleton rate"] == "?"
    assert summary["Routing profile"] == "?"


# ── Filter consistency ────────────────────────────────────────────────────────


def test_switching_profile_changes_bucket_assignment(tmp_path: Path) -> None:
    """The same cluster gets different bucket labels under different profiles."""
    _write_resolution_artifact(
        tmp_path, "filter_run", CLUSTERS_FILENAME, _build_filter_test_clusters()
    )
    frame = load_cluster_frame(tmp_path, "filter_run")

    balanced_review = filter_by_bucket(frame, "balanced_hitl", "review")
    quick_defer = filter_by_bucket(frame, "quick_low_hitl", "defer")

    # The "review" cluster in balanced == the "defer" cluster in quick
    assert balanced_review["cluster_id"].to_list() == quick_defer["cluster_id"].to_list()


def test_all_buckets_cover_all_clusters(tmp_path: Path) -> None:
    """Union of all bucket filters should cover every cluster in the frame."""
    _write_resolution_artifact(
        tmp_path, "filter_run", CLUSTERS_FILENAME, _build_filter_test_clusters()
    )
    frame = load_cluster_frame(tmp_path, "filter_run")

    for profile, buckets in BUCKETS_BY_PROFILE.items():
        total = sum(
            filter_by_bucket(frame, profile, bucket).height
            for bucket in buckets
        )
        assert total == frame.height, f"bucket union mismatch for {profile}"


def test_filter_returns_empty_for_nonexistent_bucket(tmp_path: Path) -> None:
    _write_resolution_artifact(
        tmp_path, "filter_run", CLUSTERS_FILENAME, _build_filter_test_clusters()
    )
    frame = load_cluster_frame(tmp_path, "filter_run")
    filtered = filter_by_bucket(frame, "balanced_hitl", "defer")
    assert filtered.is_empty()
