"""Query-level checks for cluster bucket counts, summaries, and listing consistency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.hitl.queries import (
    BUCKETS_BY_PROFILE,
    bucket_counts,
    bucket_summary,
    discover_run_ids,
    filter_by_bucket,
    load_cluster_frame,
    load_cluster_frame_safe,
    size_distribution,
)
from src.matching.writer import _encode_run_id_path_segment
from src.resolution.writer import CLUSTERS_FILENAME, RESOLUTION_STAGE_DIRNAME


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_test_clusters(run_id: str) -> dict[str, Any]:
    """Build a synthetic clusters.json payload with mixed bucket assignments.

    Fixture layout (5 clusters):
      - 2 x auto_merge  (both profiles)
      - 1 x review (balanced) / defer (quick)
      - 2 x keep_separate (both profiles, singletons)
    """
    return {
        "run_id": run_id,
        "cluster_count": 5,
        "clusters": [
            {
                "cluster_id": "cluster_am_1",
                "cluster_size": 3,
                "confidence": 0.92,
                "min_edge_score": 0.88,
                "density": 1.0,
                "canonical_name": "Den Norske Bank",
                "canonical_type": "ORG",
                "suspicious_merge": False,
                "routing_actions_by_profile": {
                    "balanced_hitl": "auto_merge",
                    "quick_low_hitl": "auto_merge",
                },
            },
            {
                "cluster_id": "cluster_am_2",
                "cluster_size": 2,
                "confidence": 0.90,
                "min_edge_score": 0.87,
                "density": 1.0,
                "canonical_name": "Kari Nordmann",
                "canonical_type": "PER",
                "suspicious_merge": False,
                "routing_actions_by_profile": {
                    "balanced_hitl": "auto_merge",
                    "quick_low_hitl": "auto_merge",
                },
            },
            {
                "cluster_id": "cluster_review_1",
                "cluster_size": 2,
                "confidence": 0.65,
                "min_edge_score": 0.55,
                "density": 1.0,
                "canonical_name": "Sparebank 1",
                "canonical_type": "ORG",
                "suspicious_merge": True,
                "routing_actions_by_profile": {
                    "balanced_hitl": "review",
                    "quick_low_hitl": "defer",
                },
            },
            {
                "cluster_id": "cluster_ks_1",
                "cluster_size": 1,
                "confidence": 0.0,
                "min_edge_score": 0.0,
                "density": 1.0,
                "canonical_name": "Oslo",
                "canonical_type": "LOC",
                "suspicious_merge": False,
                "routing_actions_by_profile": {
                    "balanced_hitl": "keep_separate",
                    "quick_low_hitl": "keep_separate",
                },
            },
            {
                "cluster_id": "cluster_ks_2",
                "cluster_size": 1,
                "confidence": 0.0,
                "min_edge_score": 0.0,
                "density": 1.0,
                "canonical_name": "Bergen",
                "canonical_type": "LOC",
                "suspicious_merge": False,
                "routing_actions_by_profile": {
                    "balanced_hitl": "keep_separate",
                    "quick_low_hitl": "keep_separate",
                },
            },
        ],
    }


def _write_clusters_json(
    data_dir: Path, run_id: str, payload: dict[str, Any]
) -> Path:
    """Write a clusters.json file in the expected per-run directory structure."""
    dir_name = _encode_run_id_path_segment(run_id)
    resolution_dir = data_dir / "runs" / dir_name / RESOLUTION_STAGE_DIRNAME
    resolution_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolution_dir / CLUSTERS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


@pytest.fixture()
def populated_data_dir(tmp_path: Path) -> tuple[Path, str]:
    """Create a temp data dir with one run's clusters.json."""
    run_id = "test_run_001"
    _write_clusters_json(tmp_path, run_id, _build_test_clusters(run_id))
    return tmp_path, run_id


# ── discover_run_ids ──────────────────────────────────────────────────────────


def test_discover_run_ids_finds_one_run(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    assert discover_run_ids(data_dir) == [run_id]


def test_discover_run_ids_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert discover_run_ids(tmp_path) == []


def test_discover_run_ids_ignores_incomplete_runs(tmp_path: Path) -> None:
    """A run directory without clusters.json should be skipped."""
    dir_name = _encode_run_id_path_segment("incomplete_run")
    (tmp_path / "runs" / dir_name / RESOLUTION_STAGE_DIRNAME).mkdir(parents=True)
    assert discover_run_ids(tmp_path) == []


def test_discover_run_ids_finds_multiple_runs_sorted(tmp_path: Path) -> None:
    for run_id in ["run_c", "run_a", "run_b"]:
        _write_clusters_json(tmp_path, run_id, _build_test_clusters(run_id))
    assert discover_run_ids(tmp_path) == ["run_a", "run_b", "run_c"]


# ── load_cluster_frame ────────────────────────────────────────────────────────


def test_load_cluster_frame_row_count(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    assert frame.height == 5


def test_load_cluster_frame_expected_columns(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    for col in ("cluster_id", "base_confidence", "route_balanced_hitl",
                "route_quick_low_hitl", "canonical_name", "density"):
        assert col in frame.columns


def test_load_cluster_frame_maps_confidence_field(
    populated_data_dir: tuple[Path, str],
) -> None:
    """The JSON field 'confidence' is mapped to 'base_confidence' in the frame."""
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    am1 = frame.filter(frame["cluster_id"] == "cluster_am_1")
    assert am1["base_confidence"][0] == pytest.approx(0.92)


def test_load_cluster_frame_empty_clusters_returns_empty_with_schema(
    tmp_path: Path,
) -> None:
    run_id = "empty_run"
    _write_clusters_json(
        tmp_path, run_id,
        {"run_id": run_id, "cluster_count": 0, "clusters": []},
    )
    frame = load_cluster_frame(tmp_path, run_id)
    assert frame.is_empty()
    assert "cluster_id" in frame.columns
    assert "base_confidence" in frame.columns


def test_load_cluster_frame_safe_returns_error_for_corrupt_json(tmp_path: Path) -> None:
    run_id = "corrupt_run"
    dir_name = _encode_run_id_path_segment(run_id)
    resolution_dir = tmp_path / "runs" / dir_name / RESOLUTION_STAGE_DIRNAME
    resolution_dir.mkdir(parents=True, exist_ok=True)
    (resolution_dir / CLUSTERS_FILENAME).write_text("not valid json{{{", encoding="utf-8")

    frame, error = load_cluster_frame_safe(tmp_path, run_id)

    assert frame.is_empty()
    assert "cluster_id" in frame.columns
    assert error == "Cluster data is missing or unreadable for this run."


def test_load_cluster_frame_safe_returns_error_for_malformed_cluster_row(
    tmp_path: Path,
) -> None:
    run_id = "missing_confidence"
    payload = _build_test_clusters(run_id)
    del payload["clusters"][0]["confidence"]
    _write_clusters_json(tmp_path, run_id, payload)

    frame, error = load_cluster_frame_safe(tmp_path, run_id)

    assert frame.is_empty()
    assert "base_confidence" in frame.columns
    assert error == "Cluster data is missing or unreadable for this run."


# ── bucket_counts ─────────────────────────────────────────────────────────────


def test_bucket_counts_balanced_hitl(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    counts = bucket_counts(frame, "balanced_hitl")
    assert counts == {"auto_merge": 2, "keep_separate": 2, "review": 1}


def test_bucket_counts_quick_low_hitl(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    counts = bucket_counts(frame, "quick_low_hitl")
    assert counts == {"auto_merge": 2, "defer": 1, "keep_separate": 2}


def test_bucket_counts_sum_matches_total(
    populated_data_dir: tuple[Path, str],
) -> None:
    """Total across all buckets must equal the frame row count."""
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    for profile in ("balanced_hitl", "quick_low_hitl"):
        counts = bucket_counts(frame, profile)
        assert sum(counts.values()) == frame.height


# ── bucket_summary ────────────────────────────────────────────────────────────


def test_bucket_summary_auto_merge(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    summary = bucket_summary(frame, "balanced_hitl", "auto_merge")
    assert summary["cluster_count"] == 2
    assert summary["entity_count"] == 5  # size 3 + size 2
    assert summary["max_cluster_size"] == 3


def test_bucket_summary_empty_bucket(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    # "review" does not exist in quick_low_hitl — that profile uses "defer"
    summary = bucket_summary(frame, "quick_low_hitl", "review")
    assert summary["cluster_count"] == 0
    assert summary["entity_count"] == 0


# ── filter_by_bucket ──────────────────────────────────────────────────────────


def test_filter_by_bucket_returns_correct_rows(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    filtered = filter_by_bucket(frame, "balanced_hitl", "review")
    assert filtered.height == 1
    assert filtered["cluster_id"][0] == "cluster_review_1"


def test_filter_by_bucket_returns_empty_for_nonexistent_bucket(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    filtered = filter_by_bucket(frame, "balanced_hitl", "defer")
    assert filtered.is_empty()


# ── size_distribution ─────────────────────────────────────────────────────────


def test_size_distribution_groups_correctly(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    dist = size_distribution(frame)
    # Sizes: 1 (x2), 2 (x2), 3 (x1)
    assert dist.height == 3
    size_to_count = dict(
        zip(dist["cluster_size"].to_list(), dist["cluster_count"].to_list())
    )
    assert size_to_count == {1: 2, 2: 2, 3: 1}


def test_size_distribution_on_filtered_bucket(
    populated_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = populated_data_dir
    frame = load_cluster_frame(data_dir, run_id)
    bucket_frame = filter_by_bucket(frame, "balanced_hitl", "auto_merge")
    dist = size_distribution(bucket_frame)
    size_to_count = dict(
        zip(dist["cluster_size"].to_list(), dist["cluster_count"].to_list())
    )
    assert size_to_count == {2: 1, 3: 1}
