"""Cluster query helpers for the HITL triage interface.

Loads resolution artifacts into flat Polars DataFrames and provides
bucket-level filtering and summary queries for the Streamlit explorer.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.matching.writer import RUN_OUTPUTS_DIRNAME
from src.resolution.writer import (
    CLUSTERS_FILENAME,
    RESOLUTION_STAGE_DIRNAME,
    get_clusters_output_path,
)


logger = logging.getLogger(__name__)

PROFILES = ("balanced_hitl", "quick_low_hitl")

# Maps each routing profile to the column name holding its bucket label
PROFILE_ROUTE_COLUMN: dict[str, str] = {
    "balanced_hitl": "route_balanced_hitl",
    "quick_low_hitl": "route_quick_low_hitl",
}

# Ordered routing buckets per profile (order matches severity for triage)
BUCKETS_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "balanced_hitl": ("auto_merge", "review", "keep_separate"),
    "quick_low_hitl": ("auto_merge", "defer", "keep_separate"),
}

# Columns returned by load_cluster_frame, in the expected empty-frame schema
_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "cluster_id": pl.Utf8,
    "cluster_size": pl.Int64,
    "base_confidence": pl.Float64,
    "min_edge_score": pl.Float64,
    "density": pl.Float64,
    "canonical_name": pl.Utf8,
    "canonical_type": pl.Utf8,
    "suspicious_merge": pl.Boolean,
    "route_balanced_hitl": pl.Utf8,
    "route_quick_low_hitl": pl.Utf8,
}


def _decode_run_id(dir_name: str) -> str | None:
    """Reverse the rid_<urlsafe-base64> encoding used for per-run directory names."""
    if not dir_name.startswith("rid_"):
        return None
    encoded = dir_name[4:]
    padding = (4 - len(encoded) % 4) % 4
    try:
        return base64.urlsafe_b64decode(encoded + "=" * padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def discover_run_ids(data_dir: Path) -> list[str]:
    """Find all run IDs that have completed resolution outputs.

    Scans the per-run output tree for directories containing clusters.json,
    then decodes the run_id from each directory name.

    Args:
        data_dir: Root data directory (e.g. data/processed).

    Returns:
        Sorted list of decoded run IDs.
    """
    runs_dir = Path(data_dir) / RUN_OUTPUTS_DIRNAME
    if not runs_dir.is_dir():
        return []

    run_ids: list[str] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        clusters_path = entry / RESOLUTION_STAGE_DIRNAME / CLUSTERS_FILENAME
        if not clusters_path.exists():
            continue
        run_id = _decode_run_id(entry.name)
        if run_id is not None:
            run_ids.append(run_id)

    return sorted(run_ids)


def load_cluster_frame(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load clusters.json into a flat Polars DataFrame for triage queries.

    Flattens routing_actions_by_profile into per-profile columns so that
    bucket filtering works with standard Polars expressions. The JSON field
    ``confidence`` is mapped to ``base_confidence`` for UI clarity.

    Args:
        data_dir: Root data directory containing per-run outputs.
        run_id: Pipeline run identifier.

    Returns:
        Flat DataFrame with one row per cluster.
    """
    clusters_path = get_clusters_output_path(data_dir, run_id)
    payload = json.loads(clusters_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for cluster in payload.get("clusters", []):
        routing = cluster.get("routing_actions_by_profile", {})
        rows.append({
            "cluster_id": str(cluster["cluster_id"]),
            "cluster_size": int(cluster["cluster_size"]),
            "base_confidence": float(cluster["confidence"]),
            "min_edge_score": float(cluster["min_edge_score"]),
            "density": float(cluster["density"]),
            "canonical_name": str(cluster["canonical_name"]),
            "canonical_type": str(cluster["canonical_type"]),
            "suspicious_merge": bool(cluster.get("suspicious_merge", False)),
            "route_balanced_hitl": str(routing.get("balanced_hitl", "unknown")),
            "route_quick_low_hitl": str(routing.get("quick_low_hitl", "unknown")),
        })

    if not rows:
        return pl.DataFrame(schema=_FRAME_SCHEMA)

    return pl.DataFrame(rows)


def load_cluster_frame_safe(data_dir: Path, run_id: str) -> tuple[pl.DataFrame, str | None]:
    """Load clusters for the UI without raising on corrupt or malformed payloads.

    Args:
        data_dir: Root data directory containing per-run outputs.
        run_id: Pipeline run identifier.

    Returns:
        Tuple of ``(frame, error_message)``. On success, ``error_message`` is
        ``None``. On failure, an empty frame with the expected schema is
        returned together with a short user-facing error string.
    """
    try:
        return load_cluster_frame(data_dir, run_id), None
    except FileNotFoundError:
        return pl.DataFrame(schema=_FRAME_SCHEMA), "No cluster data found for this run."
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not load clusters for %s: %s", run_id, exc)
        return (
            pl.DataFrame(schema=_FRAME_SCHEMA),
            "Cluster data is missing or unreadable for this run.",
        )


def filter_by_bucket(
    frame: pl.DataFrame,
    profile: str,
    bucket: str,
) -> pl.DataFrame:
    """Filter cluster frame to one routing bucket for the selected profile.

    Args:
        frame: Cluster DataFrame from load_cluster_frame.
        profile: One of the supported routing profiles.
        bucket: Routing action to filter by (e.g. 'auto_merge', 'review').

    Returns:
        Filtered DataFrame containing only matching clusters.
    """
    col = PROFILE_ROUTE_COLUMN[profile]
    return frame.filter(pl.col(col) == bucket)


def bucket_counts(frame: pl.DataFrame, profile: str) -> dict[str, int]:
    """Count clusters per routing bucket for one profile.

    Args:
        frame: Cluster DataFrame from load_cluster_frame.
        profile: One of the supported routing profiles.

    Returns:
        Mapping from bucket name to cluster count.
    """
    col = PROFILE_ROUTE_COLUMN[profile]
    counts = frame.group_by(col).len().sort(col)
    return {
        str(row[col]): int(row["len"])
        for row in counts.iter_rows(named=True)
    }


def bucket_summary(
    frame: pl.DataFrame,
    profile: str,
    bucket: str,
) -> dict[str, Any]:
    """Compute summary stats for one routing bucket.

    Args:
        frame: Cluster DataFrame from load_cluster_frame.
        profile: One of the supported routing profiles.
        bucket: Routing action to summarize.

    Returns:
        Dict with cluster_count, entity_count, avg_cluster_size, max_cluster_size.
    """
    filtered = filter_by_bucket(frame, profile, bucket)
    if filtered.is_empty():
        return {
            "cluster_count": 0,
            "entity_count": 0,
            "avg_cluster_size": 0.0,
            "max_cluster_size": 0,
        }
    return {
        "cluster_count": filtered.height,
        "entity_count": int(filtered["cluster_size"].sum()),
        "avg_cluster_size": round(float(filtered["cluster_size"].mean()), 2),
        "max_cluster_size": int(filtered["cluster_size"].max()),
    }


def size_distribution(frame: pl.DataFrame) -> pl.DataFrame:
    """Group clusters by size and count occurrences.

    Args:
        frame: Cluster DataFrame (full or bucket-filtered).

    Returns:
        DataFrame with columns cluster_size and cluster_count, sorted ascending.
    """
    return (
        frame.group_by("cluster_size")
        .len()
        .rename({"len": "cluster_count"})
        .sort("cluster_size")
    )
