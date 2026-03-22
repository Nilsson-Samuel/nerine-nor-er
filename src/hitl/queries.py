"""Cluster query helpers for the HITL triage interface.

Loads resolution artifacts into flat Polars DataFrames and provides
bucket-level filtering, summary queries, and cluster inspector queries
(members, edges, aliases) for the Streamlit explorer.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

import pyarrow.parquet as pq

from src.matching.writer import RUN_OUTPUTS_DIRNAME, get_scored_pairs_output_path
from src.resolution.writer import (
    CLUSTERS_FILENAME,
    RESOLUTION_STAGE_DIRNAME,
    get_clusters_output_path,
    get_resolved_entities_output_path,
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

_MEMBER_COLUMNS = [
    "entity_id",
    "normalized",
    "text",
    "type",
    "doc_id",
    "count",
    "context",
    "chunk_id",
    "char_start",
    "char_end",
]
_EDGE_COLUMNS = ["entity_id_a", "entity_id_b", "score", "blocking_source", "shap_top5"]


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


# ── Cluster inspector queries ────────────────────────────────────────────────


def load_cluster_member_ids(
    data_dir: Path,
    run_id: str,
    cluster_id: str,
) -> list[str]:
    """Load all entity IDs assigned to one cluster."""
    resolved_path = get_resolved_entities_output_path(data_dir, run_id)
    if not resolved_path.exists():
        return []

    try:
        resolved = pl.from_arrow(
            pq.read_table(
                resolved_path,
                columns=["entity_id"],
                filters=[("cluster_id", "==", cluster_id)],
            )
        )
    except Exception as exc:
        logger.warning("Could not read parquet for member IDs: %s", exc)
        return []

    if resolved.is_empty():
        return []
    return resolved["entity_id"].to_list()


def load_cluster_members(
    data_dir: Path,
    run_id: str,
    cluster_id: str,
    member_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Load members of one cluster by reading only matching entity rows.

    Returns a DataFrame with entity-level detail: entity_id, normalized, text,
    type, doc_id, count, context, chunk_id, char_start, char_end.
    Returns an empty frame if the parquet files are missing or unreadable.
    """
    entities_path = data_dir / "entities.parquet"

    if member_ids is None:
        member_ids = load_cluster_member_ids(data_dir, run_id, cluster_id)

    if not member_ids or not entities_path.exists():
        return pl.DataFrame()

    try:
        entities = pl.from_arrow(
            pq.read_table(
                entities_path,
                columns=_MEMBER_COLUMNS,
                filters=[
                    ("run_id", "==", run_id),
                    ("entity_id", "in", member_ids),
                ],
            )
        )
    except Exception as exc:
        logger.warning("Could not read parquet for members: %s", exc)
        return pl.DataFrame()

    return entities.sort("entity_id")


def load_cluster_edges(
    data_dir: Path,
    run_id: str,
    cluster_id: str,
    member_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Load all intra-cluster edges from scored_pairs for one cluster.

    Filters scored_pairs to edges where both endpoints belong to the cluster.
    Returns columns: entity_id_a, entity_id_b, score, blocking_source, shap_top5.
    Sorted ascending by score (weakest evidence first).
    """
    scored_path = get_scored_pairs_output_path(data_dir, run_id)

    if member_ids is None:
        member_ids = load_cluster_member_ids(data_dir, run_id, cluster_id)

    if len(member_ids) < 2 or not scored_path.exists():
        return pl.DataFrame()

    try:
        scored_schema = pq.read_schema(scored_path).names
        keep_cols = [col for col in _EDGE_COLUMNS if col in scored_schema]
        scored = pl.from_arrow(
            pq.read_table(
                scored_path,
                columns=keep_cols,
                filters=[
                    ("entity_id_a", "in", member_ids),
                    ("entity_id_b", "in", member_ids),
                ],
            )
        )
    except Exception as exc:
        logger.warning("Could not read parquet for edges: %s", exc)
        return pl.DataFrame()

    return scored.sort("score")


def find_weakest_edge(edges: pl.DataFrame) -> dict[str, Any] | None:
    """Return the lowest-score edge from an edges frame, or None if empty."""
    if edges.is_empty():
        return None
    return edges.sort("score").row(0, named=True)


def build_alias_table(members: pl.DataFrame) -> pl.DataFrame:
    """Aggregate members into alias groups by normalized form.

    Returns columns: normalized, surface_forms (list of distinct text values),
    mention_count (sum of count).
    """
    if members.is_empty():
        return pl.DataFrame(
            schema={"normalized": pl.Utf8, "surface_forms": pl.List(pl.Utf8),
                    "mention_count": pl.Int64}
        )

    return (
        members.group_by("normalized")
        .agg([
            pl.col("text").unique().sort().alias("surface_forms"),
            pl.col("count").sum().alias("mention_count"),
        ])
        .sort("mention_count", descending=True)
    )


def load_doc_paths(data_dir: Path, run_id: str) -> dict[str, str]:
    """Load doc_id → relative file path mapping from docs.parquet.

    Returns a dict mapping each doc_id to its source file path.
    Returns an empty dict if docs.parquet is missing or unreadable.
    """
    docs_path = data_dir / "docs.parquet"
    if not docs_path.exists():
        return {}

    try:
        table = pl.from_arrow(pq.read_table(docs_path, columns=["run_id", "doc_id", "path"]))
    except Exception as exc:
        logger.warning("Could not read docs.parquet: %s", exc)
        return {}

    filtered = table.filter(pl.col("run_id") == run_id)
    return dict(zip(filtered["doc_id"].to_list(), filtered["path"].to_list()))


def build_entity_text_lookup(members: pl.DataFrame) -> dict[str, str]:
    """Build entity_id → text mapping from a members frame.

    Used to annotate edge tables and weakest-link displays with
    human-readable entity surface forms.
    """
    if members.is_empty() or "entity_id" not in members.columns:
        return {}
    return dict(zip(members["entity_id"].to_list(), members["text"].to_list()))


def format_shap_reasons(shap_top5: list[dict[str, float]] | None) -> str:
    """Format a SHAP top-5 list into a compact human-readable string.

    Each entry is a {feature: str, value: float} struct. Returns a comma-separated
    summary like 'embedding_sim +0.12, name_jaro -0.03'. Returns '-' if empty.
    """
    if not shap_top5:
        return "-"
    parts: list[str] = []
    for entry in shap_top5:
        feat = entry.get("feature", "?")
        val = entry.get("value", 0.0)
        parts.append(f"{feat} {val:+.3f}")
    return ", ".join(parts)
