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
import pyarrow as pa

import pyarrow.parquet as pq

from src.matching.writer import get_scored_pairs_output_path
from src.resolution.clustering import MIN_OBJECTIVE_GAIN
from src.shared.config import KEEP_SCORE_THRESHOLD, OBJECTIVE_NEUTRAL_THRESHOLD
from src.shared.paths import (
    RESOLUTION_STAGE_DIRNAME,
    RUN_OUTPUTS_DIRNAME,
    get_extraction_run_output_dir,
    get_ingestion_run_output_dir,
)
from src.resolution.writer import (
    CLUSTERS_FILENAME,
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
    "positions",
]
_EDGE_COLUMNS = ["entity_id_a", "entity_id_b", "score", "blocking_source", "shap_top5"]
_PARQUET_READ_EXCEPTIONS = (OSError, pa.ArrowInvalid, pl.exceptions.PolarsError)


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


def _route_column_for_profile(profile: str) -> str:
    """Resolve one supported routing profile to its column name."""
    col = PROFILE_ROUTE_COLUMN.get(profile)
    if col is None:
        raise ValueError(f"Unsupported routing profile: {profile}")
    return col


def _available_columns(path: Path, desired: list[str]) -> list[str]:
    """Keep only columns that exist in the parquet schema."""
    schema_names = set(pq.read_schema(path).names)
    return [col for col in desired if col in schema_names]


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
    col = _route_column_for_profile(profile)
    return frame.filter(pl.col(col) == bucket)


def bucket_counts(frame: pl.DataFrame, profile: str) -> dict[str, int]:
    """Count clusters per routing bucket for one profile.

    Args:
        frame: Cluster DataFrame from load_cluster_frame.
        profile: One of the supported routing profiles.

    Returns:
        Mapping from bucket name to cluster count.
    """
    col = _route_column_for_profile(profile)
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
    except _PARQUET_READ_EXCEPTIONS as exc:
        logger.warning("Could not read parquet for member IDs: %s", exc)
        return []

    if resolved.is_empty():
        return []
    return resolved["entity_id"].to_list()


def load_entities_by_id(
    data_dir: Path,
    run_id: str,
    entity_ids: list[str],
) -> pl.DataFrame:
    """Load entity rows for the given IDs from entities.parquet.

    Returns a DataFrame with entity-level detail: entity_id, normalized, text,
    type, doc_id, count, context, chunk_id, char_start, char_end, positions.
    Returns an empty frame if the parquet file is missing or unreadable.
    """
    entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"

    if not entity_ids or not entities_path.exists():
        return pl.DataFrame()

    try:
        keep_cols = _available_columns(entities_path, _MEMBER_COLUMNS)
        if "entity_id" not in keep_cols:
            return pl.DataFrame()
        entities = pl.from_arrow(
            pq.read_table(
                entities_path,
                columns=keep_cols,
                filters=[
                    ("run_id", "==", run_id),
                    ("entity_id", "in", entity_ids),
                ],
            )
        )
    except _PARQUET_READ_EXCEPTIONS as exc:
        logger.warning("Could not read parquet for entities: %s", exc)
        return pl.DataFrame()

    return entities.sort("entity_id")


def load_cluster_members(
    data_dir: Path,
    run_id: str,
    cluster_id: str,
    member_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Load members of one cluster by reading only matching entity rows.

    Returns an empty frame if the parquet files are missing or unreadable.
    """
    if member_ids is None:
        member_ids = load_cluster_member_ids(data_dir, run_id, cluster_id)
    return load_entities_by_id(data_dir, run_id, member_ids)


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
        keep_cols = _available_columns(scored_path, _EDGE_COLUMNS)
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
    except _PARQUET_READ_EXCEPTIONS as exc:
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
    docs_path = get_ingestion_run_output_dir(data_dir, run_id) / "docs.parquet"
    if not docs_path.exists():
        return {}

    try:
        table = pl.from_arrow(pq.read_table(docs_path, columns=["run_id", "doc_id", "path"]))
    except _PARQUET_READ_EXCEPTIONS as exc:
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


# ── Merge candidates across clusters ─────────────────────────────────────────

# Why a scored pair ended up in two different clusters, in pipeline order.
UNMERGED_PRUNED = "pruned: below keep threshold"
UNMERGED_WEAK = "weak: below neutral point"
UNMERGED_SPLIT = "split: clustering kept apart"

_CROSS_PAIR_SCHEMA: dict[str, pl.DataType] = {
    "cluster_id": pl.Utf8,
    "entity_id": pl.Utf8,
    "candidate_cluster_id": pl.Utf8,
    "candidate_entity_id": pl.Utf8,
    "score": pl.Float64,
    "blocking_source": pl.Utf8,
}


def resolve_run_thresholds(diagnostics: dict[str, Any] | None) -> tuple[float, float]:
    """Return the run's (keep, neutral) thresholds, falling back to config.

    Runs record the thresholds they were resolved with in their diagnostics.
    Preferring those keeps candidate explanations correct when config has
    changed since the run.
    """
    thresholds = (diagnostics or {}).get("thresholds") or {}
    keep = thresholds.get("keep_score_threshold", KEEP_SCORE_THRESHOLD)
    neutral = thresholds.get("objective_neutral_threshold", OBJECTIVE_NEUTRAL_THRESHOLD)
    return float(keep), float(neutral)


def load_entity_cluster_ids(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load the entity_id → cluster_id assignment for every resolved entity."""
    empty = pl.DataFrame(schema={"entity_id": pl.Utf8, "cluster_id": pl.Utf8})
    resolved_path = get_resolved_entities_output_path(data_dir, run_id)
    if not resolved_path.exists():
        return empty

    try:
        return pl.from_arrow(
            pq.read_table(resolved_path, columns=["entity_id", "cluster_id"])
        )
    except _PARQUET_READ_EXCEPTIONS as exc:
        logger.warning("Could not read parquet for cluster assignments: %s", exc)
        return empty


def load_cross_cluster_pairs(
    data_dir: Path,
    run_id: str,
    min_score: float,
) -> pl.DataFrame:
    """Load scored pairs at or above min_score whose entities sit in different clusters.

    Each pair is returned twice, once from each side, so filtering on
    ``cluster_id`` yields every outside entity that was scored against that
    cluster. Returns columns: cluster_id, entity_id, candidate_cluster_id,
    candidate_entity_id, score, blocking_source.
    """
    empty = pl.DataFrame(schema=_CROSS_PAIR_SCHEMA)
    scored_path = get_scored_pairs_output_path(data_dir, run_id)
    if not scored_path.exists():
        return empty

    assignments = load_entity_cluster_ids(data_dir, run_id)
    if assignments.is_empty():
        return empty

    try:
        keep_cols = _available_columns(
            scored_path, ["entity_id_a", "entity_id_b", "score", "blocking_source"]
        )
        scored = pl.from_arrow(
            pq.read_table(
                scored_path,
                columns=keep_cols,
                filters=[("score", ">=", float(min_score))],
            )
        )
    except _PARQUET_READ_EXCEPTIONS as exc:
        logger.warning("Could not read parquet for cross-cluster pairs: %s", exc)
        return empty

    if "blocking_source" not in scored.columns:
        scored = scored.with_columns(pl.lit(None, dtype=pl.Utf8).alias("blocking_source"))

    pairs = (
        scored.join(
            assignments.rename({"entity_id": "entity_id_a", "cluster_id": "cluster_id_a"}),
            on="entity_id_a",
        )
        .join(
            assignments.rename({"entity_id": "entity_id_b", "cluster_id": "cluster_id_b"}),
            on="entity_id_b",
        )
        .filter(pl.col("cluster_id_a") != pl.col("cluster_id_b"))
        .with_columns(pl.col("score").cast(pl.Float64))
    )

    def _oriented(side: str, other: str) -> pl.DataFrame:
        return pairs.select(
            pl.col(f"cluster_id_{side}").alias("cluster_id"),
            pl.col(f"entity_id_{side}").alias("entity_id"),
            pl.col(f"cluster_id_{other}").alias("candidate_cluster_id"),
            pl.col(f"entity_id_{other}").alias("candidate_entity_id"),
            "score",
            "blocking_source",
        )

    return pl.concat([_oriented("a", "b"), _oriented("b", "a")])


def build_candidate_table(
    pairs: pl.DataFrame,
    entities: pl.DataFrame,
    cluster_frame: pl.DataFrame,
    route_column: str,
    keep_threshold: float,
    neutral_threshold: float,
) -> pl.DataFrame:
    """Attach entity text, candidate-cluster routing and an unmerged reason to pairs.

    Args:
        pairs: Rows from load_cross_cluster_pairs.
        entities: Entity rows covering both sides of every pair.
        cluster_frame: Cluster DataFrame from load_cluster_frame.
        route_column: Cluster frame column holding the active profile's bucket.
        keep_threshold: Score below which a pair never reached clustering.
        neutral_threshold: Score above which a retained pair voted to merge.

    Returns:
        One row per (entity, outside candidate), strongest score first, with
        ``candidate_rank`` numbering the candidates within each cluster.
    """
    if pairs.is_empty():
        return pl.DataFrame()

    entity_cols = [c for c in ("entity_id", "text", "type", "doc_id", "context")
                   if c in entities.columns]
    details = entities.select(entity_cols)
    candidate_details = details.rename(
        {c: f"candidate_{c}" for c in entity_cols}
    )
    candidate_clusters = cluster_frame.select(
        pl.col("cluster_id").alias("candidate_cluster_id"),
        pl.col("canonical_name").alias("candidate_canonical_name"),
        pl.col("cluster_size").alias("candidate_cluster_size"),
        pl.col(route_column).alias("candidate_route"),
    )

    unmerged_reason = (
        pl.when(pl.col("score") < keep_threshold)
        .then(pl.lit(UNMERGED_PRUNED))
        # Mirrors score_to_objective_weight: only a positive weight votes to merge.
        .when(pl.col("score") - neutral_threshold > MIN_OBJECTIVE_GAIN)
        .then(pl.lit(UNMERGED_SPLIT))
        .otherwise(pl.lit(UNMERGED_WEAK))
    )

    return (
        pairs.join(details, on="entity_id", how="left")
        .join(candidate_details, on="candidate_entity_id", how="left")
        .join(candidate_clusters, on="candidate_cluster_id", how="left")
        .with_columns(
            unmerged_reason.alias("unmerged_reason"),
            pl.col("score")
            .rank("ordinal", descending=True)
            .over("cluster_id")
            .alias("candidate_rank"),
        )
        .sort(
            ["score", "cluster_id", "entity_id", "candidate_entity_id"],
            descending=[True, False, False, False],
        )
    )


def load_candidate_table(
    data_dir: Path,
    run_id: str,
    cluster_frame: pl.DataFrame,
    profile: str,
    min_score: float,
    thresholds: tuple[float, float],
) -> pl.DataFrame:
    """Load every cross-cluster merge candidate at or above min_score for one run.

    Args:
        data_dir: Root data directory containing per-run outputs.
        run_id: Pipeline run identifier.
        cluster_frame: Cluster DataFrame from load_cluster_frame.
        profile: Routing profile whose buckets label the candidate clusters.
        min_score: Lowest pair score to include.
        thresholds: The run's (keep, neutral) thresholds, see resolve_run_thresholds.

    Returns:
        Candidate table from build_candidate_table, or an empty frame.
    """
    pairs = load_cross_cluster_pairs(data_dir, run_id, min_score)
    if pairs.is_empty():
        return pl.DataFrame()

    entity_ids = pl.concat([pairs["entity_id"], pairs["candidate_entity_id"]]).unique()
    entities = load_entities_by_id(data_dir, run_id, entity_ids.to_list())
    keep_threshold, neutral_threshold = thresholds
    return build_candidate_table(
        pairs,
        entities,
        cluster_frame,
        _route_column_for_profile(profile),
        keep_threshold,
        neutral_threshold,
    )


def top_candidate_scores(candidates: pl.DataFrame) -> pl.DataFrame:
    """Return each cluster's strongest outside candidate score.

    Gives keep_separate clusters, whose base confidence is always 0, a score
    that says how close they came to being merged.
    """
    if candidates.is_empty():
        return pl.DataFrame(schema={"cluster_id": pl.Utf8, "top_candidate_score": pl.Float64})
    return candidates.group_by("cluster_id").agg(
        pl.col("score").max().alias("top_candidate_score")
    )
