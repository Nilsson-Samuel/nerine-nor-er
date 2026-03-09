"""Parquet writers for intermediate matching artifacts."""

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.shared.schemas import SCORED_PAIRS_SCHEMA


SCORING_METADATA_FILENAME = "matching_scoring_metadata.json"


def write_string_features(df: pl.DataFrame, out_dir: Path) -> None:
    """Write string/token feature columns to string_features.parquet.

    Args:
        df: DataFrame containing feature columns (at minimum the five key
            columns from load_pairs_with_names, plus any computed features).
        out_dir: Directory to write string_features.parquet into.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "string_features.parquet")


def write_features(df: pl.DataFrame, out_dir: Path) -> None:
    """Write the integrated feature table to features.parquet.

    Args:
        df: Feature table containing pair keys and all computed feature columns.
        out_dir: Directory to write features.parquet into.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "features.parquet")


def _coerce_scored_at(scored_at: datetime) -> datetime:
    """Normalize scored_at to a timezone-aware UTC timestamp."""
    if scored_at.tzinfo is None:
        raise ValueError("scored_at must be timezone-aware")
    return scored_at.astimezone(timezone.utc)


def build_scored_pairs_table(
    candidate_pairs_df: pl.DataFrame,
    scores: list[float],
    model_version: str,
    scored_at: datetime,
    shap_top5: list[list[dict[str, float]]] | None = None,
) -> pa.Table:
    """Build a strict-schema scored pair table from candidates and scores."""
    if candidate_pairs_df.height != len(scores):
        raise ValueError("candidate pair rows must align one-to-one with scores")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")

    scored_at = _coerce_scored_at(scored_at)
    row_count = candidate_pairs_df.height
    if shap_top5 is None:
        shap_top5 = [[] for _ in range(row_count)]
    if row_count != len(shap_top5):
        raise ValueError("shap_top5 rows must align one-to-one with candidate pairs")

    return pa.Table.from_arrays(
        [
            pa.array(candidate_pairs_df["run_id"].to_list(), type=pa.string()),
            pa.array(candidate_pairs_df["entity_id_a"].to_list(), type=pa.string()),
            pa.array(candidate_pairs_df["entity_id_b"].to_list(), type=pa.string()),
            pa.array(scores, type=pa.float32()),
            pa.array([model_version.strip()] * row_count, type=pa.string()),
            pa.array([scored_at] * row_count, type=pa.timestamp("us", tz="UTC")),
            pa.array(candidate_pairs_df["blocking_methods"].to_list(), type=pa.list_(pa.string())),
            pa.array(candidate_pairs_df["blocking_source"].to_list(), type=pa.string()),
            pa.array(candidate_pairs_df["blocking_method_count"].to_list(), type=pa.int8()),
            pa.array(shap_top5, type=SCORED_PAIRS_SCHEMA.field("shap_top5").type),
        ],
        schema=SCORED_PAIRS_SCHEMA,
    )


def write_scored_pairs(table: pa.Table, out_dir: Path) -> None:
    """Write scored pair output to scored_pairs.parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "scored_pairs.parquet")


def write_scoring_metadata(metadata: Mapping[str, Any], out_dir: Path) -> None:
    """Write matching scoring metadata as formatted JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SCORING_METADATA_FILENAME).write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
