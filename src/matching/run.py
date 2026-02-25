"""Matching feature-stage orchestration and artifact writing."""

import logging
from pathlib import Path

import polars as pl

from src.matching.features import (
    COOCCURRENCE_META_FEATURE_COLUMNS,
    EMBEDDING_FEATURE_COLUMNS,
    STRING_FEATURE_COLUMNS,
    STRUCTURED_IDENTITY_FEATURE_COLUMNS,
    build_cooccurrence_meta_features,
    build_embedding_features,
    build_string_features,
    build_structured_identity_features,
    load_embedding_artifacts,
    load_pairs_with_metadata,
)
from src.matching.writer import write_features


logger = logging.getLogger(__name__)

PAIR_KEY_COLUMNS = ["run_id", "entity_id_a", "entity_id_b"]
FEATURE_COLUMNS = [
    *STRING_FEATURE_COLUMNS,
    *EMBEDDING_FEATURE_COLUMNS,
    *STRUCTURED_IDENTITY_FEATURE_COLUMNS,
    *COOCCURRENCE_META_FEATURE_COLUMNS,
]
FEATURE_OUTPUT_COLUMNS = [*PAIR_KEY_COLUMNS, *FEATURE_COLUMNS]


def _feature_diagnostic_values(
    features_df: pl.DataFrame,
    feature_column: str,
) -> tuple[float, float | int | None, float | int | None, float | None]:
    """Compute non-null rate and numeric summary stats for one feature column."""
    series = features_df[feature_column]
    row_count = series.len()
    if row_count == 0:
        return (0.0, None, None, None)

    non_null = series.drop_nulls()
    non_null_count = non_null.len()
    non_null_rate = non_null_count / row_count
    if non_null_count == 0:
        return (non_null_rate, None, None, None)

    return (
        non_null_rate,
        non_null.min(),
        non_null.max(),
        non_null.mean(),
    )


def _log_feature_diagnostics(features_df: pl.DataFrame) -> None:
    """Emit lightweight feature diagnostics for quality monitoring."""
    for column in FEATURE_COLUMNS:
        non_null_rate, min_value, max_value, mean_value = _feature_diagnostic_values(
            features_df=features_df,
            feature_column=column,
        )
        logger.info(
            "feature_diagnostics feature=%s non_null_rate=%.6f min=%s max=%s mean=%s",
            column,
            non_null_rate,
            min_value,
            max_value,
            mean_value,
        )


def _ensure_row_alignment(reference: pl.DataFrame, candidate: pl.DataFrame, name: str) -> None:
    """Raise if a feature group does not preserve pair row alignment."""
    if candidate.height != reference.height:
        raise ValueError(
            f"{name} row count mismatch: expected {reference.height}, got {candidate.height}"
        )


def _ensure_key_alignment(reference: pl.DataFrame, candidate: pl.DataFrame, name: str) -> None:
    """Raise if candidate pair keys diverge from the reference pair key sequence."""
    missing_key_columns = [column for column in PAIR_KEY_COLUMNS if column not in candidate.columns]
    if missing_key_columns:
        missing = ", ".join(missing_key_columns)
        raise ValueError(f"{name} missing required key columns: {missing}")

    reference_keys = reference.select(PAIR_KEY_COLUMNS).to_dict(as_series=False)
    candidate_keys = candidate.select(PAIR_KEY_COLUMNS).to_dict(as_series=False)
    if candidate_keys != reference_keys:
        raise ValueError(f"{name} pair keys are not aligned with input pairs")


def run_features(data_dir: Path | str, run_id: str) -> pl.DataFrame:
    """Run all feature groups and write features.parquet.

    Args:
        data_dir: Directory containing entities/candidate_pairs and embedding artifacts.
        run_id: Pipeline run identifier.

    Returns:
        Feature table with pair key columns and 14 feature columns.
    """
    data_dir = Path(data_dir)
    pairs_df = load_pairs_with_metadata(data_dir, run_id)
    artifacts = load_embedding_artifacts(data_dir)

    string_df = build_string_features(pairs_df)
    embedding_df = build_embedding_features(pairs_df, artifacts)
    structured_df = build_structured_identity_features(pairs_df)
    cooccurrence_df = build_cooccurrence_meta_features(pairs_df)

    _ensure_row_alignment(pairs_df, string_df, "string features")
    _ensure_row_alignment(pairs_df, embedding_df, "embedding features")
    _ensure_row_alignment(pairs_df, structured_df, "structured identity features")
    _ensure_row_alignment(pairs_df, cooccurrence_df, "cooccurrence metadata features")
    _ensure_key_alignment(pairs_df, string_df, "string features")

    features_df = pl.concat(
        [string_df, embedding_df, structured_df, cooccurrence_df],
        how="horizontal",
    ).select(FEATURE_OUTPUT_COLUMNS)
    _ensure_key_alignment(pairs_df, features_df, "final feature output")
    _log_feature_diagnostics(features_df)

    write_features(features_df, data_dir)
    return features_df
