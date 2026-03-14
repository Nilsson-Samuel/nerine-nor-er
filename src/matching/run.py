"""Matching feature-stage orchestration and artifact writing."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from src.matching.reranker import load_lightgbm_artifacts, score_lightgbm
from src.matching.shap_explain import explain_lightgbm_top5
from src.matching.tuning import build_tuning_summary, run_optuna_study
from src.matching.writer import (
    build_scored_pairs_table,
    write_features,
    write_scored_pairs,
    write_scoring_metadata,
)
from src.shared import schemas


logger = logging.getLogger(__name__)

PAIR_KEY_COLUMNS = ["run_id", "entity_id_a", "entity_id_b"]
FEATURE_COLUMNS = [
    *STRING_FEATURE_COLUMNS,
    *EMBEDDING_FEATURE_COLUMNS,
    *STRUCTURED_IDENTITY_FEATURE_COLUMNS,
    *COOCCURRENCE_META_FEATURE_COLUMNS,
]
FEATURE_OUTPUT_COLUMNS = [*PAIR_KEY_COLUMNS, *FEATURE_COLUMNS]
SCORED_OUTPUT_COLUMNS = [
    "run_id",
    "entity_id_a",
    "entity_id_b",
    "score",
    "model_version",
    "scored_at",
    "blocking_methods",
    "blocking_source",
    "blocking_method_count",
    "shap_top5",
]


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


def _load_feature_rows(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load one run from features.parquet, building it first when absent."""
    features_path = data_dir / "features.parquet"
    if features_path.exists():
        features_df = pl.read_parquet(features_path).filter(pl.col("run_id") == run_id)
        if not features_df.is_empty():
            return features_df
    return run_features(data_dir, run_id)


def _load_candidate_pairs_for_scoring(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load candidate pairs needed for scored output in stable pair order."""
    return (
        pl.read_parquet(data_dir / "candidate_pairs.parquet")
        .filter(pl.col("run_id") == run_id)
        .sort(["entity_id_a", "entity_id_b"])
    )


def _load_labeled_feature_matrix_if_available(
    data_dir: Path,
    run_id: str,
) -> tuple[pl.DataFrame, pl.Series] | None:
    """Load labeled features only when the requested run has labels available."""
    labels_path = data_dir / "labels.parquet"
    if not labels_path.exists():
        return None
    labels_for_run = pl.read_parquet(labels_path).filter(pl.col("run_id") == run_id)
    if labels_for_run.is_empty():
        return None

    from src.synthetic.build_matching_dataset import load_labeled_feature_matrix

    return load_labeled_feature_matrix(data_dir, run_id)


def _run_tuning_if_requested(
    data_dir: Path,
    run_id: str,
    *,
    enable_tuning: bool,
    tuning_mode: str,
    tuning_trials: int,
) -> dict[str, Any]:
    """Run tuning when labels are available, else return a metadata-only status."""
    if not enable_tuning:
        return build_tuning_summary(
            enabled=False,
            status="disabled",
            mode="disabled",
            n_trials_requested=0,
        )

    labeled = _load_labeled_feature_matrix_if_available(data_dir, run_id)
    if labeled is None:
        return build_tuning_summary(
            enabled=True,
            status="skipped_no_labels",
            mode=tuning_mode,
            n_trials_requested=tuning_trials,
        )

    X_labeled, y_labeled = labeled
    return run_optuna_study(
        X_labeled,
        y_labeled,
        enabled=True,
        mode=tuning_mode,
        n_trials=tuning_trials,
        out_dir=data_dir,
    )


def _build_scoring_metadata(
    *,
    run_id: str,
    scored_at: datetime,
    model_metadata: dict[str, Any],
    row_count: int,
    tuning_summary: dict[str, Any],
    enable_shap: bool,
    explained_row_count: int,
) -> dict[str, Any]:
    """Build one JSON metadata payload for scoring-side optional hooks."""
    return {
        "run_id": run_id,
        "scored_at": scored_at.isoformat(),
        "model_version": str(model_metadata["model_version"]),
        "params_used": str(model_metadata.get("training_param_source", "baseline")),
        "tuning": tuning_summary,
        "shap": {
            "method": "lightgbm_pred_contrib",
            "enabled": enable_shap,
            "generated": explained_row_count > 0,
            "explained_row_count": explained_row_count,
            "row_count": row_count,
        },
    }


def run_scoring(
    data_dir: Path | str,
    run_id: str,
    scored_at: datetime | None = None,
    *,
    enable_tuning: bool = False,
    tuning_mode: str = "smoke",
    tuning_trials: int = 2,
    enable_shap: bool = False,
    shap_max_rows: int | None = None,
) -> pl.DataFrame:
    """Run inference scoring and write scored_pairs.parquet for one run."""
    data_dir = Path(data_dir)
    features_df = _load_feature_rows(data_dir, run_id)
    candidate_pairs_df = _load_candidate_pairs_for_scoring(data_dir, run_id)
    effective_scored_at = scored_at or datetime.now(timezone.utc)

    _ensure_row_alignment(candidate_pairs_df, features_df, "features")
    _ensure_key_alignment(candidate_pairs_df, features_df, "features")

    tuning_summary = _run_tuning_if_requested(
        data_dir,
        run_id,
        enable_tuning=enable_tuning,
        tuning_mode=tuning_mode,
        tuning_trials=tuning_trials,
    )
    booster, metadata = load_lightgbm_artifacts(data_dir)
    scores = score_lightgbm(booster, features_df.select(FEATURE_COLUMNS)).tolist()
    shap_top5 = explain_lightgbm_top5(
        booster,
        features_df.select(FEATURE_COLUMNS),
        enabled=enable_shap,
        max_rows=shap_max_rows,
    )
    table = build_scored_pairs_table(
        candidate_pairs_df=candidate_pairs_df,
        scores=scores,
        model_version=str(metadata["model_version"]),
        scored_at=effective_scored_at,
        shap_top5=shap_top5,
    )

    validation_errors = schemas.validate_contract_rules(
        table,
        "scored_pairs",
        candidate_pairs_table=candidate_pairs_df.to_arrow(),
    )
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    write_scored_pairs(table, data_dir)
    explained_row_count = 0
    if enable_shap:
        explained_row_count = candidate_pairs_df.height if shap_max_rows is None else min(
            candidate_pairs_df.height,
            shap_max_rows,
        )
    write_scoring_metadata(
        _build_scoring_metadata(
            run_id=run_id,
            scored_at=effective_scored_at,
            model_metadata=metadata,
            row_count=candidate_pairs_df.height,
            tuning_summary=tuning_summary,
            enable_shap=enable_shap,
            explained_row_count=explained_row_count,
        ),
        data_dir,
    )
    return pl.from_arrow(table).select(SCORED_OUTPUT_COLUMNS)
