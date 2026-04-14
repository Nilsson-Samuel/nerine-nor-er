"""Helpers for held-out case-fold matching training and metric summaries.

This module keeps fold-specific training logic out of the main pipeline path.
It loads labeled rows from several isolated case runs, trains one shared
LightGBM model for the fold, and writes compact fold-level metric artifacts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.evaluation.run import get_evaluation_labels_path
from src.matching.run import FEATURE_COLUMNS, PAIR_KEY_COLUMNS
from src.matching.reranker import save_lightgbm_artifacts, train_lightgbm
from src.matching.writer import get_features_output_path

FOLD_SUMMARY_FILENAME = "fold_summary.json"
FOLD_METRICS_FILENAME = "fold_metrics.csv"
AGGREGATE_FOLD_REPORTS_FILENAME = "fold_reports.csv"
AGGREGATE_FOLD_SUMMARY_FILENAME = "fold_reports.json"
DEFAULT_FOLD_MODEL_VERSION_PREFIX = "lightgbm_case_fold"


@dataclass(frozen=True)
class FoldTrainingSource:
    """One labeled case run that contributes rows to fold training."""

    case_name: str
    data_dir: Path
    run_id: str


@dataclass(frozen=True)
class FoldTrainingMatrix:
    """Concatenated training rows plus metadata for one held-out fold."""

    X_train: pl.DataFrame
    y_train: pl.Series
    metadata: dict[str, Any]


def _raise_if_duplicate_pair_keys(frame: pl.DataFrame, source_name: str) -> None:
    """Reject duplicate pair keys before joining labels onto features."""
    duplicate_keys = (
        frame.group_by(PAIR_KEY_COLUMNS)
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_keys.height:
        raise ValueError(f"{source_name} contains duplicate pair keys")


def _load_labeled_rows(source: FoldTrainingSource) -> pl.DataFrame:
    """Load one run's labeled feature rows in stable feature-file order."""
    features_path = get_features_output_path(source.data_dir, source.run_id)
    if not features_path.exists():
        raise ValueError(
            f"missing matching features for run_id={source.run_id} at {features_path}; "
            "rerun matching features for this run"
        )

    labels_path = get_evaluation_labels_path(source.data_dir, source.run_id)
    if not labels_path.exists():
        raise ValueError(
            f"missing labels for run_id={source.run_id} at {labels_path}; "
            "write gold-bridge labels for this run before fold training"
        )

    features = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path).filter(pl.col("run_id") == source.run_id)
    if labels.is_empty():
        raise ValueError(
            f"labels.parquet does not contain rows for run_id={source.run_id}; "
            "write gold-bridge labels for this run before fold training"
        )

    _raise_if_duplicate_pair_keys(features.select(PAIR_KEY_COLUMNS), "features.parquet")
    _raise_if_duplicate_pair_keys(labels.select(PAIR_KEY_COLUMNS), "labels.parquet")

    missing_feature_keys = labels.select(PAIR_KEY_COLUMNS).join(
        features.select(PAIR_KEY_COLUMNS),
        on=PAIR_KEY_COLUMNS,
        how="anti",
    )
    if missing_feature_keys.height:
        raise ValueError("features.parquet is missing keys from labels.parquet")

    return features.join(labels, on=PAIR_KEY_COLUMNS, how="inner")


def load_multi_run_labeled_feature_matrix(
    sources: list[FoldTrainingSource],
) -> FoldTrainingMatrix:
    """Load and concatenate labeled rows from multiple case runs.

    The input order is preserved so fold assembly stays deterministic and easy
    to inspect when folds are defined manually.
    """
    if not sources:
        raise ValueError("fold training requires at least one labeled run source")

    labeled_frames: list[pl.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []

    for source in sources:
        labeled = _load_labeled_rows(source)
        positive_row_count = int(labeled["label"].sum())
        row_count = labeled.height
        run_summaries.append(
            {
                "case_name": source.case_name,
                "data_dir": str(source.data_dir),
                "run_id": source.run_id,
                "labeled_row_count": row_count,
                "positive_row_count": positive_row_count,
                "negative_row_count": row_count - positive_row_count,
                "positive_rate": round(
                    positive_row_count / row_count if row_count else 0.0,
                    6,
                ),
            }
        )
        labeled_frames.append(labeled.select([*FEATURE_COLUMNS, "label"]))

    combined = pl.concat(labeled_frames, how="vertical")
    y_train = combined["label"]
    label_values = {int(value) for value in y_train.unique().to_list()}
    if label_values != {0, 1}:
        raise ValueError(
            "fold training labels must contain both 0 and 1 across the selected train runs"
        )

    labeled_row_count = combined.height
    positive_row_count = int(y_train.sum())
    metadata = {
        "source_count": len(sources),
        "labeled_row_count": labeled_row_count,
        "positive_row_count": positive_row_count,
        "negative_row_count": labeled_row_count - positive_row_count,
        "positive_rate": round(
            positive_row_count / labeled_row_count if labeled_row_count else 0.0,
            6,
        ),
        "run_summaries": run_summaries,
    }
    return FoldTrainingMatrix(
        X_train=combined.select(FEATURE_COLUMNS),
        y_train=y_train,
        metadata=metadata,
    )


def train_and_save_fold_model(
    sources: list[FoldTrainingSource],
    model_dir: Path | str,
    *,
    model_version: str,
    training_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one fold-specific LightGBM model and persist its metadata."""
    training_matrix = load_multi_run_labeled_feature_matrix(sources)
    model = train_lightgbm(
        training_matrix.X_train,
        training_matrix.y_train,
        params=training_params,
    )
    model_metadata = save_lightgbm_artifacts(
        model,
        model_dir,
        model_version=model_version,
        training_params=training_params,
        training_param_source="fold_training",
        extra_metadata={"fold_training": training_matrix.metadata},
    )
    return {
        "model_metadata": model_metadata,
        "training_metadata": training_matrix.metadata,
    }


def build_fold_summary_row(
    *,
    fold_name: str,
    held_out_case: str,
    train_cases: list[str],
    test_run_id: str,
    training_metadata: dict[str, Any],
    evaluation_report: dict[str, Any],
) -> dict[str, Any]:
    """Extract one compact fold-level metric row from the evaluation report."""
    matching = evaluation_report["stage_metrics"]["matching"]
    blocking = evaluation_report["stage_metrics"]["blocking"]
    metrics = evaluation_report["metrics"]
    metric_scope = evaluation_report["metric_scope"]
    blocking_recall = blocking.get("gold_positive_pair_recall")
    if blocking_recall is None:
        blocking_recall = blocking["positive_pair_recall"]

    return {
        "fold_name": fold_name,
        "held_out_case": held_out_case,
        "train_cases": list(train_cases),
        "train_case_count": len(train_cases),
        "test_run_id": test_run_id,
        "train_labeled_row_count": int(training_metadata["labeled_row_count"]),
        "train_positive_rate": float(training_metadata["positive_rate"]),
        "evaluation_entity_count": int(metric_scope["evaluation_entity_count"]),
        "evaluation_candidate_pair_count": int(metric_scope["evaluation_candidate_pair_count"]),
        "pairwise_f1": float(metrics["pairwise_f1"]),
        "bcubed_f1": float(metrics["bcubed_f1"]),
        "ari": float(metrics["ari"]),
        "nmi": float(metrics["nmi"]),
        "blocking_positive_pair_recall": float(blocking_recall),
        "matching_pairwise_precision": float(matching["precision"]),
        "matching_pairwise_recall": float(matching["recall"]),
        "matching_pairwise_f1": float(matching["f1"]),
    }


def write_fold_summary_json(summary_path: Path | str, payload: dict[str, Any]) -> None:
    """Write one compact JSON summary for a fold or aggregate run."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _csv_cell(value: Any) -> str | int | float:
    """Flatten JSON-like row values into CSV-safe scalar cells."""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def write_fold_metrics_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    """Write flat fold summary rows to CSV for presentation-friendly review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("fold metrics csv requires at least one row")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(value) for key, value in row.items()})
