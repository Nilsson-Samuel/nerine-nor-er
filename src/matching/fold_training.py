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

from src.matching.run import FEATURE_COLUMNS, PAIR_KEY_COLUMNS
from src.matching.reranker import save_lightgbm_artifacts, train_lightgbm
from src.matching.writer import get_features_output_path
from src.shared.paths import get_evaluation_labels_path

FOLD_SUMMARY_FILENAME = "fold_summary.json"
FOLD_METRICS_FILENAME = "fold_metrics.csv"
FOLD_SUMMARY_MARKDOWN_FILENAME = "fold_summary.md"
AGGREGATE_FOLD_REPORTS_FILENAME = "fold_reports.csv"
AGGREGATE_FOLD_SUMMARY_FILENAME = "fold_reports.json"
AGGREGATE_FOLD_REPORTS_MARKDOWN_FILENAME = "fold_reports.md"
DEFAULT_FOLD_MODEL_VERSION_PREFIX = "lightgbm_case_fold"
FINAL_CLUSTERING_METRIC_FIELDS = (
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "ari",
    "nmi",
    "bcubed_precision",
    "bcubed_recall",
    "bcubed_f1",
)
FINAL_CLUSTERING_METRIC_LABELS = {
    "pairwise_precision": "Pairwise precision",
    "pairwise_recall": "Pairwise recall",
    "pairwise_f1": "Pairwise F1",
    "ari": "ARI",
    "nmi": "NMI",
    "bcubed_precision": "B-cubed precision",
    "bcubed_recall": "B-cubed recall",
    "bcubed_f1": "B-cubed F1",
}


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
        "pairwise_precision": float(metrics["pairwise_precision"]),
        "pairwise_recall": float(metrics["pairwise_recall"]),
        "pairwise_f1": float(metrics["pairwise_f1"]),
        "ari": float(metrics["ari"]),
        "nmi": float(metrics["nmi"]),
        "bcubed_precision": float(metrics["bcubed_precision"]),
        "bcubed_recall": float(metrics["bcubed_recall"]),
        "bcubed_f1": float(metrics["bcubed_f1"]),
        "blocking_positive_pair_recall": float(blocking_recall),
        "matching_pairwise_precision": float(matching["precision"]),
        "matching_pairwise_recall": float(matching["recall"]),
        "matching_pairwise_f1": float(matching["f1"]),
    }


def build_macro_average_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute macro averages for the primary final clustering metrics."""
    if not rows:
        raise ValueError("macro average requires at least one fold row")

    averages = {
        metric_name: sum(float(row[metric_name]) for row in rows) / len(rows)
        for metric_name in FINAL_CLUSTERING_METRIC_FIELDS
    }
    return {
        "fold_name": "macro_avg",
        "held_out_case": f"{len(rows)} folds",
        **averages,
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


def write_fold_summary_markdown(path: Path | str, payload: dict[str, Any]) -> None:
    """Write one human-readable Markdown report for a held-out fold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_row = payload["summary_row"]
    held_out_run = payload["held_out_run"]
    training = payload["training"]["training_metadata"]
    evaluation_report_paths = [
        f"`{held_out_run['evaluation_report_path']}`",
        f"`{held_out_run['evaluation_markdown_report_path']}`",
    ]
    lines = [
        f"# Fold Summary: {payload['fold_name']}",
        "",
        f"- Held-out case: `{payload['held_out_case']}`",
        f"- Train cases: `{', '.join(payload['train_cases'])}`",
        f"- Train labeled rows: {summary_row['train_labeled_row_count']}",
        f"- Train positive rate: {_format_markdown_metric(summary_row['train_positive_rate'])}",
        f"- Evaluation entities: {summary_row['evaluation_entity_count']}",
        (
            "- Evaluation candidate pairs: "
            f"{summary_row['evaluation_candidate_pair_count']}"
        ),
        "",
        "## Final Clustering Metrics",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                [FINAL_CLUSTERING_METRIC_LABELS[name], _format_markdown_metric(summary_row[name])]
                for name in FINAL_CLUSTERING_METRIC_FIELDS
            ],
        ),
        "",
        "## Stage Context",
        "",
        _markdown_table(
            ["Signal", "Value"],
            [
                [
                    "Blocking gold-positive-pair recall",
                    _format_markdown_metric(summary_row["blocking_positive_pair_recall"]),
                ],
                [
                    "Matching pairwise precision",
                    _format_markdown_metric(summary_row["matching_pairwise_precision"]),
                ],
                [
                    "Matching pairwise recall",
                    _format_markdown_metric(summary_row["matching_pairwise_recall"]),
                ],
                [
                    "Matching pairwise F1",
                    _format_markdown_metric(summary_row["matching_pairwise_f1"]),
                ],
            ],
        ),
        "",
        "## Training Context",
        "",
        _markdown_table(
            ["Item", "Value"],
            [
                ["Train case count", summary_row["train_case_count"]],
                ["Label-source run count", training["source_count"]],
                ["Held-out run ID", f"`{summary_row['test_run_id']}`"],
            ],
        ),
        "",
        "## Detailed Reports",
        "",
        f"- JSON report: {evaluation_report_paths[0]}",
        f"- Markdown report: {evaluation_report_paths[1]}",
        "",
        "## Interpretation",
        "",
        _interpret_fold_summary(summary_row),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_aggregate_fold_reports_markdown(
    path: Path | str,
    rows: list[dict[str, Any]],
) -> None:
    """Write one Markdown overview across all held-out folds."""
    if not rows:
        raise ValueError("aggregate fold Markdown requires at least one row")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    macro_row = build_macro_average_row(rows)
    lines = [
        "# Case-Fold Evaluation Overview",
        "",
        "## Fold Metrics",
        "",
        _markdown_table(
            [
                "Fold",
                "Held-out case",
                "Pairwise P",
                "Pairwise R",
                "Pairwise F1",
                "ARI",
                "NMI",
                "B-cubed P",
                "B-cubed R",
                "B-cubed F1",
                "Blocking recall",
                "Matching P",
                "Matching R",
                "Matching F1",
            ],
            [
                [
                    row["fold_name"],
                    row["held_out_case"],
                    _format_markdown_metric(row["pairwise_precision"]),
                    _format_markdown_metric(row["pairwise_recall"]),
                    _format_markdown_metric(row["pairwise_f1"]),
                    _format_markdown_metric(row["ari"]),
                    _format_markdown_metric(row["nmi"]),
                    _format_markdown_metric(row["bcubed_precision"]),
                    _format_markdown_metric(row["bcubed_recall"]),
                    _format_markdown_metric(row["bcubed_f1"]),
                    _format_markdown_metric(row["blocking_positive_pair_recall"]),
                    _format_markdown_metric(row["matching_pairwise_precision"]),
                    _format_markdown_metric(row["matching_pairwise_recall"]),
                    _format_markdown_metric(row["matching_pairwise_f1"]),
                ]
                for row in rows
            ]
            + [
                [
                    macro_row["fold_name"],
                    macro_row["held_out_case"],
                    _format_markdown_metric(macro_row["pairwise_precision"]),
                    _format_markdown_metric(macro_row["pairwise_recall"]),
                    _format_markdown_metric(macro_row["pairwise_f1"]),
                    _format_markdown_metric(macro_row["ari"]),
                    _format_markdown_metric(macro_row["nmi"]),
                    _format_markdown_metric(macro_row["bcubed_precision"]),
                    _format_markdown_metric(macro_row["bcubed_recall"]),
                    _format_markdown_metric(macro_row["bcubed_f1"]),
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_markdown_metric(value: Any) -> str:
    """Format numeric metrics for compact Markdown tables."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Build one simple GitHub-flavored Markdown table."""
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(table)


def _interpret_fold_summary(summary_row: dict[str, Any]) -> str:
    """Generate one short interpretation line from the fold metrics."""
    blocking_recall = float(summary_row["blocking_positive_pair_recall"])
    matching_precision = float(summary_row["matching_pairwise_precision"])
    matching_recall = float(summary_row["matching_pairwise_recall"])
    pairwise_precision = float(summary_row["pairwise_precision"])
    pairwise_recall = float(summary_row["pairwise_recall"])

    if blocking_recall >= 0.9 and pairwise_precision < pairwise_recall:
        return (
            "Blocking stayed high, while final precision lagged recall. "
            "That points to over-merging after candidate generation."
        )
    if blocking_recall < 0.8:
        return (
            "Blocking recall dropped early, so missed candidate pairs are likely "
            "limiting the final clustering result."
        )
    if matching_recall + 0.1 < matching_precision:
        return (
            "Matching stayed conservative relative to recall, which suggests "
            "under-linking before resolution."
        )
    return (
        "Blocking and matching stayed reasonably balanced, so this fold is best "
        "read through the final clustering metrics rather than a single stage issue."
    )
