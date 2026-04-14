"""Post-pipeline evaluation CLI for one completed run."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.error_analysis import (
    summarize_extraction_errors,
    summarize_false_merges,
    summarize_false_splits,
)
from src.evaluation.metrics import (
    clustering_metrics,
    mention_metrics,
    positive_pairs_from_memberships,
    pairwise_metrics,
)
from src.ingestion.chunking import CHUNK_OVERLAP
from src.ingestion.extraction import extract_docx_units, extract_pdf_units
from src.ingestion.normalization import normalize_text
from src.matching.writer import (
    RUN_OUTPUTS_DIRNAME,
    _encode_run_id_path_segment,
    get_scored_pairs_output_path,
)
from src.resolution.writer import get_resolved_entities_output_path
from src.shared import schemas
from src.synthetic.build_matching_dataset import LABELS_SCHEMA

EVALUATION_STAGE_DIRNAME = "evaluation"
EVALUATION_REPORT_FILENAME = "evaluation_report.json"
LABELS_FILENAME = "labels.parquet"
DEFAULT_MATCH_THRESHOLD = 0.5
DEFAULT_ALLOWED_METRIC_DROP = {
    "pairwise_f1": 0.03,
    "bcubed_f1": 0.03,
    "ari": 0.05,
    "nmi": 0.05,
}
UNLABELABLE_GOLD_GROUP_PREFIXES = ("__fp__:", "__ambiguous__:")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for one evaluation run."""
    parser = argparse.ArgumentParser(
        description="Evaluate one completed Nerine pipeline run."
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing run artifacts."
    )
    parser.add_argument("--run-id", required=True, help="Run identifier to evaluate.")
    parser.add_argument(
        "--gold-path",
        default="data/gold_annotations.csv",
        help="CSV file with gold mention annotations.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Probability threshold for score-based pair evaluation.",
    )
    parser.add_argument(
        "--baseline-report",
        default=None,
        help="Optional earlier evaluation_report.json to compare against.",
    )
    parser.add_argument(
        "--shared-labels-path",
        default=None,
        help="Optional shared labels.parquet path to merge this run into for later training.",
    )
    parser.add_argument(
        "--shared-labels-allowed-doc-ids",
        nargs="+",
        default=None,
        help="Optional doc_id allowlist for labels written to --shared-labels-path.",
    )
    return parser.parse_args(argv)


def get_evaluation_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run evaluation output directory."""
    return (
        Path(data_dir)
        / RUN_OUTPUTS_DIRNAME
        / _encode_run_id_path_segment(run_id)
        / EVALUATION_STAGE_DIRNAME
    )


def get_evaluation_report_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run evaluation report path."""
    return get_evaluation_run_output_dir(data_dir, run_id) / EVALUATION_REPORT_FILENAME


def get_evaluation_labels_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run matcher-label bridge path."""
    return get_evaluation_run_output_dir(data_dir, run_id) / LABELS_FILENAME


def _prepare_gold_label_bridge(
    data_dir: Path,
    run_id: str,
    gold_path: Path,
) -> dict[str, Any]:
    """Build gold-bridge artifacts once for label writing and full evaluation."""
    docs = _load_docs(data_dir, run_id)
    chunks = _attach_chunk_global_offsets(_load_chunks(data_dir, run_id), docs)
    gold_mentions, gold_offset_summary = _remap_gold_offsets_to_run_text(
        _remap_gold_doc_ids(_load_gold_mentions(gold_path), docs),
        chunks,
    )
    trusted_gold_mentions = (
        gold_mentions.filter(pl.col("_offset_resolved"))
        .drop("_offset_resolved")
        .sort(["doc_id", "char_start", "char_end", "mention_id"])
    )
    entities = _load_entities(data_dir, run_id)
    candidate_pairs = _load_candidate_pairs(data_dir, run_id)
    predicted_mentions = _build_predicted_mentions(entities, chunks)
    matched_mentions = _match_gold_to_predicted(
        trusted_gold_mentions,
        predicted_mentions,
    )
    gold_group_by_entity, bridge_summary = _assign_gold_groups(
        entities,
        matched_mentions,
    )
    bridge_summary["gold_mentions"] = gold_mentions.height
    bridge_summary["trusted_gold_mentions"] = trusted_gold_mentions.height
    bridge_summary.update(gold_offset_summary)
    bridge_summary["matched_mention_rate"] = (
        matched_mentions.height / trusted_gold_mentions.height
        if trusted_gold_mentions.height
        else 0.0
    )
    bridge_summary["matched_mentions_by_method"] = {
        str(row["match_method"]): int(row["count"])
        for row in matched_mentions.group_by("match_method")
        .len()
        .rename({"len": "count"})
        .iter_rows(named=True)
    }

    entity_doc_id_by_entity = {
        str(row["entity_id"]): str(row["doc_id"])
        for row in entities.select(["entity_id", "doc_id"]).iter_rows(named=True)
    }
    labels_table = _build_labels_table(candidate_pairs, gold_group_by_entity)
    bridge_summary["label_rows_written"] = labels_table.num_rows
    bridge_summary["candidate_pairs_excluded_from_labels"] = (
        candidate_pairs.height - labels_table.num_rows
    )
    return {
        "chunks": chunks,
        "gold_mentions": gold_mentions,
        "trusted_gold_mentions": trusted_gold_mentions,
        "entities": entities,
        "candidate_pairs": candidate_pairs,
        "predicted_mentions": predicted_mentions,
        "gold_group_by_entity": gold_group_by_entity,
        "entity_doc_id_by_entity": entity_doc_id_by_entity,
        "labels_table": labels_table,
        "bridge_summary": bridge_summary,
    }


def write_training_labels_from_gold(
    data_dir: Path | str,
    run_id: str,
    gold_path: Path | str,
    *,
    shared_labels_path: Path | str | None = None,
    shared_labels_allowed_doc_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write one run's gold-bridge labels without requiring scoring outputs."""
    data_dir = Path(data_dir)
    gold_path = Path(gold_path)
    labels_path = get_evaluation_labels_path(data_dir, run_id)
    bridge = _prepare_gold_label_bridge(data_dir, run_id, gold_path)

    _write_parquet_atomic(bridge["labels_table"], labels_path)
    if shared_labels_path is not None:
        shared_labels_table = _build_labels_table(
            bridge["candidate_pairs"],
            bridge["gold_group_by_entity"],
            entity_doc_id_by_entity=bridge["entity_doc_id_by_entity"],
            allowed_doc_ids=shared_labels_allowed_doc_ids,
        )
        _merge_labels_into_shared_store(
            shared_labels_table,
            Path(shared_labels_path),
            run_id,
        )

    return {
        "run_id": run_id,
        "gold_path": str(gold_path.resolve()),
        "labels_path": str(labels_path),
        "label_rows_written": int(bridge["labels_table"].num_rows),
        "candidate_pair_count": int(bridge["candidate_pairs"].height),
        "bridge_summary": dict(bridge["bridge_summary"]),
    }


def run_evaluation(
    data_dir: Path | str,
    run_id: str,
    gold_path: Path | str,
    *,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    baseline_report_path: Path | str | None = None,
    shared_labels_path: Path | str | None = None,
    shared_labels_allowed_doc_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one completed run and write the per-run report."""
    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError("match_threshold must be in [0, 1]")

    data_dir = Path(data_dir)
    gold_path = Path(gold_path)
    report_path = get_evaluation_report_path(data_dir, run_id)
    bridge = _prepare_gold_label_bridge(data_dir, run_id, gold_path)
    labels_path = get_evaluation_labels_path(data_dir, run_id)

    chunks = bridge["chunks"]
    gold_mentions = bridge["gold_mentions"]
    trusted_gold_mentions = bridge["trusted_gold_mentions"]
    entities = bridge["entities"]
    candidate_pairs = bridge["candidate_pairs"]
    scored_pairs = _load_scored_pairs(data_dir, run_id)
    resolved_entities = _load_resolved_entities(data_dir, run_id)
    predicted_mentions = bridge["predicted_mentions"]
    gold_group_by_entity = bridge["gold_group_by_entity"]
    bridge_summary = dict(bridge["bridge_summary"])
    entity_doc_id_by_entity = bridge["entity_doc_id_by_entity"]
    labels_table = bridge["labels_table"]

    _write_parquet_atomic(labels_table, labels_path)
    if shared_labels_path is not None:
        shared_labels_table = _build_labels_table(
            candidate_pairs,
            gold_group_by_entity,
            entity_doc_id_by_entity=entity_doc_id_by_entity,
            allowed_doc_ids=shared_labels_allowed_doc_ids,
        )
        _merge_labels_into_shared_store(
            shared_labels_table,
            Path(shared_labels_path),
            run_id,
        )

    predicted_cluster_by_entity = _cluster_membership_from_resolved(
        resolved_entities, entities
    )
    scoped_gold_groups = _filter_labelable_gold_groups(gold_group_by_entity)
    scoped_predicted_clusters = {
        entity_id: predicted_cluster_by_entity[entity_id]
        for entity_id in scoped_gold_groups
    }
    gold_positive_pairs = positive_pairs_from_memberships(scoped_gold_groups)
    candidate_pair_keys = _pair_key_set(candidate_pairs)
    labelable_candidate_pair_keys = _pair_key_set(pl.from_arrow(labels_table))
    score_positive_pairs = (
        _thresholded_pair_key_set(scored_pairs, match_threshold)
        & labelable_candidate_pair_keys
    )

    extraction_scores = mention_metrics(
        _mention_key_set(predicted_mentions),
        _mention_key_set(trusted_gold_mentions),
    )
    matching_scores = pairwise_metrics(score_positive_pairs, gold_positive_pairs)
    matching_scores["evaluated_candidate_pair_count"] = labels_table.num_rows
    resolution_scores = clustering_metrics(
        scoped_gold_groups, scoped_predicted_clusters
    )
    blocking_positive_coverage = _blocking_positive_coverage(
        candidate_pair_keys, gold_positive_pairs
    )

    entity_metadata = {
        str(row["entity_id"]): {
            "doc_id": row["doc_id"],
            "entity_type": row["type"],
            "text": row["text"],
            "normalized": row["normalized"],
        }
        for row in entities.iter_rows(named=True)
    }
    regression_checks = build_regression_checks(
        metrics=resolution_scores,
        input_paths={
            "gold_path": gold_path,
            "scored_pairs_path": get_scored_pairs_output_path(data_dir, run_id),
            "resolved_entities_path": get_resolved_entities_output_path(
                data_dir, run_id
            ),
            "labels_path": labels_path,
        },
        metric_scope={
            "evaluation_entity_count": len(scoped_gold_groups),
            "evaluation_candidate_pair_count": labels_table.num_rows,
        },
        baseline_report_path=(
            Path(baseline_report_path) if baseline_report_path else None
        ),
    )

    report = {
        "run_id": run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "gold_path": str(gold_path.resolve()),
            "scored_pairs_path": str(get_scored_pairs_output_path(data_dir, run_id)),
            "resolved_entities_path": str(
                get_resolved_entities_output_path(data_dir, run_id)
            ),
            "labels_path": str(labels_path),
            "baseline_report_path": (
                str(Path(baseline_report_path).resolve())
                if baseline_report_path
                else None
            ),
        },
        "counts": {
            "gold_mentions": gold_mentions.height,
            "trusted_gold_mentions": trusted_gold_mentions.height,
            "predicted_mentions": predicted_mentions.height,
            "entities": entities.height,
            "evaluation_entities": len(scoped_gold_groups),
            "candidate_pairs": candidate_pairs.height,
            "labeled_candidate_pairs": labels_table.num_rows,
            "scored_pairs": scored_pairs.height,
            "resolved_entities": resolved_entities.height,
            "gold_positive_pairs": len(gold_positive_pairs),
        },
        "alignment": bridge_summary,
        "metric_scope": {
            "uses_only_confident_gold_bridge": True,
            "evaluation_entity_count": len(scoped_gold_groups),
            "excluded_entity_count": entities.height - len(scoped_gold_groups),
            "excluded_unmatched_entity_count": bridge_summary[
                "entities_without_gold_match"
            ],
            "excluded_ambiguous_entity_count": bridge_summary[
                "entities_with_ambiguous_gold_groups"
            ],
            "evaluation_candidate_pair_count": labels_table.num_rows,
            "excluded_candidate_pair_count": candidate_pairs.height - labels_table.num_rows,
        },
        "metrics": resolution_scores,
        "stage_metrics": {
            "extraction": extraction_scores,
            "blocking": blocking_positive_coverage,
            "matching": {
                "score_threshold": match_threshold,
                **matching_scores,
            },
        },
        "error_analysis": {
            "extraction": summarize_extraction_errors(
                trusted_gold_mentions.to_dicts(),
                predicted_mentions.to_dicts(),
            ),
            "false_merges": summarize_false_merges(
                scoped_gold_groups,
                scoped_predicted_clusters,
                entity_metadata,
            ),
            "false_splits": summarize_false_splits(
                scoped_gold_groups,
                scoped_predicted_clusters,
                entity_metadata,
            ),
        },
        "regression_checks": regression_checks,
    }
    _write_json_atomic(report, report_path)
    _print_console_summary(report, report_path)
    return report


def build_regression_checks(
    *,
    metrics: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    metric_scope: Mapping[str, Any] | None = None,
    baseline_report_path: Path | None = None,
) -> dict[str, Any]:
    """Build a lightweight completeness and baseline-drift checklist."""
    checks: list[dict[str, Any]] = []
    for label, path in input_paths.items():
        checks.append(
            {"check": f"{label}_exists", "passed": path.exists(), "details": str(path)}
        )

    evaluation_entity_count = int((metric_scope or {}).get("evaluation_entity_count", 0))
    evaluation_candidate_pair_count = int(
        (metric_scope or {}).get("evaluation_candidate_pair_count", 0)
    )
    checks.append(
        {
            "check": "evaluation_entity_count_nonzero",
            "passed": evaluation_entity_count > 0,
            "details": evaluation_entity_count,
        }
    )
    checks.append(
        {
            "check": "evaluation_candidate_pair_count_nonzero",
            "passed": evaluation_candidate_pair_count > 0,
            "details": evaluation_candidate_pair_count,
        }
    )

    required_metrics = ("pairwise_f1", "bcubed_f1", "ari", "nmi")
    for metric_name in required_metrics:
        value = metrics.get(metric_name)
        passed = isinstance(value, (int, float)) and math.isfinite(float(value))
        checks.append(
            {"check": f"{metric_name}_finite", "passed": passed, "details": value}
        )

    if baseline_report_path is not None:
        baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        baseline_metrics = baseline.get("metrics", {})
        for metric_name, allowed_drop in DEFAULT_ALLOWED_METRIC_DROP.items():
            current_value = _finite_metric_value(metrics.get(metric_name))
            baseline_value = _finite_metric_value(baseline_metrics.get(metric_name))
            if current_value is None or baseline_value is None:
                checks.append(
                    {
                        "check": f"{metric_name}_drift",
                        "passed": True,
                        "skipped": True,
                        "details": {
                            "baseline": baseline_metrics.get(metric_name),
                            "current": metrics.get(metric_name),
                            "allowed_drop": allowed_drop,
                            "reason": "missing_or_nonfinite_metric",
                        },
                    }
                )
                continue
            actual_drop = baseline_value - current_value
            checks.append(
                {
                    "check": f"{metric_name}_drift",
                    "passed": actual_drop <= allowed_drop,
                    "details": {
                        "baseline": baseline_value,
                        "current": current_value,
                        "allowed_drop": allowed_drop,
                        "actual_drop": actual_drop,
                    },
                }
            )

    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def _finite_metric_value(value: Any) -> float | None:
    """Return one finite metric value or None when drift checks should be skipped."""
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _load_gold_mentions(gold_path: Path) -> pl.DataFrame:
    """Load the gold annotation CSV with typed columns and stable ordering."""
    frame = pl.read_csv(
        gold_path,
        schema_overrides={
            "doc_id": pl.String,
            "mention_id": pl.String,
            "text": pl.String,
            "entity_type": pl.String,
            "group_id": pl.String,
        },
    )
    required_columns = {
        "doc_id",
        "mention_id",
        "char_start",
        "char_end",
        "text",
        "entity_type",
        "group_id",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"gold annotation file is missing required columns: {missing}")

    return (
        frame.with_columns(
            [
                pl.col("doc_id").cast(pl.String),
                (
                    pl.col("doc_name").cast(pl.String)
                    if "doc_name" in frame.columns
                    else pl.lit(None).alias("doc_name")
                ),
                pl.col("mention_id").cast(pl.String),
                pl.col("char_start").cast(pl.Int64),
                pl.col("char_end").cast(pl.Int64),
                pl.col("text").cast(pl.String),
                pl.col("entity_type").cast(pl.String),
                pl.col("group_id").cast(pl.String),
            ]
        )
        .select(
            [
                "doc_id",
                "doc_name",
                "mention_id",
                "char_start",
                "char_end",
                "text",
                "entity_type",
                "group_id",
            ]
        )
        .sort(["doc_id", "char_start", "char_end", "mention_id"])
    )


def _load_docs(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load and validate docs metadata for one run."""
    table = pq.read_table(data_dir / "docs.parquet")
    errors = schemas.validate_contract_rules(table, "docs")
    if errors:
        raise ValueError(f"docs failed contract validation: {errors}")
    frame = pl.from_arrow(table).filter(pl.col("run_id") == run_id).sort("path")
    if frame.is_empty():
        raise ValueError(f"run_id not found in docs.parquet: {run_id}")
    return frame


def _remap_gold_doc_ids(
    gold_mentions: pl.DataFrame, docs: pl.DataFrame
) -> pl.DataFrame:
    """Prefer run-local doc_ids when gold doc_names identify the same files."""
    known_doc_ids = set(docs["doc_id"].to_list())
    if set(gold_mentions["doc_id"].unique().to_list()).issubset(known_doc_ids):
        return gold_mentions

    if "doc_name" not in gold_mentions.columns:
        return gold_mentions
    duplicate_names = _duplicate_doc_names(docs, gold_mentions)
    if duplicate_names:
        raise ValueError(
            "cannot remap gold doc_ids because doc_name is ambiguous for "
            f"{duplicate_names[:5]}"
        )

    name_to_doc_id = {
        Path(str(row["path"])).name: str(row["doc_id"])
        for row in docs.select(["doc_id", "path"]).iter_rows(named=True)
    }

    return gold_mentions.with_columns(
        pl.when(
            pl.col("doc_name").is_not_null()
            & pl.col("doc_name").is_in(list(name_to_doc_id))
        )
        .then(pl.col("doc_name").replace(name_to_doc_id))
        .otherwise(pl.col("doc_id"))
        .alias("doc_id")
    )


def _load_chunks(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load and validate chunks for one run."""
    table = pq.read_table(data_dir / "chunks.parquet")
    errors = schemas.validate_contract_rules(table, "chunks")
    if errors:
        raise ValueError(f"chunks failed contract validation: {errors}")
    frame = (
        pl.from_arrow(table)
        .filter(pl.col("run_id") == run_id)
        .sort(["doc_id", "chunk_index"])
    )
    if frame.is_empty():
        raise ValueError(f"run_id not found in chunks.parquet: {run_id}")
    return frame


def _attach_chunk_global_offsets(
    chunks: pl.DataFrame, docs: pl.DataFrame
) -> pl.DataFrame:
    """Attach one document-global chunk start offset per chunk row."""
    offsets = _chunk_global_offsets(chunks, docs)
    return chunks.with_columns(
        pl.Series(
            "global_start",
            [offsets[str(chunk_id)] for chunk_id in chunks["chunk_id"].to_list()],
            dtype=pl.Int64,
        )
    )


def _remap_gold_offsets_to_run_text(
    gold_mentions: pl.DataFrame,
    chunks: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Project gold mention offsets into the run's reconstructed document text."""
    if gold_mentions.is_empty():
        return gold_mentions.with_columns(pl.lit(True).alias("_offset_resolved")), {
            "gold_mentions_already_aligned": 0,
            "gold_mentions_remapped": 0,
            "gold_mentions_unresolved": 0,
        }

    doc_text_by_doc_id = _reconstruct_doc_text_by_doc_id(chunks)
    rows: list[dict[str, Any]] = []
    already_aligned = 0
    remapped = 0
    unresolved = 0

    for doc_id, group in gold_mentions.group_by("doc_id", maintain_order=True):
        doc_key = str(doc_id[0] if isinstance(doc_id, tuple) else doc_id)
        run_text = doc_text_by_doc_id.get(doc_key)
        search_start = 0

        for row in group.sort(["char_start", "char_end", "mention_id"]).iter_rows(
            named=True
        ):
            text = str(row["text"])
            start = int(row["char_start"])
            end = int(row["char_end"])

            if run_text is None:
                unresolved += 1
                unresolved_row = dict(row)
                unresolved_row["_offset_resolved"] = False
                rows.append(unresolved_row)
                continue

            if 0 <= start <= end <= len(run_text) and run_text[start:end] == text:
                already_aligned += 1
                search_start = max(search_start, end)
                aligned_row = dict(row)
                aligned_row["_offset_resolved"] = True
                rows.append(aligned_row)
                continue

            remapped_start = _find_text_in_doc_text(run_text, text, search_start, start)
            if remapped_start is None:
                unresolved += 1
                unresolved_row = dict(row)
                unresolved_row["_offset_resolved"] = False
                rows.append(unresolved_row)
                continue

            remapped += 1
            remapped_end = remapped_start + len(text)
            search_start = max(search_start, remapped_end)
            remapped_row = dict(row)
            remapped_row["char_start"] = remapped_start
            remapped_row["char_end"] = remapped_end
            remapped_row["_offset_resolved"] = True
            rows.append(remapped_row)

    return (
        pl.DataFrame(rows).sort(["doc_id", "char_start", "char_end", "mention_id"]),
        {
            "gold_mentions_already_aligned": already_aligned,
            "gold_mentions_remapped": remapped,
            "gold_mentions_unresolved": unresolved,
        },
    )


def _load_entities(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load and validate entities for one run."""
    table = pq.read_table(data_dir / "entities.parquet")
    errors = schemas.validate_contract_rules(table, "entities")
    if errors:
        raise ValueError(f"entities failed contract validation: {errors}")
    frame = pl.from_arrow(table).filter(pl.col("run_id") == run_id).sort("entity_id")
    if frame.is_empty():
        raise ValueError(f"run_id not found in entities.parquet: {run_id}")
    return frame


def _load_candidate_pairs(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load and validate candidate pairs for one run."""
    table = pq.read_table(data_dir / "candidate_pairs.parquet")
    errors = schemas.validate_contract_rules(table, "candidate_pairs")
    if errors:
        raise ValueError(f"candidate_pairs failed contract validation: {errors}")
    frame = (
        pl.from_arrow(table)
        .filter(pl.col("run_id") == run_id)
        .sort(["entity_id_a", "entity_id_b"])
    )
    if frame.is_empty():
        raise ValueError(f"run_id not found in candidate_pairs.parquet: {run_id}")
    return frame


def _load_scored_pairs(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load and validate scored pairs for one run."""
    scored_path = get_scored_pairs_output_path(data_dir, run_id)
    candidate_table = pq.read_table(data_dir / "candidate_pairs.parquet")
    table = pq.read_table(scored_path)
    errors = schemas.validate_contract_rules(table, "scored_pairs", candidate_table)
    if errors:
        raise ValueError(f"scored_pairs failed contract validation: {errors}")
    frame = (
        pl.from_arrow(table)
        .filter(pl.col("run_id") == run_id)
        .sort(["entity_id_a", "entity_id_b"])
    )
    if frame.is_empty():
        raise ValueError(f"run_id not found in scored_pairs.parquet: {run_id}")
    return frame


def _load_resolved_entities(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load and validate resolved entities for one run."""
    path = get_resolved_entities_output_path(data_dir, run_id)
    table = pq.read_table(path)
    errors = schemas.validate_contract_rules(table, "resolved_entities")
    if errors:
        raise ValueError(f"resolved_entities failed contract validation: {errors}")
    frame = pl.from_arrow(table).filter(pl.col("run_id") == run_id).sort("entity_id")
    if frame.is_empty():
        raise ValueError(f"run_id not found in resolved_entities.parquet: {run_id}")
    return frame


def _duplicate_doc_names(docs: pl.DataFrame, gold_mentions: pl.DataFrame) -> list[str]:
    """Return gold doc_names that collide with multiple run documents."""
    gold_doc_names = {
        str(doc_name)
        for doc_name in gold_mentions["doc_name"].to_list()
        if doc_name is not None
    }
    doc_ids_by_name: dict[str, set[str]] = {}
    for row in docs.select(["doc_id", "path"]).iter_rows(named=True):
        name = Path(str(row["path"])).name
        if name not in gold_doc_names:
            continue
        doc_ids_by_name.setdefault(name, set()).add(str(row["doc_id"]))
    return sorted(name for name, doc_ids in doc_ids_by_name.items() if len(doc_ids) > 1)


def _chunk_global_offsets(chunks: pl.DataFrame, docs: pl.DataFrame) -> dict[str, int]:
    """Reconstruct the document-global start offset for each chunk."""
    offsets: dict[str, int] = {}
    docs_by_id = {
        str(row["doc_id"]): row
        for row in docs.select(["doc_id", "path", "mime_type"]).iter_rows(named=True)
    }
    for doc_id, group in chunks.group_by("doc_id", maintain_order=True):
        doc_key = str(doc_id[0] if isinstance(doc_id, tuple) else doc_id)
        doc_meta = docs_by_id.get(doc_key)
        if doc_meta is None:
            raise ValueError(f"missing docs metadata for doc_id={doc_key}")
        doc_chunks = group.sort("chunk_index")
        source_offsets = _chunk_offsets_from_source(doc_chunks, doc_meta)
        offsets.update(source_offsets or _chunk_offsets_from_overlaps(doc_chunks))
    return offsets


def _chunk_offsets_from_source(
    doc_chunks: pl.DataFrame,
    doc_meta: Mapping[str, Any],
) -> dict[str, int] | None:
    """Replay ingestion against the source document when it can be located."""
    source_path = _resolve_source_document_path(str(doc_meta["path"]))
    if source_path is None:
        return None

    full_text = _normalized_full_text_from_source(
        source_path, str(doc_meta["mime_type"])
    )
    if not full_text and doc_chunks.height:
        return None
    try:
        return _chunk_start_offsets_from_full_text(full_text, doc_chunks)
    except ValueError:
        return None


def _resolve_source_document_path(doc_path: str) -> Path | None:
    """Resolve one source document path from a stored docs.parquet path value."""
    raw_path = Path(doc_path)
    direct_candidates: list[Path] = []
    if raw_path.is_absolute():
        direct_candidates.append(raw_path)
    else:
        direct_candidates.extend(
            [
                Path.cwd() / raw_path,
                Path.cwd() / "data" / "raw" / raw_path,
                Path.cwd() / "data" / raw_path,
            ]
        )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    basename = raw_path.name
    if not basename:
        return None

    matches: list[Path] = []
    for search_root in (Path.cwd() / "data" / "raw", Path.cwd()):
        if not search_root.exists():
            continue
        for candidate in search_root.rglob(basename):
            if candidate not in matches:
                matches.append(candidate)
            if len(matches) > 1:
                return None
    return matches[0] if matches else None


def _normalized_full_text_from_source(path: Path, mime_type: str) -> str:
    """Rebuild the normalized document text exactly as ingestion chunking sees it."""
    if mime_type == "application/pdf":
        units = extract_pdf_units(path)
    elif (
        mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        units = extract_docx_units(path)
    else:
        raise ValueError(f"unsupported mime_type for source replay: {mime_type}")
    normalized_units = [normalize_text(str(unit["text"])) for unit in units]
    return "\n\n".join(normalized_units)


def _chunk_start_offsets_from_full_text(
    full_text: str,
    doc_chunks: pl.DataFrame,
) -> dict[str, int]:
    """Recover exact chunk starts by replaying the ingestion search strategy."""
    offsets: dict[str, int] = {}
    search_start = 0
    for row in doc_chunks.sort("chunk_index").iter_rows(named=True):
        chunk_text = str(row["text"])
        chunk_start = full_text.find(chunk_text, search_start)
        if chunk_start == -1:
            chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            raise ValueError(
                f"chunk text could not be located for chunk_id={row['chunk_id']}"
            )
        offsets[str(row["chunk_id"])] = chunk_start
        search_start = chunk_start + max(1, len(chunk_text) - CHUNK_OVERLAP)
    return offsets


def _chunk_offsets_from_overlaps(doc_chunks: pl.DataFrame) -> dict[str, int]:
    """Fallback when source replay is unavailable: stitch chunks by overlap only."""
    offsets: dict[str, int] = {}
    assembled = ""
    for row in doc_chunks.sort("chunk_index").iter_rows(named=True):
        chunk_id = str(row["chunk_id"])
        text = str(row["text"])
        if not assembled:
            offsets[chunk_id] = 0
            assembled = text
            continue

        overlap = _largest_suffix_prefix_overlap(assembled, text)
        offsets[chunk_id] = len(assembled) - overlap
        assembled += text[overlap:]
    return offsets


def _reconstruct_doc_text_by_doc_id(chunks: pl.DataFrame) -> dict[str, str]:
    """Rebuild one document text per doc_id from chunk text and global starts."""
    doc_text_by_doc_id: dict[str, str] = {}
    for doc_id, group in chunks.group_by("doc_id", maintain_order=True):
        doc_key = str(doc_id[0] if isinstance(doc_id, tuple) else doc_id)
        max_end = 0
        chunk_rows = list(group.sort("global_start").iter_rows(named=True))
        for row in chunk_rows:
            max_end = max(max_end, int(row["global_start"]) + len(str(row["text"])))

        chars = [""] * max_end
        for row in chunk_rows:
            start = int(row["global_start"])
            text = str(row["text"])
            for index, char in enumerate(text):
                absolute_index = start + index
                if not chars[absolute_index]:
                    chars[absolute_index] = char
        doc_text_by_doc_id[doc_key] = "".join(char or " " for char in chars)
    return doc_text_by_doc_id


def _find_text_in_doc_text(
    doc_text: str,
    text: str,
    search_start: int,
    original_start: int,
) -> int | None:
    """Find one gold mention string in run text, preferring monotonic order."""
    start = doc_text.find(text, max(0, search_start))
    if start != -1:
        return start

    if 0 <= original_start < len(doc_text):
        window_start = max(0, original_start - 256)
        start = doc_text.find(text, window_start)
        if start != -1:
            return start

    start = doc_text.find(text)
    return start if start != -1 else None


def _largest_suffix_prefix_overlap(left: str, right: str) -> int:
    """Return the largest overlap where left suffix equals right prefix."""
    max_overlap = min(len(left), len(right))
    for overlap in range(max_overlap, -1, -1):
        if overlap == 0:
            return 0
        if left[-overlap:] == right[:overlap]:
            return overlap
    return 0


def _build_predicted_mentions(
    entities: pl.DataFrame, chunks: pl.DataFrame
) -> pl.DataFrame:
    """Explode entity provenance positions into document-global mention rows."""
    chunk_meta = {
        str(row["chunk_id"]): row
        for row in chunks.select(
            ["chunk_id", "doc_id", "chunk_index", "text", "global_start"]
        ).iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, int, int]] = set()

    for entity in entities.iter_rows(named=True):
        for position in entity["positions"]:
            meta = chunk_meta[str(position["chunk_id"])]
            global_start = int(meta["global_start"]) + int(position["char_start"])
            global_end = int(meta["global_start"]) + int(position["char_end"])
            key = (
                str(entity["entity_id"]),
                str(entity["doc_id"]),
                str(entity["type"]),
                global_start,
                global_end,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            chunk_text = str(meta["text"])
            rows.append(
                {
                    "entity_id": str(entity["entity_id"]),
                    "doc_id": str(entity["doc_id"]),
                    "mention_id": f"{entity['entity_id']}:{global_start}:{global_end}",
                    "char_start": global_start,
                    "char_end": global_end,
                    "text": chunk_text[
                        int(position["char_start"]) : int(position["char_end"])
                    ],
                    "entity_type": str(entity["type"]),
                    "chunk_id": str(position["chunk_id"]),
                    "chunk_index": int(meta["chunk_index"]),
                }
            )

    if not rows:
        return _empty_predicted_mentions_frame()
    return pl.DataFrame(rows).sort(["doc_id", "char_start", "char_end", "entity_id"])


def _match_gold_to_predicted(
    gold_mentions: pl.DataFrame,
    predicted_mentions: pl.DataFrame,
) -> pl.DataFrame:
    """Match gold mentions to predicted mentions by exact span, then text/order fallback."""
    gold_rows = gold_mentions.sort(
        ["doc_id", "char_start", "char_end", "mention_id"]
    ).to_dicts()
    predicted_rows = predicted_mentions.sort(
        ["doc_id", "char_start", "char_end", "entity_id"]
    ).to_dicts()

    predicted_by_exact: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in predicted_rows:
        exact_key = (
            str(row["doc_id"]),
            str(row["entity_type"]),
            int(row["char_start"]),
            int(row["char_end"]),
        )
        predicted_by_exact.setdefault(exact_key, []).append(row)

    matched_rows: list[dict[str, Any]] = []
    matched_gold_ids: set[str] = set()
    matched_pred_ids: set[str] = set()
    for gold_row in gold_rows:
        exact_key = (
            str(gold_row["doc_id"]),
            str(gold_row["entity_type"]),
            int(gold_row["char_start"]),
            int(gold_row["char_end"]),
        )
        candidates = predicted_by_exact.get(exact_key, [])
        if len(candidates) != 1:
            continue
        predicted_row = candidates[0]
        matched_gold_ids.add(str(gold_row["mention_id"]))
        matched_pred_ids.add(str(predicted_row["mention_id"]))
        matched_rows.append(
            {
                **gold_row,
                "entity_id": str(predicted_row["entity_id"]),
                "match_method": "exact_span",
            }
        )

    unmatched_gold_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for gold_row in gold_rows:
        if str(gold_row["mention_id"]) in matched_gold_ids:
            continue
        key = (
            str(gold_row["doc_id"]),
            str(gold_row["entity_type"]),
            str(gold_row["text"]),
        )
        unmatched_gold_groups.setdefault(key, []).append(gold_row)

    unmatched_pred_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for predicted_row in predicted_rows:
        if str(predicted_row["mention_id"]) in matched_pred_ids:
            continue
        key = (
            str(predicted_row["doc_id"]),
            str(predicted_row["entity_type"]),
            str(predicted_row["text"]),
        )
        unmatched_pred_groups.setdefault(key, []).append(predicted_row)

    for key in sorted(set(unmatched_gold_groups) & set(unmatched_pred_groups)):
        gold_group = unmatched_gold_groups[key]
        pred_group = unmatched_pred_groups[key]
        pair_count = min(len(gold_group), len(pred_group))
        for index in range(pair_count):
            gold_row = gold_group[index]
            predicted_row = pred_group[index]
            matched_rows.append(
                {
                    **gold_row,
                    "entity_id": str(predicted_row["entity_id"]),
                    "match_method": "text_order",
                }
            )

    if not matched_rows:
        return _empty_matched_mentions_frame()
    return pl.DataFrame(matched_rows).sort(
        ["doc_id", "char_start", "char_end", "entity_id"]
    )


def _empty_predicted_mentions_frame() -> pl.DataFrame:
    """Return one empty predicted-mentions frame with the expected columns."""
    return pl.DataFrame(
        schema={
            "entity_id": pl.String,
            "doc_id": pl.String,
            "mention_id": pl.String,
            "char_start": pl.Int64,
            "char_end": pl.Int64,
            "text": pl.String,
            "entity_type": pl.String,
            "chunk_id": pl.String,
            "chunk_index": pl.Int64,
        }
    )


def _empty_matched_mentions_frame() -> pl.DataFrame:
    """Return one empty gold-to-predicted bridge frame with the expected columns."""
    return pl.DataFrame(
        schema={
            "doc_id": pl.String,
            "doc_name": pl.String,
            "mention_id": pl.String,
            "char_start": pl.Int64,
            "char_end": pl.Int64,
            "text": pl.String,
            "entity_type": pl.String,
            "group_id": pl.String,
            "entity_id": pl.String,
            "match_method": pl.String,
        }
    )


def _assign_gold_groups(
    entities: pl.DataFrame,
    matched_mentions: pl.DataFrame,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign one gold group or one synthetic singleton group to every predicted entity."""
    groups_by_entity: dict[str, set[str]] = {}
    for row in matched_mentions.select(["entity_id", "group_id"]).iter_rows(named=True):
        groups_by_entity.setdefault(str(row["entity_id"]), set()).add(
            str(row["group_id"])
        )

    gold_group_by_entity: dict[str, str] = {}
    aligned_count = 0
    unmatched_count = 0
    ambiguous_count = 0
    for row in entities.select(["entity_id"]).iter_rows(named=True):
        entity_id = str(row["entity_id"])
        groups = sorted(groups_by_entity.get(entity_id, set()))
        if len(groups) == 1:
            gold_group_by_entity[entity_id] = groups[0]
            aligned_count += 1
        elif not groups:
            gold_group_by_entity[entity_id] = f"__fp__:{entity_id}"
            unmatched_count += 1
        else:
            gold_group_by_entity[entity_id] = f"__ambiguous__:{entity_id}"
            ambiguous_count += 1

    return gold_group_by_entity, {
        "matched_gold_mentions": matched_mentions.height,
        "entities_with_gold_group": aligned_count,
        "entities_without_gold_match": unmatched_count,
        "entities_with_ambiguous_gold_groups": ambiguous_count,
        "entity_alignment_rate": (
            (aligned_count / entities.height) if entities.height else 0.0
        ),
    }


def _filter_labelable_gold_groups(
    gold_group_by_entity: Mapping[str, str],
) -> dict[str, str]:
    """Keep only entities with one trusted gold-group bridge."""
    return {
        entity_id: group_id
        for entity_id, group_id in gold_group_by_entity.items()
        if _is_labelable_gold_group(group_id)
    }


def _build_labels_table(
    candidate_pairs: pl.DataFrame,
    gold_group_by_entity: Mapping[str, str],
    *,
    entity_doc_id_by_entity: Mapping[str, str] | None = None,
    allowed_doc_ids: Sequence[str] | None = None,
) -> pa.Table:
    """Build a strict-schema labels table from confidently bridged candidate pairs."""
    allowed_doc_id_set = None
    if allowed_doc_ids is not None:
        allowed_doc_id_set = {str(doc_id) for doc_id in allowed_doc_ids}

    run_ids: list[str] = []
    entity_ids_a: list[str] = []
    entity_ids_b: list[str] = []
    labels: list[int] = []
    for row in candidate_pairs.iter_rows(named=True):
        entity_id_a = str(row["entity_id_a"])
        entity_id_b = str(row["entity_id_b"])
        if allowed_doc_id_set is not None:
            if entity_doc_id_by_entity is None:
                raise ValueError(
                    "entity_doc_id_by_entity is required when allowed_doc_ids is set"
                )
            if (
                entity_doc_id_by_entity[entity_id_a] not in allowed_doc_id_set
                or entity_doc_id_by_entity[entity_id_b] not in allowed_doc_id_set
            ):
                continue
        group_id_a = gold_group_by_entity[entity_id_a]
        group_id_b = gold_group_by_entity[entity_id_b]
        if not (
            _is_labelable_gold_group(group_id_a)
            and _is_labelable_gold_group(group_id_b)
        ):
            continue
        run_ids.append(str(row["run_id"]))
        entity_ids_a.append(entity_id_a)
        entity_ids_b.append(entity_id_b)
        labels.append(int(group_id_a == group_id_b))
    return pa.Table.from_arrays(
        [
            pa.array(run_ids, type=pa.string()),
            pa.array(entity_ids_a, type=pa.string()),
            pa.array(entity_ids_b, type=pa.string()),
            pa.array(labels, type=pa.int8()),
        ],
        schema=LABELS_SCHEMA,
    )


def _is_labelable_gold_group(group_id: str) -> bool:
    """Return True when a bridge result represents one trusted gold group."""
    return not str(group_id).startswith(UNLABELABLE_GOLD_GROUP_PREFIXES)


def _merge_labels_into_shared_store(table: pa.Table, path: Path, run_id: str) -> None:
    """Merge one run's labels into an explicit shared labels parquet path."""
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        existing_rows = [
            row for row in pq.read_table(path).to_pylist() if row["run_id"] != run_id
        ]
    merged = pa.Table.from_pylist(
        existing_rows + table.to_pylist(), schema=LABELS_SCHEMA
    )
    _write_parquet_atomic(merged, path)


def _cluster_membership_from_resolved(
    resolved_entities: pl.DataFrame,
    entities: pl.DataFrame,
) -> dict[str, str]:
    """Build entity_id -> cluster_id mapping and require full entity coverage."""
    membership = {
        str(row["entity_id"]): str(row["cluster_id"])
        for row in resolved_entities.select(["entity_id", "cluster_id"]).iter_rows(
            named=True
        )
    }
    entity_ids = {str(entity_id) for entity_id in entities["entity_id"].to_list()}
    if set(membership) != entity_ids:
        raise ValueError(
            "resolved_entities does not cover exactly the same entity IDs as entities"
        )
    return membership


def _pair_key_set(frame: pl.DataFrame) -> set[tuple[str, str]]:
    """Return canonical (a < b) pair keys from a pair table."""
    pairs: set[tuple[str, str]] = set()
    for row in frame.select(["entity_id_a", "entity_id_b"]).iter_rows(named=True):
        a, b = str(row["entity_id_a"]), str(row["entity_id_b"])
        pairs.add((a, b) if a < b else (b, a))
    return pairs


def _thresholded_pair_key_set(
    scored_pairs: pl.DataFrame, threshold: float
) -> set[tuple[str, str]]:
    """Return positive pair keys at one probability threshold."""
    return _pair_key_set(scored_pairs.filter(pl.col("score") >= threshold))


def _blocking_positive_coverage(
    candidate_pair_keys: set[tuple[str, str]],
    gold_positive_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    """Summarize how many gold-positive pairs survived blocking."""
    recovered = len(candidate_pair_keys & gold_positive_pairs)
    total = len(gold_positive_pairs)
    return {
        "gold_positive_pair_recall": (recovered / total) if total else 0.0,
        "gold_positive_pairs_recovered": recovered,
        "gold_positive_pair_count": total,
    }


def _mention_key_set(frame: pl.DataFrame) -> set[tuple[str, str, int, int]]:
    """Return the exact-match mention-key set used for extraction scoring."""
    return {
        (
            str(row["doc_id"]),
            str(row["entity_type"]),
            int(row["char_start"]),
            int(row["char_end"]),
        )
        for row in frame.select(
            ["doc_id", "entity_type", "char_start", "char_end"]
        ).iter_rows(named=True)
    }


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    """Write JSON with a temp file swap to keep reports consistent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    """Write parquet with a temp file swap to keep outputs consistent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _print_console_summary(report: Mapping[str, Any], report_path: Path) -> None:
    """Emit a concise human-readable summary for one evaluation run."""
    extraction = report["stage_metrics"]["extraction"]
    matching = report["stage_metrics"]["matching"]
    metrics = report["metrics"]
    alignment = report["alignment"]
    print(
        f"run_id={report['run_id']} "
        f"extraction_f1={extraction['f1']:.3f} "
        f"matching_f1={matching['f1']:.3f} "
        f"pairwise_f1={metrics['pairwise_f1']:.3f} "
        f"bcubed_f1={metrics['bcubed_f1']:.3f}"
    )
    print(
        "alignment "
        f"entities_with_gold_group={alignment['entities_with_gold_group']} "
        f"entities_without_gold_match={alignment['entities_without_gold_match']} "
        f"entities_with_ambiguous_gold_groups={alignment['entities_with_ambiguous_gold_groups']}"
    )
    print(f"report={report_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    report = run_evaluation(
        data_dir=args.data_dir,
        run_id=args.run_id,
        gold_path=args.gold_path,
        match_threshold=args.match_threshold,
        baseline_report_path=args.baseline_report,
        shared_labels_path=args.shared_labels_path,
        shared_labels_allowed_doc_ids=args.shared_labels_allowed_doc_ids,
    )
    return 0 if report["regression_checks"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
