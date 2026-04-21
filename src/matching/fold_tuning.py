"""Case-held-out Optuna study engine for the matching reranker.

The regular tuning path optimizes pair labels inside one scoring run. This
module keeps a separate, case-aware objective: each trial trains a fold model,
scores the held-out case, runs resolution, and averages final clustering
B-cubed F0.5 across folds.
Make sure you have enough cases for to ensure a reliable generalized model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import polars as pl

from src.evaluation.run import DEFAULT_MATCH_THRESHOLD, run_evaluation
from src.matching.fold_preparation import (
    CaseFoldTuningCase,
    CaseFoldTuningFold,
    PreparedCaseRun,
    load_prepared_case_manifest,
    prepare_case_fold_artifacts,
    required_prepared_feature_artifact_paths,
)
from src.matching.run import FEATURE_OUTPUT_COLUMNS, PAIR_KEY_COLUMNS
from src.matching.fold_training import (
    DEFAULT_FOLD_MODEL_VERSION_PREFIX,
    FOLD_METRICS_FILENAME,
    FoldTrainingSource,
    _format_markdown_metric,
    _markdown_table,
    build_fold_summary_row,
    train_and_save_fold_model,
    write_fold_metrics_csv,
    write_fold_summary_json,
)
from src.matching.reranker import DEFAULT_LIGHTGBM_SEED
from src.matching.tuning import (
    VALID_TUNING_MODES,
    build_tuning_summary,
    suggest_lightgbm_params,
)
from src.matching.writer import get_features_output_path
from src.shared import schemas
from src.shared.paths import (
    get_blocking_run_output_dir,
    get_evaluation_labels_path,
    get_evaluation_run_output_dir,
    get_extraction_run_output_dir,
    get_ingestion_run_output_dir,
    get_matching_run_output_dir,
    get_resolution_run_output_dir,
)
from src.synthetic.build_matching_dataset import LABELS_SCHEMA

CASE_FOLD_BEST_PARAMS_FILENAME = "case_fold_optuna_best_params.json"
FOLD_TUNING_SUMMARY_FILENAME = "fold_tuning_summary.json"
FOLD_TUNING_TRIALS_FILENAME = "fold_tuning_trials.csv"
FOLD_TUNING_REPORT_FILENAME = "fold_tuning_report.md"
CASE_FOLD_STUDY_SUMMARY_FILENAME = FOLD_TUNING_SUMMARY_FILENAME
TRIAL_SUMMARY_FILENAME = "trial_summary.json"
CASE_FOLD_OBJECTIVE_METRIC = "bcubed_f0_5"
CASE_FOLD_FINGERPRINT_ATTR = "case_fold_fingerprint"
CASE_FOLD_FINGERPRINT_HASH_ATTR = "case_fold_fingerprint_hash"
CASE_FOLD_STUDY_FINGERPRINT_VERSION = 3
CASE_FOLD_OBJECTIVE_VERSION = 1
CASE_FOLD_SEARCH_SPACE_VERSION = 1
DEFAULT_FAILED_TRIAL_VALUE = -1.0


FoldRunner = Callable[..., dict[str, Any]]


def _validate_failed_trial_value(failed_trial_value: float) -> None:
    """Keep failed trials outside the valid B-cubed metric range."""
    if failed_trial_value >= 0.0:
        raise ValueError("failed_trial_value must be < 0.0")


def _build_case_fold_study_fingerprint(
    folds: list[CaseFoldTuningFold],
    prepared_by_fold: dict[str, dict[str, PreparedCaseRun]],
    *,
    match_threshold: float,
    enable_shap: bool,
    failed_trial_value: float,
) -> tuple[dict[str, Any], str]:
    """Build a stable payload that identifies objective-compatible studies."""
    case_names = sorted(
        {fold.held_out_case for fold in folds}
        | {case_name for fold in folds for case_name in fold.train_cases}
    )
    payload = {
        "version": CASE_FOLD_STUDY_FINGERPRINT_VERSION,
        "objective_version": CASE_FOLD_OBJECTIVE_VERSION,
        "search_space_version": CASE_FOLD_SEARCH_SPACE_VERSION,
        "objective": "macro_mean_held_out_case_bcubed_f0_5",
        "objective_metric": CASE_FOLD_OBJECTIVE_METRIC,
        "match_threshold": float(match_threshold),
        "enable_shap": bool(enable_shap),
        "failed_trial_value": float(failed_trial_value),
        "case_names": case_names,
        "case_input_identities": _case_input_identities_for_folds(
            folds,
            prepared_by_fold,
        ),
        "folds": [
            {
                "name": fold.name,
                "held_out_case": fold.held_out_case,
                "train_cases": list(fold.train_cases),
            }
            for fold in sorted(folds, key=lambda item: item.name)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def _prepared_case_input_identity(prepared_run: PreparedCaseRun) -> dict[str, Any]:
    """Return the manifest identity used to keep persistent studies compatible."""
    manifest = prepared_run.manifest or load_prepared_case_manifest(
        prepared_run.data_dir,
        prepared_run.run_id,
    )
    if manifest is None:
        return {
            "case_name": prepared_run.case_name,
            "run_id": prepared_run.run_id,
            "manifest_missing": True,
        }

    return {
        "version": manifest.get("version"),
        "artifact_semantic_version": manifest.get("artifact_semantic_version"),
        "feature_pipeline_version": manifest.get("feature_pipeline_version"),
        "case_name": manifest.get("case_name"),
        "case_root": manifest.get("case_root"),
        "gold_path": manifest.get("gold_path"),
        "input_documents": manifest.get("input_documents", []),
        "gold_file": manifest.get("gold_file"),
        "run_id": manifest.get("run_id"),
    }


def _case_input_identities_for_folds(
    folds: list[CaseFoldTuningFold],
    prepared_by_fold: dict[str, dict[str, PreparedCaseRun]],
) -> list[dict[str, Any]]:
    """Collect one deterministic input identity per case used by the folds."""
    identities: dict[str, dict[str, Any]] = {}
    for fold in sorted(folds, key=lambda item: item.name):
        for case_name in [fold.held_out_case, *fold.train_cases]:
            if case_name in identities:
                continue
            prepared_run = prepared_by_fold[fold.name][case_name]
            identities[case_name] = _prepared_case_input_identity(prepared_run)
    return [identities[case_name] for case_name in sorted(identities)]


def _validate_prepared_case_artifacts(
    fold: CaseFoldTuningFold,
    case_name: str,
    prepared_run: PreparedCaseRun,
    *,
    require_labels: bool,
) -> pl.DataFrame | None:
    """Reject missing prepared files before Optuna creates trial failures."""
    _validate_prepared_case_manifest(
        prepared_run,
        context=f"fold {fold.name} case {case_name}",
    )
    feature_paths = required_prepared_feature_artifact_paths(
        prepared_run.data_dir,
        prepared_run.run_id,
    )
    for path in feature_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"fold {fold.name} case {case_name} missing prepared artifact: {path}"
            )
        _validate_parquet_has_run_rows(
            path,
            prepared_run.run_id,
            context=f"fold {fold.name} case {case_name} prepared artifact",
        )

    features = _validate_prepared_features(
        get_features_output_path(prepared_run.data_dir, prepared_run.run_id),
        get_blocking_run_output_dir(prepared_run.data_dir, prepared_run.run_id)
        / "candidate_pairs.parquet",
        prepared_run.run_id,
        context=f"fold {fold.name} case {case_name}",
    )

    labels_path = get_evaluation_labels_path(prepared_run.data_dir, prepared_run.run_id)
    if require_labels and not labels_path.exists():
        raise FileNotFoundError(
            f"fold {fold.name} train case {case_name} missing labels: {labels_path}"
        )
    if require_labels:
        labels = _validate_prepared_labels(
            labels_path,
            prepared_run.run_id,
            context=f"fold {fold.name} train case {case_name} labels",
        )
        _validate_label_feature_alignment(
            features,
            labels,
            context=f"fold {fold.name} train case {case_name}",
        )
        return labels
    return None


def _validate_prepared_case_manifest(
    prepared_run: PreparedCaseRun,
    *,
    context: str,
) -> None:
    """Require prepared-run identity so persistent studies cannot drift silently."""
    manifest = prepared_run.manifest or load_prepared_case_manifest(
        prepared_run.data_dir,
        prepared_run.run_id,
    )
    if manifest is None:
        raise FileNotFoundError(f"{context} missing prepared manifest")
    if manifest.get("run_id") != prepared_run.run_id:
        raise ValueError(f"{context} manifest run_id does not match prepared run")
    if manifest.get("case_name") != prepared_run.case_name:
        raise ValueError(f"{context} manifest case_name does not match prepared run")


def _validate_fold_train_label_classes(
    fold: CaseFoldTuningFold,
    label_frames: list[pl.DataFrame],
) -> None:
    """Fail fold setup early when training labels cannot fit a binary model."""
    labels = pl.concat(label_frames, how="vertical")
    label_values = {int(value) for value in labels["label"].unique().to_list()}
    if label_values != {0, 1}:
        raise ValueError(
            f"fold {fold.name} train labels must contain both 0 and 1 "
            "across train cases"
        )


def _validate_required_columns(
    table: Any,
    required_columns: list[str],
    *,
    context: str,
) -> None:
    """Reject prepared parquet files that cannot satisfy downstream readers."""
    missing = [
        column for column in required_columns if column not in table.schema.names
    ]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _run_frame(table: Any, run_id: str, *, context: str) -> pl.DataFrame:
    """Return rows for one run and reject mixed-run prepared artifacts."""
    frame = pl.from_arrow(table)
    rows = frame.filter(pl.col("run_id") == run_id)
    if rows.is_empty():
        raise ValueError(f"{context} contains no rows for run_id={run_id}")
    if rows.height != frame.height:
        raise ValueError(f"{context} contains rows outside run_id={run_id}")
    return rows


def _raise_if_duplicate_pair_keys(frame: pl.DataFrame, *, context: str) -> None:
    """Reject duplicate pair keys before training or scoring sees them."""
    duplicate_keys = frame.group_by(PAIR_KEY_COLUMNS).len().filter(pl.col("len") > 1)
    if duplicate_keys.height:
        raise ValueError(f"{context} contains duplicate pair keys")


def _validate_prepared_features(
    features_path: Path,
    candidate_pairs_path: Path,
    run_id: str,
    *,
    context: str,
) -> pl.DataFrame:
    """Validate feature columns and scoring key alignment before trials start."""
    candidate_table = pq.read_table(candidate_pairs_path)
    candidate_errors = schemas.validate_contract_rules(
        candidate_table,
        "candidate_pairs",
    )
    if candidate_errors:
        raise ValueError(
            f"{context} candidate_pairs.parquet failed validation: {candidate_errors}"
        )
    candidate_frame = _run_frame(
        candidate_table,
        run_id,
        context=f"{context} candidate_pairs.parquet",
    )

    features_table = pq.read_table(features_path)
    _validate_required_columns(
        features_table,
        FEATURE_OUTPUT_COLUMNS,
        context=f"{context} features.parquet",
    )
    features = _run_frame(
        features_table,
        run_id,
        context=f"{context} features.parquet",
    )
    _raise_if_duplicate_pair_keys(features, context=f"{context} features.parquet")

    expected_keys = candidate_frame.sort(["entity_id_a", "entity_id_b"]).select(
        PAIR_KEY_COLUMNS
    )
    actual_keys = features.select(PAIR_KEY_COLUMNS)
    if actual_keys.to_dict(as_series=False) != expected_keys.to_dict(as_series=False):
        raise ValueError(
            f"{context} features.parquet pair keys do not align with candidate_pairs.parquet"
        )
    return features


def _validate_prepared_labels(
    labels_path: Path,
    run_id: str,
    *,
    context: str,
) -> pl.DataFrame:
    """Validate label bridge schema and pair-key uniqueness before trials start."""
    labels_table = pq.read_table(labels_path)
    label_errors = schemas.validate(labels_table, LABELS_SCHEMA)
    if label_errors:
        raise ValueError(f"{context} failed label schema validation: {label_errors}")
    labels = _run_frame(labels_table, run_id, context=context)
    _raise_if_duplicate_pair_keys(labels, context=context)
    return labels


def _validate_label_feature_alignment(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    context: str,
) -> None:
    """Ensure every train label can join to a prepared feature row."""
    missing_feature_keys = labels.select(PAIR_KEY_COLUMNS).join(
        features.select(PAIR_KEY_COLUMNS),
        on=PAIR_KEY_COLUMNS,
        how="anti",
    )
    if missing_feature_keys.height:
        raise ValueError(
            f"{context} features.parquet is missing keys from labels.parquet"
        )


def _validate_parquet_has_run_rows(path: Path, run_id: str, *, context: str) -> None:
    """Ensure a prepared parquet is readable and contains rows for this run."""
    try:
        table = pq.read_table(path, columns=["run_id"])
    except Exception as exc:
        raise ValueError(
            f"{context} is not readable with run_id column: {path}"
        ) from exc

    if table.num_rows == 0:
        raise ValueError(f"{context} contains no rows: {path}")
    run_ids = {str(value) for value in table.column("run_id").to_pylist()}
    if run_id not in run_ids:
        raise ValueError(f"{context} contains no rows for run_id={run_id}: {path}")


def _validate_case_fold_study_inputs(
    folds: list[CaseFoldTuningFold],
    prepared_by_fold: dict[str, dict[str, PreparedCaseRun]],
) -> None:
    """Reject fold/prepared-run mismatches before Optuna starts trials."""
    if not folds:
        raise ValueError("case-fold tuning requires at least one fold")

    seen_fold_names: set[str] = set()
    for fold in folds:
        if not fold.name:
            raise ValueError("case-fold tuning requires named folds")
        if fold.name in seen_fold_names:
            raise ValueError(f"duplicate fold name: {fold.name}")
        seen_fold_names.add(fold.name)
        if not fold.held_out_case:
            raise ValueError(f"fold {fold.name} requires one held-out case")
        if not fold.train_cases:
            raise ValueError(f"fold {fold.name} requires at least one train case")
        if len(fold.train_cases) != len(set(fold.train_cases)):
            raise ValueError(f"fold {fold.name} contains duplicate train cases")
        if fold.held_out_case in fold.train_cases:
            raise ValueError(
                f"fold {fold.name} cannot train on held-out case {fold.held_out_case}"
            )

        if fold.name not in prepared_by_fold:
            raise ValueError(f"missing prepared artifacts for fold {fold.name}")
        prepared_cases = prepared_by_fold[fold.name]
        if fold.held_out_case not in prepared_cases:
            raise ValueError(
                f"fold {fold.name} is missing held-out case {fold.held_out_case}"
            )
        _validate_prepared_case_artifacts(
            fold,
            fold.held_out_case,
            prepared_cases[fold.held_out_case],
            require_labels=False,
        )
        train_label_frames: list[pl.DataFrame] = []
        for case_name in fold.train_cases:
            if case_name not in prepared_cases:
                raise ValueError(f"fold {fold.name} is missing train case {case_name}")
            labels = _validate_prepared_case_artifacts(
                fold,
                case_name,
                prepared_cases[case_name],
                require_labels=True,
            )
            if labels is not None:
                train_label_frames.append(labels)
        _validate_fold_train_label_classes(fold, train_label_frames)


def _ensure_case_fold_study_fingerprint(
    study: Any,
    fingerprint: dict[str, Any],
    fingerprint_hash: str,
) -> None:
    """Store or compare the persistent-study compatibility fingerprint."""
    existing_hash = study.user_attrs.get(CASE_FOLD_FINGERPRINT_HASH_ATTR)
    existing_payload = study.user_attrs.get(CASE_FOLD_FINGERPRINT_ATTR)

    if existing_hash is None and existing_payload is None:
        if study.trials:
            raise ValueError(
                "Optuna study already has trials but no case-fold fingerprint"
            )
        study.set_user_attr(CASE_FOLD_FINGERPRINT_ATTR, fingerprint)
        study.set_user_attr(CASE_FOLD_FINGERPRINT_HASH_ATTR, fingerprint_hash)
        return

    if existing_hash != fingerprint_hash or existing_payload != fingerprint:
        raise ValueError(
            "Optuna study inputs differ from the stored case-fold fingerprint"
        )


def _write_failed_trial_summary(
    trial: Any,
    trial_dir: Path,
    *,
    error: Exception,
    params: dict[str, Any],
    rows: list[dict[str, Any]],
    failed_trial_value: float,
) -> float:
    """Persist enough failure context without hiding the whole study."""
    payload = {
        "status": "failed",
        "error": str(error),
        "params": params,
        "folds_completed": rows,
        "penalty_value": failed_trial_value,
    }
    write_fold_summary_json(trial_dir / TRIAL_SUMMARY_FILENAME, payload)
    trial.set_user_attr("case_fold_status", "failed")
    trial.set_user_attr("case_fold_error", str(error))
    trial.set_user_attr("penalty_value", failed_trial_value)
    return float(failed_trial_value)


def _link_or_copy_file(source: Path, destination: Path) -> None:
    """Hardlink large reusable artifacts, falling back to a normal copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _materialize_trial_case_run(
    prepared_run: PreparedCaseRun,
    trial_data_dir: Path | str,
) -> PreparedCaseRun:
    """Create a trial-local run root with prepared artifacts for write isolation."""
    trial_data_dir = Path(trial_data_dir)
    if trial_data_dir.resolve() == prepared_run.data_dir.resolve():
        raise ValueError(
            "trial data directory must differ from prepared data directory"
        )

    run_id = prepared_run.run_id
    for output_dir in [
        get_ingestion_run_output_dir(trial_data_dir, run_id),
        get_extraction_run_output_dir(trial_data_dir, run_id),
        get_blocking_run_output_dir(trial_data_dir, run_id),
        get_matching_run_output_dir(trial_data_dir, run_id),
        get_resolution_run_output_dir(trial_data_dir, run_id),
        get_evaluation_run_output_dir(trial_data_dir, run_id),
    ]:
        output_dir.mkdir(parents=True, exist_ok=True)

    required_artifacts = [
        (
            get_ingestion_run_output_dir(prepared_run.data_dir, run_id)
            / "docs.parquet",
            get_ingestion_run_output_dir(trial_data_dir, run_id) / "docs.parquet",
        ),
        (
            get_ingestion_run_output_dir(prepared_run.data_dir, run_id)
            / "chunks.parquet",
            get_ingestion_run_output_dir(trial_data_dir, run_id) / "chunks.parquet",
        ),
        (
            get_extraction_run_output_dir(prepared_run.data_dir, run_id)
            / "entities.parquet",
            get_extraction_run_output_dir(trial_data_dir, run_id) / "entities.parquet",
        ),
        (
            get_blocking_run_output_dir(prepared_run.data_dir, run_id)
            / "candidate_pairs.parquet",
            get_blocking_run_output_dir(trial_data_dir, run_id)
            / "candidate_pairs.parquet",
        ),
        (
            get_features_output_path(prepared_run.data_dir, run_id),
            get_features_output_path(trial_data_dir, run_id),
        ),
    ]
    for source, destination in required_artifacts:
        if not source.exists():
            raise FileNotFoundError(f"missing prepared artifact: {source}")
        _link_or_copy_file(source, destination)

    prepared_labels_path = get_evaluation_labels_path(prepared_run.data_dir, run_id)
    if prepared_labels_path.exists():
        _link_or_copy_file(
            prepared_labels_path,
            get_evaluation_labels_path(trial_data_dir, run_id),
        )

    return PreparedCaseRun(
        case_name=prepared_run.case_name,
        gold_path=prepared_run.gold_path,
        data_dir=trial_data_dir,
        run_id=run_id,
    )


def _run_trial_fold(
    fold: CaseFoldTuningFold,
    prepared_runs: dict[str, PreparedCaseRun],
    trial_fold_dir: Path,
    params: dict[str, Any],
    match_threshold: float,
    enable_shap: bool,
    trial_number: int,
) -> dict[str, Any]:
    """Train and evaluate one held-out fold for a single Optuna trial."""
    from src.matching.run import run_scoring
    from src.resolution.run import run_resolution

    trial_fold_dir.mkdir(parents=True, exist_ok=True)
    model_version = (
        f"{DEFAULT_FOLD_MODEL_VERSION_PREFIX}_optuna__{fold.name}__trial_{trial_number}"
    )
    train_sources = [
        FoldTrainingSource(
            case_name=case_name,
            data_dir=prepared_runs[case_name].data_dir,
            run_id=prepared_runs[case_name].run_id,
        )
        for case_name in fold.train_cases
    ]
    training_result = train_and_save_fold_model(
        train_sources,
        trial_fold_dir,
        model_version=model_version,
        training_params=params,
    )
    held_out_run = prepared_runs[fold.held_out_case]
    trial_run = _materialize_trial_case_run(
        held_out_run,
        trial_fold_dir / "case_run",
    )
    run_scoring(
        trial_run.data_dir,
        trial_run.run_id,
        model_dir=trial_fold_dir,
        enable_shap=enable_shap,
    )
    run_resolution(trial_run.data_dir, trial_run.run_id)
    evaluation_report = run_evaluation(
        trial_run.data_dir,
        trial_run.run_id,
        trial_run.gold_path,
        match_threshold=match_threshold,
    )
    row = build_fold_summary_row(
        fold_name=fold.name,
        held_out_case=fold.held_out_case,
        train_cases=fold.train_cases,
        test_run_id=trial_run.run_id,
        training_metadata=training_result["training_metadata"],
        evaluation_report=evaluation_report,
    )
    return row


def _macro_objective(rows: list[dict[str, Any]]) -> float:
    """Return macro-average B-cubed F0.5 or reject unusable fold metrics."""
    if not rows:
        raise ValueError("case-fold objective requires at least one completed fold")
    values: list[float] = []
    for row in rows:
        value = row.get(CASE_FOLD_OBJECTIVE_METRIC)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"fold {row.get('fold_name', '<unknown>')} has unusable metric"
            )
        values.append(float(value))
    return sum(values) / len(values)


def case_fold_objective(
    trial: Any,
    folds: list[CaseFoldTuningFold],
    prepared_by_fold: dict[str, dict[str, PreparedCaseRun]],
    output_root: Path | str,
    *,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    enable_shap: bool = False,
    failed_trial_value: float = DEFAULT_FAILED_TRIAL_VALUE,
    fold_runner: FoldRunner = _run_trial_fold,
) -> float:
    """Run one Optuna trial and return macro held-out case clustering quality."""
    _validate_failed_trial_value(failed_trial_value)
    _validate_case_fold_study_inputs(folds, prepared_by_fold)
    params = suggest_lightgbm_params(trial)
    trial_dir = Path(output_root) / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for fold in folds:
        prepared_runs = prepared_by_fold[fold.name]
        try:
            rows.append(
                fold_runner(
                    fold,
                    prepared_runs,
                    trial_dir / fold.name,
                    params,
                    match_threshold,
                    enable_shap,
                    int(trial.number),
                )
            )
        except Exception as exc:
            return _write_failed_trial_summary(
                trial,
                trial_dir,
                error=exc,
                params=params,
                rows=rows,
                failed_trial_value=failed_trial_value,
            )
    try:
        objective_value = _macro_objective(rows)
    except Exception as exc:
        return _write_failed_trial_summary(
            trial,
            trial_dir,
            error=exc,
            params=params,
            rows=rows,
            failed_trial_value=failed_trial_value,
        )

    payload = {
        "status": "completed",
        "objective_metric": CASE_FOLD_OBJECTIVE_METRIC,
        "objective_value": objective_value,
        "params": params,
        "fold_count": len(rows),
        "folds": rows,
    }
    write_fold_summary_json(trial_dir / TRIAL_SUMMARY_FILENAME, payload)
    write_fold_metrics_csv(trial_dir / FOLD_METRICS_FILENAME, rows)
    trial.set_user_attr("case_fold_status", "completed")
    trial.set_user_attr("fold_metrics", rows)
    trial.set_user_attr("objective_metric", CASE_FOLD_OBJECTIVE_METRIC)
    return objective_value


def _write_best_params_artifact(
    output_root: Path,
    *,
    best_value: float,
    best_trial_number: int,
    best_params: dict[str, Any],
    n_trials_completed: int,
    fold_count: int,
    storage: str | None,
    study_name: str | None,
) -> str:
    """Persist trusted study params in the same shape as scoring metadata expects."""
    payload = {
        "metric": CASE_FOLD_OBJECTIVE_METRIC,
        "objective": "macro_mean_held_out_case_bcubed_f0_5",
        "n_trials_completed": n_trials_completed,
        "fold_count": fold_count,
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "best_params": best_params,
        "storage": storage,
        "study_name": study_name,
    }
    write_fold_summary_json(output_root / CASE_FOLD_BEST_PARAMS_FILENAME, payload)
    return CASE_FOLD_BEST_PARAMS_FILENAME


def _remove_stale_best_params_artifact(output_root: Path) -> bool:
    """Delete an old trusted-params file when the current run has no trusted best."""
    artifact_path = output_root / CASE_FOLD_BEST_PARAMS_FILENAME
    if not artifact_path.exists():
        return False
    artifact_path.unlink()
    return True


def _trial_state_name(trial: Any) -> str:
    """Return a stable Optuna trial state label for reports."""
    return getattr(trial.state, "name", str(trial.state))


def _trial_params_json(trial: Any) -> str:
    """Encode trial params as a compact CSV cell."""
    return json.dumps(dict(trial.params), sort_keys=True, separators=(",", ":"))


def _trial_passes_bcubed_recall_guardrail(
    trial: Any,
    min_bcubed_recall: float | None,
) -> bool:
    """Require every held-out fold to clear the recall guardrail when set."""
    if min_bcubed_recall is None:
        return True
    fold_metrics = trial.user_attrs.get("fold_metrics", [])
    if not fold_metrics:
        return False
    return all(
        _metric_clears_minimum(row.get("bcubed_recall"), min_bcubed_recall)
        for row in fold_metrics
    )


def _metric_clears_minimum(value: Any, minimum: float) -> bool:
    """Treat missing, malformed, NaN, and infinite metrics as guardrail failures."""
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(metric) and metric >= minimum


def _trial_csv_rows(study: Any) -> list[dict[str, Any]]:
    """Flatten trial and fold metrics so every fold result stays inspectable."""
    rows: list[dict[str, Any]] = []
    for trial in sorted(study.trials, key=lambda item: int(item.number)):
        base_row = {
            "trial_number": int(trial.number),
            "trial_state": _trial_state_name(trial),
            "case_fold_status": trial.user_attrs.get("case_fold_status", ""),
            "objective_value": "" if trial.value is None else float(trial.value),
            "objective_metric": trial.user_attrs.get(
                "objective_metric",
                CASE_FOLD_OBJECTIVE_METRIC,
            ),
            "params_json": _trial_params_json(trial),
            "error": trial.user_attrs.get("case_fold_error", ""),
        }
        fold_metrics = trial.user_attrs.get("fold_metrics") or []
        if not fold_metrics:
            rows.append(
                {
                    **base_row,
                    "fold_name": "",
                    "held_out_case": "",
                    "bcubed_f0_5": "",
                    "bcubed_precision": "",
                    "bcubed_recall": "",
                    "pairwise_precision": "",
                    "pairwise_recall": "",
                    "pairwise_f1": "",
                    "matching_pairwise_precision": "",
                    "matching_pairwise_recall": "",
                    "matching_pairwise_f1": "",
                }
            )
            continue

        for fold_row in fold_metrics:
            rows.append(
                {
                    **base_row,
                    "fold_name": fold_row.get("fold_name", ""),
                    "held_out_case": fold_row.get("held_out_case", ""),
                    "bcubed_f0_5": fold_row.get("bcubed_f0_5", ""),
                    "bcubed_precision": fold_row.get("bcubed_precision", ""),
                    "bcubed_recall": fold_row.get("bcubed_recall", ""),
                    "pairwise_precision": fold_row.get("pairwise_precision", ""),
                    "pairwise_recall": fold_row.get("pairwise_recall", ""),
                    "pairwise_f1": fold_row.get("pairwise_f1", ""),
                    "matching_pairwise_precision": fold_row.get(
                        "matching_pairwise_precision",
                        "",
                    ),
                    "matching_pairwise_recall": fold_row.get(
                        "matching_pairwise_recall",
                        "",
                    ),
                    "matching_pairwise_f1": fold_row.get(
                        "matching_pairwise_f1",
                        "",
                    ),
                }
            )
    return rows


def write_fold_tuning_trials_csv(path: Path | str, study: Any) -> None:
    """Write one flat CSV row per trial/fold metric result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_number",
        "trial_state",
        "case_fold_status",
        "objective_value",
        "objective_metric",
        "fold_name",
        "held_out_case",
        "bcubed_f0_5",
        "bcubed_precision",
        "bcubed_recall",
        "pairwise_precision",
        "pairwise_recall",
        "pairwise_f1",
        "matching_pairwise_precision",
        "matching_pairwise_recall",
        "matching_pairwise_f1",
        "params_json",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _trial_csv_rows(study):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _format_params_for_markdown(params: dict[str, Any]) -> list[str]:
    """Render best params as deterministic Markdown bullets."""
    if not params:
        return ["- None"]
    return [f"- `{name}`: `{value}`" for name, value in sorted(params.items())]


def _best_trial_fold_rows(best_trial: Any | None) -> list[dict[str, Any]]:
    """Return fold metrics attached to the selected winning trial."""
    if best_trial is None:
        return []
    return list(best_trial.user_attrs.get("fold_metrics", []))


def write_fold_tuning_report_markdown(
    path: Path | str,
    *,
    summary: dict[str, Any],
    best_trial: Any | None,
    min_bcubed_recall: float | None,
) -> None:
    """Write a compact Markdown report for reviewing tuning tradeoffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_rows = _best_trial_fold_rows(best_trial)
    best_trial_number = int(best_trial.number) if best_trial is not None else "None"
    best_params = dict(best_trial.params) if best_trial is not None else {}
    guardrail_text = (
        f"B-cubed recall >= {_format_markdown_metric(min_bcubed_recall)} per fold"
        if min_bcubed_recall is not None
        else "Not set"
    )
    lines = [
        "# Case-Fold Tuning Report",
        "",
        "## Study Summary",
        "",
        _markdown_table(
            ["Item", "Value"],
            [
                ["Status", summary["status"]],
                ["Study name", summary["study_name"]],
                ["Optuna completed trial count", summary["n_trials_completed"]],
                [
                    "Objective-completed trial count",
                    summary["objective_completed_trial_count"],
                ],
                ["Trusted trial count", summary["trusted_trial_count"]],
                [
                    "Best params artifact written",
                    summary["best_params_artifact_written"],
                ],
                ["Best trial", best_trial_number],
                ["Best value", _format_markdown_metric(summary["best_value"])],
                ["Objective metric", summary["objective_metric"]],
                ["Recall guardrail", guardrail_text],
            ],
        ),
        "",
        "## Best Params",
        "",
        *_format_params_for_markdown(best_params),
        "",
        "## Winning Trial Fold Metrics",
        "",
    ]
    if best_rows:
        lines.append(
            _markdown_table(
                [
                    "Fold",
                    "Held-out case",
                    "B-cubed F0.5",
                    "B-cubed P",
                    "B-cubed R",
                    "Pairwise P",
                    "Pairwise R",
                    "Pairwise F1",
                    "Matching P",
                    "Matching R",
                    "Matching F1",
                ],
                [
                    [
                        row.get("fold_name", ""),
                        row.get("held_out_case", ""),
                        _format_markdown_metric(row.get("bcubed_f0_5", "")),
                        _format_markdown_metric(row.get("bcubed_precision", "")),
                        _format_markdown_metric(row.get("bcubed_recall", "")),
                        _format_markdown_metric(row.get("pairwise_precision", "")),
                        _format_markdown_metric(row.get("pairwise_recall", "")),
                        _format_markdown_metric(row.get("pairwise_f1", "")),
                        _format_markdown_metric(
                            row.get("matching_pairwise_precision", "")
                        ),
                        _format_markdown_metric(
                            row.get("matching_pairwise_recall", "")
                        ),
                        _format_markdown_metric(row.get("matching_pairwise_f1", "")),
                    ]
                    for row in best_rows
                ],
            )
        )
    else:
        lines.append("No trusted winning trial was available.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_case_fold_optuna_study(
    cases: dict[str, CaseFoldTuningCase],
    folds: list[CaseFoldTuningFold],
    output_root: Path | str,
    *,
    n_trials: int,
    mode: str = "study",
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    enable_shap: bool = False,
    storage: str | None = None,
    study_name: str | None = None,
    min_bcubed_recall: float | None = None,
    failed_trial_value: float = DEFAULT_FAILED_TRIAL_VALUE,
    prepared_by_fold: dict[str, dict[str, PreparedCaseRun]] | None = None,
    fold_runner: FoldRunner = _run_trial_fold,
) -> dict[str, Any]:
    """Run case-held-out Optuna tuning and write summary artifacts."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if mode not in VALID_TUNING_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_TUNING_MODES)}")
    if min_bcubed_recall is not None and not 0.0 <= min_bcubed_recall <= 1.0:
        raise ValueError("min_bcubed_recall must be between 0.0 and 1.0")
    _validate_failed_trial_value(failed_trial_value)

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required for case-fold tuning") from exc

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    effective_prepared = (
        prepared_by_fold
        if prepared_by_fold is not None
        else prepare_case_fold_artifacts(cases, folds, output_root)
    )
    _validate_case_fold_study_inputs(folds, effective_prepared)
    fingerprint, fingerprint_hash = _build_case_fold_study_fingerprint(
        folds,
        effective_prepared,
        match_threshold=match_threshold,
        enable_shap=enable_shap,
        failed_trial_value=failed_trial_value,
    )

    sampler = optuna.samplers.TPESampler(seed=DEFAULT_LIGHTGBM_SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    _ensure_case_fold_study_fingerprint(study, fingerprint, fingerprint_hash)
    study.optimize(
        lambda trial: case_fold_objective(
            trial,
            folds,
            effective_prepared,
            output_root,
            match_threshold=match_threshold,
            enable_shap=enable_shap,
            failed_trial_value=failed_trial_value,
            fold_runner=fold_runner,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    objective_completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and trial.user_attrs.get("case_fold_status") == "completed"
    ]
    trusted_trials = [
        trial
        for trial in objective_completed_trials
        if _trial_passes_bcubed_recall_guardrail(trial, min_bcubed_recall)
    ]
    best_trial = (
        max(trusted_trials, key=lambda trial: float(trial.value))
        if trusted_trials
        else None
    )
    best_value = float(best_trial.value) if best_trial is not None else None
    best_params = dict(best_trial.params) if best_trial is not None else {}
    artifact_name = None
    if best_trial is not None:
        artifact_name = _write_best_params_artifact(
            output_root,
            best_value=best_value,
            best_trial_number=int(best_trial.number),
            best_params=best_params,
            n_trials_completed=completed_trials,
            fold_count=len(folds),
            storage=storage,
            study_name=study_name,
        )
        stale_artifact_removed = False
    else:
        stale_artifact_removed = _remove_stale_best_params_artifact(output_root)

    summary = build_tuning_summary(
        enabled=True,
        status=(
            "completed"
            if best_trial is not None
            else "completed_no_trusted_best_params"
        ),
        mode=mode,
        n_trials_requested=n_trials,
        n_trials_completed=completed_trials,
        best_value=best_value,
        best_params_artifact_written=artifact_name is not None,
        best_params_artifact=artifact_name,
        best_params_found=best_trial is not None,
    )
    summary.update(
        {
            "objective_metric": CASE_FOLD_OBJECTIVE_METRIC,
            "tuning_scope": "case_fold",
            "fold_count": len(folds),
            "study_name": study.study_name,
            "storage": storage,
            "optuna_completed_trial_count": completed_trials,
            "objective_completed_trial_count": len(objective_completed_trials),
            "trusted_trial_count": len(trusted_trials),
            "successful_trial_count": len(trusted_trials),
            "best_params": best_params,
            "min_bcubed_recall": min_bcubed_recall,
            "stale_best_params_artifact_removed": stale_artifact_removed,
            "case_fold_fingerprint_hash": fingerprint_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_fold_tuning_trials_csv(output_root / FOLD_TUNING_TRIALS_FILENAME, study)
    write_fold_tuning_report_markdown(
        output_root / FOLD_TUNING_REPORT_FILENAME,
        summary=summary,
        best_trial=best_trial,
        min_bcubed_recall=min_bcubed_recall,
    )
    write_fold_summary_json(output_root / CASE_FOLD_STUDY_SUMMARY_FILENAME, summary)
    return summary
