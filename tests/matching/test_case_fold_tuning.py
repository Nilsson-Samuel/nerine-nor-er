"""Tests for case-held-out Optuna tuning over prepared fold artifacts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.matching.fold_preparation as fold_preparation
import src.matching.fold_tuning as fold_tuning
from scripts.run_case_fold_tuning import _build_default_output_root
from src.matching import run as matching_run
from src.matching.fold_preparation import get_prepared_case_manifest_path
from src.matching.fold_tuning import (
    CASE_FOLD_BEST_PARAMS_FILENAME,
    CASE_FOLD_FINGERPRINT_ATTR,
    CASE_FOLD_FINGERPRINT_HASH_ATTR,
    CASE_FOLD_OBJECTIVE_METRIC,
    DEFAULT_PAIRWISE_BETA,
    DEFAULT_FAILED_TRIAL_VALUE,
    FOLD_TUNING_REPORT_FILENAME,
    FOLD_TUNING_SUMMARY_FILENAME,
    FOLD_TUNING_TRIALS_FILENAME,
    CaseFoldTuningCase,
    CaseFoldTuningFold,
    PreparedCaseRun,
    case_fold_objective,
    run_case_fold_optuna_study,
    _macro_objective,
    _materialize_trial_case_run,
    _run_trial_fold,
    _trial_passes_pairwise_recall_guardrail,
)
from src.matching.writer import get_features_output_path, get_scored_pairs_output_path
from src.resolution import run as resolution_run
from src.resolution.writer import get_resolved_entities_output_path
from src.shared.paths import (
    get_blocking_run_output_dir,
    get_evaluation_labels_path,
    get_evaluation_report_path,
    get_evaluation_run_output_dir,
    get_extraction_run_output_dir,
    get_ingestion_run_output_dir,
    get_matching_run_output_dir,
    get_resolution_run_output_dir,
)
from src.shared import schemas
from src.synthetic.build_matching_dataset import LABELS_SCHEMA

ENTITY_ID_A = "00000000000000000000000000000001"
ENTITY_ID_B = "00000000000000000000000000000002"
ENTITY_ID_C = "00000000000000000000000000000003"
ENTITY_ID_D = "00000000000000000000000000000004"


class DummyTrial:
    """Small Optuna-like trial for objective unit tests."""

    number = 3

    def __init__(self) -> None:
        self.user_attrs: dict[str, Any] = {}

    def suggest_float(
        self, _name: str, low: float, _high: float, **_kwargs: Any
    ) -> float:
        return low

    def suggest_int(self, _name: str, low: int, _high: int, **_kwargs: Any) -> int:
        return low

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = value


def _fold_row(fold_name: str, pairwise_score: float) -> dict[str, Any]:
    """Build the complete metric surface expected by macro fold summaries."""
    return {
        "fold_name": fold_name,
        "pairwise_precision": pairwise_score,
        "pairwise_recall": pairwise_score,
        "pairwise_f1": pairwise_score,
        "pairwise_f0_5": pairwise_score,
        "ari": 0.5,
        "nmi": 0.5,
        "bcubed_precision": 0.5,
        "bcubed_recall": 0.5,
        "bcubed_f1": 0.5,
        "bcubed_f0_5": 0.5,
    }


def _write_dummy_artifact(path: Path, content: bytes = b"prepared") -> None:
    """Write a tiny placeholder artifact for path-isolation tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_run_id_parquet(path: Path, run_id: str) -> None:
    """Write a tiny readable run-scoped parquet placeholder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"run_id": [run_id]}).write_parquet(path)


def _write_candidate_pairs(path: Path, run_id: str) -> None:
    """Write a minimal candidate-pair artifact with valid key columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "run_id": run_id,
                "entity_id_a": ENTITY_ID_A,
                "entity_id_b": ENTITY_ID_B,
                "blocking_methods": ["exact"],
                "blocking_source": "exact",
                "blocking_method_count": 1,
            },
            {
                "run_id": run_id,
                "entity_id_a": ENTITY_ID_C,
                "entity_id_b": ENTITY_ID_D,
                "blocking_methods": ["exact"],
                "blocking_source": "exact",
                "blocking_method_count": 1,
            },
        ],
        schema=schemas.CANDIDATE_PAIRS_SCHEMA,
    )
    pq.write_table(table, path)


def _write_features(path: Path, run_id: str) -> None:
    """Write a minimal feature artifact aligned to the candidate pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": run_id,
            "entity_id_a": ENTITY_ID_A,
            "entity_id_b": ENTITY_ID_B,
            **{column: 0.0 for column in matching_run.FEATURE_COLUMNS},
        },
        {
            "run_id": run_id,
            "entity_id_a": ENTITY_ID_C,
            "entity_id_b": ENTITY_ID_D,
            **{column: 0.0 for column in matching_run.FEATURE_COLUMNS},
        },
    ]
    pl.DataFrame(rows).write_parquet(path)


def _write_labels(path: Path, run_id: str) -> None:
    """Write a minimal train-label artifact aligned to the feature row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "run_id": run_id,
                "entity_id_a": ENTITY_ID_A,
                "entity_id_b": ENTITY_ID_B,
                "label": 1,
            },
            {
                "run_id": run_id,
                "entity_id_a": ENTITY_ID_C,
                "entity_id_b": ENTITY_ID_D,
                "label": 0,
            },
        ],
        schema=LABELS_SCHEMA,
    )
    pq.write_table(table, path)


def _write_prepared_case_artifacts(
    data_dir: Path,
    run_id: str,
    *,
    include_labels: bool = True,
) -> None:
    """Create the prepared artifact surface that trial materialization expects."""
    for path in [
        get_ingestion_run_output_dir(data_dir, run_id) / "docs.parquet",
        get_ingestion_run_output_dir(data_dir, run_id) / "chunks.parquet",
        get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet",
    ]:
        _write_run_id_parquet(path, run_id)

    _write_candidate_pairs(
        get_blocking_run_output_dir(data_dir, run_id) / "candidate_pairs.parquet",
        run_id,
    )
    _write_features(get_features_output_path(data_dir, run_id), run_id)
    if include_labels:
        _write_labels(get_evaluation_labels_path(data_dir, run_id), run_id)


def _write_prepared_case_manifest(
    data_dir: Path,
    run_id: str,
    case_name: str,
    gold_path: Path,
    *,
    identity_tag: str = "stable",
) -> dict[str, Any]:
    """Write the compact prepared-run manifest used by study fingerprints."""
    manifest = {
        "version": fold_preparation.PREPARED_CASE_MANIFEST_VERSION,
        "artifact_semantic_version": fold_preparation.PREPARED_CASE_ARTIFACT_VERSION,
        "feature_pipeline_version": fold_preparation.MATCHING_FEATURE_PIPELINE_VERSION,
        "case_name": case_name,
        "case_root": str((data_dir / "source").resolve()),
        "gold_path": str(gold_path.resolve()),
        "input_documents": [
            {
                "path": str((data_dir / "source" / f"{case_name}.pdf").resolve()),
                "relative_path": f"{case_name}.pdf",
                "size_bytes": len(identity_tag),
                "mtime_ns": len(identity_tag) * 10,
            }
        ],
        "gold_file": {
            "path": str(gold_path.resolve()),
            "size_bytes": len(identity_tag) + 1,
            "mtime_ns": len(identity_tag) * 20,
        },
        "run_id": run_id,
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    path = get_prepared_case_manifest_path(data_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _prepared_case_run(tmp_path: Path, case_name: str = "case_a") -> PreparedCaseRun:
    """Build a prepared run with dummy static artifacts."""
    gold_path = tmp_path / f"{case_name}_gold.csv"
    run = PreparedCaseRun(
        case_name=case_name,
        gold_path=gold_path,
        data_dir=tmp_path / "prepared" / case_name,
        run_id=f"{case_name}_run",
    )
    _write_prepared_case_artifacts(run.data_dir, run.run_id)
    _write_dummy_artifact(gold_path, b"gold")
    manifest = _write_prepared_case_manifest(
        run.data_dir,
        run.run_id,
        case_name,
        gold_path,
    )
    run = PreparedCaseRun(
        case_name=run.case_name,
        gold_path=run.gold_path,
        data_dir=run.data_dir,
        run_id=run.run_id,
        manifest=manifest,
    )
    return run


def _prepared_case_stub(
    tmp_path: Path,
    case_name: str,
    *,
    identity_tag: str = "stable",
) -> PreparedCaseRun:
    """Build a lightweight prepared-run reference for validation-only tests."""
    gold_path = tmp_path / f"{case_name}_gold.csv"
    data_dir = tmp_path / "prepared" / case_name
    run_id = f"{case_name}_run"
    _write_prepared_case_artifacts(data_dir, run_id)
    _write_dummy_artifact(gold_path, b"gold")
    manifest = _write_prepared_case_manifest(
        data_dir,
        run_id,
        case_name,
        gold_path,
        identity_tag=identity_tag,
    )
    return PreparedCaseRun(
        case_name=case_name,
        gold_path=gold_path,
        data_dir=data_dir,
        run_id=run_id,
        manifest=manifest,
    )


def _prepared_by_fold_stub(
    tmp_path: Path,
    folds: list[CaseFoldTuningFold],
    *,
    identity_tag: str = "stable",
) -> dict[str, dict[str, PreparedCaseRun]]:
    """Create prepared-run coverage matching the supplied fold definitions."""
    case_runs: dict[str, PreparedCaseRun] = {}
    prepared: dict[str, dict[str, PreparedCaseRun]] = {}
    for fold in folds:
        prepared[fold.name] = {}
        for case_name in [fold.held_out_case, *fold.train_cases]:
            case_runs.setdefault(
                case_name,
                _prepared_case_stub(
                    tmp_path,
                    case_name,
                    identity_tag=identity_tag,
                ),
            )
            prepared[fold.name][case_name] = case_runs[case_name]
    return prepared


def _fake_evaluation_report() -> dict[str, Any]:
    """Return the metric fields consumed by fold summary row extraction."""
    return {
        "metrics": {
            "pairwise_precision": 0.7,
            "pairwise_recall": 0.6,
            "pairwise_f1": 0.64,
            "pairwise_f0_5": 0.68,
            "ari": 0.5,
            "nmi": 0.55,
            "bcubed_precision": 0.8,
            "bcubed_recall": 0.65,
            "bcubed_f1": 0.72,
            "bcubed_f0_5": 0.76,
        },
        "metric_scope": {
            "evaluation_entity_count": 4,
            "evaluation_candidate_pair_count": 3,
        },
        "stage_metrics": {
            "blocking": {"gold_positive_pair_recall": 1.0},
            "matching": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        },
    }


def _case_with_local_inputs(tmp_path: Path, case_name: str) -> CaseFoldTuningCase:
    """Create tiny local case inputs for manifest-focused preparation tests."""
    case_root = tmp_path / "cases" / case_name
    case_root.mkdir(parents=True, exist_ok=True)
    _write_dummy_artifact(case_root / f"{case_name}.pdf", b"case document")
    gold_path = tmp_path / "gold" / f"{case_name}.csv"
    _write_dummy_artifact(gold_path, b"gold-v1")
    return CaseFoldTuningCase(case_name, case_root, gold_path)


def _install_fast_preparation_fakes(monkeypatch: Any, calls: dict[str, int]) -> None:
    """Replace expensive pipeline stages with tiny artifact writers."""

    def fake_features(
        _case: CaseFoldTuningCase,
        data_dir: Path,
        run_id: str,
        *,
        refresh: bool = False,
    ) -> bool:
        if (
            fold_preparation._required_feature_artifacts_exist(data_dir, run_id)
            and not refresh
        ):
            return False
        calls["features"] = calls.get("features", 0) + 1
        calls["refresh"] = calls.get("refresh", 0) + int(refresh)
        _write_prepared_case_artifacts(data_dir, run_id, include_labels=False)
        return True

    def fake_labels(data_dir: Path, run_id: str, _gold_path: Path) -> dict[str, Any]:
        calls["labels"] = calls.get("labels", 0) + 1
        _write_run_id_parquet(get_evaluation_labels_path(data_dir, run_id), run_id)
        return {"label_rows_written": 1}

    monkeypatch.setattr(
        fold_preparation,
        "_ensure_case_feature_artifacts",
        fake_features,
    )
    monkeypatch.setattr(
        fold_preparation,
        "write_training_labels_from_gold",
        fake_labels,
    )


def test_prepare_case_fold_artifacts_writes_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {}
    _install_fast_preparation_fakes(monkeypatch, calls)
    case = _case_with_local_inputs(tmp_path, "case_a")
    folds = [CaseFoldTuningFold("fold_a", "case_a", [])]

    prepared = fold_preparation.prepare_case_fold_artifacts(
        {"case_a": case},
        folds,
        tmp_path / "out",
    )
    run = prepared["fold_a"]["case_a"]
    manifest = json.loads(
        get_prepared_case_manifest_path(run.data_dir, run.run_id).read_text(
            encoding="utf-8"
        )
    )

    assert calls == {"features": 1, "refresh": 1, "labels": 1}
    assert manifest["case_name"] == "case_a"
    assert manifest["version"] == fold_preparation.PREPARED_CASE_MANIFEST_VERSION
    assert (
        manifest["artifact_semantic_version"]
        == fold_preparation.PREPARED_CASE_ARTIFACT_VERSION
    )
    assert (
        manifest["feature_pipeline_version"]
        == fold_preparation.MATCHING_FEATURE_PIPELINE_VERSION
    )
    assert manifest["case_root"] == str(case.case_root.resolve())
    assert manifest["gold_path"] == str(case.gold_path.resolve())
    assert manifest["run_id"] == run.run_id
    assert manifest["input_documents"][0]["relative_path"] == "case_a.pdf"
    assert run.manifest == manifest


def test_matching_manifest_reuses_existing_prepared_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {}
    _install_fast_preparation_fakes(monkeypatch, calls)
    case = _case_with_local_inputs(tmp_path, "case_a")
    folds = [CaseFoldTuningFold("fold_a", "case_a", [])]
    output_root = tmp_path / "out"

    fold_preparation.prepare_case_fold_artifacts({"case_a": case}, folds, output_root)
    calls.clear()
    fold_preparation.prepare_case_fold_artifacts({"case_a": case}, folds, output_root)

    assert calls == {}


def test_prepare_case_fold_artifacts_rejects_duplicate_case_names(
    tmp_path: Path,
) -> None:
    case_a = _case_with_local_inputs(tmp_path, "case_a")
    case_b = CaseFoldTuningCase(
        name="case_a",
        case_root=tmp_path / "other_case",
        gold_path=tmp_path / "other_gold.csv",
    )
    folds = [CaseFoldTuningFold("fold_a", "case_a_key", [])]

    with pytest.raises(ValueError, match="unique names"):
        fold_preparation.prepare_case_fold_artifacts(
            {"case_a_key": case_a, "case_b_key": case_b},
            folds,
            tmp_path / "out",
        )


def test_changed_gold_identity_regenerates_labels_without_features(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {}
    _install_fast_preparation_fakes(monkeypatch, calls)
    case = _case_with_local_inputs(tmp_path, "case_a")
    folds = [CaseFoldTuningFold("fold_a", "case_a", [])]
    output_root = tmp_path / "out"

    first = fold_preparation.prepare_case_fold_artifacts(
        {"case_a": case},
        folds,
        output_root,
    )
    calls.clear()
    case.gold_path.write_bytes(b"gold-v2-with-new-review")
    second = fold_preparation.prepare_case_fold_artifacts(
        {"case_a": case},
        folds,
        output_root,
    )

    assert calls == {"labels": 1}
    assert (
        first["fold_a"]["case_a"].manifest["gold_file"]
        != second["fold_a"]["case_a"].manifest["gold_file"]
    )


def test_changed_document_content_same_size_refreshes_features(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {}
    _install_fast_preparation_fakes(monkeypatch, calls)
    case = _case_with_local_inputs(tmp_path, "case_a")
    folds = [CaseFoldTuningFold("fold_a", "case_a", [])]
    output_root = tmp_path / "out"
    document_path = case.case_root / "case_a.pdf"

    first = fold_preparation.prepare_case_fold_artifacts(
        {"case_a": case},
        folds,
        output_root,
    )
    original_stat = document_path.stat()
    calls.clear()
    document_path.write_bytes(b"case documenz")
    os.utime(
        document_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    second = fold_preparation.prepare_case_fold_artifacts(
        {"case_a": case},
        folds,
        output_root,
    )

    assert calls == {"features": 1, "refresh": 1, "labels": 1}
    assert (
        first["fold_a"]["case_a"].manifest["input_documents"]
        != second["fold_a"]["case_a"].manifest["input_documents"]
    )


def test_missing_manifest_forces_prepared_feature_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {}
    _install_fast_preparation_fakes(monkeypatch, calls)
    case = _case_with_local_inputs(tmp_path, "case_a")
    folds = [CaseFoldTuningFold("fold_a", "case_a", [])]
    output_root = tmp_path / "out"
    data_dir = output_root / "prepared" / "cases" / "case_a"
    run_id = fold_preparation._stable_run_id("case_a")
    _write_prepared_case_artifacts(data_dir, run_id)

    prepared = fold_preparation.prepare_case_fold_artifacts(
        {"case_a": case},
        folds,
        output_root,
    )

    assert calls == {"features": 1, "refresh": 1, "labels": 1}
    run = prepared["fold_a"]["case_a"]
    assert get_prepared_case_manifest_path(run.data_dir, run.run_id).exists()


def test_materialize_trial_case_run_links_or_copies_static_artifacts(
    tmp_path: Path,
) -> None:
    prepared_run = _prepared_case_run(tmp_path)
    trial_data_dir = tmp_path / "trials" / "trial_0000" / "fold_a" / "case_run"

    trial_run = _materialize_trial_case_run(prepared_run, trial_data_dir)

    assert trial_run.case_name == prepared_run.case_name
    assert trial_run.run_id == prepared_run.run_id
    assert trial_run.gold_path == prepared_run.gold_path
    assert trial_run.data_dir == trial_data_dir
    for output_dir in [
        get_ingestion_run_output_dir(trial_data_dir, prepared_run.run_id),
        get_extraction_run_output_dir(trial_data_dir, prepared_run.run_id),
        get_blocking_run_output_dir(trial_data_dir, prepared_run.run_id),
        get_matching_run_output_dir(trial_data_dir, prepared_run.run_id),
        get_resolution_run_output_dir(trial_data_dir, prepared_run.run_id),
        get_evaluation_run_output_dir(trial_data_dir, prepared_run.run_id),
    ]:
        assert output_dir.exists()

    copied_paths = [
        get_ingestion_run_output_dir(trial_data_dir, prepared_run.run_id)
        / "docs.parquet",
        get_ingestion_run_output_dir(trial_data_dir, prepared_run.run_id)
        / "chunks.parquet",
        get_extraction_run_output_dir(trial_data_dir, prepared_run.run_id)
        / "entities.parquet",
        get_blocking_run_output_dir(trial_data_dir, prepared_run.run_id)
        / "candidate_pairs.parquet",
        get_features_output_path(trial_data_dir, prepared_run.run_id),
        get_evaluation_labels_path(trial_data_dir, prepared_run.run_id),
    ]
    assert all(path.exists() for path in copied_paths)
    assert (
        get_features_output_path(trial_data_dir, prepared_run.run_id).read_bytes()
        == get_features_output_path(
            prepared_run.data_dir, prepared_run.run_id
        ).read_bytes()
    )


def test_run_trial_fold_uses_trial_local_case_run_for_write_stages(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    prepared_run = _prepared_case_run(tmp_path)
    train_run = _prepared_case_run(tmp_path, "case_b")
    fold = CaseFoldTuningFold("fold_a", "case_a", ["case_b"])
    trial_fold_dir = tmp_path / "trials" / "trial_0007" / "fold_a"
    calls: list[tuple[str, Path, str]] = []

    def fake_train_and_save_fold_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "training_metadata": {
                "labeled_row_count": 10,
                "positive_rate": 0.4,
            }
        }

    def fake_scoring(
        data_dir: Path,
        run_id: str,
        **_kwargs: Any,
    ) -> None:
        calls.append(("scoring", Path(data_dir), run_id))

    def fake_resolution(data_dir: Path, run_id: str) -> dict[str, Any]:
        calls.append(("resolution", Path(data_dir), run_id))
        return {"cluster_count": 2}

    def fake_evaluation(
        data_dir: Path,
        run_id: str,
        _gold_path: Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(("evaluation", Path(data_dir), run_id))
        return _fake_evaluation_report()

    monkeypatch.setattr(
        fold_tuning,
        "train_and_save_fold_model",
        fake_train_and_save_fold_model,
    )
    monkeypatch.setattr(matching_run, "run_scoring", fake_scoring)
    monkeypatch.setattr(resolution_run, "run_resolution", fake_resolution)
    monkeypatch.setattr(fold_tuning, "run_evaluation", fake_evaluation)

    row = _run_trial_fold(
        fold,
        {"case_a": prepared_run, "case_b": train_run},
        trial_fold_dir,
        params={"learning_rate": 0.1},
        match_threshold=0.6,
        enable_shap=True,
        trial_number=7,
    )

    trial_data_dir = trial_fold_dir / "case_run"
    assert calls == [
        ("scoring", trial_data_dir, prepared_run.run_id),
        ("resolution", trial_data_dir, prepared_run.run_id),
        ("evaluation", trial_data_dir, prepared_run.run_id),
    ]
    assert all(call[1] != prepared_run.data_dir for call in calls)
    assert row["fold_name"] == "fold_a"
    assert row["pairwise_f0_5"] == 0.68


def test_repeated_trial_folds_keep_outputs_out_of_prepared_case_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    prepared_run = _prepared_case_run(tmp_path)
    train_run = _prepared_case_run(tmp_path, "case_b")
    fold = CaseFoldTuningFold("fold_a", "case_a", ["case_b"])

    def fake_train_and_save_fold_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "training_metadata": {
                "labeled_row_count": 10,
                "positive_rate": 0.4,
            }
        }

    def fake_scoring(data_dir: Path, run_id: str, **_kwargs: Any) -> None:
        _write_dummy_artifact(get_scored_pairs_output_path(data_dir, run_id), b"scored")

    def fake_resolution(data_dir: Path, run_id: str) -> dict[str, Any]:
        _write_dummy_artifact(
            get_resolved_entities_output_path(data_dir, run_id),
            b"resolved",
        )
        return {"cluster_count": 2}

    def fake_evaluation(
        data_dir: Path,
        run_id: str,
        _gold_path: Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        _write_dummy_artifact(get_evaluation_report_path(data_dir, run_id), b"report")
        return _fake_evaluation_report()

    monkeypatch.setattr(
        fold_tuning,
        "train_and_save_fold_model",
        fake_train_and_save_fold_model,
    )
    monkeypatch.setattr(matching_run, "run_scoring", fake_scoring)
    monkeypatch.setattr(resolution_run, "run_resolution", fake_resolution)
    monkeypatch.setattr(fold_tuning, "run_evaluation", fake_evaluation)

    for trial_number in [0, 1]:
        _run_trial_fold(
            fold,
            {"case_a": prepared_run, "case_b": train_run},
            tmp_path / "trials" / f"trial_{trial_number:04d}" / "fold_a",
            params={"learning_rate": 0.1},
            match_threshold=0.6,
            enable_shap=False,
            trial_number=trial_number,
        )

    for trial_number in [0, 1]:
        trial_data_dir = (
            tmp_path / "trials" / f"trial_{trial_number:04d}" / "fold_a" / "case_run"
        )
        assert get_scored_pairs_output_path(
            trial_data_dir, prepared_run.run_id
        ).exists()
        assert get_resolved_entities_output_path(
            trial_data_dir,
            prepared_run.run_id,
        ).exists()
        assert get_evaluation_report_path(trial_data_dir, prepared_run.run_id).exists()

    assert not get_scored_pairs_output_path(
        prepared_run.data_dir,
        prepared_run.run_id,
    ).exists()
    assert not get_resolved_entities_output_path(
        prepared_run.data_dir,
        prepared_run.run_id,
    ).exists()
    assert not get_evaluation_report_path(
        prepared_run.data_dir,
        prepared_run.run_id,
    ).exists()


def test_case_fold_objective_macro_averages_fake_fold_results(tmp_path: Path) -> None:
    folds = [
        CaseFoldTuningFold("fold_a", "case_a", ["case_b"]),
        CaseFoldTuningFold("fold_b", "case_b", ["case_a"]),
    ]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    trial = DummyTrial()

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        return _fold_row(fold.name, {"fold_a": 0.4, "fold_b": 0.8}[fold.name])

    value = case_fold_objective(
        trial,
        folds,
        prepared,
        tmp_path,
        fold_runner=fake_runner,
    )
    summary = json.loads(
        (tmp_path / "trials" / "trial_0003" / "trial_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert value == 0.6000000000000001
    assert summary["status"] == "completed"
    assert summary["objective_metric"] == CASE_FOLD_OBJECTIVE_METRIC
    assert summary["objective_beta"] == DEFAULT_PAIRWISE_BETA
    assert summary["fold_count"] == 2
    assert trial.user_attrs["case_fold_status"] == "completed"


def test_case_fold_objective_penalizes_failed_fold(tmp_path: Path) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    trial = DummyTrial()

    def failing_runner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("fold could not produce usable metrics")

    value = case_fold_objective(
        trial,
        folds,
        _prepared_by_fold_stub(tmp_path, folds),
        tmp_path,
        failed_trial_value=-0.25,
        fold_runner=failing_runner,
    )
    summary = json.loads(
        (tmp_path / "trials" / "trial_0003" / "trial_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert value == -0.25
    assert summary["status"] == "failed"
    assert summary["objective_metric"] == CASE_FOLD_OBJECTIVE_METRIC
    assert summary["objective_beta"] == DEFAULT_PAIRWISE_BETA
    assert summary["penalty_value"] == -0.25
    assert "usable metrics" in summary["error"]
    assert trial.user_attrs["case_fold_status"] == "failed"


def test_default_failed_trial_value_is_out_of_metric_range() -> None:
    assert DEFAULT_FAILED_TRIAL_VALUE == -1.0


def test_macro_objective_uses_pairwise_precision_recall_and_default_beta() -> None:
    row = _fold_row("fold_a", 0.9)
    row.update(
        {
            "pairwise_precision": 0.8,
            "pairwise_recall": 0.2,
            "bcubed_f0_5": 0.99,
        }
    )

    value = _macro_objective([row])

    assert DEFAULT_PAIRWISE_BETA == 0.5
    assert value == pytest.approx(0.5)
    assert row[CASE_FOLD_OBJECTIVE_METRIC] == pytest.approx(0.5)
    assert row["pairwise_beta"] == DEFAULT_PAIRWISE_BETA


def test_macro_objective_supports_non_default_pairwise_beta() -> None:
    row = _fold_row("fold_a", 0.9)
    row.update({"pairwise_precision": 0.8, "pairwise_recall": 0.2})

    value = _macro_objective([row], pairwise_beta=1.0)

    assert value == pytest.approx(0.32)
    assert row[CASE_FOLD_OBJECTIVE_METRIC] == pytest.approx(0.32)
    assert row["pairwise_beta"] == 1.0


def test_case_fold_objective_rejects_non_negative_failed_trial_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="failed_trial_value"):
        case_fold_objective(
            DummyTrial(),
            [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])],
            {"fold_a": {}},
            tmp_path,
            failed_trial_value=0.0,
        )


def test_case_fold_objective_rejects_missing_prepared_case_before_penalty(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = {"fold_a": {"case_a": _prepared_case_stub(tmp_path, "case_a")}}

    with pytest.raises(ValueError, match="missing train case case_b"):
        case_fold_objective(
            DummyTrial(),
            folds,
            prepared,
            tmp_path,
        )

    assert not (tmp_path / "trials" / "trial_0003" / "trial_summary.json").exists()


@pytest.mark.parametrize("pairwise_beta", [0.0, -0.1, float("nan"), float("inf")])
def test_case_fold_study_rejects_invalid_pairwise_beta(
    tmp_path: Path,
    pairwise_beta: float,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]

    with pytest.raises(ValueError, match="pairwise_beta"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            pairwise_beta=pairwise_beta,
            prepared_by_fold=_prepared_by_fold_stub(tmp_path, folds),
        )


def test_case_fold_study_rejects_missing_prepared_fold(tmp_path: Path) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]

    with pytest.raises(ValueError, match="missing prepared artifacts"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            prepared_by_fold={},
        )


def test_case_fold_study_rejects_missing_held_out_prepared_case(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = {"fold_a": {"case_b": _prepared_case_stub(tmp_path, "case_b")}}

    with pytest.raises(ValueError, match="missing held-out case case_a"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            prepared_by_fold=prepared,
        )


def test_case_fold_study_rejects_missing_train_prepared_case(tmp_path: Path) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = {"fold_a": {"case_a": _prepared_case_stub(tmp_path, "case_a")}}

    with pytest.raises(ValueError, match="missing train case case_b"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            prepared_by_fold=prepared,
        )


def test_case_fold_study_rejects_duplicate_train_cases(tmp_path: Path) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b", "case_b"])]

    with pytest.raises(ValueError, match="duplicate train cases"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            prepared_by_fold=_prepared_by_fold_stub(tmp_path, folds),
        )


def test_case_fold_study_rejects_missing_prepared_artifact_before_optuna(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    storage_path = tmp_path / "optuna.db"
    get_features_output_path(
        prepared["fold_a"]["case_a"].data_dir,
        prepared["fold_a"]["case_a"].run_id,
    ).unlink()

    with pytest.raises(FileNotFoundError, match="missing prepared artifact"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=f"sqlite:///{storage_path}",
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
        )

    assert not storage_path.exists()


def test_case_fold_study_rejects_missing_train_labels_before_optuna(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    storage_path = tmp_path / "optuna.db"
    get_evaluation_labels_path(
        prepared["fold_a"]["case_b"].data_dir,
        prepared["fold_a"]["case_b"].run_id,
    ).unlink()

    with pytest.raises(FileNotFoundError, match="missing labels"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=f"sqlite:///{storage_path}",
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
        )

    assert not storage_path.exists()


def test_case_fold_study_rejects_missing_prepared_manifest_before_optuna(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    storage_path = tmp_path / "optuna.db"
    original_run = prepared["fold_a"]["case_a"]
    get_prepared_case_manifest_path(
        original_run.data_dir,
        original_run.run_id,
    ).unlink()
    prepared["fold_a"]["case_a"] = PreparedCaseRun(
        case_name=original_run.case_name,
        gold_path=original_run.gold_path,
        data_dir=original_run.data_dir,
        run_id=original_run.run_id,
        manifest=None,
    )

    with pytest.raises(FileNotFoundError, match="missing prepared manifest"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=f"sqlite:///{storage_path}",
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
        )

    assert not storage_path.exists()


def test_case_fold_study_rejects_single_class_train_labels_before_optuna(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    storage_path = tmp_path / "optuna.db"
    train_run = prepared["fold_a"]["case_b"]
    labels_path = get_evaluation_labels_path(train_run.data_dir, train_run.run_id)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "run_id": train_run.run_id,
                    "entity_id_a": ENTITY_ID_A,
                    "entity_id_b": ENTITY_ID_B,
                    "label": 1,
                }
            ],
            schema=LABELS_SCHEMA,
        ),
        labels_path,
    )

    with pytest.raises(ValueError, match="both 0 and 1"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=f"sqlite:///{storage_path}",
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
        )

    assert not storage_path.exists()


def test_case_fold_study_rejects_unreadable_prepared_artifact_before_optuna(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    storage_path = tmp_path / "optuna.db"
    get_features_output_path(
        prepared["fold_a"]["case_a"].data_dir,
        prepared["fold_a"]["case_a"].run_id,
    ).write_bytes(b"not parquet")

    with pytest.raises(ValueError, match="not readable"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=f"sqlite:///{storage_path}",
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
        )

    assert not storage_path.exists()


def test_case_fold_study_rejects_prepared_artifact_without_run_rows(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    storage_path = tmp_path / "optuna.db"
    _write_features(
        get_features_output_path(
            prepared["fold_a"]["case_a"].data_dir,
            prepared["fold_a"]["case_a"].run_id,
        ),
        "other_run",
    )

    with pytest.raises(ValueError, match="contains no rows for run_id"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=f"sqlite:///{storage_path}",
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
        )

    assert not storage_path.exists()


def test_case_fold_study_rejects_schema_invalid_features_before_optuna(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)
    storage_path = tmp_path / "optuna.db"
    _write_run_id_parquet(
        get_features_output_path(
            prepared["fold_a"]["case_a"].data_dir,
            prepared["fold_a"]["case_a"].run_id,
        ),
        prepared["fold_a"]["case_a"].run_id,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=f"sqlite:///{storage_path}",
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
        )

    assert not storage_path.exists()


def test_case_fold_study_rejects_duplicate_fold_names(tmp_path: Path) -> None:
    folds = [
        CaseFoldTuningFold("fold_a", "case_a", ["case_b"]),
        CaseFoldTuningFold("fold_a", "case_b", ["case_a"]),
    ]
    prepared = _prepared_by_fold_stub(tmp_path, folds[:1])

    with pytest.raises(ValueError, match="duplicate fold name"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            prepared_by_fold=prepared,
        )


def test_case_fold_study_writes_best_params_and_uses_persistent_storage(
    tmp_path: Path,
) -> None:
    folds = [
        CaseFoldTuningFold("fold_a", "case_a", ["case_b"]),
        CaseFoldTuningFold("fold_b", "case_b", ["case_a"]),
    ]
    storage_path = tmp_path / "optuna.db"
    storage = f"sqlite:///{storage_path}"
    prepared = _prepared_by_fold_stub(tmp_path, folds)

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        trial_number: int,
    ) -> dict[str, Any]:
        base_value = 0.4 + (0.2 * trial_number)
        return _fold_row(
            fold.name, base_value + (0.1 if fold.name == "fold_b" else 0.0)
        )

    summary = run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=2,
        storage=storage,
        study_name="case_fold_demo",
        prepared_by_fold=prepared,
        fold_runner=fake_runner,
    )
    artifact = json.loads(
        (tmp_path / CASE_FOLD_BEST_PARAMS_FILENAME).read_text(encoding="utf-8")
    )

    loaded = optuna.load_study(study_name="case_fold_demo", storage=storage)

    assert storage_path.exists()
    assert len(loaded.trials) == 2
    assert CASE_FOLD_FINGERPRINT_ATTR in loaded.user_attrs
    assert CASE_FOLD_FINGERPRINT_HASH_ATTR in loaded.user_attrs
    assert summary["best_params_artifact_written"] is True
    assert summary["best_params_artifact"] == CASE_FOLD_BEST_PARAMS_FILENAME
    assert summary["objective_metric"] == CASE_FOLD_OBJECTIVE_METRIC
    assert summary["objective_beta"] == DEFAULT_PAIRWISE_BETA
    assert artifact["metric"] == CASE_FOLD_OBJECTIVE_METRIC
    assert artifact["objective"] == "macro_mean_held_out_case_pairwise_f_beta"
    assert artifact["objective_beta"] == DEFAULT_PAIRWISE_BETA
    assert artifact["fold_count"] == 2
    assert artifact["n_trials_completed"] == 2
    assert set(artifact["best_params"]) == {
        "learning_rate",
        "n_estimators",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
    }


def test_case_fold_study_writes_readable_tuning_reports(tmp_path: Path) -> None:
    folds = [
        CaseFoldTuningFold("fold_a", "case_a", ["case_b"]),
        CaseFoldTuningFold("fold_b", "case_b", ["case_a"]),
    ]

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        trial_number: int,
    ) -> dict[str, Any]:
        row = _fold_row(fold.name, 0.45 + (0.2 * trial_number))
        row.update(
            {
                "held_out_case": fold.held_out_case,
                "matching_pairwise_precision": 0.91,
                "matching_pairwise_recall": 0.82,
                "matching_pairwise_f1": 0.86,
            }
        )
        return row

    summary = run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=2,
        mode="study",
        study_name="case_fold_demo",
        prepared_by_fold=_prepared_by_fold_stub(tmp_path, folds),
        fold_runner=fake_runner,
    )
    summary_payload = json.loads(
        (tmp_path / FOLD_TUNING_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    with (tmp_path / FOLD_TUNING_TRIALS_FILENAME).open(
        encoding="utf-8",
        newline="",
    ) as handle:
        trial_rows = list(csv.DictReader(handle))
    report = (tmp_path / FOLD_TUNING_REPORT_FILENAME).read_text(encoding="utf-8")

    assert summary["best_value"] == pytest.approx(0.65)
    assert summary_payload["best_params"]
    assert summary_payload["n_trials_completed"] == 2
    assert len(trial_rows) == 4
    assert trial_rows[0]["fold_name"] == "fold_a"
    assert float(trial_rows[0]["pairwise_f_beta"]) == pytest.approx(0.45)
    assert trial_rows[0]["pairwise_beta"] == "0.5"
    assert float(trial_rows[0]["pairwise_f0_5"]) == pytest.approx(0.45)
    assert "Best Params" in report
    assert "Optuna completed trial count" in report
    assert "Objective-completed trial count" in report
    assert "Trusted trial count" in report
    assert "Best params artifact written" in report
    assert "Pairwise F-beta" in report
    assert "Pairwise beta" in report
    assert "B-cubed R" in report
    assert "Pairwise P" in report


def test_case_fold_study_recall_guardrail_withholds_best_params(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]

    def low_recall_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        row = _fold_row(fold.name, 0.7)
        row["pairwise_recall"] = 0.4
        return row

    summary = run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        min_pairwise_recall=0.8,
        prepared_by_fold=_prepared_by_fold_stub(tmp_path, folds),
        fold_runner=low_recall_runner,
    )

    assert summary["status"] == "completed_no_trusted_best_params"
    assert summary["best_params_found"] is False
    assert summary["objective_completed_trial_count"] == 1
    assert summary["trusted_trial_count"] == 0
    assert summary["successful_trial_count"] == 0
    assert not (tmp_path / CASE_FOLD_BEST_PARAMS_FILENAME).exists()
    assert (tmp_path / FOLD_TUNING_TRIALS_FILENAME).exists()


def test_case_fold_study_pairwise_recall_guardrail_accepts_all_passing_folds(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]

    def passing_recall_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        row = _fold_row(fold.name, 0.7)
        row["pairwise_recall"] = 0.8
        return row

    summary = run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        min_pairwise_recall=0.8,
        prepared_by_fold=_prepared_by_fold_stub(tmp_path, folds),
        fold_runner=passing_recall_runner,
    )

    assert summary["best_params_found"] is True
    assert summary["trusted_trial_count"] == 1
    assert (tmp_path / CASE_FOLD_BEST_PARAMS_FILENAME).exists()


def test_case_fold_study_removes_stale_best_params_when_guardrail_rejects_all(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    prepared = _prepared_by_fold_stub(tmp_path, folds)

    def trusted_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        row = _fold_row(fold.name, 0.7)
        row["pairwise_recall"] = 0.9
        return row

    first_summary = run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        prepared_by_fold=prepared,
        fold_runner=trusted_runner,
    )
    assert first_summary["best_params_artifact_written"] is True
    assert (tmp_path / CASE_FOLD_BEST_PARAMS_FILENAME).exists()

    def low_recall_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        row = _fold_row(fold.name, 0.8)
        row["pairwise_recall"] = 0.2
        return row

    second_summary = run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        min_pairwise_recall=0.8,
        prepared_by_fold=prepared,
        fold_runner=low_recall_runner,
    )

    assert second_summary["status"] == "completed_no_trusted_best_params"
    assert second_summary["objective_completed_trial_count"] == 1
    assert second_summary["trusted_trial_count"] == 0
    assert second_summary["stale_best_params_artifact_removed"] is True
    assert not (tmp_path / CASE_FOLD_BEST_PARAMS_FILENAME).exists()


@pytest.mark.parametrize("recall_value", [None, "not-a-number", "nan", "inf", "-inf"])
def test_pairwise_guardrail_treats_malformed_recall_as_failure(
    recall_value: Any,
) -> None:
    row = _fold_row("fold_a", 0.7)
    if recall_value is None:
        del row["pairwise_recall"]
    else:
        row["pairwise_recall"] = recall_value
    trial = type("TrialWithMetrics", (), {"user_attrs": {"fold_metrics": [row]}})()

    assert _trial_passes_pairwise_recall_guardrail(trial, 0.8) is False


def test_case_fold_tuning_default_output_root_is_timestamped_and_fresh(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 4, 21, 12, 30, 15, tzinfo=timezone.utc)
    first_path = _build_default_output_root(tmp_path, now=now)
    first_path.mkdir(parents=True)

    next_path = _build_default_output_root(tmp_path, now=now)

    assert first_path == tmp_path / "case_fold_tuning_20260421T123015Z"
    assert next_path == tmp_path / "case_fold_tuning_20260421T123015Z_02"


def test_case_fold_tuning_runner_help_bootstraps_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, "scripts/run_case_fold_tuning.py", "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run case-held-out Optuna tuning" in result.stdout
    assert "--pairwise-beta" in result.stdout
    assert "--min-pairwise-recall" in result.stdout
    assert "--storage" in result.stdout


def test_case_fold_study_reuses_matching_persistent_fingerprint(
    tmp_path: Path,
) -> None:
    folds = [
        CaseFoldTuningFold("fold_a", "case_a", ["case_b"]),
        CaseFoldTuningFold("fold_b", "case_b", ["case_a"]),
    ]
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    prepared = _prepared_by_fold_stub(tmp_path, folds)

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        return _fold_row(fold.name, 0.3)

    run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        storage=storage,
        study_name="case_fold_demo",
        prepared_by_fold=prepared,
        fold_runner=fake_runner,
    )
    run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        storage=storage,
        study_name="case_fold_demo",
        prepared_by_fold=prepared,
        fold_runner=fake_runner,
    )

    loaded = optuna.load_study(study_name="case_fold_demo", storage=storage)
    assert len(loaded.trials) == 2


def test_case_fold_study_rejects_existing_unfingerprinted_trials(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="case_fold_demo",
    ).optimize(lambda _trial: 0.1, n_trials=1)

    with pytest.raises(ValueError, match="no case-fold fingerprint"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=storage,
            study_name="case_fold_demo",
            prepared_by_fold=_prepared_by_fold_stub(tmp_path, folds),
            fold_runner=lambda *_args, **_kwargs: _fold_row("fold_a", 0.4),
        )


def test_case_fold_study_rejects_changed_case_inputs_for_same_folds(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        return _fold_row(fold.name, 0.4)

    run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        storage=storage,
        study_name="case_fold_demo",
        prepared_by_fold=_prepared_by_fold_stub(
            tmp_path / "original",
            folds,
            identity_tag="original",
        ),
        fold_runner=fake_runner,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=storage,
            study_name="case_fold_demo",
            prepared_by_fold=_prepared_by_fold_stub(
                tmp_path / "changed",
                folds,
                identity_tag="changed",
            ),
            fold_runner=fake_runner,
        )


def test_case_fold_study_rejects_changed_search_space_version(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    prepared = _prepared_by_fold_stub(tmp_path, folds)

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        return _fold_row(fold.name, 0.4)

    run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        storage=storage,
        study_name="case_fold_demo",
        prepared_by_fold=prepared,
        fold_runner=fake_runner,
    )

    monkeypatch.setattr(
        fold_tuning,
        "CASE_FOLD_SEARCH_SPACE_VERSION",
        fold_tuning.CASE_FOLD_SEARCH_SPACE_VERSION + 1,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=storage,
            study_name="case_fold_demo",
            prepared_by_fold=prepared,
            fold_runner=fake_runner,
        )


@pytest.mark.parametrize(
    ("pairwise_beta", "min_pairwise_recall"),
    [
        (1.0, None),
        (DEFAULT_PAIRWISE_BETA, 0.7),
    ],
)
def test_case_fold_study_rejects_changed_pairwise_objective_settings(
    tmp_path: Path,
    pairwise_beta: float,
    min_pairwise_recall: float | None,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    prepared = _prepared_by_fold_stub(tmp_path, folds)

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        return _fold_row(fold.name, 0.4)

    run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        storage=storage,
        study_name="case_fold_demo",
        prepared_by_fold=prepared,
        fold_runner=fake_runner,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        run_case_fold_optuna_study(
            cases={},
            folds=folds,
            output_root=tmp_path,
            n_trials=1,
            storage=storage,
            study_name="case_fold_demo",
            pairwise_beta=pairwise_beta,
            min_pairwise_recall=min_pairwise_recall,
            prepared_by_fold=prepared,
            fold_runner=fake_runner,
        )


@pytest.mark.parametrize(
    ("changed_folds", "match_threshold"),
    [
        ([CaseFoldTuningFold("fold_x", "case_a", ["case_b"])], 0.5),
        ([CaseFoldTuningFold("fold_a", "case_c", ["case_b"])], 0.5),
        ([CaseFoldTuningFold("fold_a", "case_a", ["case_c"])], 0.5),
        ([CaseFoldTuningFold("fold_a", "case_a", ["case_b"])], 0.7),
    ],
)
def test_case_fold_study_rejects_incompatible_persistent_fingerprint(
    tmp_path: Path,
    changed_folds: list[CaseFoldTuningFold],
    match_threshold: float,
) -> None:
    original_folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"

    def fake_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        return _fold_row(fold.name, 0.4)

    run_case_fold_optuna_study(
        cases={},
        folds=original_folds,
        output_root=tmp_path,
        n_trials=1,
        storage=storage,
        study_name="case_fold_demo",
        prepared_by_fold=_prepared_by_fold_stub(tmp_path / "original", original_folds),
        fold_runner=fake_runner,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        run_case_fold_optuna_study(
            cases={},
            folds=changed_folds,
            output_root=tmp_path,
            n_trials=1,
            match_threshold=match_threshold,
            storage=storage,
            study_name="case_fold_demo",
            prepared_by_fold=_prepared_by_fold_stub(
                tmp_path / "changed", changed_folds
            ),
            fold_runner=fake_runner,
        )


def test_case_fold_study_treats_zero_score_trial_as_trusted_best(
    tmp_path: Path,
) -> None:
    folds = [CaseFoldTuningFold("fold_a", "case_a", ["case_b"])]

    def zero_score_runner(
        fold: CaseFoldTuningFold,
        _prepared_runs: dict,
        _trial_fold_dir: Path,
        _params: dict,
        _match_threshold: float,
        _enable_shap: bool,
        _trial_number: int,
    ) -> dict[str, Any]:
        return _fold_row(fold.name, 0.0)

    summary = run_case_fold_optuna_study(
        cases={},
        folds=folds,
        output_root=tmp_path,
        n_trials=1,
        prepared_by_fold=_prepared_by_fold_stub(tmp_path, folds),
        fold_runner=zero_score_runner,
    )
    artifact = json.loads(
        (tmp_path / CASE_FOLD_BEST_PARAMS_FILENAME).read_text(encoding="utf-8")
    )

    assert summary["best_params_found"] is True
    assert summary["best_value"] == 0.0
    assert artifact["best_value"] == 0.0
