"""Prepare reusable case-fold artifacts before tuning starts.

Case-held-out tuning should spend trial time only on model training, scoring,
resolution, and evaluation. These helpers create or reuse the slower per-case
feature and label artifacts once before Optuna begins.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.run import write_training_labels_from_gold
from src.ingestion.discovery import discover_documents
from src.matching.writer import get_features_output_path
from src.shared.paths import (
    get_blocking_run_output_dir,
    get_evaluation_labels_path,
    get_extraction_run_output_dir,
    get_ingestion_run_output_dir,
    get_run_root_dir,
)

logger = logging.getLogger(__name__)
PREPARED_CASE_MANIFEST_FILENAME = "prepared_case_manifest.json"
PREPARED_CASE_MANIFEST_VERSION = 2
PREPARED_CASE_ARTIFACT_VERSION = 1
MATCHING_FEATURE_PIPELINE_VERSION = 1


@dataclass(frozen=True)
class CaseFoldTuningCase:
    """Case input for feature preparation and gold-label bridging."""

    name: str
    case_root: Path
    gold_path: Path


@dataclass(frozen=True)
class CaseFoldTuningFold:
    """Manual fold with one held-out case and one or more train cases."""

    name: str
    held_out_case: str
    train_cases: list[str]


@dataclass(frozen=True)
class PreparedCaseRun:
    """Prepared per-case artifacts that can be reused by all Optuna trials."""

    case_name: str
    gold_path: Path
    data_dir: Path
    run_id: str
    manifest: dict[str, Any] | None = None


def _stable_run_id(case_name: str) -> str:
    """Build a deterministic run ID so prepared feature artifacts are reusable."""
    seed = f"case_fold_optuna:shared:{case_name}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _ordered_case_names(fold: CaseFoldTuningFold) -> list[str]:
    """Preserve train-case order and append the held-out case once."""
    ordered = list(fold.train_cases)
    if fold.held_out_case not in ordered:
        ordered.append(fold.held_out_case)
    return ordered


def get_prepared_case_manifest_path(data_dir: Path | str, run_id: str) -> Path:
    """Return the manifest path for one prepared case run."""
    return get_run_root_dir(data_dir, run_id) / PREPARED_CASE_MANIFEST_FILENAME


def _file_identity(path: Path, *, case_root: Path | None = None) -> dict[str, Any]:
    """Capture file identity fields that change when local inputs change."""
    resolved = path.resolve()
    stat = resolved.stat()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(resolved),
    }
    if case_root is not None:
        identity["relative_path"] = str(resolved.relative_to(case_root))
    return identity


def _file_sha256(path: Path) -> str:
    """Hash one input file so same-size rewrites cannot reuse stale artifacts."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_prepared_case_manifest(
    case: CaseFoldTuningCase,
    data_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Build the expected input identity for one reusable prepared run."""
    case_root = case.case_root.resolve()
    documents = [
        _file_identity(path, case_root=case_root)
        for path in discover_documents(case_root)
    ]
    return {
        "version": PREPARED_CASE_MANIFEST_VERSION,
        "artifact_semantic_version": PREPARED_CASE_ARTIFACT_VERSION,
        "feature_pipeline_version": MATCHING_FEATURE_PIPELINE_VERSION,
        "case_name": case.name,
        "case_root": str(case_root),
        "gold_path": str(case.gold_path.resolve()),
        "input_documents": documents,
        "gold_file": _file_identity(case.gold_path),
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_prepared_case_manifest(
    data_dir: Path | str,
    run_id: str,
) -> dict[str, Any] | None:
    """Load a prepared-run manifest if one has already been written."""
    path = get_prepared_case_manifest_path(data_dir, run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_prepared_case_manifest(
    data_dir: Path,
    run_id: str,
    manifest: dict[str, Any],
) -> None:
    """Persist the manifest only after features and labels are prepared."""
    path = get_prepared_case_manifest_path(data_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _manifest_without_generated_at(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare manifests by input identity, not by write timestamp."""
    comparable = dict(manifest)
    comparable.pop("generated_at", None)
    return comparable


def _manifest_matches(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
    """Return True when all prepared inputs still match."""
    if actual is None:
        return False
    return _manifest_without_generated_at(actual) == _manifest_without_generated_at(
        expected
    )


def _feature_inputs_match(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
    """Return True when documents stayed stable even if gold labels changed."""
    if actual is None:
        return False
    keys = (
        "version",
        "artifact_semantic_version",
        "feature_pipeline_version",
        "case_name",
        "case_root",
        "input_documents",
        "run_id",
    )
    return all(actual.get(key) == expected.get(key) for key in keys)


_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _require_safe_name(name: str, kind: str) -> None:
    """Reject names that would escape their intended output directory."""
    if not _SAFE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"{kind} name must contain only letters, digits, hyphens, and underscores: {name!r}"
        )


def _reject_duplicate_case_names(cases: dict[str, CaseFoldTuningCase]) -> None:
    """Keep reusable prepared run roots one-to-one with configured cases."""
    seen: dict[str, str] = {}
    for case_key, case in cases.items():
        _require_safe_name(case.name, "case")
        previous_key = seen.get(case.name)
        if previous_key is not None:
            raise ValueError(
                "case-fold tuning cases must have unique names: "
                f"{case.name!r} is used by {previous_key!r} and {case_key!r}"
            )
        seen[case.name] = case_key


def required_prepared_feature_artifact_paths(
    data_dir: Path | str,
    run_id: str,
) -> list[Path]:
    """List the prepared artifacts needed before scoring or label training."""
    data_dir = Path(data_dir)
    return [
        get_ingestion_run_output_dir(data_dir, run_id) / "docs.parquet",
        get_ingestion_run_output_dir(data_dir, run_id) / "chunks.parquet",
        get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet",
        get_blocking_run_output_dir(data_dir, run_id) / "candidate_pairs.parquet",
        get_features_output_path(data_dir, run_id),
    ]


def _required_feature_artifacts_exist(data_dir: Path, run_id: str) -> bool:
    """Check the full prepared artifact surface needed by scoring and labels."""
    required_paths = required_prepared_feature_artifact_paths(data_dir, run_id)
    return all(path.exists() for path in required_paths)


def _ensure_case_feature_artifacts(
    case: CaseFoldTuningCase,
    data_dir: Path,
    run_id: str,
    *,
    refresh: bool = False,
) -> bool:
    """Create expensive case features once before trial optimization starts."""
    if _required_feature_artifacts_exist(data_dir, run_id) and not refresh:
        return False

    from src.blocking.run import run_blocking
    from src.extraction.run import run_extraction
    from src.ingestion.run import run_ingestion
    from src.matching.run import run_features

    logger.info("Preparing feature artifacts for case %s", case.name)
    data_dir.mkdir(parents=True, exist_ok=True)
    if refresh:
        shutil.rmtree(get_run_root_dir(data_dir, run_id), ignore_errors=True)
    run_ingestion(case.case_root, data_dir, run_id=run_id)
    run_extraction(data_dir, run_id)
    run_blocking(data_dir, run_id)
    run_features(data_dir, run_id)
    return True


def prepare_case_fold_artifacts(
    cases: dict[str, CaseFoldTuningCase],
    folds: list[CaseFoldTuningFold],
    output_root: Path | str,
) -> dict[str, dict[str, PreparedCaseRun]]:
    """Ensure per-case features and labels exist before the Optuna study starts."""
    _reject_duplicate_case_names(cases)
    prepared_root = Path(output_root) / "prepared" / "cases"
    prepared_by_fold: dict[str, dict[str, PreparedCaseRun]] = {}
    prepared_by_case: dict[str, PreparedCaseRun] = {}

    for fold in folds:
        prepared_by_fold[fold.name] = {}
        for case_name in _ordered_case_names(fold):
            if case_name not in cases:
                raise ValueError(
                    f"fold {fold.name} references unknown case: {case_name}"
                )
            if case_name not in prepared_by_case:
                case = cases[case_name]
                data_dir = prepared_root / case.name
                run_id = _stable_run_id(case.name)
                expected_manifest = _build_prepared_case_manifest(
                    case, data_dir, run_id
                )
                existing_manifest = load_prepared_case_manifest(data_dir, run_id)
                features_reusable = _required_feature_artifacts_exist(
                    data_dir, run_id
                ) and _feature_inputs_match(existing_manifest, expected_manifest)
                features_created = _ensure_case_feature_artifacts(
                    case,
                    data_dir,
                    run_id,
                    refresh=not features_reusable,
                )
                labels_path = get_evaluation_labels_path(data_dir, run_id)
                manifest_matches = _manifest_matches(
                    existing_manifest,
                    expected_manifest,
                )
                if features_created or not labels_path.exists() or not manifest_matches:
                    logger.info("Preparing label bridge for case %s", case.name)
                    write_training_labels_from_gold(data_dir, run_id, case.gold_path)
                _write_prepared_case_manifest(data_dir, run_id, expected_manifest)
                prepared_by_case[case_name] = PreparedCaseRun(
                    case_name=case.name,
                    gold_path=case.gold_path,
                    data_dir=data_dir,
                    run_id=run_id,
                    manifest=expected_manifest,
                )
            prepared_by_fold[fold.name][case_name] = prepared_by_case[case_name]
    return prepared_by_fold
