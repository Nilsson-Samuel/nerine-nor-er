#!/usr/bin/env python3
"""Run pragmatic held-out case-fold evaluation with isolated per-case outputs.

Each case runs through ingestion, extraction, blocking, and matching features in
its own data directory. Train-case gold annotations are turned into labels,
those labeled rows are concatenated for one fold-specific LightGBM model, and
only the held-out case is scored, resolved, and evaluated with that frozen model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.run import (
    DEFAULT_MATCH_THRESHOLD,
    run_evaluation,
    write_training_labels_from_gold,
)
from src.matching.fold_training import (
    AGGREGATE_FOLD_REPORTS_FILENAME,
    AGGREGATE_FOLD_REPORTS_MARKDOWN_FILENAME,
    AGGREGATE_FOLD_SUMMARY_FILENAME,
    DEFAULT_FOLD_MODEL_VERSION_PREFIX,
    FOLD_METRICS_FILENAME,
    FOLD_SUMMARY_MARKDOWN_FILENAME,
    FOLD_SUMMARY_FILENAME,
    FoldTrainingSource,
    build_fold_summary_row,
    train_and_save_fold_model,
    write_fold_metrics_csv,
    write_aggregate_fold_reports_markdown,
    write_fold_summary_markdown,
    write_fold_summary_json,
)
from src.shared.paths import (
    get_evaluation_markdown_report_path,
    get_evaluation_report_path,
)
from src.resolution.clustering import validate_resolution_thresholds
from src.shared.config import KEEP_SCORE_THRESHOLD, OBJECTIVE_NEUTRAL_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseConfig:
    """Case-specific inputs needed for one fold run."""

    name: str
    case_root: Path
    gold_path: Path


@dataclass(frozen=True)
class FoldConfig:
    """Manual fold definition with one held-out case and explicit train cases."""

    name: str
    held_out_case: str
    train_cases: list[str]


@dataclass(frozen=True)
class CaseRun:
    """Resolved runtime paths for one case inside one fold."""

    case_name: str
    gold_path: Path
    data_dir: Path
    run_id: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manual fold execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="JSON file describing cases and folds."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cv"),
        help="Root directory for fold outputs.",
    )
    parser.add_argument(
        "--fold-name",
        action="append",
        default=None,
        help="Optional fold name filter. Repeat to run several named folds.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Score threshold for held-out matching evaluation.",
    )
    parser.add_argument(
        "--keep-score-threshold",
        type=float,
        default=KEEP_SCORE_THRESHOLD,
        help="Minimum score retained as a resolution graph edge.",
    )
    parser.add_argument(
        "--objective-neutral-threshold",
        type=float,
        default=OBJECTIVE_NEUTRAL_THRESHOLD,
        help="Retained-edge score where resolution evidence turns merge-positive.",
    )
    parser.add_argument(
        "--enable-shap",
        action="store_true",
        help="Generate SHAP explanations for held-out scoring only.",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    """Install a basic logging handler when the runner is used directly."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _resolve_config_path(
    base_dir: Path, raw_path: str | None, default_path: Path
) -> Path:
    """Resolve a config-relative path, falling back to the fold-case default."""
    if raw_path is None:
        return default_path.resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_case_configs(
    payload: dict[str, Any], config_path: Path
) -> dict[str, CaseConfig]:
    """Load named case inputs and apply repo-friendly defaults."""
    base_dir = config_path.parent.resolve()
    source_root_raw = payload.get("source_root", "data/raw")
    source_root = _resolve_config_path(
        base_dir, str(source_root_raw), Path.cwd() / "data/raw"
    )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, dict) or not raw_cases:
        raise ValueError("config must define a non-empty 'cases' object")

    cases: dict[str, CaseConfig] = {}
    for case_name, raw_case in raw_cases.items():
        if not isinstance(raw_case, dict):
            raise ValueError(f"case entry must be an object: {case_name}")
        default_case_root = source_root / case_name
        default_gold_path = (
            default_case_root / "annotation" / "gold_annotations.group_id_reviewed.csv"
        )
        case_root = _resolve_config_path(
            base_dir, raw_case.get("case_root"), default_case_root
        )
        gold_path = _resolve_config_path(
            base_dir, raw_case.get("gold_path"), default_gold_path
        )
        if not case_root.exists():
            raise FileNotFoundError(
                f"case_root does not exist for {case_name}: {case_root}"
            )
        if not gold_path.exists():
            raise FileNotFoundError(
                f"gold_path does not exist for {case_name}: {gold_path}"
            )
        cases[case_name] = CaseConfig(
            name=case_name,
            case_root=case_root,
            gold_path=gold_path,
        )
    return cases


def _parse_test_case_configs(
    payload: dict[str, Any],
    config_path: Path,
) -> dict[str, CaseConfig]:
    """Load optional final hold-out case inputs from the runner config."""
    raw_test_cases = payload.get("test_cases")
    if not raw_test_cases:
        return {}

    base_dir = config_path.parent.resolve()
    source_root_raw = payload.get("source_root", "data/raw")
    source_root = _resolve_config_path(
        base_dir, str(source_root_raw), Path.cwd() / "data/raw"
    )
    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_test_cases, dict):
        entries = [
            (str(case_name), raw_case)
            for case_name, raw_case in raw_test_cases.items()
            if isinstance(raw_case, dict)
        ]
        if len(entries) != len(raw_test_cases):
            raise ValueError("each test_cases entry must be an object")
    elif isinstance(raw_test_cases, list):
        for raw_case in raw_test_cases:
            if not isinstance(raw_case, dict):
                raise ValueError("each test_cases entry must be an object")
            case_name = str(raw_case.get("name", "")).strip()
            if not case_name:
                raise ValueError("each test_cases entry must define name")
            entries.append((case_name, raw_case))
    else:
        raise ValueError("test_cases must be a list or object when provided")

    test_cases: dict[str, CaseConfig] = {}
    for case_name, raw_case in entries:
        configured_name = str(raw_case.get("name", case_name)).strip()
        if configured_name and configured_name != case_name:
            raise ValueError(
                f"test case key and name must match: {case_name} != {configured_name}"
            )
        if case_name in test_cases:
            raise ValueError(f"duplicate test case name: {case_name}")
        default_case_root = source_root / case_name
        default_gold_path = (
            default_case_root / "annotation" / "gold_annotations.group_id_reviewed.csv"
        )
        case_root = _resolve_config_path(
            base_dir, raw_case.get("case_root"), default_case_root
        )
        gold_path = _resolve_config_path(
            base_dir, raw_case.get("gold_path"), default_gold_path
        )
        if not case_root.exists():
            raise FileNotFoundError(
                f"case_root does not exist for test case {case_name}: {case_root}"
            )
        if not gold_path.exists():
            raise FileNotFoundError(
                f"gold_path does not exist for test case {case_name}: {gold_path}"
            )
        test_cases[case_name] = CaseConfig(
            name=case_name,
            case_root=case_root,
            gold_path=gold_path,
        )
    return test_cases


def _parse_fold_configs(payload: dict[str, Any]) -> list[FoldConfig]:
    """Validate explicit manual folds from JSON config."""
    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError("config must define a non-empty 'folds' list")

    folds: list[FoldConfig] = []
    seen_names: set[str] = set()
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, dict):
            raise ValueError("each fold entry must be an object")
        fold_name = str(raw_fold.get("name", "")).strip()
        held_out_case = str(raw_fold.get("held_out_case", "")).strip()
        train_cases = [
            str(case_name).strip() for case_name in raw_fold.get("train_cases", [])
        ]
        if not fold_name:
            raise ValueError("fold name must be a non-empty string")
        if fold_name in seen_names:
            raise ValueError(f"duplicate fold name: {fold_name}")
        if not held_out_case:
            raise ValueError(f"fold {fold_name} must define held_out_case")
        if not train_cases:
            raise ValueError(f"fold {fold_name} must define at least one train case")
        if len(set(train_cases)) != len(train_cases):
            raise ValueError(f"fold {fold_name} must not contain duplicate train cases")
        if held_out_case in train_cases:
            raise ValueError(f"fold {fold_name} must not train on its held-out case")
        seen_names.add(fold_name)
        folds.append(
            FoldConfig(
                name=fold_name,
                held_out_case=held_out_case,
                train_cases=train_cases,
            )
        )
    return folds


def _load_runner_config(
    config_path: Path,
) -> tuple[dict[str, CaseConfig], list[FoldConfig], dict[str, CaseConfig]]:
    """Read and validate the case-fold runner config JSON."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cases = _parse_case_configs(payload, config_path)
    folds = _parse_fold_configs(payload)
    for fold in folds:
        referenced_cases = [*fold.train_cases, fold.held_out_case]
        for case_name in referenced_cases:
            if case_name not in cases:
                raise ValueError(
                    f"fold {fold.name} references unknown case: {case_name}"
                )
    test_cases = _parse_test_case_configs(payload, config_path)
    case_overlap = sorted(set(test_cases) & set(cases))
    if case_overlap:
        raise ValueError(
            f"test_cases must not overlap with cases: {case_overlap}"
        )
    fold_case_names = {
        case_name for fold in folds for case_name in [fold.held_out_case, *fold.train_cases]
    }
    fold_overlap = sorted(set(test_cases) & fold_case_names)
    if fold_overlap:
        raise ValueError(
            f"test_cases must not overlap with fold cases: {fold_overlap}"
        )
    return cases, folds, test_cases


def _filter_folds(
    folds: list[FoldConfig], selected_names: list[str] | None
) -> list[FoldConfig]:
    """Keep only requested folds when the user narrows execution."""
    if not selected_names:
        return folds
    selected = set(selected_names)
    filtered = [fold for fold in folds if fold.name in selected]
    if not filtered:
        raise ValueError(
            f"no configured folds matched --fold-name values: {sorted(selected)}"
        )
    return filtered


def _stable_run_id(fold_name: str, case_name: str) -> str:
    """Build a deterministic 32-char run ID for one case within one fold."""
    seed = f"case_fold:{fold_name}:{case_name}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _ordered_case_names(fold: FoldConfig) -> list[str]:
    """Preserve train-case order and append the held-out case once."""
    ordered = list(fold.train_cases)
    if fold.held_out_case not in ordered:
        ordered.append(fold.held_out_case)
    return ordered


def _run_case_feature_pipeline(
    case: CaseConfig, fold_dir: Path, fold_name: str
) -> CaseRun:
    """Run feature-producing stages for one case inside one fold."""
    from src.blocking.run import run_blocking
    from src.extraction.run import run_extraction
    from src.ingestion.run import run_ingestion
    from src.matching.run import run_features

    case_data_dir = fold_dir / case.name
    run_id = _stable_run_id(fold_name, case.name)
    case_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fold %s: running feature stages for case %s", fold_name, case.name)
    run_ingestion(case.case_root, case_data_dir, run_id=run_id)
    run_extraction(case_data_dir, run_id)
    run_blocking(case_data_dir, run_id)
    run_features(case_data_dir, run_id)
    return CaseRun(
        case_name=case.name,
        gold_path=case.gold_path,
        data_dir=case_data_dir,
        run_id=run_id,
    )


def _train_fold_model(
    fold: FoldConfig,
    fold_dir: Path,
    case_runs: dict[str, CaseRun],
) -> dict[str, Any]:
    """Build labels from train cases and fit one fold-specific LightGBM model."""
    shared_labels_path = fold_dir / "shared_train_labels.parquet"
    train_sources: list[FoldTrainingSource] = []
    label_summaries: list[dict[str, Any]] = []

    for case_name in fold.train_cases:
        case_run = case_runs[case_name]
        logger.info("Fold %s: writing train labels for case %s", fold.name, case_name)
        label_summaries.append(
            write_training_labels_from_gold(
                case_run.data_dir,
                case_run.run_id,
                case_run.gold_path,
                shared_labels_path=shared_labels_path,
            )
        )
        train_sources.append(
            FoldTrainingSource(
                case_name=case_name,
                data_dir=case_run.data_dir,
                run_id=case_run.run_id,
            )
        )

    model_version = f"{DEFAULT_FOLD_MODEL_VERSION_PREFIX}__{fold.name}"
    training_result = train_and_save_fold_model(
        train_sources,
        fold_dir,
        model_version=model_version,
    )
    training_result["shared_labels_path"] = str(shared_labels_path)
    training_result["label_summaries"] = label_summaries
    return training_result


def _evaluate_held_out_case(
    fold: FoldConfig,
    fold_dir: Path,
    case_run: CaseRun,
    *,
    match_threshold: float,
    keep_score_threshold: float,
    objective_neutral_threshold: float,
    enable_shap: bool,
) -> dict[str, Any]:
    """Score, resolve, and evaluate the held-out case with the frozen fold model."""
    from src.matching.run import run_scoring
    from src.resolution.run import run_resolution

    logger.info("Fold %s: scoring held-out case %s", fold.name, case_run.case_name)
    run_scoring(
        case_run.data_dir,
        case_run.run_id,
        model_dir=fold_dir,
        enable_shap=enable_shap,
    )
    run_resolution(
        case_run.data_dir,
        case_run.run_id,
        keep_score_threshold=keep_score_threshold,
        objective_neutral_threshold=objective_neutral_threshold,
    )
    return run_evaluation(
        case_run.data_dir,
        case_run.run_id,
        case_run.gold_path,
        match_threshold=match_threshold,
    )


def run_fold(
    fold: FoldConfig,
    cases: dict[str, CaseConfig],
    output_root: Path,
    *,
    match_threshold: float,
    keep_score_threshold: float,
    objective_neutral_threshold: float,
    enable_shap: bool,
) -> dict[str, Any]:
    """Run one fold end to end and write per-fold summary artifacts."""
    keep_score_threshold, objective_neutral_threshold = validate_resolution_thresholds(
        keep_score_threshold,
        objective_neutral_threshold,
    )
    fold_dir = output_root / fold.name
    fold_dir.mkdir(parents=True, exist_ok=True)
    case_runs = {
        case_name: _run_case_feature_pipeline(cases[case_name], fold_dir, fold.name)
        for case_name in _ordered_case_names(fold)
    }
    training_result = _train_fold_model(fold, fold_dir, case_runs)
    held_out_run = case_runs[fold.held_out_case]
    evaluation_report = _evaluate_held_out_case(
        fold,
        fold_dir,
        held_out_run,
        match_threshold=match_threshold,
        keep_score_threshold=keep_score_threshold,
        objective_neutral_threshold=objective_neutral_threshold,
        enable_shap=enable_shap,
    )
    summary_row = build_fold_summary_row(
        fold_name=fold.name,
        held_out_case=fold.held_out_case,
        train_cases=fold.train_cases,
        test_run_id=held_out_run.run_id,
        training_metadata=training_result["training_metadata"],
        evaluation_report=evaluation_report,
    )
    fold_summary = {
        "fold_name": fold.name,
        "held_out_case": fold.held_out_case,
        "train_cases": fold.train_cases,
        "fold_dir": str(fold_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resolution_thresholds": {
            "keep_score_threshold": keep_score_threshold,
            "objective_neutral_threshold": objective_neutral_threshold,
        },
        "training": training_result,
        "held_out_run": {
            "case_name": held_out_run.case_name,
            "data_dir": str(held_out_run.data_dir),
            "run_id": held_out_run.run_id,
            "evaluation_report_path": str(
                get_evaluation_report_path(held_out_run.data_dir, held_out_run.run_id)
            ),
            "evaluation_markdown_report_path": str(
                get_evaluation_markdown_report_path(
                    held_out_run.data_dir, held_out_run.run_id
                )
            ),
        },
        "summary_row": summary_row,
    }
    write_fold_summary_json(fold_dir / FOLD_SUMMARY_FILENAME, fold_summary)
    write_fold_metrics_csv(fold_dir / FOLD_METRICS_FILENAME, [summary_row])
    write_fold_summary_markdown(fold_dir / FOLD_SUMMARY_MARKDOWN_FILENAME, fold_summary)
    return summary_row


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    _configure_logging()
    cases, folds, _test_cases = _load_runner_config(args.config.resolve())
    selected_folds = _filter_folds(folds, args.fold_name)
    keep_score_threshold, objective_neutral_threshold = validate_resolution_thresholds(
        args.keep_score_threshold,
        args.objective_neutral_threshold,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        run_fold(
            fold,
            cases,
            output_root,
            match_threshold=args.match_threshold,
            keep_score_threshold=keep_score_threshold,
            objective_neutral_threshold=objective_neutral_threshold,
            enable_shap=args.enable_shap,
        )
        for fold in selected_folds
    ]
    write_fold_metrics_csv(output_root / AGGREGATE_FOLD_REPORTS_FILENAME, summary_rows)
    write_aggregate_fold_reports_markdown(
        output_root / AGGREGATE_FOLD_REPORTS_MARKDOWN_FILENAME,
        summary_rows,
    )
    write_fold_summary_json(
        output_root / AGGREGATE_FOLD_SUMMARY_FILENAME,
        {
            "config_path": str(args.config.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "resolution_thresholds": {
                "keep_score_threshold": keep_score_threshold,
                "objective_neutral_threshold": objective_neutral_threshold,
            },
            "fold_count": len(summary_rows),
            "folds": summary_rows,
        },
    )
    logger.info(
        "Case-fold evaluation complete: %d folds written under %s",
        len(summary_rows),
        output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
