#!/usr/bin/env python3
"""Run case-held-out Optuna tuning for the matching reranker.

The wrapper is intentionally thin: it reuses the held-out case-fold config
validation, prepares reusable case artifacts, then delegates trials and report
writing to the matching-stage tuning module.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_case_fold_eval import (
    CaseConfig,
    FoldConfig,
    _filter_folds,
    _load_runner_config,
)
from src.evaluation.run import DEFAULT_MATCH_THRESHOLD
from src.matching.fold_preparation import CaseFoldTuningCase, CaseFoldTuningFold
from src.matching.fold_tuning import run_case_fold_optuna_study
from src.matching.tuning import DEFAULT_SMOKE_TRIALS, VALID_TUNING_MODES

DEFAULT_STUDY_NAME = "case_fold_lightgbm"
DEFAULT_OUTPUT_PARENT = Path("data/cv_tuning")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for case-fold Optuna tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="JSON file describing cases and folds."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Explicit directory for prepared artifacts, trials, and reports. "
            "When omitted, a timestamped child is created under data/cv_tuning."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_SMOKE_TRIALS,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_TUNING_MODES),
        default="smoke",
        help="Tuning mode label recorded in reports.",
    )
    parser.add_argument(
        "--fold-name",
        action="append",
        default=None,
        help="Optional fold name filter. Repeat to tune several named folds.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Score threshold used when evaluating held-out resolved clusters.",
    )
    parser.add_argument(
        "--min-bcubed-recall",
        type=float,
        default=None,
        help="Optional per-fold recall guardrail for trusted best params.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, for example sqlite:///data/cv_tuning/optuna.db.",
    )
    parser.add_argument(
        "--study-name",
        default=DEFAULT_STUDY_NAME,
        help="Stable Optuna study name for persistent storage.",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    """Install simple logging for direct CLI use."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _build_default_output_root(
    parent: Path | str = DEFAULT_OUTPUT_PARENT,
    *,
    now: datetime | None = None,
) -> Path:
    """Build a fresh timestamped output directory for ordinary CLI runs."""
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stem = timestamp.strftime("case_fold_tuning_%Y%m%dT%H%M%SZ")
    parent = Path(parent)
    candidate = parent / stem
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = parent / f"{stem}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _resolve_output_root(output_root: Path | None) -> Path:
    """Return the explicit output root or a fresh default run directory."""
    if output_root is not None:
        return output_root.resolve()
    return _build_default_output_root().resolve()


def _to_tuning_cases(cases: dict[str, CaseConfig]) -> dict[str, CaseFoldTuningCase]:
    """Convert validated runner cases into tuning preparation inputs."""
    return {
        name: CaseFoldTuningCase(
            name=case.name,
            case_root=case.case_root,
            gold_path=case.gold_path,
        )
        for name, case in cases.items()
    }


def _to_tuning_folds(folds: list[FoldConfig]) -> list[CaseFoldTuningFold]:
    """Convert validated runner folds into tuning fold definitions."""
    return [
        CaseFoldTuningFold(
            name=fold.name,
            held_out_case=fold.held_out_case,
            train_cases=list(fold.train_cases),
        )
        for fold in folds
    ]


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    _configure_logging()
    cases, folds = _load_runner_config(args.config.resolve())
    selected_folds = _filter_folds(folds, args.fold_name)
    output_root = _resolve_output_root(args.output_root)
    logger.info("Writing case-fold tuning outputs under %s", output_root)
    summary = run_case_fold_optuna_study(
        _to_tuning_cases(cases),
        _to_tuning_folds(selected_folds),
        output_root,
        n_trials=args.n_trials,
        mode=args.mode,
        match_threshold=args.match_threshold,
        min_bcubed_recall=args.min_bcubed_recall,
        storage=args.storage,
        study_name=args.study_name,
    )
    logger.info(
        "Case-fold tuning complete: %s trials written under %s",
        summary["n_trials_completed"],
        output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
