"""Optuna tuning plumbing for the matching-stage reranker.

This module keeps tuning optional and bounded. Normal runs can record
disabled or smoke-test status without turning tuning into a required
part of baseline scoring.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src.matching.reranker import (
    DEFAULT_LIGHTGBM_SEED,
    DEFAULT_MATCH_F_BETA,
    DEFAULT_MATCH_THRESHOLD,
    evaluate_lightgbm,
    train_lightgbm,
)


BEST_PARAMS_FILENAME = "optuna_best_params.json"
DEFAULT_SMOKE_TRIALS = 2
NON_TRIVIAL_STUDY_MIN_TRIALS = 5
VALID_TUNING_MODES = {"smoke", "study"}


def build_tuning_summary(
    *,
    enabled: bool,
    status: str,
    mode: str,
    n_trials_requested: int,
    n_trials_completed: int = 0,
    best_value: float | None = None,
    best_params_artifact_written: bool = False,
    best_params_artifact: str | None = None,
    best_params_found: bool = False,
) -> dict[str, Any]:
    """Build a compact JSON-safe summary for run metadata."""
    return {
        "enabled": enabled,
        "status": status,
        "mode": mode,
        "n_trials_requested": n_trials_requested,
        "n_trials_completed": n_trials_completed,
        "best_value": best_value,
        "best_params_found": best_params_found,
        "best_params_artifact_written": best_params_artifact_written,
        "best_params_artifact": best_params_artifact,
    }


def split_labeled_feature_matrix(
    X: pl.DataFrame | np.ndarray,
    y: pl.Series | np.ndarray,
    *,
    validation_size: float = 0.25,
    random_state: int = DEFAULT_LIGHTGBM_SEED,
) -> tuple[pl.DataFrame | np.ndarray, pl.DataFrame | np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/validation split."""
    labels = np.asarray(y.to_list() if isinstance(y, pl.Series) else y)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError(
            "Hyperparameter tuning requires labeled rows from both classes for the "
            "stratified train/validation split."
        )
    min_class_count = int(counts.min())
    if min_class_count < 2:
        raise ValueError(
            "Hyperparameter tuning requires at least 2 labeled rows in each class "
            "for the stratified train/validation split."
        )
    row_indices = np.arange(len(labels))
    train_idx, val_idx, y_train, y_val = train_test_split(
        row_indices,
        labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )
    if isinstance(X, pl.DataFrame):
        return X[train_idx.tolist()], X[val_idx.tolist()], y_train, y_val
    matrix = np.asarray(X)
    return matrix[train_idx], matrix[val_idx], y_train, y_val


def suggest_lightgbm_params(trial: Any) -> dict[str, Any]:
    """Suggested LightGBM search space for reranker tuning."""
    return {
        # Keep learning conservative so small feature sets do not overreact to noise.
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        # Moderate tree counts are enough for smoke/study runs without turning cost up too fast.
        "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
        # Control tree complexity while still allowing useful interaction splits.
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        # Prevent tiny leaves from fitting sparse pairwise quirks too aggressively.
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
        # Row subsampling adds a bit of regularization without making runs unstable.
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        # Column subsampling reduces overreliance on a few strong similarity features.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
    }


def objective(
    trial: Any,
    X_train: pl.DataFrame | np.ndarray,
    y_train: pl.Series | np.ndarray,
    X_val: pl.DataFrame | np.ndarray,
    y_val: pl.Series | np.ndarray,
    *,
    beta: float = DEFAULT_MATCH_F_BETA,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Train one LightGBM trial and return the precision-weighted objective."""
    params = suggest_lightgbm_params(trial)
    model = train_lightgbm(X_train, y_train, params=params)
    metrics = evaluate_lightgbm(model, X_val, y_val, beta=beta, threshold=threshold)
    trial.set_user_attr("validation_metrics", metrics)
    return float(metrics["f_beta"])


def _write_best_params_artifact(
    out_dir: Path,
    *,
    best_params: Mapping[str, Any],
    best_value: float,
    n_trials_completed: int,
    beta: float,
    threshold: float,
) -> str:
    """Persist best-params metadata for non-trivial studies."""
    payload = {
        "metric": "f_beta",
        "beta": beta,
        "threshold": threshold,
        "n_trials_completed": n_trials_completed,
        "best_value": float(best_value),
        "best_params": dict(best_params),
    }
    (out_dir / BEST_PARAMS_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return BEST_PARAMS_FILENAME


def run_optuna_study(
    X: pl.DataFrame | np.ndarray,
    y: pl.Series | np.ndarray,
    *,
    enabled: bool = False,
    mode: str = "smoke",
    n_trials: int = DEFAULT_SMOKE_TRIALS,
    out_dir: Path | str | None = None,
    beta: float = DEFAULT_MATCH_F_BETA,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """Run a bounded Optuna study or return disabled metadata cleanly."""
    if not enabled:
        return build_tuning_summary(
            enabled=False,
            status="disabled",
            mode="disabled",
            n_trials_requested=0,
        )
    if mode not in VALID_TUNING_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_TUNING_MODES)}")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1 when tuning is enabled")

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required when tuning is enabled") from exc

    X_train, X_val, y_train, y_val = split_labeled_feature_matrix(X, y)
    sampler = optuna.samplers.TPESampler(seed=DEFAULT_LIGHTGBM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            beta=beta,
            threshold=threshold,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    completed_trials = sum(trial.value is not None for trial in study.trials)
    best_value = float(study.best_value)
    best_params = dict(study.best_params)

    artifact_name: str | None = None
    artifact_written = False
    if (
        mode == "study"
        and completed_trials >= NON_TRIVIAL_STUDY_MIN_TRIALS
        and out_dir is not None
    ):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifact_name = _write_best_params_artifact(
            out_path,
            best_params=best_params,
            best_value=best_value,
            n_trials_completed=completed_trials,
            beta=beta,
            threshold=threshold,
        )
        artifact_written = True

    return build_tuning_summary(
        enabled=True,
        status="completed",
        mode=mode,
        n_trials_requested=n_trials,
        n_trials_completed=completed_trials,
        best_value=best_value,
        best_params_artifact_written=artifact_written,
        best_params_artifact=artifact_name,
        best_params_found=bool(best_params),
    )
