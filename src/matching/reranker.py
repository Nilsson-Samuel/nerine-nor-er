"""Baseline LightGBM training, persistence, and inference helpers for pair matching."""

from __future__ import annotations

import logging
from pathlib import Path
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl
from lightgbm import Booster, LGBMClassifier
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score


logger = logging.getLogger(__name__)

DEFAULT_LIGHTGBM_SEED = 7
DEFAULT_MATCH_F_BETA = 0.5
DEFAULT_MATCH_THRESHOLD = 0.5
DEFAULT_MODEL_VERSION = "lightgbm_baseline"
MODEL_FILENAME = "reranker_model.txt"
MODEL_METADATA_FILENAME = "reranker_model_metadata.json"
BASELINE_LIGHTGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "n_estimators": 120,
    "num_leaves": 31,
    "min_child_samples": 5,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "random_state": DEFAULT_LIGHTGBM_SEED,
    "feature_fraction_seed": DEFAULT_LIGHTGBM_SEED,
    "bagging_seed": DEFAULT_LIGHTGBM_SEED,
    "data_random_seed": DEFAULT_LIGHTGBM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbosity": -1,
}


def _coerce_feature_matrix(
    X: pl.DataFrame | np.ndarray,
    matrix_name: str,
) -> tuple[np.ndarray, list[str] | None]:
    """Return a 2D NumPy feature matrix and optional feature names."""
    if isinstance(X, pl.DataFrame):
        return X.to_numpy(), X.columns

    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{matrix_name} must be a 2D feature matrix")
    return matrix, None


def _coerce_binary_labels(
    y: pl.Series | np.ndarray,
    label_name: str,
) -> np.ndarray:
    """Return validated binary labels as an int8 NumPy array."""
    values = np.asarray(y.to_list() if isinstance(y, pl.Series) else y)
    if values.ndim != 1:
        raise ValueError(f"{label_name} must be a 1D label vector")

    unique_values = set(np.unique(values).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{label_name} must contain only binary labels 0/1")
    if unique_values != {0, 1}:
        raise ValueError(f"{label_name} must contain both 0 and 1 labels")

    return values.astype(np.int8, copy=False)


def _build_lightgbm_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge baseline parameters with optional caller overrides."""
    merged = dict(BASELINE_LIGHTGBM_PARAMS)
    if params is not None:
        merged.update(dict(params))
    return merged


def _booster_from_model(model: LGBMClassifier | Booster) -> Booster:
    """Return a LightGBM booster from either supported model wrapper."""
    return model.booster_ if isinstance(model, LGBMClassifier) else model


def _jsonable_param_value(value: Any) -> Any:
    """Convert numpy-backed parameter values into JSON-safe primitives."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_lightgbm(
    X_train: pl.DataFrame | np.ndarray,
    y_train: pl.Series | np.ndarray,
    params: Mapping[str, Any] | None = None,
) -> LGBMClassifier:
    """Train a deterministic baseline LightGBM classifier.

    Args:
        X_train: Training feature matrix with the 14 matching features.
        y_train: Binary training labels aligned to ``X_train``.
        params: Optional LightGBM parameter overrides.

    Returns:
        Trained ``LGBMClassifier`` instance.
    """
    matrix, feature_names = _coerce_feature_matrix(X_train, "X_train")
    labels = _coerce_binary_labels(y_train, "y_train")

    model = LGBMClassifier(**_build_lightgbm_params(params))
    model.fit(matrix, labels, feature_name=feature_names)
    return model


def save_lightgbm_artifacts(
    model: LGBMClassifier | Booster,
    out_dir: Path | str,
    model_version: str = DEFAULT_MODEL_VERSION,
    training_params: Mapping[str, Any] | None = None,
    training_param_source: str = "baseline",
) -> dict[str, Any]:
    """Persist a trained LightGBM model and minimal inference metadata."""
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    if not isinstance(training_param_source, str) or not training_param_source.strip():
        raise ValueError("training_param_source must be a non-empty string")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = _booster_from_model(model)
    booster.save_model(str(out_dir / MODEL_FILENAME))

    params_used = {
        key: _jsonable_param_value(value)
        for key, value in _build_lightgbm_params(training_params).items()
    }
    metadata = {
        "model_version": model_version.strip(),
        "feature_names": list(booster.feature_name()),
        "training_param_source": training_param_source.strip(),
        "training_params": params_used,
    }
    (out_dir / MODEL_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def load_lightgbm_artifacts(model_dir: Path | str) -> tuple[Booster, dict[str, Any]]:
    """Load a persisted LightGBM booster and its inference metadata."""
    model_dir = Path(model_dir)
    booster = Booster(model_file=str(model_dir / MODEL_FILENAME))
    metadata = json.loads((model_dir / MODEL_METADATA_FILENAME).read_text(encoding="utf-8"))
    return booster, metadata


def score_lightgbm(
    model: LGBMClassifier | Booster,
    X: pl.DataFrame | np.ndarray,
) -> np.ndarray:
    """Score a feature matrix and return probabilities clipped to [0, 1]."""
    matrix, feature_names = _coerce_feature_matrix(X, "X_score")
    booster = _booster_from_model(model)
    expected_feature_names = list(booster.feature_name())
    if feature_names is not None and expected_feature_names and feature_names != expected_feature_names:
        raise ValueError("X_score columns must match trained LightGBM feature order")

    probabilities = np.asarray(booster.predict(matrix), dtype=np.float64)
    if probabilities.ndim != 1:
        raise RuntimeError("inference probabilities must be a 1D array")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("inference probabilities must be finite")

    return np.clip(probabilities, 0.0, 1.0).astype(np.float32, copy=False)


def evaluate_lightgbm(
    model: LGBMClassifier,
    X_val: pl.DataFrame | np.ndarray,
    y_val: pl.Series | np.ndarray,
    beta: float = DEFAULT_MATCH_F_BETA,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, float]:
    """Evaluate a LightGBM classifier on held-out validation data.

    Args:
        model: Trained LightGBM classifier.
        X_val: Validation feature matrix aligned to ``y_val``.
        y_val: Binary validation labels.
        beta: F-beta value for the precision-weighted decision metric.
        threshold: Probability threshold used for precision/recall/F-beta.

    Returns:
        Dictionary with precision, recall, F-beta, and PR-AUC metrics.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    matrix, _ = _coerce_feature_matrix(X_val, "X_val")
    labels = _coerce_binary_labels(y_val, "y_val")

    probabilities = score_lightgbm(model, matrix).astype(np.float64, copy=False)
    predictions = (probabilities >= threshold).astype(np.int8)
    metrics = {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f_beta": float(fbeta_score(labels, predictions, beta=beta, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }
    logger.info(
        "lightgbm_validation precision=%.6f recall=%.6f f_beta=%.6f pr_auc=%.6f threshold=%.2f",
        metrics["precision"],
        metrics["recall"],
        metrics["f_beta"],
        metrics["pr_auc"],
        threshold,
    )
    return metrics
