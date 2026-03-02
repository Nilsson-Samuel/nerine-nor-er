"""Baseline LightGBM training and validation helpers for pair matching."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score


logger = logging.getLogger(__name__)

DEFAULT_LIGHTGBM_SEED = 7
DEFAULT_MATCH_F_BETA = 0.5
DEFAULT_MATCH_THRESHOLD = 0.5
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

    probabilities = np.asarray(model.booster_.predict(matrix), dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("validation probabilities must be finite")

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
