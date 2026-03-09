"""SHAP explanation plumbing for LightGBM scored-pair output.

This module keeps SHAP generation optional and formats explanations into the
strict top-5 contract required by the scored-pairs schema.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from lightgbm import Booster, LGBMClassifier


DEFAULT_SHAP_TOP_K = 5


def empty_shap_top5(row_count: int) -> list[list[dict[str, float]]]:
    """Return contract-safe empty SHAP lists for one scored table."""
    return [[] for _ in range(row_count)]


def _coerce_matrix(
    X: pl.DataFrame | np.ndarray,
) -> tuple[np.ndarray, list[str] | None]:
    """Return a 2D matrix and optional feature names for explanation."""
    if isinstance(X, pl.DataFrame):
        return X.to_numpy(), X.columns

    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    return matrix, None


def _booster_from_model(model: LGBMClassifier | Booster) -> Booster:
    """Normalize supported LightGBM model wrappers to a Booster."""
    return model.booster_ if isinstance(model, LGBMClassifier) else model


def _normalize_shap_values(raw_values: object) -> np.ndarray:
    """Normalize SHAP output variants into a 2D float array."""
    values = getattr(raw_values, "values", raw_values)
    if isinstance(values, list):
        values = values[-1]

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 3:
        if matrix.shape[-1] == 2:
            matrix = matrix[:, :, -1]
        elif matrix.shape[1] == 2:
            matrix = matrix[:, -1, :]
    if matrix.ndim != 2:
        raise RuntimeError("SHAP values must resolve to a 2D matrix")
    return matrix


def format_shap_top5(
    feature_names: Sequence[str],
    shap_values: Sequence[float],
    *,
    top_k: int = DEFAULT_SHAP_TOP_K,
) -> list[dict[str, float]]:
    """Format one SHAP row into sorted, unique top-k feature/value structs."""
    if len(feature_names) != len(shap_values):
        raise ValueError("feature_names and shap_values must have the same length")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    ranked = sorted(
        (
            (str(feature), float(value))
            for feature, value in zip(feature_names, shap_values, strict=True)
            if str(feature) and np.isfinite(value)
        ),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return [
        {"feature": feature, "value": float(np.float32(value))}
        for feature, value in ranked[:top_k]
    ]


def explain_lightgbm_top5(
    model: LGBMClassifier | Booster,
    X: pl.DataFrame | np.ndarray,
    *,
    enabled: bool = False,
    max_rows: int | None = None,
    top_k: int = DEFAULT_SHAP_TOP_K,
) -> list[list[dict[str, float]]]:
    """Generate contract-safe SHAP top-k lists or return empty placeholders."""
    matrix, feature_names = _coerce_matrix(X)
    row_count = matrix.shape[0]
    if not enabled:
        return empty_shap_top5(row_count)
    if row_count == 0:
        return []
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be >= 0")

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap is required when explanations are enabled") from exc

    booster = _booster_from_model(model)
    resolved_feature_names = feature_names or list(booster.feature_name())
    explain_row_count = row_count if max_rows is None else min(row_count, max_rows)
    if explain_row_count == 0:
        return empty_shap_top5(row_count)
    explainer = shap.TreeExplainer(booster)
    raw_values = explainer(matrix[:explain_row_count], check_additivity=False)
    shap_matrix = _normalize_shap_values(raw_values)

    top5 = [
        format_shap_top5(resolved_feature_names, row_values, top_k=top_k)
        for row_values in shap_matrix
    ]
    if explain_row_count < row_count:
        top5.extend(empty_shap_top5(row_count - explain_row_count))
    return top5
