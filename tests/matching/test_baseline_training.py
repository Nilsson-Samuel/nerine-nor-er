"""Tests for baseline LightGBM training and validation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.matching.run import run_features
from src.matching.reranker import (
    BASELINE_LIGHTGBM_PARAMS,
    evaluate_lightgbm,
    save_lightgbm_artifacts,
    train_lightgbm,
)
from src.synthetic.build_matching_dataset import build_matching_dataset, load_labeled_feature_matrix


def _variant_triplet(base: str, alias: str, alt: str, context_prefix: str, index: int) -> list[dict]:
    """Build three text/context variants for one synthetic identity group."""
    return [
        {
            "text": base,
            "context": f"{context_prefix} {base} appeared in report {index}.",
        },
        {
            "text": alias,
            "context": f"{context_prefix} alias {alias} appeared in report {index}.",
        },
        {
            "text": alt,
            "context": f"{context_prefix} alternate {alt} appeared in report {index}.",
        },
    ]


def _training_payload() -> dict:
    """Return a compact synthetic payload with enough label diversity for training."""
    groups: list[dict] = []

    person_groups = [
        ("per_01", "Alice Hansen", "A. Hansen", "Alice H."),
        ("per_02", "Bjarne Olsen", "B. Olsen", "Bjarne O."),
        ("per_03", "Caroline Berg", "C. Berg", "Caroline B."),
        ("per_04", "Daniel Vik", "D. Vik", "Daniel V."),
    ]
    for index, (group_id, base, alias, alt) in enumerate(person_groups, start=1):
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "PER",
                "doc_ids": [f"doc_{group_id}_a", f"doc_{group_id}_b"],
                "variants": _variant_triplet(base, alias, alt, "Interview", index),
            }
        )

    org_groups = [
        ("org_01", "Den Norske Bank", "DNB", "DNB ASA"),
        ("org_02", "Oslo Taxi", "Oslo Taxi AS", "Taxi Oslo"),
    ]
    for index, (group_id, base, alias, alt) in enumerate(org_groups, start=1):
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "ORG",
                "doc_ids": [f"doc_{group_id}_a", f"doc_{group_id}_b"],
                "variants": _variant_triplet(base, alias, alt, "Company", index),
            }
        )

    return {
        "run_id": "run_baseline_lgbm",
        "groups": groups,
        "hard_negatives": [
            {"group_id_a": "per_01", "group_id_b": "per_02"},
            {"group_id_a": "org_01", "group_id_b": "org_02"},
        ],
    }


@pytest.fixture()
def synthetic_training_dir(tmp_path: Path) -> tuple[Path, str]:
    """Build synthetic artifacts for baseline reranker training tests."""
    data_dir = tmp_path / "synthetic_training"
    identity_groups_path = tmp_path / "identity_groups.json"
    identity_groups_path.write_text(json.dumps(_training_payload()), encoding="utf-8")

    build_matching_dataset(identity_groups_path, data_dir, max_pairs=2500, seed=7)
    run_features(data_dir, "run_baseline_lgbm")
    return data_dir, "run_baseline_lgbm"


def _split_labeled_data(
    data_dir: Path,
    run_id: str,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
    """Load the labeled feature matrix and create a deterministic train/validation split."""
    X, y = load_labeled_feature_matrix(data_dir, run_id)
    labels = y.to_numpy()
    row_indices = np.arange(X.height)
    train_idx, val_idx, y_train, y_val = train_test_split(
        row_indices,
        labels,
        test_size=0.25,
        random_state=7,
        stratify=labels,
    )
    X_train = X[train_idx.tolist()]
    X_val = X[val_idx.tolist()]
    return X_train, X_val, y_train, y_val


def test_train_lightgbm_runs_end_to_end_on_synthetic_features(
    synthetic_training_dir: tuple[Path, str],
    tmp_path: Path,
) -> None:
    data_dir, run_id = synthetic_training_dir
    X, y = load_labeled_feature_matrix(data_dir, run_id)

    model = train_lightgbm(X, y)
    metadata = save_lightgbm_artifacts(model, tmp_path / "model")

    assert isinstance(model, LGBMClassifier)
    assert model.booster_.num_feature() == X.width
    assert model.booster_.feature_name() == X.columns
    assert metadata["training_params"]["num_leaves"] == 15
    assert metadata["training_params"]["min_child_samples"] == 40
    assert metadata["training_params"]["reg_lambda"] == 3.0
    assert metadata["training_params"]["subsample"] == 0.9
    assert metadata["training_params"]["colsample_bytree"] == 0.9


def test_baseline_lightgbm_defaults_are_conservative() -> None:
    assert BASELINE_LIGHTGBM_PARAMS["learning_rate"] == 0.05
    assert BASELINE_LIGHTGBM_PARAMS["n_estimators"] == 120
    assert BASELINE_LIGHTGBM_PARAMS["num_leaves"] == 15
    assert BASELINE_LIGHTGBM_PARAMS["min_child_samples"] == 40
    assert BASELINE_LIGHTGBM_PARAMS["reg_lambda"] == 3.0
    assert BASELINE_LIGHTGBM_PARAMS["subsample"] == 0.9
    assert BASELINE_LIGHTGBM_PARAMS["subsample_freq"] == 1
    assert BASELINE_LIGHTGBM_PARAMS["colsample_bytree"] == 0.9
    assert BASELINE_LIGHTGBM_PARAMS["random_state"] == 7
    assert BASELINE_LIGHTGBM_PARAMS["feature_fraction_seed"] == 7
    assert BASELINE_LIGHTGBM_PARAMS["bagging_seed"] == 7
    assert BASELINE_LIGHTGBM_PARAMS["data_random_seed"] == 7
    assert BASELINE_LIGHTGBM_PARAMS["force_col_wise"] is True
    assert BASELINE_LIGHTGBM_PARAMS["n_jobs"] == 1


def test_evaluate_lightgbm_returns_finite_metrics_in_unit_interval(
    synthetic_training_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_training_dir
    X_train, X_val, y_train, y_val = _split_labeled_data(data_dir, run_id)

    model = train_lightgbm(X_train, y_train)
    metrics = evaluate_lightgbm(model, X_val, y_val)

    assert set(metrics) == {"precision", "recall", "f_beta", "pr_auc"}
    for value in metrics.values():
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_fixed_seed_yields_stable_validation_metrics(
    synthetic_training_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_training_dir
    X_train, X_val, y_train, y_val = _split_labeled_data(data_dir, run_id)

    first_model = train_lightgbm(X_train, y_train)
    second_model = train_lightgbm(X_train, y_train)

    first_metrics = evaluate_lightgbm(first_model, X_val, y_val)
    second_metrics = evaluate_lightgbm(second_model, X_val, y_val)

    for metric_name, first_value in first_metrics.items():
        assert second_metrics[metric_name] == pytest.approx(first_value, abs=1e-12)
