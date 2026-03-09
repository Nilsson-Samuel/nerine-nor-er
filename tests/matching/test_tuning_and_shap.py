"""Tests for matching-stage tuning and SHAP plumbing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from src.matching.run import run_features, run_scoring
from src.matching.reranker import save_lightgbm_artifacts, train_lightgbm
from src.matching.shap_explain import format_shap_top5
from src.matching.tuning import BEST_PARAMS_FILENAME, run_optuna_study, split_labeled_feature_matrix
from src.matching.writer import SCORING_METADATA_FILENAME
from src.shared import schemas
from src.synthetic.build_matching_dataset import build_matching_dataset, load_labeled_feature_matrix


_IDENTITY_GROUPS_PAYLOAD = {
    "run_id": "run_tuning_and_shap",
    "groups": [
        {
            "group_id": "per_alice",
            "entity_type": "PER",
            "doc_ids": ["case_doc_1", "case_doc_2"],
            "variants": [
                {"text": "Alice Hansen", "context": "Alice Hansen forklarte seg i avhoret."},
                {"text": "A. Hansen", "context": "A. Hansen ble observert ved adressen."},
                {"text": "Alice H.", "context": "Alice H. ble nevnt i telefonloggen."},
            ],
        },
        {
            "group_id": "per_bob",
            "entity_type": "PER",
            "doc_ids": ["case_doc_3", "case_doc_4"],
            "variants": [
                {"text": "Bjarne Olsen", "context": "Bjarne Olsen ble nevnt i rapporten."},
                {"text": "B. Olsen", "context": "B. Olsen signerte dokumentet."},
                {"text": "Bjarne O.", "context": "Bjarne O. ble sett ved bilen."},
            ],
        },
        {
            "group_id": "org_dnb",
            "entity_type": "ORG",
            "doc_ids": ["case_doc_5", "case_doc_6"],
            "variants": [
                {"text": "DNB ASA", "context": "DNB ASA behandlet betalingen."},
                {"text": "DNB", "context": "DNB ble brukt som bankforbindelse."},
                {"text": "Den Norske Bank", "context": "Den Norske Bank nevnes i bilaget."},
            ],
        },
        {
            "group_id": "org_taxi",
            "entity_type": "ORG",
            "doc_ids": ["case_doc_7", "case_doc_8"],
            "variants": [
                {"text": "Oslo Taxi", "context": "Oslo Taxi mottok bestillingen."},
                {"text": "Taxi Oslo", "context": "Taxi Oslo ble brukt i loggen."},
                {"text": "Oslo Taxi AS", "context": "Oslo Taxi AS står på kvitteringen."},
            ],
        },
    ],
    "hard_negatives": [
        {"group_id_a": "per_alice", "group_id_b": "per_bob"},
        {"group_id_a": "org_dnb", "group_id_b": "org_taxi"},
    ],
}


@pytest.fixture()
def scoring_data_dir(tmp_path: Path) -> tuple[Path, str]:
    """Build a synthetic scoring fixture with persisted LightGBM artifacts."""
    data_dir = tmp_path / "scoring_data"
    identity_groups_path = tmp_path / "identity_groups.json"
    identity_groups_path.write_text(json.dumps(_IDENTITY_GROUPS_PAYLOAD), encoding="utf-8")

    build_matching_dataset(identity_groups_path, data_dir, max_pairs=2500, seed=7)
    run_features(data_dir, _IDENTITY_GROUPS_PAYLOAD["run_id"])

    X, y = load_labeled_feature_matrix(data_dir, _IDENTITY_GROUPS_PAYLOAD["run_id"])
    model = train_lightgbm(X, y)
    save_lightgbm_artifacts(model, data_dir, model_version="lightgbm_baseline_s516")

    return data_dir, _IDENTITY_GROUPS_PAYLOAD["run_id"]


def _load_scoring_metadata(data_dir: Path) -> dict:
    return json.loads((data_dir / SCORING_METADATA_FILENAME).read_text(encoding="utf-8"))


def _assert_shap_contract(row: list[dict]) -> None:
    assert len(row) <= 5
    features = [entry["feature"] for entry in row]
    values = [entry["value"] for entry in row]
    assert len(features) == len(set(features))
    assert all(isinstance(feature, str) and feature for feature in features)
    assert all(np.isfinite(value) for value in values)
    assert [abs(value) for value in values] == sorted((abs(value) for value in values), reverse=True)


def test_format_shap_top5_orders_by_absolute_value_and_limits_to_five() -> None:
    top5 = format_shap_top5(
        ["zeta", "alpha", "beta", "gamma", "delta", "epsilon"],
        [0.1, -0.9, 0.4, -0.4, np.inf, 0.2],
    )

    assert [entry["feature"] for entry in top5] == ["alpha", "beta", "gamma", "epsilon", "zeta"]
    assert all(isinstance(entry["value"], float) for entry in top5)
    _assert_shap_contract(top5)


def test_run_scoring_disabled_hooks_keep_empty_shap_and_write_metadata(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    scored = run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
    )
    metadata = _load_scoring_metadata(data_dir)

    assert all(row == [] for row in scored["shap_top5"].to_list())
    assert metadata["params_used"] == "baseline"
    assert metadata["tuning"]["status"] == "disabled"
    assert metadata["shap"] == {
        "enabled": False,
        "generated": False,
        "explained_row_count": 0,
        "row_count": scored.height,
    }


def test_run_scoring_with_shap_enabled_produces_contract_safe_top5(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    scored = run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
        enable_shap=True,
        shap_max_rows=3,
    )
    scored_table = pq.read_table(data_dir / "scored_pairs.parquet")
    candidate_table = pq.read_table(data_dir / "candidate_pairs.parquet")
    metadata = _load_scoring_metadata(data_dir)

    assert schemas.validate_contract_rules(
        scored_table,
        "scored_pairs",
        candidate_pairs_table=candidate_table,
    ) == []
    explained_rows = scored["shap_top5"].to_list()[:3]
    skipped_rows = scored["shap_top5"].to_list()[3:]
    assert explained_rows
    for row in explained_rows:
        assert row
        _assert_shap_contract(row)
    assert all(row == [] for row in skipped_rows)
    assert metadata["shap"] == {
        "enabled": True,
        "generated": True,
        "explained_row_count": 3,
        "row_count": scored.height,
    }


def test_run_scoring_with_shap_max_rows_zero_writes_empty_explanations(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    scored = run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
        enable_shap=True,
        shap_max_rows=0,
    )
    metadata = _load_scoring_metadata(data_dir)

    assert all(row == [] for row in scored["shap_top5"].to_list())
    assert metadata["shap"] == {
        "enabled": True,
        "generated": False,
        "explained_row_count": 0,
        "row_count": scored.height,
    }


def test_run_scoring_tuning_smoke_records_metadata_without_best_params_artifact(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
        enable_tuning=True,
        tuning_mode="smoke",
        tuning_trials=2,
    )
    metadata = _load_scoring_metadata(data_dir)

    assert metadata["tuning"]["enabled"] is True
    assert metadata["tuning"]["status"] == "completed"
    assert metadata["tuning"]["mode"] == "smoke"
    assert metadata["tuning"]["n_trials_requested"] == 2
    assert metadata["tuning"]["best_params_artifact_written"] is False
    assert not (data_dir / BEST_PARAMS_FILENAME).exists()


def test_non_trivial_optuna_study_writes_best_params_artifact(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    X, y = load_labeled_feature_matrix(data_dir, run_id)

    summary = run_optuna_study(
        X,
        y,
        enabled=True,
        mode="study",
        n_trials=5,
        out_dir=data_dir,
    )

    assert summary["status"] == "completed"
    assert summary["best_params_artifact_written"] is True
    assert summary["best_params_artifact"] == BEST_PARAMS_FILENAME
    assert (data_dir / BEST_PARAMS_FILENAME).exists()


def test_split_labeled_feature_matrix_rejects_tiny_class_counts() -> None:
    X = np.asarray([[0.1, 0.2], [0.9, 0.8]], dtype=np.float64)
    y = np.asarray([0, 1], dtype=np.int8)

    with pytest.raises(
        ValueError,
        match=(
            "Hyperparameter tuning requires at least 2 labeled rows in each class "
            "for the stratified train/validation split\\."
        ),
    ):
        split_labeled_feature_matrix(X, y)


def test_run_scoring_tuning_skips_when_labels_exist_only_for_other_runs(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    (
        pl.read_parquet(data_dir / "labels.parquet")
        .with_columns(pl.lit("other_run").alias("run_id"))
        .write_parquet(data_dir / "labels.parquet")
    )

    run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
        enable_tuning=True,
        tuning_mode="smoke",
        tuning_trials=2,
    )
    metadata = _load_scoring_metadata(data_dir)

    assert metadata["tuning"]["enabled"] is True
    assert metadata["tuning"]["status"] == "skipped_no_labels"
    assert metadata["tuning"]["n_trials_completed"] == 0
