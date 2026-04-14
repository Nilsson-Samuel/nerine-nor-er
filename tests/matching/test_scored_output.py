"""Integration tests for inference scoring and scored pair output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from src.matching.run import run_features, run_scoring
from src.matching.reranker import save_lightgbm_artifacts, train_lightgbm
from src.matching.writer import get_features_output_path, get_scored_pairs_output_path
from src.shared import schemas
from src.synthetic.build_matching_dataset import build_matching_dataset, load_labeled_feature_matrix


_IDENTITY_GROUPS_PAYLOAD = {
    "run_id": "run_scored_pairs",
    "groups": [
        {
            "group_id": "per_alice",
            "entity_type": "PER",
            "doc_ids": ["case_doc_1", "case_doc_2"],
            "variants": [
                {"text": "Alice Hansen", "context": "Alice Hansen forklarte seg i avhoret."},
                {"text": "A. Hansen", "context": "A. Hansen ble observert ved adressen."},
            ],
        },
        {
            "group_id": "per_bob",
            "entity_type": "PER",
            "doc_ids": ["case_doc_3", "case_doc_4"],
            "variants": [
                {"text": "Bjarne Olsen", "context": "Bjarne Olsen ble nevnt i rapporten."},
                {"text": "B. Olsen", "context": "B. Olsen signerte dokumentet."},
            ],
        },
        {
            "group_id": "org_dnb",
            "entity_type": "ORG",
            "doc_ids": ["case_doc_5", "case_doc_6"],
            "variants": [
                {"text": "DNB ASA", "context": "DNB ASA behandlet betalingen."},
                {"text": "DNB", "context": "DNB ble brukt som bankforbindelse."},
            ],
        },
        {
            "group_id": "org_taxi",
            "entity_type": "ORG",
            "doc_ids": ["case_doc_7", "case_doc_8"],
            "variants": [
                {"text": "Oslo Taxi", "context": "Oslo Taxi mottok bestillingen."},
                {"text": "Taxi Oslo", "context": "Taxi Oslo ble brukt i loggen."},
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
    """Build a synthetic dataset with trained model artifacts for scoring."""
    data_dir = tmp_path / "scoring_data"
    identity_groups_path = tmp_path / "identity_groups.json"
    identity_groups_path.write_text(json.dumps(_IDENTITY_GROUPS_PAYLOAD), encoding="utf-8")

    build_matching_dataset(identity_groups_path, data_dir, max_pairs=2500, seed=7)
    run_features(data_dir, _IDENTITY_GROUPS_PAYLOAD["run_id"])

    X, y = load_labeled_feature_matrix(data_dir, _IDENTITY_GROUPS_PAYLOAD["run_id"])
    model = train_lightgbm(X, y)
    save_lightgbm_artifacts(model, data_dir, model_version="lightgbm_baseline_s514")

    return data_dir, _IDENTITY_GROUPS_PAYLOAD["run_id"]


def test_run_scoring_writes_scored_pairs_parquet(scoring_data_dir: tuple[Path, str]) -> None:
    data_dir, run_id = scoring_data_dir

    run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
    )

    assert get_scored_pairs_output_path(data_dir, run_id).exists()
    assert not (data_dir / "scored_pairs.parquet").exists()


def test_scored_pairs_output_matches_contract(scoring_data_dir: tuple[Path, str]) -> None:
    data_dir, run_id = scoring_data_dir
    scored_at = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)

    scored = run_scoring(data_dir, run_id, scored_at=scored_at)
    scored_table = pq.read_table(get_scored_pairs_output_path(data_dir, run_id))
    candidate_table = pq.read_table(data_dir / "candidate_pairs.parquet")

    assert scored.columns == [
        "run_id",
        "entity_id_a",
        "entity_id_b",
        "score",
        "model_version",
        "scored_at",
        "blocking_methods",
        "blocking_source",
        "blocking_method_count",
        "shap_top5",
    ]
    assert schemas.validate(scored_table, schemas.SCORED_PAIRS_SCHEMA) == []
    assert (
        schemas.validate_contract_rules(
            scored_table,
            "scored_pairs",
            candidate_pairs_table=candidate_table,
        )
        == []
    )

    candidates = pl.read_parquet(data_dir / "candidate_pairs.parquet").filter(pl.col("run_id") == run_id)
    expected_keys = candidates.sort(["entity_id_a", "entity_id_b"]).select(
        ["run_id", "entity_id_a", "entity_id_b"]
    )
    assert scored.select(["run_id", "entity_id_a", "entity_id_b"]).to_dict(
        as_series=False
    ) == expected_keys.to_dict(as_series=False)
    assert scored.height == candidates.height
    assert scored["score"].is_between(0.0, 1.0, closed="both").all()
    assert set(scored["model_version"].unique().to_list()) == {"lightgbm_baseline_s514"}
    assert set(scored["scored_at"].to_list()) == {scored_at}
    assert scored["blocking_method_count"].min() >= 1
    assert scored["blocking_method_count"].max() <= 3
    assert scored_table.column("shap_top5").to_pylist() == [[] for _ in range(scored.height)]


def test_run_scoring_can_load_model_from_explicit_model_dir(
    scoring_data_dir: tuple[Path, str],
    tmp_path: Path,
) -> None:
    data_dir, run_id = scoring_data_dir
    model_dir = tmp_path / "model_store"
    model_dir.mkdir()
    (data_dir / "reranker_model.txt").replace(model_dir / "reranker_model.txt")
    (data_dir / "reranker_model_metadata.json").replace(
        model_dir / "reranker_model_metadata.json"
    )

    scored = run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
        model_dir=model_dir,
    )

    assert scored.height > 0
    assert scored["score"].is_between(0.0, 1.0, closed="both").all()


def test_scored_pairs_contract_requires_all_candidate_pairs_to_be_scored(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    run_scoring(
        data_dir,
        run_id,
        scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
    )

    scored_table = pq.read_table(get_scored_pairs_output_path(data_dir, run_id))
    candidate_table = pq.read_table(data_dir / "candidate_pairs.parquet")

    truncated_scored_table = scored_table.slice(0, scored_table.num_rows - 1)
    errors = schemas.validate_contract_rules(
        truncated_scored_table,
        "scored_pairs",
        candidate_pairs_table=candidate_table,
    )

    assert any("candidate_pairs row is missing from scored_pairs" in error for error in errors)


def test_run_scoring_requires_per_run_features_output(
    scoring_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = scoring_data_dir
    features_path = get_features_output_path(data_dir, run_id)
    features_path.unlink()

    with pytest.raises(
        ValueError,
        match="missing matching features for run_id=.*rerun matching features for this run",
    ):
        run_scoring(
            data_dir,
            run_id,
            scored_at=datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
        )
