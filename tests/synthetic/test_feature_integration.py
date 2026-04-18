"""Integration tests for synthetic data validation and feature loading."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.matching.run import FEATURE_COLUMNS, FEATURE_OUTPUT_COLUMNS, run_features
from src.matching.writer import get_features_output_path
from src.shared import schemas
from src.shared.paths import get_blocking_run_output_dir, get_extraction_run_output_dir
from src.synthetic.build_matching_dataset import (
    LABELS_SCHEMA,
    build_matching_dataset,
    load_labeled_feature_matrix,
)
from src.synthetic.validate import validate_synthetic_data

_IDENTITY_GROUPS_PAYLOAD = {
    "run_id": "run_synthetic",
    "groups": [
        {
            "group_id": "per_alice",
            "entity_type": "PER",
            "doc_ids": ["case_doc_1", "case_doc_2"],
            "variants": [
                {
                    "text": "Alice Hansen",
                    "context": "Alice Hansen forklarte seg i avhoret.",
                },
                {
                    "text": "A. Hansen",
                    "context": "A. Hansen ble observert ved adressen.",
                },
            ],
        },
        {
            "group_id": "per_bob",
            "entity_type": "PER",
            "doc_ids": ["case_doc_3", "case_doc_4"],
            "variants": [
                {
                    "text": "Bjarne Olsen",
                    "context": "Bjarne Olsen ble nevnt i rapporten.",
                },
                {
                    "text": "B. Olsen",
                    "context": "B. Olsen signerte dokumentet.",
                },
            ],
        },
        {
            "group_id": "org_dnb",
            "entity_type": "ORG",
            "doc_ids": ["case_doc_5", "case_doc_6"],
            "variants": [
                {
                    "text": "DNB ASA",
                    "context": "DNB ASA behandlet betalingen.",
                },
                {
                    "text": "DNB",
                    "context": "DNB ble brukt som bankforbindelse.",
                },
            ],
        },
    ],
    "hard_negatives": [
        {
            "group_id_a": "per_alice",
            "group_id_b": "per_bob",
        }
    ],
}
UNIT_INTERVAL_COLUMNS = [
    "jaro_winkler_similarity",
    "levenshtein_ratio_similarity",
    "token_jaccard_similarity",
    "token_containment_ratio",
    "char_trigram_jaccard_similarity",
]
COSINE_COLUMNS = [
    "cosine_sim_entity",
    "cosine_sim_context",
]
BINARY_COLUMNS = [
    "abbreviation_match_flag",
    "double_metaphone_overlap_flag",
    "norwegian_id_match",
    "first_name_match",
    "last_name_match",
    "shared_doc_count",
]


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> tuple[Path, str]:
    """Build a synthetic dataset from a local test payload."""
    data_dir = tmp_path / "synthetic_data"
    identity_groups_path = tmp_path / "identity_groups.json"
    identity_groups_path.write_text(
        json.dumps(_IDENTITY_GROUPS_PAYLOAD),
        encoding="utf-8",
    )
    build_matching_dataset(identity_groups_path, data_dir, max_pairs=2500, seed=7)
    run_id = _IDENTITY_GROUPS_PAYLOAD["run_id"]
    return data_dir, run_id


def test_validate_synthetic_data_passes_for_generated_dataset(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, _ = synthetic_data_dir

    assert validate_synthetic_data(data_dir) == []


def test_validate_synthetic_data_detects_orphan_and_unlabeled_pairs(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    labels = pq.read_table(data_dir / "labels.parquet").to_pylist()
    broken_rows = labels[1:]
    broken_rows.append(
        {
            "run_id": run_id,
            "entity_id_a": "0" * 32,
            "entity_id_b": "f" * 32,
            "label": 0,
        }
    )
    pq.write_table(
        pa.Table.from_pylist(broken_rows, schema=LABELS_SCHEMA),
        data_dir / "labels.parquet",
    )

    issues = validate_synthetic_data(data_dir)

    assert any("orphan labels" in issue for issue in issues)
    assert any("candidate pairs without labels" in issue for issue in issues)


def test_validate_synthetic_data_detects_pair_entity_reference_drift(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    blocking_dir = get_blocking_run_output_dir(data_dir, run_id)
    candidates = pq.read_table(blocking_dir / "candidate_pairs.parquet").to_pylist()
    labels = pq.read_table(data_dir / "labels.parquet").to_pylist()

    fake_a = "0" * 32
    fake_b = "1" * 32
    candidates[0]["entity_id_a"] = fake_a
    candidates[0]["entity_id_b"] = fake_b
    labels[0]["entity_id_a"] = fake_a
    labels[0]["entity_id_b"] = fake_b
    pq.write_table(
        pa.Table.from_pylist(candidates, schema=schemas.CANDIDATE_PAIRS_SCHEMA),
        blocking_dir / "candidate_pairs.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(labels, schema=LABELS_SCHEMA),
        data_dir / "labels.parquet",
    )

    issues = validate_synthetic_data(data_dir)

    assert any(
        issue
        == "candidate_pairs.parquet: row 1: entity_id_a must reference entities.parquet"
        for issue in issues
    )
    assert any(
        issue
        == "candidate_pairs.parquet: row 1: entity_id_b must reference entities.parquet"
        for issue in issues
    )
    assert any(
        issue == "labels.parquet: row 1: entity_id_a must reference entities.parquet"
        for issue in issues
    )
    assert any(
        issue == "labels.parquet: row 1: entity_id_b must reference entities.parquet"
        for issue in issues
    )


def test_validate_synthetic_data_reports_nullable_label_rows_without_crashing(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, _ = synthetic_data_dir
    labels = pq.read_table(data_dir / "labels.parquet").to_pylist()
    labels[0]["entity_id_a"] = None

    pq.write_table(
        pa.Table.from_pylist(labels, schema=LABELS_SCHEMA),
        data_dir / "labels.parquet",
    )

    issues = validate_synthetic_data(data_dir)

    assert any(
        issue == "labels.parquet: row 1: entity_id_a must be 32-char lowercase hex"
        for issue in issues
    )


def test_validate_synthetic_data_reports_missing_label_columns_without_crashing(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    pq.write_table(
        pa.table(
            {
                "run_id": pa.array([run_id]),
                "entity_id_a": pa.array(["0" * 32]),
            }
        ),
        data_dir / "labels.parquet",
    )

    issues = validate_synthetic_data(data_dir)

    assert "labels.parquet: missing column: entity_id_b" in issues
    assert "labels.parquet: missing column: label" in issues


def test_run_features_on_synthetic_data_produces_complete_feature_matrix(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir

    features = run_features(data_dir, run_id)

    assert features.columns == FEATURE_OUTPUT_COLUMNS
    assert len(FEATURE_COLUMNS) == 14
    assert features.height > 0
    for column in FEATURE_COLUMNS:
        assert features[column].null_count() == 0
    for column in UNIT_INTERVAL_COLUMNS:
        assert features[column].is_between(0.0, 1.0, closed="both").all()
    for column in COSINE_COLUMNS:
        assert features[column].is_between(-1.0, 1.0, closed="both").all()
    for column in BINARY_COLUMNS:
        assert set(features[column].unique().to_list()).issubset({0, 1})
    assert features["blocking_method_count"].min() >= 1
    assert features["blocking_method_count"].max() <= 3


def test_run_features_logs_diagnostics_for_all_columns_on_synthetic_data(
    synthetic_data_dir: tuple[Path, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_dir, run_id = synthetic_data_dir

    caplog.set_level(logging.INFO, logger="src.matching.run")
    run_features(data_dir, run_id)

    pattern = re.compile(r"feature_diagnostics feature=([a-z0-9_]+)\b")
    logged_columns = {
        match.group(1)
        for record in caplog.records
        if (match := pattern.search(record.getMessage()))
    }

    assert logged_columns == set(FEATURE_COLUMNS)


def test_load_labeled_feature_matrix_has_zero_row_loss(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    labels = pl.read_parquet(data_dir / "labels.parquet").filter(
        pl.col("run_id") == run_id
    )
    run_features(data_dir, run_id)

    x_matrix, y_vector = load_labeled_feature_matrix(data_dir, run_id)
    features = pl.read_parquet(get_features_output_path(data_dir, run_id))
    joined = features.join(
        labels, on=["run_id", "entity_id_a", "entity_id_b"], how="inner"
    )

    assert x_matrix.columns == FEATURE_COLUMNS
    assert x_matrix.height == y_vector.len() == labels.height
    assert joined.height == features.height == labels.height


def test_load_labeled_feature_matrix_allows_unlabeled_feature_rows(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    run_features(data_dir, run_id)

    labels = pl.read_parquet(data_dir / "labels.parquet")
    labels.slice(1).write_parquet(data_dir / "labels.parquet")

    x_matrix, y_vector = load_labeled_feature_matrix(data_dir, run_id)

    assert x_matrix.columns == FEATURE_COLUMNS
    assert x_matrix.height == y_vector.len() == labels.height - 1


def test_load_labeled_feature_matrix_rejects_duplicate_feature_keys(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    features = run_features(data_dir, run_id)
    corrupted = pl.concat(
        [features.slice(0, 1), features.slice(0, 1), features.slice(2)]
    )
    corrupted.write_parquet(get_features_output_path(data_dir, run_id))

    with pytest.raises(
        ValueError, match="features.parquet contains duplicate pair keys"
    ):
        load_labeled_feature_matrix(data_dir, run_id)


def test_load_labeled_feature_matrix_requires_per_run_features_output(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    features_path = get_features_output_path(data_dir, run_id)

    run_features(data_dir, run_id)
    features_path.unlink()

    with pytest.raises(
        ValueError,
        match="missing matching features for run_id=.*rerun matching features for this run",
    ):
        load_labeled_feature_matrix(data_dir, run_id)


def test_load_labeled_feature_matrix_rejects_missing_feature_keys_even_when_row_count_matches(
    synthetic_data_dir: tuple[Path, str],
) -> None:
    data_dir, run_id = synthetic_data_dir
    features = run_features(data_dir, run_id)
    corrupted = (
        features.with_row_index("row_idx")
        .with_columns(
            pl.when(pl.col("row_idx") == 0)
            .then(pl.lit("0" * 32))
            .otherwise(pl.col("entity_id_a"))
            .alias("entity_id_a"),
            pl.when(pl.col("row_idx") == 0)
            .then(pl.lit("1" * 32))
            .otherwise(pl.col("entity_id_b"))
            .alias("entity_id_b"),
        )
        .drop("row_idx")
    )
    corrupted.write_parquet(get_features_output_path(data_dir, run_id))

    with pytest.raises(
        ValueError, match="features.parquet is missing keys from labels.parquet"
    ):
        load_labeled_feature_matrix(data_dir, run_id)
