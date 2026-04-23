"""Unit tests for co-occurrence and blocking metadata features."""

from pathlib import Path

import polars as pl
import pytest

from src.matching.features import (
    COOCCURRENCE_META_FEATURE_COLUMNS,
    EMBEDDING_FEATURE_COLUMNS,
    PAIR_METADATA_COLUMNS,
    STRING_FEATURE_COLUMNS,
    STRUCTURED_IDENTITY_FEATURE_COLUMNS,
    build_cooccurrence_meta_features,
    build_embedding_features,
    build_string_features,
    build_structured_identity_features,
    load_embedding_artifacts,
    load_pairs_with_metadata,
    shared_doc_count,
)
from src.shared.fixtures import DEFAULT_RUN_ID, write_mock_handoff


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Write mock handoff fixture files and return the directory."""
    write_mock_handoff(tmp_path)
    return tmp_path


def test_shared_doc_count_overlap() -> None:
    assert shared_doc_count("doc-a", "doc-a") == 1


def test_shared_doc_count_no_overlap() -> None:
    assert shared_doc_count("doc-a", "doc-b") == 0


def test_metadata_loader_returns_required_columns(handoff_dir: Path) -> None:
    pairs = load_pairs_with_metadata(handoff_dir, DEFAULT_RUN_ID)
    assert pairs.columns == PAIR_METADATA_COLUMNS


def test_build_cooccurrence_meta_features_counts_and_passthrough() -> None:
    pairs = pl.DataFrame(
        {
            "doc_id_a": ["doc-a", "doc-a", "doc-z"],
            "doc_id_b": ["doc-a", "doc-c", "doc-z"],
            "blocking_method_count": [1, 2, 3],
        }
    )

    result = build_cooccurrence_meta_features(pairs)

    assert result.columns == COOCCURRENCE_META_FEATURE_COLUMNS
    assert result["shared_doc_count"].to_list() == [1, 0, 1]
    assert result["blocking_method_count"].to_list() == pairs["blocking_method_count"].to_list()


def test_build_cooccurrence_meta_features_handles_null_and_empty_doc_ids() -> None:
    pairs = pl.DataFrame(
        {
            "doc_id_a": ["doc-a", None, "", "doc-d"],
            "doc_id_b": ["doc-a", "doc-b", "", None],
            "blocking_method_count": [1, 2, 3, 4],
        }
    )

    result = build_cooccurrence_meta_features(pairs)

    assert result["shared_doc_count"].to_list() == [1, 0, 0, 0]
    assert result["blocking_method_count"].to_list() == [1, 2, 3, 4]


def test_full_feature_contract_contains_15_feature_columns(handoff_dir: Path) -> None:
    pairs = load_pairs_with_metadata(handoff_dir, DEFAULT_RUN_ID)
    artifacts = load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)

    string_df = build_string_features(pairs)
    embedding_df = build_embedding_features(pairs, artifacts)
    structured_df = build_structured_identity_features(pairs)
    cooccurrence_df = build_cooccurrence_meta_features(pairs)

    all_feature_columns = [
        *STRING_FEATURE_COLUMNS,
        *EMBEDDING_FEATURE_COLUMNS,
        *STRUCTURED_IDENTITY_FEATURE_COLUMNS,
        *COOCCURRENCE_META_FEATURE_COLUMNS,
    ]
    assert len(all_feature_columns) == 15

    merged = pl.concat(
        [string_df, embedding_df, structured_df, cooccurrence_df],
        how="horizontal",
    )

    assert merged.columns == ["run_id", "entity_id_a", "entity_id_b", *all_feature_columns]
    assert merged.height == pairs.height
    for col in all_feature_columns:
        assert merged[col].null_count() == 0

    assert (
        cooccurrence_df["blocking_method_count"].to_list()
        == pairs["blocking_method_count"].to_list()
    )
