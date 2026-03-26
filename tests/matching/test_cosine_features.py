"""Unit tests for cosine embedding feature helpers and builder."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.matching.features import (
    EMBEDDING_FEATURE_COLUMNS,
    EmbeddingArtifacts,
    build_embedding_features,
    cosine_sim_from_lookup,
    load_embedding_artifacts,
    load_pairs_with_names,
)
from src.shared.fixtures import DEFAULT_RUN_ID, write_mock_handoff


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Write mock handoff fixture files and return the directory path."""
    write_mock_handoff(tmp_path)
    return tmp_path


def test_cosine_sim_from_lookup_identical_vectors_is_one() -> None:
    matrix = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    index = {"a": 0, "b": 1}
    assert cosine_sim_from_lookup("a", "b", matrix, index) == 1.0


def test_cosine_sim_from_lookup_orthogonal_vectors_is_near_zero() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = {"a": 0, "b": 1}
    assert abs(cosine_sim_from_lookup("a", "b", matrix, index)) < 1e-12


def test_cosine_sim_from_lookup_negative_similarity_is_preserved() -> None:
    matrix = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    index = {"a": 0, "b": 1}
    assert cosine_sim_from_lookup("a", "b", matrix, index) == -1.0


def test_build_embedding_features_controlled_values() -> None:
    pairs = pl.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "entity_id_a": ["e1", "e1"],
            "entity_id_b": ["e2", "e3"],
            "name_a": ["x", "x"],
            "name_b": ["y", "z"],
        }
    )
    artifacts = EmbeddingArtifacts(
        embeddings=np.array(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
            dtype=np.float32,
        ),
        context_embeddings=np.array(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
            dtype=np.float32,
        ),
        embedding_entity_ids=np.array(["e1", "e2", "e3"]),
    )

    result = build_embedding_features(pairs, artifacts)

    assert result.columns == EMBEDDING_FEATURE_COLUMNS
    assert result["cosine_sim_entity"].to_list() == [1.0, -1.0]
    assert result["cosine_sim_context"].to_list() == [1.0, -1.0]


def test_build_embedding_features_fixture_output_is_bounded_and_non_null(
    handoff_dir: Path,
) -> None:
    pairs = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    artifacts = load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)
    result = build_embedding_features(pairs, artifacts)

    assert len(result) == len(pairs)
    assert result.columns == EMBEDDING_FEATURE_COLUMNS
    for col in EMBEDDING_FEATURE_COLUMNS:
        assert result[col].null_count() == 0
        assert result[col].min() >= -1.0
        assert result[col].max() <= 1.0


def test_build_embedding_features_empty_input_schema() -> None:
    pairs = pl.DataFrame(
        schema={
            "run_id": pl.Utf8,
            "entity_id_a": pl.Utf8,
            "entity_id_b": pl.Utf8,
            "name_a": pl.Utf8,
            "name_b": pl.Utf8,
        }
    )
    artifacts = EmbeddingArtifacts(
        embeddings=np.empty((0, 768), dtype=np.float32),
        context_embeddings=np.empty((0, 768), dtype=np.float32),
        embedding_entity_ids=np.array([], dtype=str),
    )
    result = build_embedding_features(pairs, artifacts)
    assert len(result) == 0
    assert result.columns == EMBEDDING_FEATURE_COLUMNS


def test_build_embedding_features_rejects_missing_entity_id() -> None:
    pairs = pl.DataFrame(
        {
            "run_id": ["run-1"],
            "entity_id_a": ["e1"],
            "entity_id_b": ["missing"],
            "name_a": ["x"],
            "name_b": ["y"],
        }
    )
    artifacts = EmbeddingArtifacts(
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        context_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        embedding_entity_ids=np.array(["e1", "e2"]),
    )

    with pytest.raises(ValueError, match="missing from embedding_entity_ids"):
        build_embedding_features(pairs, artifacts)
