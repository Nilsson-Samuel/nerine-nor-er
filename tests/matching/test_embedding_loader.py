"""Tests for embedding artifact loader and alignment guards."""

from pathlib import Path

import numpy as np
import pytest

from src.matching.features import build_embedding_id_index, load_embedding_artifacts
from src.shared.fixtures import write_mock_handoff


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Create mock handoff artifacts for loader tests."""
    write_mock_handoff(tmp_path)
    return tmp_path


def test_load_embedding_artifacts_happy_path(handoff_dir: Path) -> None:
    artifacts = load_embedding_artifacts(handoff_dir)
    assert artifacts.embeddings.shape == artifacts.context_embeddings.shape
    assert artifacts.embeddings.shape[1] == 768
    assert artifacts.embeddings.shape[0] == len(artifacts.embedding_entity_ids)


def test_load_embedding_artifacts_rejects_row_count_mismatch(handoff_dir: Path) -> None:
    context_embeddings = np.load(handoff_dir / "context_embeddings.npy", allow_pickle=False)
    np.save(handoff_dir / "context_embeddings.npy", context_embeddings[:-1])

    with pytest.raises(ValueError, match="row count"):
        load_embedding_artifacts(handoff_dir)


def test_load_embedding_artifacts_rejects_wrong_dim(handoff_dir: Path) -> None:
    embeddings = np.load(handoff_dir / "embeddings.npy", allow_pickle=False)
    np.save(handoff_dir / "embeddings.npy", embeddings[:, :-1])

    with pytest.raises(ValueError, match="dim mismatch"):
        load_embedding_artifacts(handoff_dir)


def test_load_embedding_artifacts_rejects_mismatched_id_order(handoff_dir: Path) -> None:
    embedding_entity_ids = np.load(handoff_dir / "embedding_entity_ids.npy", allow_pickle=False)
    corrupted = embedding_entity_ids.copy()
    corrupted[[0, 1]] = corrupted[[1, 0]]
    np.save(handoff_dir / "embedding_entity_ids.npy", corrupted)

    with pytest.raises(ValueError, match="row order"):
        load_embedding_artifacts(handoff_dir)


def test_load_embedding_artifacts_rejects_non_normalized_rows(handoff_dir: Path) -> None:
    embeddings = np.load(handoff_dir / "embeddings.npy", allow_pickle=False)
    embeddings[0] = embeddings[0] * 2.0
    np.save(handoff_dir / "embeddings.npy", embeddings)

    with pytest.raises(ValueError, match="L2-normalized"):
        load_embedding_artifacts(handoff_dir)


def test_build_embedding_id_index_is_complete_and_stable(handoff_dir: Path) -> None:
    artifacts = load_embedding_artifacts(handoff_dir)
    index = build_embedding_id_index(artifacts.embedding_entity_ids)
    expected_ids = [str(entity_id) for entity_id in artifacts.embedding_entity_ids.tolist()]

    assert len(index) == len(expected_ids)
    assert set(index.keys()) == set(expected_ids)
    for i, entity_id in enumerate(expected_ids):
        assert index[entity_id] == i


def test_build_embedding_id_index_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique IDs"):
        build_embedding_id_index(np.array(["a", "a"]))
