"""Tests for embedding artifact loader and alignment guards."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.blocking.embeddings import persist_embedding_artifacts
from src.matching.features import build_embedding_id_index, load_embedding_artifacts
from src.shared.fixtures import (
    DEFAULT_RUN_ID,
    build_mock_embedding_artifacts,
    build_mock_entities,
    write_mock_handoff,
)
from src.shared.paths import get_blocking_run_output_dir, get_extraction_run_output_dir


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Create mock handoff artifacts for loader tests."""
    write_mock_handoff(tmp_path)
    return tmp_path


def test_load_embedding_artifacts_happy_path(handoff_dir: Path) -> None:
    artifacts = load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)
    assert artifacts.embeddings.shape == artifacts.context_embeddings.shape
    assert artifacts.embeddings.shape[1] == 768
    assert artifacts.embeddings.shape[0] == len(artifacts.embedding_entity_ids)


def test_load_embedding_artifacts_accepts_blocking_writer_output(tmp_path: Path) -> None:
    entities = build_mock_entities()
    embeddings, context_embeddings, embedding_entity_ids = build_mock_embedding_artifacts(
        entities
    )
    extraction_dir = get_extraction_run_output_dir(tmp_path, DEFAULT_RUN_ID)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(entities, extraction_dir / "entities.parquet")

    blocking_dir = get_blocking_run_output_dir(tmp_path, DEFAULT_RUN_ID)
    blocking_dir.mkdir(parents=True, exist_ok=True)
    persist_embedding_artifacts(
        entity_ids=embedding_entity_ids.tolist(),
        embeddings=embeddings,
        context_embeddings=context_embeddings,
        out_dir=blocking_dir,
    )

    artifacts = load_embedding_artifacts(tmp_path, DEFAULT_RUN_ID)

    assert artifacts.embedding_entity_ids.dtype.kind == "U"
    assert artifacts.embedding_entity_ids.tolist() == embedding_entity_ids.tolist()
    assert np.allclose(artifacts.embeddings, embeddings)
    assert np.allclose(artifacts.context_embeddings, context_embeddings)


def test_load_embedding_artifacts_uses_per_run_entity_id_order(handoff_dir: Path) -> None:
    extraction_dir = get_extraction_run_output_dir(handoff_dir, DEFAULT_RUN_ID)
    entities = pq.read_table(extraction_dir / "entities.parquet")
    reversed_entities = entities.take(pa.array([3, 2, 1, 0]))
    pq.write_table(reversed_entities, extraction_dir / "entities.parquet")

    artifacts = load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)

    expected_ids = sorted(reversed_entities.column("entity_id").to_pylist())
    assert artifacts.embedding_entity_ids.tolist() == expected_ids


def test_load_embedding_artifacts_rejects_row_count_mismatch(handoff_dir: Path) -> None:
    blocking_dir = get_blocking_run_output_dir(handoff_dir, DEFAULT_RUN_ID)
    context_embeddings = np.load(blocking_dir / "context_embeddings.npy", allow_pickle=False)
    np.save(blocking_dir / "context_embeddings.npy", context_embeddings[:-1])

    with pytest.raises(ValueError, match="row count"):
        load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)


def test_load_embedding_artifacts_rejects_wrong_dim(handoff_dir: Path) -> None:
    blocking_dir = get_blocking_run_output_dir(handoff_dir, DEFAULT_RUN_ID)
    embeddings = np.load(blocking_dir / "embeddings.npy", allow_pickle=False)
    np.save(blocking_dir / "embeddings.npy", embeddings[:, :-1])

    with pytest.raises(ValueError, match="dim mismatch"):
        load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)


def test_load_embedding_artifacts_rejects_mismatched_id_order(handoff_dir: Path) -> None:
    blocking_dir = get_blocking_run_output_dir(handoff_dir, DEFAULT_RUN_ID)
    embedding_entity_ids = np.load(blocking_dir / "embedding_entity_ids.npy", allow_pickle=False)
    corrupted = embedding_entity_ids.copy()
    corrupted[[0, 1]] = corrupted[[1, 0]]
    np.save(blocking_dir / "embedding_entity_ids.npy", corrupted)

    with pytest.raises(ValueError, match="row order"):
        load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)


def test_load_embedding_artifacts_rejects_non_normalized_rows(handoff_dir: Path) -> None:
    blocking_dir = get_blocking_run_output_dir(handoff_dir, DEFAULT_RUN_ID)
    embeddings = np.load(blocking_dir / "embeddings.npy", allow_pickle=False)
    embeddings[0] = embeddings[0] * 2.0
    np.save(blocking_dir / "embeddings.npy", embeddings)

    with pytest.raises(ValueError, match="L2-normalized"):
        load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)


def test_build_embedding_id_index_is_complete_and_stable(handoff_dir: Path) -> None:
    artifacts = load_embedding_artifacts(handoff_dir, DEFAULT_RUN_ID)
    index = build_embedding_id_index(artifacts.embedding_entity_ids)
    expected_ids = [str(entity_id) for entity_id in artifacts.embedding_entity_ids.tolist()]

    assert len(index) == len(expected_ids)
    assert set(index.keys()) == set(expected_ids)
    for i, entity_id in enumerate(expected_ids):
        assert index[entity_id] == i


def test_load_embedding_artifacts_rejects_unknown_run_id(handoff_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_embedding_artifacts(handoff_dir, "missing_run")


def test_build_embedding_id_index_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique IDs"):
        build_embedding_id_index(np.array(["a", "a"]))
