"""Tests for embedding fixture artifacts and alignment validation."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.shared.fixtures import DEFAULT_RUN_ID, write_mock_handoff
from src.shared.paths import get_blocking_run_output_dir, get_extraction_run_output_dir
from src.shared.validators import validate_embedding_alignment


def test_embedding_alignment_happy_path(tmp_path: Path) -> None:
    write_mock_handoff(tmp_path)
    extraction_dir = get_extraction_run_output_dir(tmp_path, DEFAULT_RUN_ID)
    blocking_dir = get_blocking_run_output_dir(tmp_path, DEFAULT_RUN_ID)
    entities = pq.read_table(extraction_dir / "entities.parquet")

    validate_embedding_alignment(
        embeddings=np.load(blocking_dir / "embeddings.npy"),
        context_embeddings=np.load(blocking_dir / "context_embeddings.npy"),
        embedding_entity_ids=np.load(blocking_dir / "embedding_entity_ids.npy"),
        entity_ids=entities.column("entity_id").to_pylist(),
    )


def test_embedding_alignment_rejects_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="row count"):
        validate_embedding_alignment(
            embeddings=np.zeros((4, 768), dtype="float32"),
            context_embeddings=np.zeros((3, 768), dtype="float32"),
            embedding_entity_ids=np.array(["a"] * 4),
            entity_ids=["a"] * 4,
        )


def test_embedding_alignment_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="dim mismatch"):
        validate_embedding_alignment(
            embeddings=np.zeros((4, 767), dtype="float32"),
            context_embeddings=np.zeros((4, 768), dtype="float32"),
            embedding_entity_ids=np.array(["a"] * 4),
            entity_ids=["a"] * 4,
        )


def test_embedding_alignment_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="row order"):
        validate_embedding_alignment(
            embeddings=np.zeros((4, 768), dtype="float32"),
            context_embeddings=np.zeros((4, 768), dtype="float32"),
            embedding_entity_ids=np.array(["a", "b", "c", "d"]),
            entity_ids=["a", "b", "x", "d"],
        )
