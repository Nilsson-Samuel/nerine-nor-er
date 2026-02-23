"""Validation helpers shared across pipeline stages."""

import numpy as np


def validate_embedding_alignment(
    embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    embedding_entity_ids: np.ndarray,
    entity_ids: list[str] | np.ndarray,
    expected_dim: int = 768,
) -> None:
    """Validate that embedding artifacts align with each other and with entities.

    Args:
        embeddings: Entity-name embedding matrix.
        context_embeddings: Context embedding matrix.
        embedding_entity_ids: Entity IDs aligned to embedding rows.
        entity_ids: Expected entity IDs from entities.parquet row order.
        expected_dim: Required embedding width.

    Raises:
        ValueError: If row counts, dimensions, or ID ordering do not align.
    """
    if embeddings.ndim != 2 or context_embeddings.ndim != 2:
        raise ValueError("embeddings and context_embeddings must be 2D matrices")
    if embeddings.shape[0] != context_embeddings.shape[0] or embeddings.shape[0] != len(
        embedding_entity_ids
    ):
        raise ValueError("embedding row count mismatch across matrices and ID array")
    if embeddings.shape[1] != expected_dim or context_embeddings.shape[1] != expected_dim:
        raise ValueError(f"embedding dim mismatch; expected {expected_dim}")
    if not np.array_equal(np.asarray(embedding_entity_ids), np.asarray(entity_ids)):
        raise ValueError("embedding_entity_ids must match entity row order exactly")
