"""FAISS HNSW index — build, query, and extract candidate neighbor pairs.

Builds an IndexHNSWFlat over L2-normalized entity embeddings and retrieves
top-k nearest neighbors per entity. Self-hits are filtered out. Results are
returned as (entity_id_a, entity_id_b) tuples for downstream union logic.
"""

import logging
import time

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# HNSW hyperparameters
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64
DEFAULT_K = 100


def build_hnsw_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """Build a FAISS HNSW index from L2-normalized embeddings.

    Args:
        embeddings: Float32 array of shape (N, dim), already L2-normalized.

    Returns:
        Populated FAISS HNSW index ready for search.
    """
    dim = embeddings.shape[1]
    t0 = time.monotonic()

    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(embeddings)

    elapsed = time.monotonic() - t0
    logger.info(
        "FAISS HNSW built: %d vectors, dim=%d, M=%d, efConstruction=%d (%.1fs)",
        embeddings.shape[0], dim, HNSW_M, EF_CONSTRUCTION, elapsed,
    )
    return index


def query_neighbors(
    index: faiss.IndexHNSWFlat,
    embeddings: np.ndarray,
    entity_ids: list[str],
    k: int = DEFAULT_K,
) -> list[tuple[str, str]]:
    """Query the HNSW index for top-k neighbors and return candidate pairs.

    Each entity is queried against the index. Self-hits are removed. Pairs are
    returned with canonical ordering (a < b) but NOT yet deduplicated — that
    happens in candidates.py where all blocking methods are unioned.

    Args:
        index: Built FAISS HNSW index.
        embeddings: Same embeddings used to build the index (for querying).
        entity_ids: Entity ID strings aligned with embedding rows.
        k: Number of neighbors to retrieve per query.

    Returns:
        List of (entity_id_a, entity_id_b) candidate pairs (may contain dupes).
    """
    index.hnsw.efSearch = EF_SEARCH
    t0 = time.monotonic()

    # k+1 because top-1 result is often the query vector itself
    _, indices = index.search(embeddings, k + 1)

    elapsed = time.monotonic() - t0
    n = len(entity_ids)

    pairs: list[tuple[str, str]] = []
    for i in range(n):
        query_id = entity_ids[i]
        for j in indices[i]:
            if j < 0 or j == i:
                continue
            neighbor_id = entity_ids[j]
            pairs.append((query_id, neighbor_id))

    logger.info(
        "FAISS query: %d entities, k=%d, efSearch=%d, %d raw pairs (%.1fs)",
        n, k, EF_SEARCH, len(pairs), elapsed,
    )
    return pairs
