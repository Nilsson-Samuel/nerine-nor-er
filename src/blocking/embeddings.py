"""SBERT embedding generation for blocking — encode entity names and contexts.

Loads entities.parquet for a given run, encodes the `normalized` and `context`
fields with NbAiLab/nb-sbert-base (768-dim), L2-normalizes both matrices, and
persists aligned .npy artifacts for downstream FAISS and matching stages.
"""

import logging
import time
from pathlib import Path

import duckdb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.shared.paths import get_extraction_run_output_dir

logger = logging.getLogger(__name__)

SBERT_MODEL = "NbAiLab/nb-sbert-base"
EMBEDDING_DIM = 768
BATCH_SIZE = 128


def load_entities_for_embedding(
    data_dir: Path,
    run_id: str,
    con: duckdb.DuckDBPyConnection,
) -> tuple[list[str], list[str], list[str]]:
    """Load entity IDs, normalized names, and contexts in deterministic order.

    Args:
        data_dir: Directory containing entities.parquet.
        run_id: Run identifier.
        con: DuckDB connection.

    Returns:
        Tuple of (entity_ids, normalized_names, contexts) lists, all same length
        and ordered by entity_id for deterministic alignment.
    """
    entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"
    con.execute(
        f"CREATE OR REPLACE TABLE entities AS SELECT * FROM '{entities_path}'"
    )
    rows = con.execute(
        "SELECT entity_id, normalized, context "
        "FROM entities WHERE run_id = ? "
        "ORDER BY entity_id",
        [run_id],
    ).fetchall()

    entity_ids = [r[0] for r in rows]
    names = [r[1] for r in rows]
    contexts = [r[2] for r in rows]
    return entity_ids, names, contexts


def encode_and_persist(
    entity_ids: list[str],
    names: list[str],
    contexts: list[str],
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode texts with SBERT, L2-normalize, and save aligned .npy artifacts.

    Args:
        entity_ids: Entity ID strings (alignment key).
        names: Normalized entity name strings.
        contexts: Context window strings.
        out_dir: Directory to write .npy files into.

    Returns:
        Tuple of (embeddings, context_embeddings) as float32 arrays,
        both L2-normalized and shape (N, 768).
    """
    n = len(entity_ids)
    logger.info("Encoding %d entities with %s", n, SBERT_MODEL)

    model = SentenceTransformer(SBERT_MODEL)

    t0 = time.monotonic()
    emb = model.encode(names, batch_size=BATCH_SIZE, show_progress_bar=True)
    emb = emb.astype("float32")
    t_name = time.monotonic() - t0

    t0 = time.monotonic()
    ctx_emb = model.encode(contexts, batch_size=BATCH_SIZE, show_progress_bar=True)
    ctx_emb = ctx_emb.astype("float32")
    t_ctx = time.monotonic() - t0

    # L2 normalize for cosine-compatible FAISS (inner product = cosine after norm)
    faiss.normalize_L2(emb)
    faiss.normalize_L2(ctx_emb)

    # Alignment check before persisting
    if emb.shape != (n, EMBEDDING_DIM):
        raise ValueError(f"emb shape {emb.shape} != ({n}, {EMBEDDING_DIM})")
    if ctx_emb.shape != (n, EMBEDDING_DIM):
        raise ValueError(f"ctx_emb shape {ctx_emb.shape} != ({n}, {EMBEDDING_DIM})")

    persist_embedding_artifacts(
        entity_ids=entity_ids,
        embeddings=emb,
        context_embeddings=ctx_emb,
        out_dir=out_dir,
    )

    logger.info(
        "Embeddings persisted: shape=%s, name_encode=%.1fs, ctx_encode=%.1fs",
        emb.shape, t_name, t_ctx,
    )
    return emb, ctx_emb


def persist_embedding_artifacts(
    entity_ids: list[str] | np.ndarray,
    embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    out_dir: Path,
) -> None:
    """Persist blocking embedding artifacts in the matching-safe on-disk format.

    Args:
        entity_ids: Entity IDs aligned to embedding rows.
        embeddings: L2-normalized entity-name embeddings.
        context_embeddings: L2-normalized context embeddings.
        out_dir: Directory to write `.npy` artifacts into.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "context_embeddings.npy", context_embeddings)
    np.save(out_dir / "embedding_entity_ids.npy", np.asarray(entity_ids, dtype=np.str_))
