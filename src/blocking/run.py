"""Blocking stage orchestrator — exact, structured, FAISS, phonetic, MinHash, union, write.

Reads entities.parquet, generates SBERT embeddings, runs five blocking
strategies (exact name, structured ID, FAISS HNSW, Double Metaphone,
MinHash LSH), unions candidate pairs with source tracking, writes
candidate_pairs.parquet, and generates handoff_manifest.json for the
Developer A → B boundary.
"""

import logging
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from src.blocking.candidates import union_candidates
from src.blocking.embeddings import (
    EMBEDDING_DIM,
    encode_and_persist,
    load_entities_for_embedding,
)
from src.blocking.exact import build_exact_name_pairs, build_structured_id_pairs
from src.blocking.faiss_index import DEFAULT_K, build_hnsw_index, query_neighbors
from src.blocking.minhash import build_minhash_index, query_minhash_pairs
from src.blocking.phonetic import build_phonetic_index, query_phonetic_pairs
from src.blocking.writer import write_candidate_pairs, write_handoff_manifest
from src.shared.paths import get_blocking_run_output_dir
from src.shared.schemas import validate_contract_rules

logger = logging.getLogger(__name__)


def run_blocking(
    data_dir: Path,
    run_id: str,
    con: duckdb.DuckDBPyConnection | None = None,
    k: int = DEFAULT_K,
) -> str:
    """Run the full blocking pipeline: exact → embed → block → union → write → validate.

    Args:
        data_dir: Directory containing entities.parquet (and output target).
        run_id: Run identifier.
        con: Optional DuckDB connection.
        k: FAISS top-k neighbors per entity.

    Returns:
        The run_id used for this execution.
    """
    t0 = time.monotonic()

    if con is None:
        con = duckdb.connect()

    data_dir = Path(data_dir)

    # ── Step 1: Load entities ──────────────────────────────────────────
    entity_ids, names, contexts = load_entities_for_embedding(
        data_dir, run_id, con,
    )
    if not entity_ids:
        logger.warning("No entities found for run_id=%s — skipping blocking.", run_id)
        return run_id

    logger.info(
        "Blocking stage: %d entities loaded (run_id=%s)", len(entity_ids), run_id,
    )

    # Build entity type lookup for same-type filtering
    entity_types = _load_entity_types(data_dir, run_id, con)
    types_list = [entity_types[eid] for eid in entity_ids]

    # ── Step 2: Exact blocking (cheap, high-value) ─────────────────────
    exact_pairs = build_exact_name_pairs(entity_ids, names, types_list)
    structured_pairs = build_structured_id_pairs(entity_ids, names, types_list)

    # ── Step 3: Embed ──────────────────────────────────────────────────
    blocking_out_dir = get_blocking_run_output_dir(data_dir, run_id)
    embeddings, _ctx_emb = encode_and_persist(
        entity_ids, names, contexts, blocking_out_dir,
    )

    # ── Step 4: FAISS blocking ─────────────────────────────────────────
    index = build_hnsw_index(embeddings)
    faiss_pairs = query_neighbors(index, embeddings, entity_ids, k=k)

    # ── Step 5: Phonetic blocking ──────────────────────────────────────
    phonetic_index = build_phonetic_index(entity_ids, names, types_list)
    phonetic_pairs = query_phonetic_pairs(phonetic_index)

    # ── Step 6: MinHash blocking ───────────────────────────────────────
    lsh, signatures = build_minhash_index(entity_ids, names)
    minhash_pairs = query_minhash_pairs(lsh, signatures, entity_ids)

    # ── Step 7: Union candidates ───────────────────────────────────────
    candidates = union_candidates(
        faiss_pairs, phonetic_pairs, minhash_pairs,
        exact_pairs, structured_pairs,
        entity_types,
    )

    if not candidates:
        logger.warning("No candidate pairs after union — check entity count / types.")
        return run_id

    # ── Step 8: Write candidate_pairs.parquet ──────────────────────────
    candidate_count = write_candidate_pairs(candidates, run_id, data_dir, con)

    # ── Step 9: Write handoff manifest ─────────────────────────────────
    entity_types_present = sorted(set(entity_types.values()))
    write_handoff_manifest(
        run_id=run_id,
        entity_count=len(entity_ids),
        candidate_count=candidate_count,
        entity_types_present=entity_types_present,
        data_dir=data_dir,
        embedding_dim=EMBEDDING_DIM,
        k=k,
    )

    # ── Step 10: Final validation ──────────────────────────────────────
    _validate_outputs(data_dir, run_id)

    elapsed = time.monotonic() - t0
    logger.info(
        "Blocking complete: %d entities → %d candidate pairs in %.1fs (run_id=%s)",
        len(entity_ids), candidate_count, elapsed, run_id,
    )
    return run_id


def _load_entity_types(
    data_dir: Path, run_id: str, con: duckdb.DuckDBPyConnection,
) -> dict[str, str]:
    """Build entity_id → type mapping from entities table."""
    rows = con.execute(
        "SELECT entity_id, type FROM entities WHERE run_id = ?",
        [run_id],
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _validate_outputs(data_dir: Path, run_id: str) -> None:
    """Run contract validation on candidate_pairs.parquet."""
    cp_path = get_blocking_run_output_dir(data_dir, run_id) / "candidate_pairs.parquet"
    if not cp_path.exists():
        raise FileNotFoundError(f"Expected {cp_path} after blocking")

    table = pq.read_table(cp_path)
    errors = validate_contract_rules(table, "candidate_pairs")
    if errors:
        for e in errors:
            logger.error("candidate_pairs contract violation: %s", e)
        raise ValueError(
            f"candidate_pairs.parquet post-write validation failed "
            f"with {len(errors)} error(s)"
        )
    logger.info("candidate_pairs.parquet contract validation passed.")
