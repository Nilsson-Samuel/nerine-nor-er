"""Mock handoff fixture writer for the matching stage bootstrap.

Writes deterministic, contract-valid parquet + embedding fixture artifacts so
matching development can proceed before real blocking output is available.
All IDs are derived from fixed seeds so repeated calls produce identical files.
"""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.shared import schemas
from src.shared.validators import validate_embedding_alignment


# Default run identifier used by all fixture helpers.
DEFAULT_RUN_ID = "mock_run_001"
_EMBEDDING_DIM = 768
_EMBEDDING_SEED = 42

# Pyarrow type for a single position struct in the positions list column.
_POSITION_TYPE = pa.struct([
    ("chunk_id", pa.string()),
    ("char_start", pa.int32()),
    ("char_end", pa.int32()),
    ("page_num", pa.int32()),
    ("source_unit_kind", pa.string()),
])

# --- Stable deterministic IDs (seed → 32-char lowercase hex) ---

def _hex_id(seed: str) -> str:
    """Derive a stable 32-char lowercase hex ID from an arbitrary seed string."""
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


# Pre-computed fixture IDs — fixed so every run produces identical bytes.
_DOC_A = _hex_id("fixture_doc_a")
_DOC_B = _hex_id("fixture_doc_b")
_CK_A1 = _hex_id("fixture_chunk_a1")
_CK_A2 = _hex_id("fixture_chunk_a2")
_CK_B1 = _hex_id("fixture_chunk_b1")
_E_PER1 = _hex_id("fixture_entity_per1")
_E_PER2 = _hex_id("fixture_entity_per2")
_E_ORG1 = _hex_id("fixture_entity_org1")
_E_ORG2 = _hex_id("fixture_entity_org2")


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    """Return a canonicalized (entity_id_a, entity_id_b) pair with a < b.

    Args:
        a: First entity ID.
        b: Second entity ID.

    Returns:
        Tuple ordered so the smaller ID is first.

    Raises:
        ValueError: If the two IDs are identical (self-pair not allowed).
    """
    if a == b:
        raise ValueError("self-pair not allowed")
    return (a, b) if a < b else (b, a)


def _pos(chunk_id: str, cs: int, ce: int, page: int = 0) -> dict:
    """Build a single position struct dict (pdf_page source kind assumed)."""
    return {
        "chunk_id": chunk_id,
        "char_start": cs,
        "char_end": ce,
        "page_num": page,
        "source_unit_kind": "pdf_page",
    }


def build_mock_entities(run_id: str = DEFAULT_RUN_ID) -> pa.Table:
    """Build a minimal set of mock entities: 2 PER + 2 ORG, all contract-valid.

    The entity set is intentionally small to keep fixture overhead low. The two
    PER entities share a normalized form to model a typical abbreviation match
    case ("P. Hansen" ↔ "Per Hansen").

    Args:
        run_id: Pipeline run identifier for all rows.

    Returns:
        PyArrow Table conforming to ENTITIES_SCHEMA.
    """
    # Each tuple: (entity_id, doc_id, chunk_id, text, normalized, type,
    #              char_start, char_end, context, count, [positions])
    rows = [
        (_E_PER1, _DOC_A, _CK_A1, "Per Hansen",     "per hansen",      "PER",
         0,  10, "Per Hansen avhørtes den 12. mars.",       1, [_pos(_CK_A1,  0, 10)]),
        (_E_PER2, _DOC_A, _CK_A2, "P. Hansen",       "per hansen",      "PER",
         5,  14, "P. Hansen ankom kl. 09:00.",               1, [_pos(_CK_A2,  5, 14)]),
        (_E_ORG1, _DOC_B, _CK_B1, "DNB ASA",         "dnb asa",         "ORG",
         20, 27, "Overføring til DNB ASA ble bekreftet.",    1, [_pos(_CK_B1, 20, 27)]),
        (_E_ORG2, _DOC_B, _CK_B1, "Den Norske Bank", "den norske bank", "ORG",
         50, 65, "Konto i Den Norske Bank ble sperret.",     1, [_pos(_CK_B1, 50, 65)]),
    ]

    n = len(rows)
    return pa.table(
        {
            "run_id":      pa.array([run_id] * n,            type=pa.string()),
            "entity_id":   pa.array([r[0] for r in rows],   type=pa.string()),
            "doc_id":      pa.array([r[1] for r in rows],   type=pa.string()),
            "chunk_id":    pa.array([r[2] for r in rows],   type=pa.string()),
            "text":        pa.array([r[3] for r in rows],   type=pa.string()),
            "normalized":  pa.array([r[4] for r in rows],   type=pa.string()),
            "type":        pa.array([r[5] for r in rows],   type=pa.string()),
            "char_start":  pa.array([r[6] for r in rows],   type=pa.int32()),
            "char_end":    pa.array([r[7] for r in rows],   type=pa.int32()),
            "context":     pa.array([r[8] for r in rows],   type=pa.string()),
            "count":       pa.array([r[9] for r in rows],   type=pa.int32()),
            "positions":   pa.array([r[10] for r in rows],  type=pa.list_(_POSITION_TYPE)),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )


def build_mock_candidates(run_id: str, entities: pa.Table) -> pa.Table:
    """Build mock candidate pairs from an entities table (same-type pairs only).

    Produces one PER pair and one ORG pair, each from a single blocking method,
    so all consistency invariants are trivially satisfied.

    Args:
        run_id: Pipeline run identifier for all rows.
        entities: PyArrow Table produced by build_mock_entities().

    Returns:
        PyArrow Table conforming to CANDIDATE_PAIRS_SCHEMA.
    """
    ids = {row["type"]: [] for row in entities.to_pylist()}
    for row in entities.to_pylist():
        ids.setdefault(row["type"], []).append(row["entity_id"])

    per_a, per_b = normalize_pair(ids["PER"][0], ids["PER"][1])
    org_a, org_b = normalize_pair(ids["ORG"][0], ids["ORG"][1])

    # One PER pair found by phonetic blocking, one ORG pair found by FAISS.
    # blocking_source == sole method when blocking_method_count == 1.
    pair_rows = [
        (run_id, per_a, per_b, ["phonetic"], "phonetic", 1),
        (run_id, org_a, org_b, ["faiss"],    "faiss",    1),
    ]

    n = len(pair_rows)
    return pa.table(
        {
            "run_id":               pa.array([r[0] for r in pair_rows], type=pa.string()),
            "entity_id_a":          pa.array([r[1] for r in pair_rows], type=pa.string()),
            "entity_id_b":          pa.array([r[2] for r in pair_rows], type=pa.string()),
            "blocking_methods":     pa.array([r[3] for r in pair_rows],
                                             type=pa.list_(pa.string())),
            "blocking_source":      pa.array([r[4] for r in pair_rows], type=pa.string()),
            "blocking_method_count":pa.array([r[5] for r in pair_rows], type=pa.int8()),
        },
        schema=schemas.CANDIDATE_PAIRS_SCHEMA,
    )


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row, keeping float32 output."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / (norms + 1e-12)).astype("float32")


def build_mock_embedding_artifacts(
    entities: pa.Table,
    seed: int = _EMBEDDING_SEED,
    embedding_dim: int = _EMBEDDING_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic embedding artifacts aligned to fixture entity row order.

    Args:
        entities: Entity table from build_mock_entities().
        seed: RNG seed for deterministic values.
        embedding_dim: Embedding width (default 768).

    Returns:
        Tuple of (embeddings, context_embeddings, embedding_entity_ids).
    """
    entity_ids = np.array(entities.column("entity_id").to_pylist())
    n_entities = len(entity_ids)

    rng = np.random.default_rng(seed=seed)
    embeddings = rng.normal(size=(n_entities, embedding_dim)).astype("float32")
    context_embeddings = rng.normal(size=(n_entities, embedding_dim)).astype("float32")
    return (
        _l2_normalize_rows(embeddings),
        _l2_normalize_rows(context_embeddings),
        entity_ids,
    )


def write_mock_handoff(out_dir: Path, run_id: str = DEFAULT_RUN_ID) -> None:
    """Write handoff fixture files to out_dir.

    The output is fully deterministic: the same run_id always produces
    bit-identical files. Intended for use in tests and early matching
    development before real blocking output is available.

    Args:
        out_dir: Directory to write fixture files into (created if absent).
        run_id: Pipeline run identifier embedded in every row.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entities = build_mock_entities(run_id)
    candidates = build_mock_candidates(run_id, entities)
    embeddings, context_embeddings, embedding_entity_ids = build_mock_embedding_artifacts(
        entities
    )
    validate_embedding_alignment(
        embeddings=embeddings,
        context_embeddings=context_embeddings,
        embedding_entity_ids=embedding_entity_ids,
        entity_ids=entities.column("entity_id").to_pylist(),
    )

    # Validate before writing — catch contract violations early rather than silently
    # persisting bad fixtures that would cause confusing failures downstream.
    entity_errors = schemas.validate_contract_rules(entities, "entities")
    if entity_errors:
        raise ValueError(f"Mock entities failed contract validation: {entity_errors}")

    candidate_errors = schemas.validate_contract_rules(candidates, "candidate_pairs")
    if candidate_errors:
        raise ValueError(f"Mock candidates failed contract validation: {candidate_errors}")

    pq.write_table(entities,   out_dir / "entities.parquet")
    pq.write_table(candidates, out_dir / "candidate_pairs.parquet")
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "context_embeddings.npy", context_embeddings)
    np.save(out_dir / "embedding_entity_ids.npy", embedding_entity_ids)
