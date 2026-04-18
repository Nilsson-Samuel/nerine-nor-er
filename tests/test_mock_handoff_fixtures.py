"""Tests for the mock handoff fixture writer.

Covers:
- Schema/contract validity of generated entities and candidate pairs.
- Determinism: identical bytes on repeated calls with the same run_id.
- DuckDB-based uniqueness invariants (as would be used in real pipeline queries).
- Broken fixture detection: validate_contract_rules() catches known violations.
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.shared import schemas
from src.shared.fixtures import (
    DEFAULT_RUN_ID,
    build_mock_candidates,
    build_mock_entities,
    normalize_pair,
    write_mock_handoff,
)
from src.shared.paths import get_blocking_run_output_dir, get_extraction_run_output_dir


# ---------------------------------------------------------------------------
# normalize_pair helper
# ---------------------------------------------------------------------------

def test_normalize_pair_orders_correctly() -> None:
    a = "a" * 32
    b = "b" * 32
    assert normalize_pair(a, b) == (a, b)
    assert normalize_pair(b, a) == (a, b)  # reversed input → same canonical order


def test_normalize_pair_rejects_self_pair() -> None:
    eid = "a" * 32
    with pytest.raises(ValueError, match="self-pair"):
        normalize_pair(eid, eid)


# ---------------------------------------------------------------------------
# Entity fixture — schema and contract rules
# ---------------------------------------------------------------------------

def test_entities_schema_valid() -> None:
    table = build_mock_entities()
    errors = schemas.validate(table, schemas.ENTITIES_SCHEMA)
    assert errors == [], f"Schema errors: {errors}"


def test_entities_contract_rules_pass() -> None:
    table = build_mock_entities()
    errors = schemas.validate_contract_rules(table, "entities")
    assert errors == [], f"Contract errors: {errors}"


def test_entities_has_per_and_org_types() -> None:
    table = build_mock_entities()
    types = set(table.column("type").to_pylist())
    assert "PER" in types and "ORG" in types


# ---------------------------------------------------------------------------
# Candidate pairs fixture — schema and contract rules
# ---------------------------------------------------------------------------

def test_candidates_schema_valid() -> None:
    entities = build_mock_entities()
    candidates = build_mock_candidates(DEFAULT_RUN_ID, entities)
    errors = schemas.validate(candidates, schemas.CANDIDATE_PAIRS_SCHEMA)
    assert errors == [], f"Schema errors: {errors}"


def test_candidates_contract_rules_pass() -> None:
    entities = build_mock_entities()
    candidates = build_mock_candidates(DEFAULT_RUN_ID, entities)
    errors = schemas.validate_contract_rules(candidates, "candidate_pairs")
    assert errors == [], f"Contract errors: {errors}"


def test_candidates_ordering_and_no_self_pairs() -> None:
    entities = build_mock_entities()
    candidates = build_mock_candidates(DEFAULT_RUN_ID, entities)
    rows = candidates.to_pylist()
    for row in rows:
        assert row["entity_id_a"] < row["entity_id_b"], "Pair ordering violated"
        assert row["entity_id_a"] != row["entity_id_b"], "Self-pair found"


# ---------------------------------------------------------------------------
# DuckDB-based uniqueness checks (mirrors real pipeline query patterns)
# ---------------------------------------------------------------------------

def test_entities_unique_keys_via_duckdb() -> None:
    """Use DuckDB to verify (run_id, entity_id) uniqueness — same pattern as pipeline."""
    table = build_mock_entities()
    con = duckdb.connect()
    con.register("entities", table)
    dup_count = con.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT run_id, entity_id FROM entities"
        "  GROUP BY run_id, entity_id HAVING COUNT(*) > 1"
        ")"
    ).fetchone()[0]
    assert dup_count == 0, f"Found {dup_count} duplicate (run_id, entity_id) keys"


def test_candidates_unique_keys_via_duckdb() -> None:
    """DuckDB uniqueness check on (run_id, entity_id_a, entity_id_b)."""
    entities = build_mock_entities()
    candidates = build_mock_candidates(DEFAULT_RUN_ID, entities)
    con = duckdb.connect()
    con.register("candidate_pairs", candidates)
    dup_count = con.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT run_id, entity_id_a, entity_id_b FROM candidate_pairs"
        "  GROUP BY run_id, entity_id_a, entity_id_b HAVING COUNT(*) > 1"
        ")"
    ).fetchone()[0]
    assert dup_count == 0, f"Found {dup_count} duplicate pair keys"


# ---------------------------------------------------------------------------
# File I/O and determinism
# ---------------------------------------------------------------------------

def test_write_mock_handoff_creates_files(tmp_path) -> None:
    write_mock_handoff(tmp_path)
    extraction_dir = get_extraction_run_output_dir(tmp_path, DEFAULT_RUN_ID)
    blocking_dir = get_blocking_run_output_dir(tmp_path, DEFAULT_RUN_ID)
    assert (extraction_dir / "entities.parquet").exists()
    assert (blocking_dir / "candidate_pairs.parquet").exists()
    assert (blocking_dir / "embeddings.npy").exists()
    assert (blocking_dir / "context_embeddings.npy").exists()
    assert (blocking_dir / "embedding_entity_ids.npy").exists()


def test_written_files_pass_contract_rules(tmp_path) -> None:
    write_mock_handoff(tmp_path)
    entities   = pq.read_table(get_extraction_run_output_dir(tmp_path, DEFAULT_RUN_ID) / "entities.parquet")
    candidates = pq.read_table(get_blocking_run_output_dir(tmp_path, DEFAULT_RUN_ID) / "candidate_pairs.parquet")
    assert schemas.validate_contract_rules(entities,   "entities")        == []
    assert schemas.validate_contract_rules(candidates, "candidate_pairs") == []


def test_determinism(tmp_path) -> None:
    """Same run_id must produce bit-identical parquet files on every write."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    write_mock_handoff(dir_a)
    write_mock_handoff(dir_b)

    extraction_a = get_extraction_run_output_dir(dir_a, DEFAULT_RUN_ID)
    extraction_b = get_extraction_run_output_dir(dir_b, DEFAULT_RUN_ID)
    blocking_a = get_blocking_run_output_dir(dir_a, DEFAULT_RUN_ID)
    blocking_b = get_blocking_run_output_dir(dir_b, DEFAULT_RUN_ID)

    assert (extraction_a / "entities.parquet").read_bytes() == (extraction_b / "entities.parquet").read_bytes()
    for fname in (
        "candidate_pairs.parquet",
        "embeddings.npy",
        "context_embeddings.npy",
        "embedding_entity_ids.npy",
    ):
        bytes_a = (blocking_a / fname).read_bytes()
        bytes_b = (blocking_b / fname).read_bytes()
        assert bytes_a == bytes_b, f"{fname}: output differs between runs"


# ---------------------------------------------------------------------------
# Broken fixture — validate_contract_rules() must catch the violation
# ---------------------------------------------------------------------------

def test_broken_self_pair_caught_by_contract_rules() -> None:
    """A self-pair must be caught with a clear error from validate_contract_rules."""
    eid = "a" * 32  # any valid hex32 string
    broken = pa.table(
        {
            "run_id":               pa.array(["mock_run_001"],  type=pa.string()),
            "entity_id_a":          pa.array([eid],             type=pa.string()),
            "entity_id_b":          pa.array([eid],             type=pa.string()),  # self-pair
            "blocking_methods":     pa.array([["faiss"]],       type=pa.list_(pa.string())),
            "blocking_source":      pa.array(["faiss"],         type=pa.string()),
            "blocking_method_count":pa.array([1],               type=pa.int8()),
        },
        schema=schemas.CANDIDATE_PAIRS_SCHEMA,
    )
    errors = schemas.validate_contract_rules(broken, "candidate_pairs")
    assert any("self-pair" in e for e in errors), (
        f"Expected a self-pair error, got: {errors}"
    )


def test_broken_count_mismatch_caught_by_contract_rules() -> None:
    """count != len(positions) must be caught by entity contract validation."""
    eid    = "a" * 32
    doc_id = "b" * 32
    ck_id  = "c" * 32
    pos    = {"chunk_id": ck_id, "char_start": 0, "char_end": 5,
              "page_num": 0, "source_unit_kind": "pdf_page"}

    broken = pa.table(
        {
            "run_id":     pa.array(["mock_run_001"], type=pa.string()),
            "entity_id":  pa.array([eid],           type=pa.string()),
            "doc_id":     pa.array([doc_id],        type=pa.string()),
            "chunk_id":   pa.array([ck_id],         type=pa.string()),
            "text":       pa.array(["Test"],         type=pa.string()),
            "normalized": pa.array(["test"],         type=pa.string()),
            "type":       pa.array(["PER"],          type=pa.string()),
            "char_start": pa.array([0],              type=pa.int32()),
            "char_end":   pa.array([5],              type=pa.int32()),
            "context":    pa.array(["ctx"],          type=pa.string()),
            "count":      pa.array([3],              type=pa.int32()),  # wrong: 3 ≠ 1 position
            "positions":  pa.array(
                [[pos]],
                type=pa.list_(pa.struct([
                    ("chunk_id", pa.string()),
                    ("char_start", pa.int32()),
                    ("char_end", pa.int32()),
                    ("page_num", pa.int32()),
                    ("source_unit_kind", pa.string()),
                ])),
            ),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )
    errors = schemas.validate_contract_rules(broken, "entities")
    assert any("count" in e and "positions" in e for e in errors), (
        f"Expected count/positions mismatch error, got: {errors}"
    )
