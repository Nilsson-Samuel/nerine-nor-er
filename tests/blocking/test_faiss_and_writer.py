"""Tests for FAISS HNSW index and candidate parquet/manifest writer.

FAISS tests use small random embeddings — index mechanics work the same
regardless of whether vectors are meaningful SBERT output.

Writer tests verify parquet schema conformance and handoff manifest content.

Covers:
- FAISS HNSW build + query returns neighbors
- Self-hits filtered from FAISS output
- Writer produces schema-valid candidate_pairs.parquet
- Writer raises on contract violation
- Manifest contains all required keys with correct types
"""

import json
import tempfile
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pyarrow.parquet as pq
import pytest

from src.blocking.faiss_index import build_hnsw_index, query_neighbors
from src.blocking.writer import write_candidate_pairs, write_handoff_manifest
from src.shared.paths import get_blocking_run_output_dir
from src.shared.schemas import HANDOFF_MANIFEST_KEYS, validate_contract_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32


def _random_embeddings(n: int, dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, dim)).astype("float32")
    faiss.normalize_L2(emb)
    return emb


# ---------------------------------------------------------------------------
# FAISS HNSW tests
# ---------------------------------------------------------------------------

class TestFaissIndex:
    def test_build_and_query_returns_pairs(self):
        emb = _random_embeddings(10)
        ids = [f"{i:032x}" for i in range(10)]
        index = build_hnsw_index(emb)
        pairs = query_neighbors(index, emb, ids, k=3)
        assert len(pairs) > 0

    def test_no_self_hits(self):
        emb = _random_embeddings(10)
        ids = [f"{i:032x}" for i in range(10)]
        index = build_hnsw_index(emb)
        pairs = query_neighbors(index, emb, ids, k=5)
        for a, b in pairs:
            assert a != b

    def test_k_limits_neighbors(self):
        emb = _random_embeddings(20)
        ids = [f"{i:032x}" for i in range(20)]
        index = build_hnsw_index(emb)
        pairs = query_neighbors(index, emb, ids, k=2)
        # Each entity gets at most k neighbors, so max pairs = n * k
        assert len(pairs) <= 20 * 2


# ---------------------------------------------------------------------------
# Writer: candidate_pairs.parquet
# ---------------------------------------------------------------------------

class TestWriteCandidatePairs:
    def test_writes_valid_parquet(self, tmp_path):
        con = duckdb.connect()
        candidates = [
            {
                "entity_id_a": ID_A,
                "entity_id_b": ID_B,
                "blocking_methods": ["faiss"],
                "blocking_source": "faiss",
                "blocking_method_count": 1,
            },
        ]
        count = write_candidate_pairs(candidates, "run_test", tmp_path, con)
        assert count == 1

        table = pq.read_table(get_blocking_run_output_dir(tmp_path, "run_test") / "candidate_pairs.parquet")
        errors = validate_contract_rules(table, "candidate_pairs")
        assert errors == []

    def test_multi_method_pair_valid(self, tmp_path):
        con = duckdb.connect()
        candidates = [
            {
                "entity_id_a": ID_A,
                "entity_id_b": ID_B,
                "blocking_methods": ["exact", "faiss"],
                "blocking_source": "multi",
                "blocking_method_count": 2,
            },
        ]
        count = write_candidate_pairs(candidates, "run_test", tmp_path, con)
        assert count == 1

        table = pq.read_table(get_blocking_run_output_dir(tmp_path, "run_test") / "candidate_pairs.parquet")
        errors = validate_contract_rules(table, "candidate_pairs")
        assert errors == []

    def test_invalid_pair_raises(self, tmp_path):
        con = duckdb.connect()
        # entity_id_a > entity_id_b violates ordering contract
        candidates = [
            {
                "entity_id_a": ID_B,  # b > a
                "entity_id_b": ID_A,
                "blocking_methods": ["faiss"],
                "blocking_source": "faiss",
                "blocking_method_count": 1,
            },
        ]
        with pytest.raises(ValueError, match="contract validation"):
            write_candidate_pairs(candidates, "run_test", tmp_path, con)


# ---------------------------------------------------------------------------
# Writer: handoff_manifest.json
# ---------------------------------------------------------------------------

class TestWriteHandoffManifest:
    def test_manifest_contains_required_keys(self, tmp_path):
        path = write_handoff_manifest(
            run_id="run_test",
            entity_count=100,
            candidate_count=500,
            entity_types_present=["ORG", "PER"],
            data_dir=tmp_path,
        )
        manifest = json.loads(path.read_text())
        for key in HANDOFF_MANIFEST_KEYS:
            assert key in manifest, f"Missing manifest key: {key}"

    def test_manifest_values_correct(self, tmp_path):
        path = write_handoff_manifest(
            run_id="run_test",
            entity_count=100,
            candidate_count=500,
            entity_types_present=["PER", "ORG"],
            data_dir=tmp_path,
            embedding_dim=768,
            k=100,
        )
        manifest = json.loads(path.read_text())
        assert manifest["schema_version"] == "1.1"
        assert manifest["run_id"] == "run_test"
        assert manifest["mention_count"] == 100
        assert manifest["candidate_count"] == 500
        assert manifest["embedding_dim"] == 768
        assert manifest["entity_types_present"] == ["ORG", "PER"]  # sorted
        assert manifest["k"] == 100

    def test_manifest_created_at_is_iso(self, tmp_path):
        path = write_handoff_manifest(
            run_id="run_test",
            entity_count=10,
            candidate_count=5,
            entity_types_present=["PER"],
            data_dir=tmp_path,
        )
        manifest = json.loads(path.read_text())
        # Should be parseable ISO 8601
        from datetime import datetime
        datetime.fromisoformat(manifest["created_at"])
