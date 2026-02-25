"""Tests for chunking and full ingestion orchestration.

Covers the validation gate criteria:
- Chunk size and overlap follow configured values
- Deterministic output across repeated runs on unchanged input
- chunks.parquet required fields present and correctly typed
- (run_id, doc_id, chunk_index) uniqueness
- All chunk doc_id values exist in docs.parquet for same run
- chunk_id is stable 32-char hex derived from doc_id + chunk_index
- run_metadata.json written with expected keys
- Full end-to-end ingestion pipeline produces valid artifacts
"""

import json
from pathlib import Path

import fitz
import pyarrow.parquet as pq
import pytest
from docx import Document

from src.ingestion.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_splitter,
    chunk_document,
    make_chunk_id,
)
from src.ingestion.run import (
    run_discovery_and_registration,
    run_extraction_and_normalization,
    run_ingestion,
)
from src.shared.schemas import validate, validate_contract_rules, CHUNKS_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def case_root(tmp_path: Path) -> Path:
    """Case folder with a PDF and a DOCX for end-to-end tests."""
    root = tmp_path / "case"
    root.mkdir()

    # PDF with enough text to produce multiple chunks
    pdf = fitz.open()
    page = pdf.new_page()
    long_text = ("Politiet etterforsker saken om DNB ASA. " * 40).strip()
    page.insert_text((50, 72), long_text)
    pdf.save(str(root / "rapport.pdf"))
    pdf.close()

    # DOCX with multiple paragraphs
    doc = Document()
    for i in range(10):
        doc.add_paragraph(
            f"Avhør nummer {i}. Kari Nordmann forklarte seg om hendelsen. "
            "Det ble observert flere personer på stedet. "
            "Vitnet beskrev situasjonen i detalj."
        )
    doc.save(str(root / "avhoer.docx"))

    return root


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Unit tests — make_chunk_id
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def test_returns_32_hex(self):
        cid = make_chunk_id("abc123", 0)
        assert len(cid) == 32
        assert all(c in "0123456789abcdef" for c in cid)

    def test_deterministic(self):
        assert make_chunk_id("doc1", 5) == make_chunk_id("doc1", 5)

    def test_different_index_gives_different_id(self):
        assert make_chunk_id("doc1", 0) != make_chunk_id("doc1", 1)

    def test_different_doc_gives_different_id(self):
        assert make_chunk_id("doc1", 0) != make_chunk_id("doc2", 0)


# ---------------------------------------------------------------------------
# Unit tests — build_splitter
# ---------------------------------------------------------------------------

class TestBuildSplitter:
    def test_chunk_size(self):
        splitter = build_splitter()
        assert splitter._chunk_size == CHUNK_SIZE

    def test_chunk_overlap(self):
        splitter = build_splitter()
        assert splitter._chunk_overlap == CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Unit tests — chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_empty_units_returns_empty(self):
        splitter = build_splitter()
        assert chunk_document("doc1", [], splitter) == []

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size produces exactly one chunk."""
        splitter = build_splitter()
        units = [{"page_num": 0, "text": "Short text.", "source_unit_kind": "pdf_page"}]
        chunks = chunk_document("doc1", units, splitter)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_chunk_fields_present(self):
        splitter = build_splitter()
        units = [{"page_num": 0, "text": "Hello world.", "source_unit_kind": "pdf_page"}]
        chunks = chunk_document("doc1", units, splitter)
        expected_keys = {"chunk_id", "doc_id", "chunk_index", "text",
                         "source_unit_kind", "page_num"}
        assert set(chunks[0].keys()) == expected_keys

    def test_chunk_index_sequential(self):
        """chunk_index starts at 0 and increments."""
        splitter = build_splitter()
        text = "Word " * 300  # ~1500 chars, should produce multiple chunks
        units = [{"page_num": 0, "text": text, "source_unit_kind": "pdf_page"}]
        chunks = chunk_document("doc1", units, splitter)
        assert len(chunks) > 1
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_id_matches_make_chunk_id(self):
        splitter = build_splitter()
        units = [{"page_num": 0, "text": "Some text.", "source_unit_kind": "pdf_page"}]
        chunks = chunk_document("doc1", units, splitter)
        for c in chunks:
            assert c["chunk_id"] == make_chunk_id("doc1", c["chunk_index"])

    def test_source_unit_kind_preserved(self):
        splitter = build_splitter()
        units = [
            {"page_num": 0, "text": "Paragraph.", "source_unit_kind": "docx_paragraph"},
        ]
        chunks = chunk_document("doc1", units, splitter)
        assert chunks[0]["source_unit_kind"] == "docx_paragraph"

    def test_page_num_preserved(self):
        splitter = build_splitter()
        units = [
            {"page_num": 3, "text": "Page three.", "source_unit_kind": "pdf_page"},
        ]
        chunks = chunk_document("doc1", units, splitter)
        assert chunks[0]["page_num"] == 3

    def test_no_empty_text_chunks(self):
        """Empty text units should not produce chunks with empty text."""
        splitter = build_splitter()
        units = [
            {"page_num": 0, "text": "", "source_unit_kind": "pdf_page"},
            {"page_num": 1, "text": "Real content here.", "source_unit_kind": "pdf_page"},
        ]
        chunks = chunk_document("doc1", units, splitter)
        for c in chunks:
            assert c["text"].strip() != ""

    def test_overlap_present_in_long_text(self):
        """When text is long enough for multiple chunks, adjacent chunks share text."""
        splitter = build_splitter()
        text = "Setning nummer en. " * 200
        units = [{"page_num": 0, "text": text, "source_unit_kind": "pdf_page"}]
        chunks = chunk_document("doc1", units, splitter)
        if len(chunks) >= 2:
            # Last part of chunk N should overlap with start of chunk N+1
            end_of_first = chunks[0]["text"][-CHUNK_OVERLAP:]
            assert end_of_first in chunks[1]["text"]

    def test_deterministic_output(self):
        """Same input produces identical chunks across two calls."""
        splitter = build_splitter()
        text = "Ola Nordmann er mistenkt i saken. " * 50
        units = [{"page_num": 0, "text": text, "source_unit_kind": "pdf_page"}]
        run1 = chunk_document("doc1", units, splitter)
        run2 = chunk_document("doc1", units, splitter)
        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2):
            assert c1 == c2

    def test_multi_unit_page_num_tracking(self):
        """Chunks from multi-page docs get the correct starting page_num."""
        splitter = build_splitter()
        # Two pages with enough text that chunks span into page 1
        units = [
            {"page_num": 0, "text": "A " * 300, "source_unit_kind": "pdf_page"},
            {"page_num": 1, "text": "B " * 300, "source_unit_kind": "pdf_page"},
        ]
        chunks = chunk_document("doc1", units, splitter)
        # At least one chunk should reference page_num 1
        page_nums = {c["page_num"] for c in chunks}
        assert 0 in page_nums
        assert 1 in page_nums


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestChunksSchemaValidation:
    def _make_valid_chunk_table(self):
        """Build a minimal valid chunks table for testing."""
        import pyarrow as pa
        return pa.table({
            "run_id": ["testrun"],
            "chunk_id": [make_chunk_id("a" * 32, 0)],
            "doc_id": ["a" * 32],
            "chunk_index": [0],
            "text": ["Some chunk text."],
            "source_unit_kind": ["pdf_page"],
            "page_num": [0],
        }, schema=CHUNKS_SCHEMA)

    def test_valid_table_passes_schema(self):
        table = self._make_valid_chunk_table()
        errors = validate(table, CHUNKS_SCHEMA)
        assert errors == []

    def test_valid_table_passes_contract(self):
        table = self._make_valid_chunk_table()
        errors = validate_contract_rules(table, "chunks")
        assert errors == []

    def test_empty_text_fails_contract(self):
        import pyarrow as pa
        table = pa.table({
            "run_id": ["testrun"],
            "chunk_id": [make_chunk_id("a" * 32, 0)],
            "doc_id": ["a" * 32],
            "chunk_index": [0],
            "text": [""],
            "source_unit_kind": ["pdf_page"],
            "page_num": [0],
        }, schema=CHUNKS_SCHEMA)
        errors = validate_contract_rules(table, "chunks")
        assert any("text" in e for e in errors)

    def test_duplicate_chunk_id_fails_contract(self):
        import pyarrow as pa
        cid = make_chunk_id("a" * 32, 0)
        table = pa.table({
            "run_id": ["testrun", "testrun"],
            "chunk_id": [cid, cid],
            "doc_id": ["a" * 32, "a" * 32],
            "chunk_index": [0, 1],
            "text": ["Text one.", "Text two."],
            "source_unit_kind": ["pdf_page", "pdf_page"],
            "page_num": [0, 0],
        }, schema=CHUNKS_SCHEMA)
        errors = validate_contract_rules(table, "chunks")
        assert any("duplicate (run_id, chunk_id)" in e for e in errors)

    def test_duplicate_doc_chunk_index_fails_contract(self):
        import pyarrow as pa
        table = pa.table({
            "run_id": ["testrun", "testrun"],
            "chunk_id": [make_chunk_id("a" * 32, 0), make_chunk_id("b" * 32, 0)],
            "doc_id": ["a" * 32, "a" * 32],
            "chunk_index": [0, 0],
            "text": ["Text one.", "Text two."],
            "source_unit_kind": ["pdf_page", "pdf_page"],
            "page_num": [0, 0],
        }, schema=CHUNKS_SCHEMA)
        errors = validate_contract_rules(table, "chunks")
        assert any("duplicate (run_id, doc_id, chunk_index)" in e for e in errors)

    def test_negative_chunk_index_fails(self):
        import pyarrow as pa
        table = pa.table({
            "run_id": ["testrun"],
            "chunk_id": [make_chunk_id("a" * 32, 0)],
            "doc_id": ["a" * 32],
            "chunk_index": [-1],
            "text": ["Text."],
            "source_unit_kind": ["pdf_page"],
            "page_num": [0],
        }, schema=CHUNKS_SCHEMA)
        errors = validate_contract_rules(table, "chunks")
        assert any("chunk_index" in e for e in errors)

    def test_invalid_source_unit_kind_fails(self):
        import pyarrow as pa
        table = pa.table({
            "run_id": ["testrun"],
            "chunk_id": [make_chunk_id("a" * 32, 0)],
            "doc_id": ["a" * 32],
            "chunk_index": [0],
            "text": ["Text."],
            "source_unit_kind": ["invalid_kind"],
            "page_num": [0],
        }, schema=CHUNKS_SCHEMA)
        errors = validate_contract_rules(table, "chunks")
        assert any("source_unit_kind" in e for e in errors)


# ---------------------------------------------------------------------------
# End-to-end orchestrator tests
# ---------------------------------------------------------------------------

class TestRunIngestion:
    def test_produces_docs_and_chunks_parquet(
        self, case_root: Path, data_dir: Path,
    ):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        assert (data_dir / "docs.parquet").exists()
        assert (data_dir / "chunks.parquet").exists()

    def test_chunks_parquet_schema_valid(self, case_root: Path, data_dir: Path):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        table = pq.read_table(data_dir / "chunks.parquet")
        errors = validate(table, CHUNKS_SCHEMA)
        assert errors == []

    def test_chunks_parquet_contract_valid(self, case_root: Path, data_dir: Path):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        table = pq.read_table(data_dir / "chunks.parquet")
        errors = validate_contract_rules(table, "chunks")
        assert errors == []

    def test_docs_parquet_still_contract_valid(
        self, case_root: Path, data_dir: Path,
    ):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        table = pq.read_table(data_dir / "docs.parquet")
        errors = validate_contract_rules(table, "docs")
        assert errors == []

    def test_chunk_doc_ids_exist_in_docs(self, case_root: Path, data_dir: Path):
        """All chunk doc_id values must reference a doc in docs.parquet."""
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        docs = pq.read_table(data_dir / "docs.parquet").to_pylist()
        chunks = pq.read_table(data_dir / "chunks.parquet").to_pylist()
        doc_ids = {d["doc_id"] for d in docs}
        for c in chunks:
            assert c["doc_id"] in doc_ids

    def test_all_chunks_have_correct_run_id(
        self, case_root: Path, data_dir: Path,
    ):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        table = pq.read_table(data_dir / "chunks.parquet")
        for row in table.to_pylist():
            assert row["run_id"] == "e2e_test"

    def test_all_chunks_have_non_empty_text(
        self, case_root: Path, data_dir: Path,
    ):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        table = pq.read_table(data_dir / "chunks.parquet")
        for row in table.to_pylist():
            assert row["text"].strip() != ""

    def test_chunk_index_unique_per_doc(self, case_root: Path, data_dir: Path):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        table = pq.read_table(data_dir / "chunks.parquet")
        seen: set[tuple[str, str, int]] = set()
        for row in table.to_pylist():
            key = (row["run_id"], row["doc_id"], row["chunk_index"])
            assert key not in seen
            seen.add(key)

    def test_run_metadata_written(self, case_root: Path, data_dir: Path):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        meta_path = data_dir / "run_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == "e2e_test"
        assert meta["docs_processed"] > 0
        assert meta["chunk_count"] > 0
        assert "elapsed_seconds" in meta

    def test_run_metadata_has_chunk_stats(self, case_root: Path, data_dir: Path):
        run_ingestion(case_root, data_dir, run_id="e2e_test")
        meta = json.loads((data_dir / "run_metadata.json").read_text())
        assert "chunks_per_doc_min" in meta
        assert "chunks_per_doc_max" in meta
        assert "chunks_per_doc_median" in meta
        assert meta["chunks_per_doc_min"] >= 1

    def test_deterministic_across_runs(self, case_root: Path, tmp_path: Path):
        """Same input produces identical chunks.parquet content."""
        dir1 = tmp_path / "out1"
        dir1.mkdir()
        dir2 = tmp_path / "out2"
        dir2.mkdir()

        run_ingestion(case_root, dir1, run_id="det_test")
        run_ingestion(case_root, dir2, run_id="det_test")

        chunks1 = pq.read_table(dir1 / "chunks.parquet").to_pylist()
        chunks2 = pq.read_table(dir2 / "chunks.parquet").to_pylist()

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1["chunk_id"] == c2["chunk_id"]
            assert c1["text"] == c2["text"]
            assert c1["chunk_index"] == c2["chunk_index"]

    def test_returns_run_id(self, case_root: Path, data_dir: Path):
        result = run_ingestion(case_root, data_dir, run_id="myrun")
        assert result == "myrun"

    def test_auto_generates_run_id(self, case_root: Path, data_dir: Path):
        result = run_ingestion(case_root, data_dir)
        assert len(result) == 32

    def test_existing_tests_still_work_discovery_registration(
        self, case_root: Path, data_dir: Path,
    ):
        """Sub-stage functions remain callable for backward compatibility."""
        run_id = run_discovery_and_registration(
            case_root, data_dir, run_id="compat_test"
        )
        assert run_id == "compat_test"
        assert (data_dir / "docs.parquet").exists()

    def test_existing_tests_still_work_extraction(
        self, case_root: Path, data_dir: Path,
    ):
        """Sub-stage extraction function remains callable."""
        run_id = run_discovery_and_registration(
            case_root, data_dir, run_id="compat_test"
        )
        result = run_extraction_and_normalization(case_root, data_dir, run_id)
        assert len(result) > 0
