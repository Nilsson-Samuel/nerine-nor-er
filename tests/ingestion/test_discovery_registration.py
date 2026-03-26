"""Tests for file discovery, document registration, and docs.parquet output.

Covers important (validation gate) criteria:
- No duplicate doc_id for same run_id on re-run
- No duplicate path for same run_id on re-run
- doc_id is 32 hex chars
- Correct mime_type and source_unit_kind
- path is relative, not absolute
- file_size > 0
- extracted_at is a UTC timestamp
"""

import hashlib
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from src.ingestion.discovery import discover_documents
from src.ingestion.registration import (
    build_doc_row,
    compute_doc_id,
    register_documents,
)
from src.ingestion.run import run_discovery_and_registration
from src.shared.schemas import DOCS_SCHEMA, validate, validate_contract_rules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def case_root(tmp_path: Path) -> Path:
    """Create a temp case folder with sample PDF and DOCX files."""
    root = tmp_path / "case"
    root.mkdir()

    # Create subdirectory with files
    sub = root / "subdir"
    sub.mkdir()

    (root / "report.pdf").write_bytes(b"fake-pdf-content-1")
    (root / "notes.docx").write_bytes(b"fake-docx-content-2")
    (sub / "evidence.pdf").write_bytes(b"fake-pdf-content-3")

    # Non-matching files (should be ignored)
    (root / "image.png").write_bytes(b"png-data")
    (root / "readme.txt").write_text("hello")

    return root


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_finds_pdf_and_docx_only(self, case_root: Path):
        files = discover_documents(case_root)
        extensions = {f.suffix.lower() for f in files}
        assert extensions == {".pdf", ".docx"}

    def test_finds_files_in_subdirs(self, case_root: Path):
        files = discover_documents(case_root)
        names = {f.name for f in files}
        assert "evidence.pdf" in names

    def test_returns_sorted_list(self, case_root: Path):
        files = discover_documents(case_root)
        assert files == sorted(files)

    def test_returns_three_files(self, case_root: Path):
        files = discover_documents(case_root)
        assert len(files) == 3

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        empty = tmp_path / "empty_case"
        empty.mkdir()
        assert discover_documents(empty) == []

    def test_case_insensitive_extension(self, tmp_path: Path):
        root = tmp_path / "case_ext"
        root.mkdir()
        (root / "upper.PDF").write_bytes(b"pdf-data")
        (root / "mixed.Docx").write_bytes(b"docx-data")
        files = discover_documents(root)
        assert len(files) == 2


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestComputeDocId:
    def test_length_is_32(self, case_root: Path):
        doc_id = compute_doc_id(case_root / "report.pdf")
        assert len(doc_id) == 32

    def test_all_lowercase_hex(self, case_root: Path):
        doc_id = compute_doc_id(case_root / "report.pdf")
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_deterministic(self, case_root: Path):
        path = case_root / "report.pdf"
        assert compute_doc_id(path) == compute_doc_id(path)

    def test_matches_manual_sha256(self, case_root: Path):
        path = case_root / "report.pdf"
        expected = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        assert compute_doc_id(path) == expected

    def test_different_content_different_id(self, case_root: Path):
        id1 = compute_doc_id(case_root / "report.pdf")
        id2 = compute_doc_id(case_root / "notes.docx")
        assert id1 != id2


class TestBuildDocRow:
    def test_relative_path(self, case_root: Path):
        from datetime import datetime, timezone
        row = build_doc_row(
            case_root / "report.pdf", case_root, "testrun", datetime.now(timezone.utc)
        )
        assert not row["path"].startswith("/")
        assert row["path"] == "report.pdf"

    def test_subdirectory_path_is_posix(self, case_root: Path):
        from datetime import datetime, timezone
        row = build_doc_row(
            case_root / "subdir" / "evidence.pdf",
            case_root,
            "testrun",
            datetime.now(timezone.utc),
        )
        assert row["path"] == "subdir/evidence.pdf"

    def test_pdf_metadata(self, case_root: Path):
        from datetime import datetime, timezone
        row = build_doc_row(
            case_root / "report.pdf", case_root, "testrun", datetime.now(timezone.utc)
        )
        assert row["mime_type"] == "application/pdf"
        assert row["source_unit_kind"] == "pdf_page"

    def test_docx_metadata(self, case_root: Path):
        from datetime import datetime, timezone
        row = build_doc_row(
            case_root / "notes.docx", case_root, "testrun", datetime.now(timezone.utc)
        )
        assert row["mime_type"] == (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
        assert row["source_unit_kind"] == "docx_paragraph"

    def test_file_size_positive(self, case_root: Path):
        from datetime import datetime, timezone
        row = build_doc_row(
            case_root / "report.pdf", case_root, "testrun", datetime.now(timezone.utc)
        )
        assert row["file_size"] > 0

    def test_page_count_is_none(self, case_root: Path):
        from datetime import datetime, timezone
        row = build_doc_row(
            case_root / "report.pdf", case_root, "testrun", datetime.now(timezone.utc)
        )
        assert row["page_count"] is None


class TestRegisterDocuments:
    def test_registers_all_new(self, case_root: Path, con: duckdb.DuckDBPyConnection):
        files = discover_documents(case_root)
        table = register_documents(files, case_root, "run1", con)
        assert len(table) == 3

    def test_schema_valid(self, case_root: Path, con: duckdb.DuckDBPyConnection):
        files = discover_documents(case_root)
        table = register_documents(files, case_root, "run1", con)
        errors = validate(table, DOCS_SCHEMA)
        assert errors == []

    def test_contract_valid(self, case_root: Path, con: duckdb.DuckDBPyConnection):
        files = discover_documents(case_root)
        table = register_documents(files, case_root, "run1", con)
        errors = validate_contract_rules(table, "docs")
        assert errors == []

    def test_dedup_skips_existing(self, case_root: Path, con: duckdb.DuckDBPyConnection):
        """Re-registering the same files produces no new rows."""
        files = discover_documents(case_root)
        run_id = "run1"

        # First registration
        table1 = register_documents(files, case_root, run_id, con)
        con.execute("CREATE OR REPLACE TABLE docs AS SELECT * FROM table1")

        # Second registration — should skip all
        table2 = register_documents(files, case_root, run_id, con)
        assert len(table2) == 0

    def test_empty_input(self, case_root: Path, con: duckdb.DuckDBPyConnection):
        table = register_documents([], case_root, "run1", con)
        assert len(table) == 0
        errors = validate(table, DOCS_SCHEMA)
        assert errors == []

    def test_changed_file_same_path_no_duplicate(
        self, case_root: Path, con: duckdb.DuckDBPyConnection,
    ):
        """If a file's content changes but path stays the same, skip it."""
        files = discover_documents(case_root)
        run_id = "run1"

        # First registration
        table1 = register_documents(files, case_root, run_id, con)
        con.execute("CREATE OR REPLACE TABLE docs AS SELECT * FROM table1")

        # Change content of report.pdf (new doc_id, same path)
        (case_root / "report.pdf").write_bytes(b"changed-pdf-content")

        # Second registration — should skip the changed file
        table2 = register_documents(files, case_root, run_id, con)
        paths = table2.column("path").to_pylist()
        assert "report.pdf" not in paths

    def test_identical_files_in_same_batch_deduped(
        self, tmp_path: Path, con: duckdb.DuckDBPyConnection,
    ):
        """Two files with identical bytes in the same batch must not both register.

        a.pdf and b.pdf contain the same bytes,
        producing the same doc_id. Only the first should be accepted;
        the second must be skipped to preserve (run_id, doc_id) uniqueness.
        """
        case_root = tmp_path / "case_dup"
        case_root.mkdir()

        identical_content = b"identical-pdf-content"
        (case_root / "a.pdf").write_bytes(identical_content)
        (case_root / "b.pdf").write_bytes(identical_content)

        files = discover_documents(case_root)
        table = register_documents(files, case_root, "run1", con)

        # Only one row should be registered
        assert len(table) == 1

        # Contract validation must pass (no duplicate run_id+doc_id)
        errors = validate_contract_rules(table, "docs")
        assert errors == []


# ---------------------------------------------------------------------------
# End-to-end orchestrator tests
# ---------------------------------------------------------------------------

class TestRunDiscoveryAndRegistration:
    def test_creates_docs_parquet(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="e2etest1")
        docs_path = data_dir / "docs.parquet"
        assert docs_path.exists()

    def test_parquet_schema_valid(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="e2etest2")
        table = pq.read_table(data_dir / "docs.parquet")
        errors = validate(table, DOCS_SCHEMA)
        assert errors == []

    def test_parquet_contract_valid(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="e2etest3")
        table = pq.read_table(data_dir / "docs.parquet")
        errors = validate_contract_rules(table, "docs")
        assert errors == []

    def test_correct_row_count(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="e2etest4")
        table = pq.read_table(data_dir / "docs.parquet")
        assert len(table) == 3

    def test_incremental_no_duplicates(self, case_root: Path, data_dir: Path):
        """Running twice on the same input produces no duplicate rows."""
        run_id = "incremental1"
        run_discovery_and_registration(case_root, data_dir, run_id=run_id)
        run_discovery_and_registration(case_root, data_dir, run_id=run_id)

        table = pq.read_table(data_dir / "docs.parquet")
        assert len(table) == 3  # Not 6

        # Contract validation catches uniqueness violations
        errors = validate_contract_rules(table, "docs")
        assert errors == []

    def test_run_id_returned(self, case_root: Path, data_dir: Path):
        rid = run_discovery_and_registration(case_root, data_dir, run_id="myrun")
        assert rid == "myrun"

    def test_auto_generated_run_id(self, case_root: Path, data_dir: Path):
        rid = run_discovery_and_registration(case_root, data_dir)
        assert len(rid) == 32
        assert all(c in "0123456789abcdef" for c in rid)

    def test_no_files_no_parquet(self, tmp_path: Path, data_dir: Path):
        """Empty case root fails early with a clear user-facing error."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No PDF/DOCX files found under case_root"):
            run_discovery_and_registration(empty, data_dir, run_id="emptyrun")
        assert not (data_dir / "docs.parquet").exists()

    def test_all_paths_relative(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="relpath")
        table = pq.read_table(data_dir / "docs.parquet")
        paths = table.column("path").to_pylist()
        for p in paths:
            assert not p.startswith("/"), f"Path should be relative: {p}"

    def test_all_doc_ids_hex32(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="hex32chk")
        table = pq.read_table(data_dir / "docs.parquet")
        for doc_id in table.column("doc_id").to_pylist():
            assert len(doc_id) == 32
            assert all(c in "0123456789abcdef" for c in doc_id)

    def test_file_sizes_positive(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="fsize")
        table = pq.read_table(data_dir / "docs.parquet")
        for size in table.column("file_size").to_pylist():
            assert size > 0

    def test_extracted_at_is_utc(self, case_root: Path, data_dir: Path):
        run_discovery_and_registration(case_root, data_dir, run_id="utcchk")
        table = pq.read_table(data_dir / "docs.parquet")
        # PyArrow timestamp type should have tz="UTC"
        ts_field = table.schema.field("extracted_at")
        assert str(ts_field.type) == "timestamp[us, tz=UTC]"
