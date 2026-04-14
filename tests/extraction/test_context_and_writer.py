"""Tests for context extraction, entity writer, and full extraction pipeline.

Covers validation gate criteria:
- Context window is ±50 chars around primary mention span
- Context slices correctly at chunk boundaries (no IndexError)
- entity_id is stable 32-char hex derived from primary mention coordinates
- entities.parquet has correct schema and passes contract validation
- unique (run_id, entity_id) keys
- count == len(positions) for every row
- Primary span is included in positions list
- char_start/char_end point to expected text within context
- Full pipeline: mentions → normalize → dedup → context → write → validate
"""

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.extraction.context import extract_context
from src.extraction.writer import make_entity_id, write_entities_parquet
from src.shared.paths import get_extraction_run_output_dir, get_ingestion_run_output_dir
from src.shared.schemas import (
    CHUNKS_SCHEMA,
    ENTITIES_SCHEMA,
    validate,
    validate_contract_rules,
)


# ---------------------------------------------------------------------------
# Context extraction tests
# ---------------------------------------------------------------------------

class TestExtractContext:
    def test_basic_window(self):
        text = "A" * 20 + "ENTITY" + "B" * 20
        ctx = extract_context(text, 20, 26, window=10)
        assert ctx == "A" * 10 + "ENTITY" + "B" * 10

    def test_window_at_start(self):
        text = "ENTITY" + "B" * 100
        ctx = extract_context(text, 0, 6, window=50)
        # Left side: max(0, 0-50)=0, so no padding beyond start
        assert ctx.startswith("ENTITY")
        assert len(ctx) == 6 + 50

    def test_window_at_end(self):
        text = "A" * 100 + "ENTITY"
        ctx = extract_context(text, 100, 106, window=50)
        assert ctx.endswith("ENTITY")
        assert len(ctx) == 50 + 6

    def test_default_window_is_50(self):
        text = "X" * 200
        ctx = extract_context(text, 80, 90)
        # [80-50, 90+50] = [30, 140]
        assert len(ctx) == 110

    def test_short_text(self):
        text = "Hi"
        ctx = extract_context(text, 0, 2, window=50)
        assert ctx == "Hi"

    def test_empty_text(self):
        ctx = extract_context("", 0, 0, window=50)
        assert ctx == ""

    def test_entity_text_preserved_in_context(self):
        text = "Politiet etterforsker saken om DNB ASA i Oslo."
        start = text.index("DNB ASA")
        end = start + len("DNB ASA")
        ctx = extract_context(text, start, end, window=10)
        assert "DNB ASA" in ctx


# ---------------------------------------------------------------------------
# Entity ID tests
# ---------------------------------------------------------------------------

class TestMakeEntityId:
    def test_returns_32_hex(self):
        eid = make_entity_id("a" * 32, "PER", "b" * 32, 10, 20, "kari nordmann")
        assert len(eid) == 32
        assert all(c in "0123456789abcdef" for c in eid)

    def test_deterministic(self):
        args = ("a" * 32, "PER", "b" * 32, 10, 20, "kari nordmann")
        assert make_entity_id(*args) == make_entity_id(*args)

    def test_different_inputs_differ(self):
        base = ("a" * 32, "PER", "b" * 32, 10, 20, "kari nordmann")
        other = ("a" * 32, "PER", "b" * 32, 10, 20, "per hansen")
        assert make_entity_id(*base) != make_entity_id(*other)

    def test_different_type_differs(self):
        a = make_entity_id("a" * 32, "PER", "b" * 32, 10, 20, "oslo")
        b = make_entity_id("a" * 32, "LOC", "b" * 32, 10, 20, "oslo")
        assert a != b


# ---------------------------------------------------------------------------
# Writer tests (using synthetic entity dicts)
# ---------------------------------------------------------------------------

def _make_entity(
    doc_id: str = "a" * 32,
    chunk_id: str = "b" * 32,
    text: str = "Kari Nordmann",
    normalized: str = "Kari Nordmann",
    entity_type: str = "PER",
    char_start: int = 10,
    char_end: int = 23,
    context: str = "sa at Kari Nordmann forklarte seg",
    count: int = 1,
    positions: list | None = None,
) -> dict:
    if positions is None:
        positions = [{
            "chunk_id": chunk_id,
            "char_start": char_start,
            "char_end": char_end,
            "page_num": 0,
            "source_unit_kind": "pdf_page",
        }]
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "normalized": normalized,
        "type": entity_type,
        "char_start": char_start,
        "char_end": char_end,
        "context": context,
        "count": count,
        "positions": positions,
    }


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    return d


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


class TestWriteEntitiesParquet:
    def test_creates_file(self, data_dir, con):
        entities = [_make_entity()]
        path = write_entities_parquet(entities, "testrun", data_dir, con)
        assert path.exists()

    def test_schema_valid(self, data_dir, con):
        entities = [_make_entity()]
        path = write_entities_parquet(entities, "testrun", data_dir, con)
        table = pq.read_table(path)
        errors = validate(table, ENTITIES_SCHEMA)
        assert errors == [], errors

    def test_contract_valid(self, data_dir, con):
        entities = [_make_entity()]
        path = write_entities_parquet(entities, "testrun", data_dir, con)
        table = pq.read_table(path)
        errors = validate_contract_rules(table, "entities")
        assert errors == [], errors

    def test_entity_id_is_hex32(self, data_dir, con):
        entities = [_make_entity()]
        path = write_entities_parquet(entities, "testrun", data_dir, con)
        table = pq.read_table(path)
        eid = table.column("entity_id")[0].as_py()
        assert len(eid) == 32
        assert all(c in "0123456789abcdef" for c in eid)

    def test_run_id_attached(self, data_dir, con):
        entities = [_make_entity()]
        path = write_entities_parquet(entities, "myrun123", data_dir, con)
        table = pq.read_table(path)
        assert table.column("run_id")[0].as_py() == "myrun123"

    def test_positions_nested(self, data_dir, con):
        entities = [_make_entity(count=2, positions=[
            {"chunk_id": "b" * 32, "char_start": 10, "char_end": 23,
             "page_num": 0, "source_unit_kind": "pdf_page"},
            {"chunk_id": "c" * 32, "char_start": 5, "char_end": 18,
             "page_num": 1, "source_unit_kind": "pdf_page"},
        ])]
        path = write_entities_parquet(entities, "testrun", data_dir, con)
        table = pq.read_table(path)
        positions = table.column("positions")[0].as_py()
        assert len(positions) == 2
        assert positions[0]["chunk_id"] == "b" * 32
        assert positions[1]["chunk_id"] == "c" * 32

    def test_count_matches_positions(self, data_dir, con):
        entities = [_make_entity()]
        path = write_entities_parquet(entities, "testrun", data_dir, con)
        table = pq.read_table(path)
        row = table.to_pylist()[0]
        assert row["count"] == len(row["positions"])

    def test_unique_entity_ids(self, data_dir, con):
        # Two different entities should get different IDs
        e1 = _make_entity(text="Kari", normalized="Kari", char_start=0, char_end=4)
        e2 = _make_entity(text="Per", normalized="Per", char_start=10, char_end=13)
        path = write_entities_parquet([e1, e2], "testrun", data_dir, con)
        table = pq.read_table(path)
        ids = table.column("entity_id").to_pylist()
        assert len(set(ids)) == 2

    def test_duckdb_registration(self, data_dir, con):
        entities = [_make_entity()]
        write_entities_parquet(entities, "testrun", data_dir, con)
        result = con.execute("SELECT COUNT(*) FROM entities").fetchone()
        assert result[0] == 1

    def test_empty_entities_no_file(self, data_dir, con):
        path = write_entities_parquet([], "testrun", data_dir, con)
        assert not path.exists()

    def test_multiple_entities_all_contract_valid(self, data_dir, con):
        entities = [
            _make_entity(text="Kari", normalized="Kari", char_start=0, char_end=4),
            _make_entity(
                text="Oslo", normalized="Oslo", entity_type="LOC",
                char_start=20, char_end=24,
                context="arbeider i Oslo kommune",
            ),
            _make_entity(
                text="91234567", normalized="91234567", entity_type="COMM",
                char_start=30, char_end=38,
                context="ring 91234567 for info",
                chunk_id="c" * 32,
            ),
        ]
        path = write_entities_parquet(entities, "testrun", data_dir, con)
        table = pq.read_table(path)
        errors = validate_contract_rules(table, "entities")
        assert errors == [], errors
        assert table.num_rows == 3


# ---------------------------------------------------------------------------
# Full pipeline integration (using synthetic chunks.parquet)
# ---------------------------------------------------------------------------

def _write_synthetic_chunks(data_dir: Path, run_id: str, con: duckdb.DuckDBPyConnection):
    """Write a minimal chunks.parquet with text containing known entities."""
    chunk_text = (
        "Politiet etterforsker saken. Kari Nordmann forklarte at "
        "DNB ASA overførte penger til konto NO9386011117947. "
        "Ring 91234567 for mer informasjon. Bil AB12345 observert."
    )
    rows = [{
        "run_id": run_id,
        "chunk_id": "b" * 32,
        "doc_id": "a" * 32,
        "chunk_index": 0,
        "text": chunk_text,
        "source_unit_kind": "pdf_page",
        "page_num": 0,
    }]
    arrays = {field.name: [] for field in CHUNKS_SCHEMA}
    for row in rows:
        for field in CHUNKS_SCHEMA:
            arrays[field.name].append(row[field.name])
    table = pa.table(arrays, schema=CHUNKS_SCHEMA)
    ingestion_dir = get_ingestion_run_output_dir(data_dir, run_id)
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = ingestion_dir / "chunks.parquet"
    pq.write_table(table, chunks_path)
    con.execute(f"CREATE OR REPLACE TABLE chunks AS SELECT * FROM '{chunks_path}'")
    return chunk_text


class TestFullPipeline:
    """Integration tests using run_extraction on synthetic chunks."""

    def test_imports_cleanly(self):
        """Verify the full pipeline can be imported without errors."""
        from src.extraction.run import run_extraction  # noqa: F401

    def test_run_extraction_produces_entities_parquet(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "integrationtest"
        _write_synthetic_chunks(data_dir, run_id, con)
        run_extraction(data_dir, run_id, con)
        entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"
        assert entities_path.exists()

    def test_entities_schema_valid(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "integrationtest"
        _write_synthetic_chunks(data_dir, run_id, con)
        run_extraction(data_dir, run_id, con)
        entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"
        table = pq.read_table(entities_path)
        errors = validate(table, ENTITIES_SCHEMA)
        assert errors == [], errors

    def test_entities_contract_valid(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "integrationtest"
        _write_synthetic_chunks(data_dir, run_id, con)
        run_extraction(data_dir, run_id, con)
        entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"
        table = pq.read_table(entities_path)
        errors = validate_contract_rules(table, "entities")
        assert errors == [], errors

    def test_all_entities_have_context(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "integrationtest"
        _write_synthetic_chunks(data_dir, run_id, con)
        run_extraction(data_dir, run_id, con)
        entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"
        table = pq.read_table(entities_path)
        for row in table.to_pylist():
            assert row["context"], f"Empty context for entity {row['entity_id']}"

    def test_primary_span_in_positions(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "integrationtest"
        _write_synthetic_chunks(data_dir, run_id, con)
        run_extraction(data_dir, run_id, con)
        entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"
        table = pq.read_table(entities_path)
        for row in table.to_pylist():
            positions = row["positions"]
            assert any(
                p["chunk_id"] == row["chunk_id"]
                and p["char_start"] == row["char_start"]
                and p["char_end"] == row["char_end"]
                for p in positions
            ), f"Primary span missing from positions for entity {row['entity_id']}"

    def test_count_equals_positions_length(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "integrationtest"
        _write_synthetic_chunks(data_dir, run_id, con)
        run_extraction(data_dir, run_id, con)
        entities_path = get_extraction_run_output_dir(data_dir, run_id) / "entities.parquet"
        table = pq.read_table(entities_path)
        for row in table.to_pylist():
            assert row["count"] == len(row["positions"])

    def test_duckdb_entities_registered(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "integrationtest"
        _write_synthetic_chunks(data_dir, run_id, con)
        run_extraction(data_dir, run_id, con)
        count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count > 0

    def test_no_chunks_returns_run_id(self, data_dir, con):
        from src.extraction.run import run_extraction
        result = run_extraction(data_dir, "emptyrun", con)
        assert result == "emptyrun"
        entities_path = get_extraction_run_output_dir(data_dir, "emptyrun") / "entities.parquet"
        assert not entities_path.exists()

    def test_returns_run_id(self, data_dir, con):
        from src.extraction.run import run_extraction
        run_id = "myrun42"
        _write_synthetic_chunks(data_dir, run_id, con)
        result = run_extraction(data_dir, run_id, con)
        assert result == run_id

    def test_deterministic_across_runs(self, data_dir, con):
        """Same input should produce identical entity IDs."""
        from src.extraction.run import run_extraction
        run_id = "dettest"

        # First run
        dir1 = data_dir / "run1"
        dir1.mkdir()
        con1 = duckdb.connect()
        _write_synthetic_chunks(dir1, run_id, con1)
        run_extraction(dir1, run_id, con1)
        t1 = pq.read_table(get_extraction_run_output_dir(dir1, run_id) / "entities.parquet")

        # Second run
        dir2 = data_dir / "run2"
        dir2.mkdir()
        con2 = duckdb.connect()
        _write_synthetic_chunks(dir2, run_id, con2)
        run_extraction(dir2, run_id, con2)
        t2 = pq.read_table(get_extraction_run_output_dir(dir2, run_id) / "entities.parquet")

        ids1 = sorted(t1.column("entity_id").to_pylist())
        ids2 = sorted(t2.column("entity_id").to_pylist())
        assert ids1 == ids2
