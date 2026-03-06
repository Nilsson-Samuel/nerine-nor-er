"""Ingestion stage orchestrator — discovery, registration, extraction, normalization, chunking.

Orchestrates the full ingestion pipeline: discover files, register docs,
extract and normalize text, chunk into overlapping segments, and persist
docs.parquet + chunks.parquet with run metadata.
"""

import json
import logging
import statistics
import time
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.chunking import build_splitter, chunk_document
from src.ingestion.discovery import discover_documents
from src.ingestion.extraction import extract_docx_units, extract_pdf_units
from src.ingestion.normalization import normalize_text
from src.ingestion.registration import register_documents
from src.shared.schemas import CHUNKS_SCHEMA, DOCS_SCHEMA

logger = logging.getLogger(__name__)

# Threshold for flagging extraction as too short (chars after normalization)
_MIN_EXTRACTED_CHARS = 20


def run_ingestion(
    case_root: Path,
    data_dir: Path,
    run_id: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> str:
    """Run the full ingestion pipeline: discover → register → extract → chunk → write.

    Args:
        case_root: Root directory containing input PDF/DOCX files.
        data_dir: Output directory for parquet files (e.g. data/processed/).
        run_id: Optional run identifier. Generated as 32-char hex if not provided.
        con: Optional DuckDB connection. Created in-memory if not provided.

    Returns:
        The run_id used for this execution.
    """
    t0 = time.monotonic()

    if run_id is None:
        run_id = uuid4().hex[:32]
    if con is None:
        con = duckdb.connect()

    # Step 1–2: Discover and register
    run_id = run_discovery_and_registration(case_root, data_dir, run_id, con)

    # Step 3–5: Extract and normalize
    units_by_doc = run_extraction_and_normalization(case_root, data_dir, run_id, con)

    if not units_by_doc:
        logger.warning("No text units extracted — skipping chunking.")
        _write_run_metadata(data_dir, run_id, 0, 0, [], time.monotonic() - t0)
        return run_id

    # Step 6: Chunk all documents
    all_chunks = _run_chunking(units_by_doc, run_id)

    # Step 7: Write chunks.parquet
    _write_chunks_parquet(all_chunks, run_id, data_dir, con)

    # Run metadata
    chunks_per_doc = _chunks_per_doc(all_chunks)
    doc_count = len(units_by_doc)
    _write_run_metadata(
        data_dir, run_id, doc_count, len(all_chunks),
        chunks_per_doc, time.monotonic() - t0,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Ingestion complete: %d docs, %d chunks in %.1fs (run_id=%s)",
        doc_count, len(all_chunks), elapsed, run_id,
    )
    return run_id


def _run_chunking(
    units_by_doc: dict[str, list[dict]], run_id: str,
) -> list[dict]:
    """Chunk all documents and return flat list of chunk rows."""
    splitter = build_splitter()
    all_chunks: list[dict] = []

    for doc_id in sorted(units_by_doc.keys()):
        units = units_by_doc[doc_id]
        doc_chunks = chunk_document(doc_id, units, splitter)
        for chunk in doc_chunks:
            chunk["run_id"] = run_id
        all_chunks.extend(doc_chunks)

    logger.info("Chunked %d documents → %d chunks", len(units_by_doc), len(all_chunks))
    return all_chunks


def _write_chunks_parquet(
    chunks: list[dict],
    run_id: str,
    data_dir: Path,
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Write chunk rows to chunks.parquet (atomic temp+rename).

    Deduplicates against existing chunks: if a (run_id, doc_id) pair
    already has chunks on disk, those doc's chunks are skipped to
    prevent duplication on rerun.
    """
    data_dir = Path(data_dir)
    chunks_path = data_dir / "chunks.parquet"

    if not chunks:
        logger.info("No chunks to write.")
        return

    # Filter out chunks for doc_ids already present in this run
    if chunks_path.exists():
        existing = pq.read_table(chunks_path)
        existing_keys = set()
        run_id_col = existing.column("run_id").to_pylist()
        doc_id_col = existing.column("doc_id").to_pylist()
        for r, d in zip(run_id_col, doc_id_col):
            if r == run_id:
                existing_keys.add(d)
        if existing_keys:
            before = len(chunks)
            chunks = [c for c in chunks if c["doc_id"] not in existing_keys]
            skipped = before - len(chunks)
            if skipped:
                logger.info(
                    "Skipped %d chunks (doc already chunked in run %s)",
                    skipped, run_id,
                )

    if not chunks:
        logger.info("No new chunks to write after dedup.")
        return

    # Build columnar arrays in schema order
    arrays: dict[str, list] = {field.name: [] for field in CHUNKS_SCHEMA}
    for row in chunks:
        for field in CHUNKS_SCHEMA:
            arrays[field.name].append(row[field.name])

    table = pa.table(arrays, schema=CHUNKS_SCHEMA)

    # Append to existing file if present
    if chunks_path.exists():
        existing = pq.read_table(chunks_path)
        table = pa.concat_tables([existing, table])

    # Atomic write via temp file
    tmp_path = chunks_path.with_suffix(".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.rename(chunks_path)

    # Register in DuckDB
    con.execute(
        f"CREATE OR REPLACE TABLE chunks AS SELECT * FROM '{chunks_path}'"
    )

    logger.info("Wrote %d chunk rows to %s", len(chunks), chunks_path)


def _chunks_per_doc(chunks: list[dict]) -> list[int]:
    """Count chunks per doc_id for metadata stats."""
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["doc_id"]] = counts.get(c["doc_id"], 0) + 1
    return list(counts.values())


def _write_run_metadata(
    data_dir: Path,
    run_id: str,
    doc_count: int,
    chunk_count: int,
    chunks_per_doc: list[int],
    elapsed_seconds: float,
) -> None:
    """Write run_metadata.json with summary counts and timing."""
    meta = {
        "run_id": run_id,
        "docs_processed": doc_count,
        "chunk_count": chunk_count,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    if chunks_per_doc:
        meta["chunks_per_doc_min"] = min(chunks_per_doc)
        meta["chunks_per_doc_max"] = max(chunks_per_doc)
        meta["chunks_per_doc_median"] = round(statistics.median(chunks_per_doc), 1)

    meta_path = Path(data_dir) / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote run metadata to %s", meta_path)


# ---------------------------------------------------------------------------
# Sub-stage functions (used by run_ingestion and existing tests)
# ---------------------------------------------------------------------------

def run_discovery_and_registration(
    case_root: Path,
    data_dir: Path,
    run_id: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> str:
    """Run file discovery and document registration.

    Discovers PDF/DOCX files under case_root, computes content-hash doc_ids,
    deduplicates against existing records, and persists new rows to
    docs.parquet (append-safe for incremental runs).

    Args:
        case_root: Root directory containing input PDF/DOCX files.
        data_dir: Output directory for parquet files (e.g. data/processed/).
        run_id: Optional run identifier. Generated as 32-char hex if not provided.
        con: Optional DuckDB connection. Created in-memory if not provided.

    Returns:
        The run_id used for this execution.
    """
    if run_id is None:
        run_id = uuid4().hex[:32]
    if con is None:
        con = duckdb.connect()

    case_root = Path(case_root).resolve()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    docs_path = data_dir / "docs.parquet"

    # Load existing parquet into DuckDB for dedup queries
    if docs_path.exists():
        con.execute(
            f"CREATE OR REPLACE TABLE docs AS SELECT * FROM '{docs_path}'"
        )

    # Step 1: Discover files
    file_paths = discover_documents(case_root)
    logger.info("Discovered %d files in %s", len(file_paths), case_root)

    if not file_paths:
        logger.warning("No PDF/DOCX files found in %s", case_root)
        return run_id

    # Step 2: Register documents (with dedup)
    new_docs = register_documents(file_paths, case_root, run_id, con)

    if len(new_docs) == 0:
        logger.info("No new documents to register.")
        return run_id

    # Step 3: Persist docs.parquet (append if file already exists)
    if docs_path.exists():
        existing = pq.read_table(docs_path)
        combined = pa.concat_tables([existing, new_docs])
        pq.write_table(combined, docs_path)
    else:
        pq.write_table(new_docs, docs_path)

    # Re-register in DuckDB after write
    con.execute(
        f"CREATE OR REPLACE TABLE docs AS SELECT * FROM '{docs_path}'"
    )

    logger.info("Wrote %d new doc rows to %s", len(new_docs), docs_path)
    return run_id


def run_extraction_and_normalization(
    case_root: Path,
    data_dir: Path,
    run_id: str,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, list[dict]]:
    """Extract and normalize text from all registered documents for a run.

    For each document in docs.parquet with the given run_id:
    1. Extract text units (page-level for PDF, paragraph-level for DOCX).
    2. Normalize each unit (ftfy, NFC, control-char cleanup, whitespace collapse).
    3. Track extraction warnings per category.
    4. Backfill page_count in docs.parquet.

    Args:
        case_root: Root directory containing input PDF/DOCX files.
        data_dir: Directory containing docs.parquet.
        run_id: Run identifier to filter documents.
        con: Optional DuckDB connection. Created in-memory if not provided.

    Returns:
        Dict mapping doc_id → list of normalized text units.
        Each unit: {page_num: int, text: str, source_unit_kind: str}.
    """
    if con is None:
        con = duckdb.connect()

    case_root = Path(case_root).resolve()
    data_dir = Path(data_dir)
    docs_path = data_dir / "docs.parquet"

    if not docs_path.exists():
        logger.warning("No docs.parquet found at %s", docs_path)
        return {}

    # Load docs for this run
    con.execute(
        f"CREATE OR REPLACE TABLE docs AS SELECT * FROM '{docs_path}'"
    )
    rows = con.execute(
        "SELECT doc_id, path, mime_type FROM docs WHERE run_id = ?", [run_id]
    ).fetchall()

    if not rows:
        logger.warning("No documents found for run_id=%s", run_id)
        return {}

    logger.info("Extracting text from %d documents (run_id=%s)", len(rows), run_id)

    # Warning counters
    warnings: dict[str, int] = {}
    # page_count updates: doc_id → count
    page_counts: dict[str, int] = {}
    # Extracted + normalized units per doc
    result: dict[str, list[dict]] = {}

    for doc_id, rel_path, mime_type in rows:
        abs_path = case_root / rel_path

        if not abs_path.exists():
            _warn(warnings, "file_not_found")
            logger.warning("file_not_found: %s", rel_path)
            result[doc_id] = []
            page_counts[doc_id] = 0
            continue

        # Extract raw text units
        if mime_type == "application/pdf":
            units = extract_pdf_units(abs_path)
        else:
            units = extract_docx_units(abs_path)

        page_counts[doc_id] = len(units)

        # Normalize each unit and check extraction quality
        normalized_units = []
        doc_char_total = 0
        for unit in units:
            clean_text = normalize_text(unit["text"])
            doc_char_total += len(clean_text)
            normalized_units.append({
                "page_num": unit["page_num"],
                "text": clean_text,
                "source_unit_kind": unit["source_unit_kind"],
            })

        # Flag documents with very little extracted text
        if doc_char_total < _MIN_EXTRACTED_CHARS:
            _warn(warnings, "extraction_failed_or_short")
            logger.warning(
                "extraction_failed_or_short: %s (%d chars)", rel_path, doc_char_total
            )

        result[doc_id] = normalized_units

    # Backfill page_count in docs.parquet
    _update_page_counts(docs_path, run_id, page_counts)

    # Log warning summary
    if warnings:
        logger.info("Extraction warnings: %s", warnings)
    logger.info(
        "Extraction complete: %d docs, %d total units",
        len(result),
        sum(len(u) for u in result.values()),
    )

    return result


def _warn(counters: dict[str, int], category: str) -> None:
    """Increment a warning counter."""
    counters[category] = counters.get(category, 0) + 1


def _update_page_counts(
    docs_path: Path, run_id: str, page_counts: dict[str, int]
) -> None:
    """Backfill page_count in docs.parquet for the given run."""
    if not page_counts:
        return

    table = pq.read_table(docs_path)
    rows = table.to_pylist()

    for row in rows:
        if row["run_id"] == run_id and row["doc_id"] in page_counts:
            row["page_count"] = page_counts[row["doc_id"]]

    # Rebuild table with original schema to preserve types
    arrays = {field.name: [] for field in DOCS_SCHEMA}
    for row in rows:
        for field in DOCS_SCHEMA:
            arrays[field.name].append(row[field.name])

    updated = pa.table(arrays, schema=DOCS_SCHEMA)
    pq.write_table(updated, docs_path)
    logger.info("Backfilled page_count for %d docs", len(page_counts))
