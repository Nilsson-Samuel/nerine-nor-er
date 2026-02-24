"""Ingestion stage orchestrator — discovery, registration, extraction, normalization.

Covers S2.1 (discovery + registration → docs.parquet) and
S2.2 (extraction + normalization → text units with page_count backfill).
Will be extended with chunking in S2.3.
"""

import logging
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.discovery import discover_documents
from src.ingestion.extraction import extract_docx_units, extract_pdf_units
from src.ingestion.normalization import normalize_text
from src.ingestion.registration import register_documents
from src.shared.schemas import DOCS_SCHEMA

logger = logging.getLogger(__name__)

# Threshold for flagging extraction as too short (chars after normalization)
_MIN_EXTRACTED_CHARS = 20


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
