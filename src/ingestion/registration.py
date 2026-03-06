"""Document registration — compute doc_id, gather metadata, dedup against DuckDB."""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa

from src.shared.schemas import DOCS_SCHEMA

logger = logging.getLogger(__name__)

# Extension → (mime_type, source_unit_kind)
_EXT_META = {
    ".pdf": ("application/pdf", "pdf_page"),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx_paragraph",
    ),
}


def compute_doc_id(file_path: Path) -> str:
    """Compute doc_id as first 32 hex chars of SHA-256 of file contents."""
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return digest[:32]


def build_doc_row(
    file_path: Path,
    case_root: Path,
    run_id: str,
    extracted_at: datetime,
) -> dict:
    """Build a single document metadata row.

    Args:
        file_path: Absolute path to the document file.
        case_root: Case root directory (for relative path computation).
        run_id: Current run identifier.
        extracted_at: UTC timestamp for this ingestion run.

    Returns:
        Dict with all docs.parquet fields.
    """
    doc_id = compute_doc_id(file_path)
    rel_path = file_path.relative_to(case_root).as_posix()
    ext = file_path.suffix.lower()
    mime_type, source_unit_kind = _EXT_META[ext]
    file_size = file_path.stat().st_size

    return {
        "run_id": run_id,
        "doc_id": doc_id,
        "path": rel_path,
        "mime_type": mime_type,
        "source_unit_kind": source_unit_kind,
        "page_count": None,  # Filled during extraction
        "file_size": file_size,
        "extracted_at": extracted_at,
    }


def register_documents(
    file_paths: list[Path],
    case_root: Path,
    run_id: str,
    con: duckdb.DuckDBPyConnection,
) -> pa.Table:
    """Register discovered files, skipping those already present for this run.

    Args:
        file_paths: List of absolute file paths to register.
        case_root: Case root directory.
        run_id: Current run identifier.
        con: DuckDB connection (docs table may or may not exist yet).

    Returns:
        PyArrow Table of newly registered document rows (schema: DOCS_SCHEMA).
    """
    extracted_at = datetime.now(timezone.utc)

    existing_doc_ids, existing_paths = _get_existing_keys(con, run_id)

    rows = []
    skipped = 0
    for fp in file_paths:
        row = build_doc_row(fp, case_root, run_id, extracted_at)
        if row["doc_id"] in existing_doc_ids:
            logger.info("Skipping already-registered doc: %s", row["path"])
            skipped += 1
            continue
        if row["path"] in existing_paths:
            logger.warning(
                "Skipping doc with existing path but new content: %s",
                row["path"],
            )
            skipped += 1
            continue
        rows.append(row)

    logger.info(
        "Registration: %d discovered, %d new, %d skipped (dedup)",
        len(file_paths),
        len(rows),
        skipped,
    )

    if not rows:
        return DOCS_SCHEMA.empty_table()

    # Build columnar arrays from row dicts
    arrays = {field.name: [] for field in DOCS_SCHEMA}
    for row in rows:
        for field in DOCS_SCHEMA:
            arrays[field.name].append(row[field.name])

    return pa.table(arrays, schema=DOCS_SCHEMA)


def _get_existing_keys(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> tuple[set[str], set[str]]:
    """Query existing doc_ids and paths for a given run_id from the docs table."""
    try:
        result = con.execute(
            "SELECT doc_id, path FROM docs WHERE run_id = ?", [run_id]
        ).fetchall()
        doc_ids = {row[0] for row in result}
        paths = {row[1] for row in result}
        return doc_ids, paths
    except duckdb.CatalogException:
        # Table doesn't exist yet — no duplicates possible
        return set(), set()
