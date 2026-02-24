"""Ingestion stage orchestrator — file discovery and document registration.

Currently covers S2.1 (discovery + registration → docs.parquet).
Will be extended with extraction, normalization, and chunking in S2.2–S2.3.
"""

import logging
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.discovery import discover_documents
from src.ingestion.registration import register_documents

logger = logging.getLogger(__name__)


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
