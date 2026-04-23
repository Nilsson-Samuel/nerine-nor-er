"""Entities parquet writer — assembles entity rows and writes schema-valid output.

Computes stable entity_id hashes, attaches context windows, and writes
entities.parquet with the nested positions list<struct> column.
"""

import hashlib
import logging
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.shared.paths import get_extraction_run_output_dir
from src.shared.schemas import ENTITIES_SCHEMA

logger = logging.getLogger(__name__)


def make_entity_id(
    doc_id: str, entity_type: str, chunk_id: str,
    start: int, end: int, normalized: str,
) -> str:
    """Compute a stable 32-char hex entity ID.

    Deterministic hash of the entity's primary mention coordinates so
    the same entity always gets the same ID across runs.
    """
    raw = f"{doc_id}|{entity_type}|{chunk_id}|{start}|{end}|{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def write_entities_parquet(
    entities: list[dict],
    run_id: str,
    data_dir: Path,
    con: duckdb.DuckDBPyConnection,
) -> Path:
    """Write entity rows to entities.parquet with atomic temp+rename.

    Each entity dict must already have: doc_id, text, normalized, type,
    chunk_id, char_start, char_end, context, count, positions.

    Args:
        entities: Deduped entity dicts with context already attached.
        run_id: Run identifier.
        data_dir: Output directory.
        con: DuckDB connection for table registration.

    Returns:
        Path to the written entities.parquet file.
    """
    data_dir = Path(data_dir)
    out_dir = get_extraction_run_output_dir(data_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    entities_path = out_dir / "entities.parquet"

    if not entities:
        logger.info("No entities to write.")
        return entities_path

    # Build entity rows with stable IDs
    rows = _build_rows(entities, run_id)

    # Build columnar arrays in schema order
    arrays: dict[str, list] = {field.name: [] for field in ENTITIES_SCHEMA}
    for row in rows:
        for field in ENTITIES_SCHEMA:
            arrays[field.name].append(row[field.name])

    table = pa.table(arrays, schema=ENTITIES_SCHEMA)

    # Atomic write via temp file
    tmp_path = entities_path.with_suffix(".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.rename(entities_path)

    # Register in DuckDB for downstream blocking
    con.execute(
        f"CREATE OR REPLACE TABLE entities AS SELECT * FROM '{entities_path}'"
    )

    logger.info("Wrote %d entity rows to %s", len(rows), entities_path)
    return entities_path


def _build_rows(entities: list[dict], run_id: str) -> list[dict]:
    """Attach run_id and entity_id to each entity dict."""
    rows: list[dict] = []
    for e in entities:
        entity_id = make_entity_id(
            e["doc_id"], e["type"], e["chunk_id"],
            e["char_start"], e["char_end"], e["normalized"],
        )
        rows.append({
            "run_id": run_id,
            "entity_id": entity_id,
            "doc_id": e["doc_id"],
            "chunk_id": e["chunk_id"],
            "text": e["text"],
            "normalized": e["normalized"],
            "type": e["type"],
            "char_start": e["char_start"],
            "char_end": e["char_end"],
            "context": e["context"],
            "count": e["count"],
            "positions": e["positions"],
        })
    return rows
