"""Extraction stage orchestrator — NER, normalization, dedup, context, entities.parquet.

Reads chunks.parquet, runs NER + regex extraction on each chunk, and produces
a unified mention stream with char-level provenance.  Later steps
add normalization, dedup, context extraction, and parquet writing.
"""

import logging
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from src.extraction.ner import build_ner, extract_ner_mentions
from src.extraction.regex_supplements import (
    extract_regex_mentions,
    filter_overlapping_with_ner,
)

logger = logging.getLogger(__name__)


def run_mention_extraction(
    data_dir: Path,
    run_id: str,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[dict]:
    """Extract entity mentions from all chunks in a run.

    Loads chunks.parquet, runs NER + regex on each chunk in deterministic
    order (doc_id, chunk_index), and returns a unified mention list.

    Args:
        data_dir: Directory containing chunks.parquet.
        run_id: Run identifier to filter chunks.
        con: Optional DuckDB connection.

    Returns:
        List of mention dicts, each with keys: doc_id, chunk_id, text, type,
        char_start, char_end, page_num, source_unit_kind, source.
    """
    if con is None:
        con = duckdb.connect()

    data_dir = Path(data_dir)
    chunks_path = data_dir / "chunks.parquet"

    if not chunks_path.exists():
        logger.warning("No chunks.parquet found at %s", chunks_path)
        return []

    chunks = _load_chunks(chunks_path, run_id, con)
    if not chunks:
        logger.warning("No chunks found for run_id=%s", run_id)
        return []

    logger.info("Extracting mentions from %d chunks (run_id=%s)", len(chunks), run_id)

    # Load NER pipeline once before the loop
    ner_pipe = build_ner()

    all_mentions: list[dict] = []
    counts = {"ner": 0, "regex": 0}
    type_counts: dict[str, int] = {}

    for chunk in chunks:
        chunk_mentions = _extract_chunk_mentions(chunk, ner_pipe)
        for m in chunk_mentions:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
            type_counts[m["type"]] = type_counts.get(m["type"], 0) + 1
        all_mentions.extend(chunk_mentions)

    logger.info(
        "Mention extraction complete: %d mentions (ner=%d, regex=%d)",
        len(all_mentions), counts["ner"], counts["regex"],
    )
    logger.info("Mentions by type: %s", type_counts)

    return all_mentions


def _load_chunks(
    chunks_path: Path, run_id: str, con: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """Load chunks for a run, sorted by (doc_id, chunk_index)."""
    con.execute(
        f"CREATE OR REPLACE TABLE chunks AS SELECT * FROM '{chunks_path}'"
    )
    rows = con.execute(
        "SELECT chunk_id, doc_id, text, page_num, source_unit_kind "
        "FROM chunks WHERE run_id = ? "
        "ORDER BY doc_id, chunk_index",
        [run_id],
    ).fetchall()

    return [
        {
            "chunk_id": r[0],
            "doc_id": r[1],
            "text": r[2],
            "page_num": r[3],
            "source_unit_kind": r[4],
        }
        for r in rows
    ]


def _extract_chunk_mentions(chunk: dict, ner_pipe: object) -> list[dict]:
    """Run NER + regex on a single chunk and merge results."""
    text = chunk["text"]
    doc_id = chunk["doc_id"]
    chunk_id = chunk["chunk_id"]
    page_num = chunk["page_num"]
    source_unit_kind = chunk["source_unit_kind"]

    # NER extraction
    ner_mentions = extract_ner_mentions(
        text, doc_id, chunk_id, page_num, source_unit_kind, ner_pipe,
    )

    # Regex extraction, filtered to not overlap with NER spans
    regex_mentions = extract_regex_mentions(
        text, doc_id, chunk_id, page_num, source_unit_kind,
    )
    regex_mentions = filter_overlapping_with_ner(regex_mentions, ner_mentions)

    return ner_mentions + regex_mentions
