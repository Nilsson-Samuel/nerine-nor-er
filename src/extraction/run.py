"""Extraction stage orchestrator — NER, normalization, dedup, context, entities.parquet.

Reads chunks.parquet, runs NER + regex extraction on each chunk, normalizes
mentions by entity type, deduplicates within each document, attaches context
windows, and writes schema-valid entities.parquet with DuckDB registration.
"""

import logging
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from src.extraction.context import extract_context
from src.extraction.dedup import dedup_mentions
from src.extraction.entity_normalizer import normalize_entity
from src.extraction.ner import build_ner, extract_ner_mentions
from src.extraction.regex_supplements import (
    extract_regex_mentions,
    merge_regex_with_ner,
)
from src.extraction.writer import write_entities_parquet
from src.shared.paths import get_extraction_run_output_dir, get_ingestion_run_output_dir
from src.shared.schemas import validate_contract_rules

logger = logging.getLogger(__name__)


def run_extraction(
    data_dir: Path,
    run_id: str,
    con: duckdb.DuckDBPyConnection | None = None,
) -> str:
    """Run the full extraction pipeline: NER → normalize → dedup → context → write.

    Args:
        data_dir: Directory containing chunks.parquet (and output for entities.parquet).
        run_id: Run identifier.
        con: Optional DuckDB connection.

    Returns:
        The run_id used for this execution.
    """
    t0 = time.monotonic()

    if con is None:
        con = duckdb.connect()

    data_dir = Path(data_dir)

    # Step 1: Extract raw mentions from chunks
    mentions = run_mention_extraction(data_dir, run_id, con)
    if not mentions:
        logger.warning("No mentions extracted — skipping remaining steps.")
        return run_id

    # Step 2: Normalize and deduplicate
    entities = run_normalization_and_dedup(mentions)
    if not entities:
        logger.warning("No entities after dedup — skipping write.")
        return run_id

    # Step 3: Attach context windows (needs chunk text lookup)
    chunks_path = get_ingestion_run_output_dir(data_dir, run_id) / "chunks.parquet"
    chunk_texts = _build_chunk_text_lookup(chunks_path, run_id, con)
    _attach_contexts(entities, chunk_texts)

    # Step 4: Write entities.parquet
    entities_path = write_entities_parquet(entities, run_id, data_dir, con)

    # Step 5: Validate contract
    if entities_path.exists():
        table = pq.read_table(entities_path)
        errors = validate_contract_rules(table, "entities")
        if errors:
            logger.error("entities.parquet contract validation failed:")
            for err in errors:
                logger.error("  %s", err)
            raise ValueError(
                f"entities.parquet contract validation failed with {len(errors)} errors"
            )
        logger.info("entities.parquet contract validation passed.")

    elapsed = time.monotonic() - t0
    logger.info(
        "Extraction complete: %d entities in %.1fs (run_id=%s)",
        len(entities), elapsed, run_id,
    )
    return run_id


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
    chunks_path = get_ingestion_run_output_dir(data_dir, run_id) / "chunks.parquet"

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


def run_normalization_and_dedup(mentions: list[dict]) -> list[dict]:
    """Normalize mentions by entity type and deduplicate within each document.

    Args:
        mentions: Raw mention dicts from run_mention_extraction().

    Returns:
        List of deduped entity dicts with normalized text and position provenance.
    """
    if not mentions:
        return []

    # Add normalized text to each mention
    for m in mentions:
        m["normalized"] = normalize_entity(m["text"], m["type"])

    # Skip mentions that normalized to empty
    mentions = [m for m in mentions if m["normalized"]]

    return dedup_mentions(mentions)


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

    # Regex extraction — regex supersets of same-type NER fragments win and
    # replace the covered subspans; all other overlaps defer to NER as before.
    regex_mentions = extract_regex_mentions(
        text, doc_id, chunk_id, page_num, source_unit_kind,
    )
    regex_mentions, ner_mentions = merge_regex_with_ner(regex_mentions, ner_mentions)

    return ner_mentions + regex_mentions


def _build_chunk_text_lookup(
    chunks_path: Path, run_id: str, con: duckdb.DuckDBPyConnection,
) -> dict[str, str]:
    """Build a chunk_id → chunk_text lookup for context extraction."""
    rows = con.execute(
        "SELECT chunk_id, text FROM chunks WHERE run_id = ?", [run_id],
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _attach_contexts(
    entities: list[dict], chunk_texts: dict[str, str],
) -> None:
    """Attach context windows to each entity's primary mention in place."""
    for e in entities:
        chunk_text = chunk_texts.get(e["chunk_id"], "")
        e["context"] = extract_context(chunk_text, e["char_start"], e["char_end"])
