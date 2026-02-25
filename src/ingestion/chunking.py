"""Deterministic chunking of normalized text units.

Splits extracted text into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter. Each chunk gets a stable chunk_id
derived from its doc_id and positional index.
"""

import hashlib
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Chunking parameters — kept as module constants per project convention
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " "]


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Build a text splitter with configured size, overlap, and separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a stable chunk_id from doc_id and chunk_index.

    Uses SHA-256 of "{doc_id}:{chunk_index}" truncated to 32 hex chars.
    Deterministic across runs for the same input.
    """
    raw = f"{doc_id}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def chunk_document(
    doc_id: str,
    units: list[dict],
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict]:
    """Chunk all normalized text units for a single document.

    Concatenates all unit texts with double newlines (to let the splitter
    respect paragraph boundaries), then splits into chunks. Each chunk
    inherits the source_unit_kind from the first unit and gets the
    page_num of the unit where it starts.

    Args:
        doc_id: Document identifier.
        units: Normalized text units from extraction. Each has
            page_num, text, source_unit_kind.
        splitter: Configured text splitter.

    Returns:
        List of chunk row dicts ready for parquet writing.
    """
    if not units:
        return []

    # Build the full text and track character offsets for each unit
    # so we can map chunk start positions back to page_num/source_unit_kind
    unit_starts: list[int] = []
    parts: list[str] = []
    offset = 0
    for unit in units:
        unit_starts.append(offset)
        text = unit["text"]
        parts.append(text)
        offset += len(text) + 2  # +2 for the "\n\n" join separator

    full_text = "\n\n".join(parts)

    if not full_text.strip():
        logger.warning("No text to chunk for doc_id=%s", doc_id)
        return []

    # Split into chunks
    chunk_texts = splitter.split_text(full_text)

    chunks = []
    for chunk_index, chunk_text in enumerate(chunk_texts):
        if not chunk_text.strip():
            continue

        # Find which unit this chunk starts in
        chunk_start_pos = full_text.find(chunk_text)
        page_num, source_unit_kind = _locate_unit(
            chunk_start_pos, unit_starts, units
        )

        chunks.append({
            "chunk_id": make_chunk_id(doc_id, chunk_index),
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "source_unit_kind": source_unit_kind,
            "page_num": page_num,
        })

    return chunks


def _locate_unit(
    char_pos: int, unit_starts: list[int], units: list[dict]
) -> tuple[int, str]:
    """Find the page_num and source_unit_kind for a character position.

    Walks through unit start offsets to find which unit contains char_pos.
    Falls back to the first unit if char_pos is before all starts.
    """
    matched_idx = 0
    for i, start in enumerate(unit_starts):
        if start <= char_pos:
            matched_idx = i
        else:
            break
    return units[matched_idx]["page_num"], units[matched_idx]["source_unit_kind"]
