"""Extended mention context for the HITL inspector.

Extraction stores only a ±50 character context per entity, cut from a single
chunk. Reviewers need more than that to judge whether two mentions refer to the
same real-world entity, so this module rebuilds each document's text from its
overlapping chunks and cuts arbitrarily wide windows around any mention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.shared.paths import get_ingestion_run_output_dir


logger = logging.getLogger(__name__)

# Upper bound on the text shared by two consecutive chunks. Ingestion uses a
# 50 character overlap; the margin covers separators the splitter keeps.
_MAX_STITCH_OVERLAP = 200
# Shorter suffix/prefix matches are treated as coincidence, not overlap.
_MIN_STITCH_OVERLAP = 5
# Mentions shown per entity before the rest are summarised as a count.
MAX_MENTIONS_PER_ENTITY = 5

_CHUNK_COLUMNS = ["chunk_id", "doc_id", "chunk_index", "text"]
_PARQUET_READ_EXCEPTIONS = (OSError, pa.ArrowInvalid, pl.exceptions.PolarsError)


@dataclass(frozen=True, slots=True)
class MentionWindow:
    """One mention split into the text before it, the mention, and the text after."""

    before: str
    mention: str
    after: str
    clipped_start: bool
    clipped_end: bool


@dataclass(frozen=True, slots=True)
class DocTextIndex:
    """Stitched document texts and where each chunk starts inside them."""

    doc_texts: dict[str, str]
    chunk_offsets: dict[str, tuple[str, int]]


def load_run_chunks(data_dir: Path, run_id: str) -> pl.DataFrame:
    """Load chunk texts for one run, ordered by document and chunk index.

    Returns an empty frame if chunks.parquet is missing or unreadable.
    """
    chunks_path = get_ingestion_run_output_dir(data_dir, run_id) / "chunks.parquet"
    if not chunks_path.exists():
        return pl.DataFrame()

    try:
        chunks = pl.from_arrow(
            pq.read_table(
                chunks_path,
                columns=_CHUNK_COLUMNS,
                filters=[("run_id", "==", run_id)],
            )
        )
    except _PARQUET_READ_EXCEPTIONS as exc:
        logger.warning("Could not read chunks.parquet: %s", exc)
        return pl.DataFrame()

    return chunks.sort(["doc_id", "chunk_index"])


def _suffix_prefix_overlap(left: str, right: str) -> int:
    """Length of the longest suffix of ``left`` that is a prefix of ``right``."""
    upper = min(len(left), len(right), _MAX_STITCH_OVERLAP)
    for size in range(upper, _MIN_STITCH_OVERLAP - 1, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def stitch_chunk_texts(texts: list[str]) -> tuple[str, list[int]]:
    """Rebuild one document's text from its ordered, overlapping chunks.

    Consecutive chunks share up to ingestion's chunk overlap. That shared text
    is kept once; chunks without a detectable overlap are joined by a newline.

    Returns:
        Tuple of ``(document_text, chunk_start_offsets)`` where offset ``i`` is
        the position of chunk ``i``'s first character in the document text.
    """
    doc_text = ""
    offsets: list[int] = []
    for text in texts:
        if not doc_text:
            offsets.append(0)
            doc_text = text
            continue

        overlap = _suffix_prefix_overlap(doc_text, text)
        if overlap:
            offsets.append(len(doc_text) - overlap)
            doc_text += text[overlap:]
        else:
            doc_text += "\n"
            offsets.append(len(doc_text))
            doc_text += text

    return doc_text, offsets


def build_doc_text_index(chunks: pl.DataFrame) -> DocTextIndex:
    """Stitch every document in a chunk frame from load_run_chunks."""
    doc_texts: dict[str, str] = {}
    chunk_offsets: dict[str, tuple[str, int]] = {}
    if chunks.is_empty():
        return DocTextIndex(doc_texts, chunk_offsets)

    for (doc_id,), doc_chunks in chunks.group_by("doc_id", maintain_order=True):
        doc_text, offsets = stitch_chunk_texts(doc_chunks["text"].to_list())
        doc_texts[doc_id] = doc_text
        for chunk_id, offset in zip(doc_chunks["chunk_id"].to_list(), offsets):
            chunk_offsets[chunk_id] = (doc_id, offset)

    return DocTextIndex(doc_texts, chunk_offsets)


def cut_mention_window(doc_text: str, start: int, end: int, window: int) -> MentionWindow:
    """Cut ``window`` characters on each side of the span ``[start, end)``."""
    left = max(0, start - window)
    right = min(len(doc_text), end + window)
    return MentionWindow(
        before=doc_text[left:start],
        mention=doc_text[start:end],
        after=doc_text[end:right],
        clipped_start=left > 0,
        clipped_end=right < len(doc_text),
    )


def entity_mention_windows(
    record: dict[str, Any],
    index: DocTextIndex,
    window: int,
) -> list[MentionWindow]:
    """Build context windows for every mention merged into one entity row.

    Uses the ``positions`` provenance list when present and falls back to the
    primary ``chunk_id``/``char_start``/``char_end`` span. Mentions whose chunk
    is not in the index are skipped, and mentions that land on the same
    document span (from overlapping chunks) are shown once.
    """
    positions = record.get("positions") or [
        {
            "chunk_id": record.get("chunk_id"),
            "char_start": record.get("char_start"),
            "char_end": record.get("char_end"),
        }
    ]

    windows: list[MentionWindow] = []
    seen_spans: set[tuple[str, int, int]] = set()
    for position in positions:
        located = index.chunk_offsets.get(position.get("chunk_id"))
        char_start = position.get("char_start")
        char_end = position.get("char_end")
        if located is None or char_start is None or char_end is None:
            continue

        doc_id, offset = located
        span = (doc_id, offset + int(char_start), offset + int(char_end))
        if span in seen_spans:
            continue
        seen_spans.add(span)
        windows.append(cut_mention_window(index.doc_texts[doc_id], span[1], span[2], window))

    return windows
