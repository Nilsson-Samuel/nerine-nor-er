"""Tests for extended mention context: chunk stitching and mention windows."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.hitl.context import (
    MentionWindow,
    build_doc_text_index,
    cut_mention_window,
    entity_mention_windows,
    load_run_chunks,
    stitch_chunk_texts,
)
from src.hitl.ui_utils import format_mention_html
from src.shared.paths import get_ingestion_run_output_dir
from src.shared.schemas import CHUNKS_SCHEMA


RUN_ID = "context_test_run"

# One document split into two chunks sharing the text "Ola Nordmann".
CHUNK_0 = "Vitnet forklarte at Ola Nordmann"
CHUNK_1 = "Ola Nordmann kjørte bilen til Bergen."
DOC_TEXT = "Vitnet forklarte at Ola Nordmann kjørte bilen til Bergen."


def _chunk_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "chunk_id": ["c0", "c1", "c2"],
        "doc_id": ["doc_a", "doc_a", "doc_b"],
        "chunk_index": [0, 1, 0],
        "text": [CHUNK_0, CHUNK_1, "Kari Hansen bor i Oslo."],
    })


# ── stitch_chunk_texts ───────────────────────────────────────────────────────


def test_stitch_removes_shared_overlap() -> None:
    doc_text, offsets = stitch_chunk_texts([CHUNK_0, CHUNK_1])
    assert doc_text == DOC_TEXT
    assert offsets == [0, len(CHUNK_0) - len("Ola Nordmann")]


def test_stitch_offsets_locate_every_chunk() -> None:
    chunks = ["alpha beta gamma delta", "gamma delta epsilon zeta", "zeta eta theta"]
    doc_text, offsets = stitch_chunk_texts(chunks)
    for chunk, offset in zip(chunks, offsets):
        assert doc_text[offset:offset + len(chunk)] == chunk


def test_stitch_joins_non_overlapping_chunks_with_newline() -> None:
    doc_text, offsets = stitch_chunk_texts(["First paragraph.", "Second paragraph."])
    assert doc_text == "First paragraph.\nSecond paragraph."
    assert offsets == [0, len("First paragraph.\n")]


def test_stitch_ignores_coincidental_short_overlap() -> None:
    # A 3-character match is below the minimum, so both chunks are kept whole
    doc_text, _ = stitch_chunk_texts(["Han het Ola", "Ola var der"])
    assert doc_text == "Han het Ola\nOla var der"


def test_stitch_empty_input() -> None:
    assert stitch_chunk_texts([]) == ("", [])


# ── build_doc_text_index ─────────────────────────────────────────────────────


def test_doc_text_index_stitches_per_document() -> None:
    index = build_doc_text_index(_chunk_frame())
    assert index.doc_texts["doc_a"] == DOC_TEXT
    assert index.doc_texts["doc_b"] == "Kari Hansen bor i Oslo."
    assert index.chunk_offsets["c1"] == ("doc_a", len(CHUNK_0) - len("Ola Nordmann"))
    assert index.chunk_offsets["c2"] == ("doc_b", 0)


def test_doc_text_index_empty_frame() -> None:
    index = build_doc_text_index(pl.DataFrame())
    assert index.doc_texts == {}
    assert index.chunk_offsets == {}


# ── cut_mention_window ───────────────────────────────────────────────────────


def test_cut_mention_window_splits_around_span() -> None:
    window = cut_mention_window("abc NAME def", 4, 8, 2)
    assert window == MentionWindow("c ", "NAME", " d", True, True)


def test_cut_mention_window_not_clipped_at_document_edges() -> None:
    window = cut_mention_window("abc NAME def", 4, 8, 100)
    assert window.before == "abc "
    assert window.after == " def"
    assert not window.clipped_start
    assert not window.clipped_end


# ── entity_mention_windows ───────────────────────────────────────────────────


def test_mention_window_extends_past_chunk_boundary() -> None:
    index = build_doc_text_index(_chunk_frame())
    # "Bergen" inside chunk c1, chunk-relative offsets
    start = CHUNK_1.index("Bergen")
    record = {"chunk_id": "c1", "char_start": start, "char_end": start + 6}

    [window] = entity_mention_windows(record, index, window=1000)

    assert window.mention == "Bergen"
    # Context reaches back into chunk c0
    assert window.before.startswith("Vitnet forklarte")


def test_mention_windows_cover_all_positions() -> None:
    index = build_doc_text_index(_chunk_frame())
    record = {
        "chunk_id": "c0", "char_start": 0, "char_end": 6,
        "positions": [
            {"chunk_id": "c0", "char_start": 0, "char_end": 6},
            {"chunk_id": "c2", "char_start": 0, "char_end": 11},
        ],
    }
    windows = entity_mention_windows(record, index, window=10)
    assert [w.mention for w in windows] == ["Vitnet", "Kari Hansen"]


def test_mention_windows_dedupe_span_seen_through_chunk_overlap() -> None:
    index = build_doc_text_index(_chunk_frame())
    overlap_start = CHUNK_0.index("Ola Nordmann")
    record = {
        "positions": [
            {"chunk_id": "c0", "char_start": overlap_start, "char_end": overlap_start + 12},
            {"chunk_id": "c1", "char_start": 0, "char_end": 12},
        ],
    }
    windows = entity_mention_windows(record, index, window=5)
    assert len(windows) == 1
    assert windows[0].mention == "Ola Nordmann"


def test_mention_windows_skip_unknown_chunks() -> None:
    index = build_doc_text_index(_chunk_frame())
    record = {"chunk_id": "missing", "char_start": 0, "char_end": 3, "positions": None}
    assert entity_mention_windows(record, index, window=10) == []


# ── load_run_chunks ──────────────────────────────────────────────────────────


def test_load_run_chunks_filters_run_and_sorts(tmp_path: Path) -> None:
    rows = [
        {"run_id": RUN_ID, "chunk_id": "c1", "doc_id": "doc_a", "chunk_index": 1,
         "text": CHUNK_1, "source_unit_kind": "pdf_page", "page_num": 1},
        {"run_id": RUN_ID, "chunk_id": "c0", "doc_id": "doc_a", "chunk_index": 0,
         "text": CHUNK_0, "source_unit_kind": "pdf_page", "page_num": 1},
        {"run_id": "other_run", "chunk_id": "x", "doc_id": "doc_a", "chunk_index": 2,
         "text": "ignored", "source_unit_kind": "pdf_page", "page_num": 1},
    ]
    ingestion_dir = get_ingestion_run_output_dir(tmp_path, RUN_ID)
    ingestion_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CHUNKS_SCHEMA), ingestion_dir / "chunks.parquet"
    )

    chunks = load_run_chunks(tmp_path, RUN_ID)

    assert chunks["chunk_id"].to_list() == ["c0", "c1"]


def test_load_run_chunks_missing_file(tmp_path: Path) -> None:
    assert load_run_chunks(tmp_path, RUN_ID).is_empty()


# ── format_mention_html ──────────────────────────────────────────────────────


def test_format_mention_html_escapes_and_highlights() -> None:
    html = format_mention_html(MentionWindow("<b>x", "A&B", "y>", True, False))
    assert "&lt;b&gt;x" in html
    assert "<b>A&amp;B</b></mark>" in html
    assert "y&gt;" in html
    assert html.count("…") == 1
