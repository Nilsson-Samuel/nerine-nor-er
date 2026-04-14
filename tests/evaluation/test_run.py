"""Integration tests for the evaluation runner and regression checklist."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.run import (
    _build_predicted_mentions,
    _build_labels_table,
    _chunk_start_offsets_from_full_text,
    _match_gold_to_predicted,
    _remap_gold_offsets_to_run_text,
    _remap_gold_doc_ids,
    build_regression_checks,
    get_evaluation_labels_path,
    get_evaluation_report_path,
    run_evaluation,
)
from src.matching.writer import (
    get_matching_run_output_dir,
    get_scored_pairs_output_path,
)
from src.resolution.writer import get_resolved_entities_output_path
from src.shared import schemas
from src.shared.paths import get_blocking_run_output_dir, get_extraction_run_output_dir, get_ingestion_run_output_dir


def _hex32(number: int) -> str:
    """Build one deterministic lowercase 32-char hex identifier."""
    return f"{number:032x}"


def _write_case_artifacts(tmp_path: Path, run_id: str) -> Path:
    """Write a compact completed-run fixture for evaluation tests."""
    data_dir = tmp_path / "eval_data"
    data_dir.mkdir()

    doc_1 = _hex32(1)
    doc_2 = _hex32(2)
    chunk_1 = _hex32(11)
    chunk_2 = _hex32(12)
    chunk_3 = _hex32(13)
    entity_1 = _hex32(21)
    entity_2 = _hex32(22)
    entity_3 = _hex32(23)
    entity_4 = _hex32(24)

    docs = pa.table(
        {
            "run_id": pa.array([run_id, run_id], type=pa.string()),
            "doc_id": pa.array([doc_1, doc_2], type=pa.string()),
            "path": pa.array(["doc_1.docx", "doc_2.docx"], type=pa.string()),
            "mime_type": pa.array(
                [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ],
                type=pa.string(),
            ),
            "source_unit_kind": pa.array(
                ["docx_paragraph", "docx_paragraph"], type=pa.string()
            ),
            "page_count": pa.array([2, 1], type=pa.int32()),
            "file_size": pa.array([100, 100], type=pa.int64()),
            "extracted_at": pa.array(
                [datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)] * 2,
                type=pa.timestamp("us", tz="UTC"),
            ),
        },
        schema=schemas.DOCS_SCHEMA,
    )
    ingestion_dir = get_ingestion_run_output_dir(data_dir, run_id)
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(docs, ingestion_dir / "docs.parquet")

    chunks = pa.table(
        {
            "run_id": pa.array([run_id, run_id, run_id], type=pa.string()),
            "chunk_id": pa.array([chunk_1, chunk_2, chunk_3], type=pa.string()),
            "doc_id": pa.array([doc_1, doc_1, doc_2], type=pa.string()),
            "chunk_index": pa.array([0, 1, 0], type=pa.int32()),
            "text": pa.array(
                [
                    "Alice Johnson met Bob",
                    "met Bob Stone in Oslo.",
                    "B. Stone called Alice.",
                ],
                type=pa.string(),
            ),
            "source_unit_kind": pa.array(
                ["docx_paragraph", "docx_paragraph", "docx_paragraph"],
                type=pa.string(),
            ),
            "page_num": pa.array([0, 1, 0], type=pa.int32()),
        },
        schema=schemas.CHUNKS_SCHEMA,
    )
    pq.write_table(chunks, ingestion_dir / "chunks.parquet")

    entities = pa.table(
        {
            "run_id": pa.array([run_id] * 4, type=pa.string()),
            "entity_id": pa.array(
                [entity_1, entity_2, entity_3, entity_4], type=pa.string()
            ),
            "doc_id": pa.array([doc_1, doc_1, doc_2, doc_2], type=pa.string()),
            "chunk_id": pa.array(
                [chunk_1, chunk_2, chunk_3, chunk_3], type=pa.string()
            ),
            "text": pa.array(
                ["Alice Johnson", "Bob Stone", "B. Stone", "Alice"],
                type=pa.string(),
            ),
            "normalized": pa.array(
                ["alice johnson", "bob stone", "b. stone", "alice"],
                type=pa.string(),
            ),
            "type": pa.array(["PER", "PER", "PER", "PER"], type=pa.string()),
            "char_start": pa.array([0, 4, 0, 16], type=pa.int32()),
            "char_end": pa.array([13, 13, 8, 21], type=pa.int32()),
            "context": pa.array(
                [
                    "Alice Johnson met Bob",
                    "met Bob Stone in Oslo.",
                    "B. Stone called Alice.",
                    "B. Stone called Alice.",
                ],
                type=pa.string(),
            ),
            "count": pa.array([1, 1, 1, 1], type=pa.int32()),
            "positions": pa.array(
                [
                    [
                        {
                            "chunk_id": chunk_1,
                            "char_start": 0,
                            "char_end": 13,
                            "page_num": 0,
                            "source_unit_kind": "docx_paragraph",
                        }
                    ],
                    [
                        {
                            "chunk_id": chunk_2,
                            "char_start": 4,
                            "char_end": 13,
                            "page_num": 1,
                            "source_unit_kind": "docx_paragraph",
                        }
                    ],
                    [
                        {
                            "chunk_id": chunk_3,
                            "char_start": 0,
                            "char_end": 8,
                            "page_num": 0,
                            "source_unit_kind": "docx_paragraph",
                        }
                    ],
                    [
                        {
                            "chunk_id": chunk_3,
                            "char_start": 16,
                            "char_end": 21,
                            "page_num": 0,
                            "source_unit_kind": "docx_paragraph",
                        }
                    ],
                ],
                type=pa.list_(schemas.POSITION_STRUCT_TYPE),
            ),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )
    extraction_dir = get_extraction_run_output_dir(data_dir, run_id)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(entities, extraction_dir / "entities.parquet")

    candidate_pairs = pa.table(
        {
            "run_id": pa.array([run_id] * 6, type=pa.string()),
            "entity_id_a": pa.array(
                [entity_1, entity_1, entity_1, entity_2, entity_2, entity_3],
                type=pa.string(),
            ),
            "entity_id_b": pa.array(
                [entity_2, entity_3, entity_4, entity_3, entity_4, entity_4],
                type=pa.string(),
            ),
            "blocking_methods": pa.array([["faiss"]] * 6, type=pa.list_(pa.string())),
            "blocking_source": pa.array(["faiss"] * 6, type=pa.string()),
            "blocking_method_count": pa.array([1] * 6, type=pa.int8()),
        },
        schema=schemas.CANDIDATE_PAIRS_SCHEMA,
    )
    blocking_dir = get_blocking_run_output_dir(data_dir, run_id)
    blocking_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(candidate_pairs, blocking_dir / "candidate_pairs.parquet")

    matching_dir = get_matching_run_output_dir(data_dir, run_id)
    matching_dir.mkdir(parents=True, exist_ok=True)
    scored_pairs = pa.table(
        {
            "run_id": pa.array([run_id] * 6, type=pa.string()),
            "entity_id_a": pa.array(
                [entity_1, entity_1, entity_1, entity_2, entity_2, entity_3],
                type=pa.string(),
            ),
            "entity_id_b": pa.array(
                [entity_2, entity_3, entity_4, entity_3, entity_4, entity_4],
                type=pa.string(),
            ),
            "score": pa.array([0.8, 0.1, 0.9, 0.9, 0.1, 0.1], type=pa.float32()),
            "model_version": pa.array(["toy_model"] * 6, type=pa.string()),
            "scored_at": pa.array(
                [datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)] * 6,
                type=pa.timestamp("us", tz="UTC"),
            ),
            "blocking_methods": pa.array([["faiss"]] * 6, type=pa.list_(pa.string())),
            "blocking_source": pa.array(["faiss"] * 6, type=pa.string()),
            "blocking_method_count": pa.array([1] * 6, type=pa.int8()),
            "shap_top5": pa.array(
                [[]] * 6, type=schemas.SCORED_PAIRS_SCHEMA.field("shap_top5").type
            ),
        },
        schema=schemas.SCORED_PAIRS_SCHEMA,
    )
    pq.write_table(scored_pairs, get_scored_pairs_output_path(data_dir, run_id))

    resolved_entities = pa.Table.from_pylist(
        [
            _resolved_row(
                run_id,
                _hex32(101),
                entity_1,
                doc_1,
                "alice johnson",
                3,
                0.99,
                "auto_merge",
                [doc_1, doc_2],
                _hex32(201),
            ),
            _resolved_row(
                run_id,
                _hex32(101),
                entity_2,
                doc_1,
                "alice johnson",
                3,
                0.99,
                "auto_merge",
                [doc_1, doc_2],
                _hex32(201),
            ),
            _resolved_row(
                run_id,
                _hex32(101),
                entity_4,
                doc_2,
                "alice johnson",
                3,
                0.99,
                "auto_merge",
                [doc_1, doc_2],
                _hex32(201),
            ),
            _resolved_row(
                run_id,
                _hex32(102),
                entity_3,
                doc_2,
                "b. stone",
                1,
                0.2,
                "keep_separate",
                [doc_2],
                _hex32(202),
            ),
        ],
        schema=schemas.RESOLVED_ENTITIES_SCHEMA,
    )
    resolved_path = get_resolved_entities_output_path(data_dir, run_id)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(resolved_entities, resolved_path)

    return data_dir


def _resolved_row(
    run_id: str,
    cluster_id: str,
    entity_id: str,
    doc_id: str,
    canonical_name: str,
    cluster_size: int,
    confidence: float,
    route_action: str,
    doc_ids: list[str],
    component_id: str,
) -> dict[str, object]:
    """Build one strict-schema resolved row."""
    return {
        "run_id": run_id,
        "cluster_id": cluster_id,
        "entity_id": entity_id,
        "doc_id": doc_id,
        "entity_type": "PER",
        "canonical_name": canonical_name,
        "canonical_type": "PER",
        "cluster_size": cluster_size,
        "confidence": confidence,
        "needs_review": False,
        "route_action": route_action,
        "clustering_method": "pivot",
        "component_id": component_id,
        "doc_ids": sorted(doc_ids),
        "most_recent_doc_id": None,
        "most_recent_doc_date": None,
        "attributes": {},
    }


def _write_subset_metric_case_artifacts(tmp_path: Path, run_id: str) -> Path:
    """Write a run where one merged entity lacks any trusted gold bridge."""
    data_dir = tmp_path / "subset_eval_data"
    data_dir.mkdir()

    doc_1 = _hex32(301)
    chunk_1 = _hex32(311)
    entity_1 = _hex32(321)
    entity_2 = _hex32(322)
    entity_3 = _hex32(323)

    docs = pa.table(
        {
            "run_id": pa.array([run_id], type=pa.string()),
            "doc_id": pa.array([doc_1], type=pa.string()),
            "path": pa.array(["subset_case.docx"], type=pa.string()),
            "mime_type": pa.array(
                [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ],
                type=pa.string(),
            ),
            "source_unit_kind": pa.array(["docx_paragraph"], type=pa.string()),
            "page_count": pa.array([1], type=pa.int32()),
            "file_size": pa.array([100], type=pa.int64()),
            "extracted_at": pa.array(
                [datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)],
                type=pa.timestamp("us", tz="UTC"),
            ),
        },
        schema=schemas.DOCS_SCHEMA,
    )
    ingestion_dir = get_ingestion_run_output_dir(data_dir, run_id)
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(docs, ingestion_dir / "docs.parquet")

    chunk_text = "Alice Brown met A. Brown and Observer."
    chunks = pa.table(
        {
            "run_id": pa.array([run_id], type=pa.string()),
            "chunk_id": pa.array([chunk_1], type=pa.string()),
            "doc_id": pa.array([doc_1], type=pa.string()),
            "chunk_index": pa.array([0], type=pa.int32()),
            "text": pa.array([chunk_text], type=pa.string()),
            "source_unit_kind": pa.array(["docx_paragraph"], type=pa.string()),
            "page_num": pa.array([0], type=pa.int32()),
        },
        schema=schemas.CHUNKS_SCHEMA,
    )
    pq.write_table(chunks, ingestion_dir / "chunks.parquet")

    entities = pa.table(
        {
            "run_id": pa.array([run_id] * 3, type=pa.string()),
            "entity_id": pa.array([entity_1, entity_2, entity_3], type=pa.string()),
            "doc_id": pa.array([doc_1, doc_1, doc_1], type=pa.string()),
            "chunk_id": pa.array([chunk_1, chunk_1, chunk_1], type=pa.string()),
            "text": pa.array(
                ["Alice Brown", "A. Brown", "Observer"],
                type=pa.string(),
            ),
            "normalized": pa.array(
                ["alice brown", "a. brown", "observer"],
                type=pa.string(),
            ),
            "type": pa.array(["PER", "PER", "PER"], type=pa.string()),
            "char_start": pa.array([0, 16, 29], type=pa.int32()),
            "char_end": pa.array([11, 24, 37], type=pa.int32()),
            "context": pa.array([chunk_text] * 3, type=pa.string()),
            "count": pa.array([1, 1, 1], type=pa.int32()),
            "positions": pa.array(
                [
                    [
                        {
                            "chunk_id": chunk_1,
                            "char_start": 0,
                            "char_end": 11,
                            "page_num": 0,
                            "source_unit_kind": "docx_paragraph",
                        }
                    ],
                    [
                        {
                            "chunk_id": chunk_1,
                            "char_start": 16,
                            "char_end": 24,
                            "page_num": 0,
                            "source_unit_kind": "docx_paragraph",
                        }
                    ],
                    [
                        {
                            "chunk_id": chunk_1,
                            "char_start": 29,
                            "char_end": 37,
                            "page_num": 0,
                            "source_unit_kind": "docx_paragraph",
                        }
                    ],
                ],
                type=pa.list_(schemas.POSITION_STRUCT_TYPE),
            ),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )
    extraction_dir = get_extraction_run_output_dir(data_dir, run_id)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(entities, extraction_dir / "entities.parquet")

    candidate_pairs = pa.table(
        {
            "run_id": pa.array([run_id] * 3, type=pa.string()),
            "entity_id_a": pa.array([entity_1, entity_1, entity_2], type=pa.string()),
            "entity_id_b": pa.array([entity_2, entity_3, entity_3], type=pa.string()),
            "blocking_methods": pa.array([["faiss"]] * 3, type=pa.list_(pa.string())),
            "blocking_source": pa.array(["faiss"] * 3, type=pa.string()),
            "blocking_method_count": pa.array([1] * 3, type=pa.int8()),
        },
        schema=schemas.CANDIDATE_PAIRS_SCHEMA,
    )
    blocking_dir = get_blocking_run_output_dir(data_dir, run_id)
    blocking_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(candidate_pairs, blocking_dir / "candidate_pairs.parquet")

    matching_dir = get_matching_run_output_dir(data_dir, run_id)
    matching_dir.mkdir(parents=True, exist_ok=True)
    scored_pairs = pa.table(
        {
            "run_id": pa.array([run_id] * 3, type=pa.string()),
            "entity_id_a": pa.array([entity_1, entity_1, entity_2], type=pa.string()),
            "entity_id_b": pa.array([entity_2, entity_3, entity_3], type=pa.string()),
            "score": pa.array([0.95, 0.95, 0.95], type=pa.float32()),
            "model_version": pa.array(["toy_model"] * 3, type=pa.string()),
            "scored_at": pa.array(
                [datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)] * 3,
                type=pa.timestamp("us", tz="UTC"),
            ),
            "blocking_methods": pa.array([["faiss"]] * 3, type=pa.list_(pa.string())),
            "blocking_source": pa.array(["faiss"] * 3, type=pa.string()),
            "blocking_method_count": pa.array([1] * 3, type=pa.int8()),
            "shap_top5": pa.array(
                [[]] * 3, type=schemas.SCORED_PAIRS_SCHEMA.field("shap_top5").type
            ),
        },
        schema=schemas.SCORED_PAIRS_SCHEMA,
    )
    pq.write_table(scored_pairs, get_scored_pairs_output_path(data_dir, run_id))

    resolved_entities = pa.Table.from_pylist(
        [
            _resolved_row(
                run_id,
                _hex32(401),
                entity_1,
                doc_1,
                "alice brown",
                3,
                0.99,
                "auto_merge",
                [doc_1],
                _hex32(501),
            ),
            _resolved_row(
                run_id,
                _hex32(401),
                entity_2,
                doc_1,
                "alice brown",
                3,
                0.99,
                "auto_merge",
                [doc_1],
                _hex32(501),
            ),
            _resolved_row(
                run_id,
                _hex32(401),
                entity_3,
                doc_1,
                "alice brown",
                3,
                0.99,
                "auto_merge",
                [doc_1],
                _hex32(501),
            ),
        ],
        schema=schemas.RESOLVED_ENTITIES_SCHEMA,
    )
    resolved_path = get_resolved_entities_output_path(data_dir, run_id)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(resolved_entities, resolved_path)

    return data_dir


def _write_gold_csv(path: Path) -> None:
    """Write a small gold annotation CSV matching the fixture artifacts."""
    rows = [
        {
            "case_id": "case_a",
            "doc_id": _hex32(1),
            "doc_name": "doc_1.docx",
            "mention_id": "m1",
            "char_start": 0,
            "char_end": 13,
            "text": "Alice Johnson",
            "entity_type": "PER",
            "group_id": "per_alice",
            "canonical_text": "Alice Johnson",
            "notes": "",
        },
        {
            "case_id": "case_a",
            "doc_id": _hex32(1),
            "doc_name": "doc_1.docx",
            "mention_id": "m2",
            "char_start": 18,
            "char_end": 27,
            "text": "Bob Stone",
            "entity_type": "PER",
            "group_id": "per_bob",
            "canonical_text": "Bob Stone",
            "notes": "",
        },
        {
            "case_id": "case_a",
            "doc_id": _hex32(2),
            "doc_name": "doc_2.docx",
            "mention_id": "m3",
            "char_start": 0,
            "char_end": 8,
            "text": "B. Stone",
            "entity_type": "PER",
            "group_id": "per_bob",
            "canonical_text": "Bob Stone",
            "notes": "",
        },
        {
            "case_id": "case_a",
            "doc_id": _hex32(2),
            "doc_name": "doc_2.docx",
            "mention_id": "m4",
            "char_start": 16,
            "char_end": 21,
            "text": "Alice",
            "entity_type": "PER",
            "group_id": "per_alice",
            "canonical_text": "Alice Johnson",
            "notes": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_shifted_gold_csv(path: Path) -> None:
    """Write gold mentions with legacy offsets from a different text view."""
    rows = [
        {
            "case_id": "case_a",
            "doc_id": _hex32(1),
            "doc_name": "doc_1.docx",
            "mention_id": "m1",
            "char_start": 0,
            "char_end": 13,
            "text": "Alice Johnson",
            "entity_type": "PER",
            "group_id": "per_alice",
            "canonical_text": "Alice Johnson",
            "notes": "",
        },
        {
            "case_id": "case_a",
            "doc_id": _hex32(1),
            "doc_name": "doc_1.docx",
            "mention_id": "m2",
            "char_start": 19,
            "char_end": 28,
            "text": "Bob Stone",
            "entity_type": "PER",
            "group_id": "per_bob",
            "canonical_text": "Bob Stone",
            "notes": "",
        },
        {
            "case_id": "case_a",
            "doc_id": _hex32(2),
            "doc_name": "doc_2.docx",
            "mention_id": "m3",
            "char_start": 0,
            "char_end": 8,
            "text": "B. Stone",
            "entity_type": "PER",
            "group_id": "per_bob",
            "canonical_text": "Bob Stone",
            "notes": "",
        },
        {
            "case_id": "case_a",
            "doc_id": _hex32(2),
            "doc_name": "doc_2.docx",
            "mention_id": "m4",
            "char_start": 17,
            "char_end": 22,
            "text": "Alice",
            "entity_type": "PER",
            "group_id": "per_alice",
            "canonical_text": "Alice Johnson",
            "notes": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_subset_metric_gold_csv(path: Path) -> None:
    """Write gold mentions for only the confidently bridged subset."""
    rows = [
        {
            "case_id": "case_subset",
            "doc_id": _hex32(301),
            "doc_name": "subset_case.docx",
            "mention_id": "m1",
            "char_start": 0,
            "char_end": 11,
            "text": "Alice Brown",
            "entity_type": "PER",
            "group_id": "per_alice",
            "canonical_text": "Alice Brown",
            "notes": "",
        },
        {
            "case_id": "case_subset",
            "doc_id": _hex32(301),
            "doc_name": "subset_case.docx",
            "mention_id": "m2",
            "char_start": 16,
            "char_end": 24,
            "text": "A. Brown",
            "entity_type": "PER",
            "group_id": "per_alice",
            "canonical_text": "Alice Brown",
            "notes": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_unmatched_gold_csv(path: Path) -> None:
    """Write gold mentions that do not bridge to any predicted entity."""
    rows = [
        {
            "case_id": "case_unmatched",
            "doc_id": _hex32(301),
            "doc_name": "subset_case.docx",
            "mention_id": "m1",
            "char_start": 100,
            "char_end": 108,
            "text": "Nobody",
            "entity_type": "PER",
            "group_id": "per_nobody",
            "canonical_text": "Nobody",
            "notes": "",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_labels_table_skips_unmatched_and_ambiguous_entities() -> None:
    candidate_pairs = pl.DataFrame(
        {
            "run_id": ["eval_run_001"] * 3,
            "entity_id_a": [_hex32(1), _hex32(1), _hex32(3)],
            "entity_id_b": [_hex32(2), _hex32(3), _hex32(4)],
        }
    )

    labels = _build_labels_table(
        candidate_pairs,
        {
            _hex32(1): "group_alpha",
            _hex32(2): "group_alpha",
            _hex32(3): "__fp__:" + _hex32(3),
            _hex32(4): "__ambiguous__:" + _hex32(4),
        },
    ).to_pylist()

    assert labels == [
        {
            "run_id": "eval_run_001",
            "entity_id_a": _hex32(1),
            "entity_id_b": _hex32(2),
            "label": 1,
        }
    ]


def test_chunk_start_offsets_replay_preserves_join_separators() -> None:
    doc_chunks = pl.DataFrame(
        {
            "chunk_id": [_hex32(11), _hex32(12)],
            "chunk_index": [0, 1],
            "text": ["Alpha", "Beta"],
        }
    )

    offsets = _chunk_start_offsets_from_full_text("Alpha\n\nBeta", doc_chunks)

    assert offsets == {_hex32(11): 0, _hex32(12): 7}


def test_remap_gold_doc_ids_rejects_ambiguous_doc_names() -> None:
    docs = pl.DataFrame(
        {
            "doc_id": [_hex32(1), _hex32(2)],
            "path": ["case_a/report.docx", "case_b/report.docx"],
        }
    )
    gold_mentions = pl.DataFrame(
        {
            "doc_id": ["legacy_doc"],
            "doc_name": ["report.docx"],
            "mention_id": ["m1"],
            "char_start": [0],
            "char_end": [4],
            "text": ["Test"],
            "entity_type": ["PER"],
            "group_id": ["g1"],
        }
    )

    with pytest.raises(ValueError, match="doc_name is ambiguous"):
        _remap_gold_doc_ids(gold_mentions, docs)


def test_remap_gold_offsets_to_run_text_realigns_shifted_mentions() -> None:
    chunks = pl.DataFrame(
        {
            "doc_id": [_hex32(1)],
            "chunk_id": [_hex32(11)],
            "chunk_index": [0],
            "text": ["Alpha  Beta"],
            "global_start": [0],
        }
    )
    gold_mentions = pl.DataFrame(
        {
            "doc_id": [_hex32(1), _hex32(1)],
            "doc_name": ["doc_1.docx", "doc_1.docx"],
            "mention_id": ["m1", "m2"],
            "char_start": [0, 6],
            "char_end": [5, 10],
            "text": ["Alpha", "Beta"],
            "entity_type": ["PER", "PER"],
            "group_id": ["g1", "g2"],
        }
    )

    remapped, summary = _remap_gold_offsets_to_run_text(gold_mentions, chunks)

    assert remapped["char_start"].to_list() == [0, 7]
    assert remapped["char_end"].to_list() == [5, 11]
    assert remapped["_offset_resolved"].to_list() == [True, True]
    assert summary == {
        "gold_mentions_already_aligned": 1,
        "gold_mentions_remapped": 1,
        "gold_mentions_unresolved": 0,
    }


def test_run_evaluation_writes_report_and_labels(tmp_path: Path) -> None:
    run_id = "eval_run_001"
    data_dir = _write_case_artifacts(tmp_path, run_id)
    gold_path = tmp_path / "gold_annotations.csv"
    shared_labels_path = tmp_path / "shared_labels.parquet"
    _write_gold_csv(gold_path)

    report = run_evaluation(
        data_dir=data_dir,
        run_id=run_id,
        gold_path=gold_path,
        shared_labels_path=shared_labels_path,
    )

    report_path = get_evaluation_report_path(data_dir, run_id)
    labels_path = get_evaluation_labels_path(data_dir, run_id)
    assert report_path.exists()
    assert labels_path.exists()
    assert shared_labels_path.exists()

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    labels = pq.read_table(labels_path).to_pylist()
    assert persisted["counts"]["gold_mentions"] == 4
    assert persisted["counts"]["evaluation_entities"] == 4
    assert persisted["counts"]["labeled_candidate_pairs"] == 6
    assert persisted["alignment"]["matched_gold_mentions"] == 4
    assert persisted["metric_scope"]["uses_only_confident_gold_bridge"] is True
    assert persisted["stage_metrics"]["extraction"]["f1"] == 1.0
    assert persisted["stage_metrics"]["matching"]["f1"] == 0.8
    assert persisted["stage_metrics"]["matching"]["evaluated_candidate_pair_count"] == 6
    assert persisted["metrics"]["pairwise_f1"] == 0.4
    assert persisted["metrics"]["bcubed_f1"] > 0.0
    assert persisted["regression_checks"]["passed"] is True
    assert len(labels) == 6
    assert {row["label"] for row in labels} == {0, 1}
    assert report["metrics"]["pairwise_f1"] == 0.4


def test_run_evaluation_filters_shared_training_labels_by_allowed_doc_ids(
    tmp_path: Path,
) -> None:
    run_id = "eval_run_filtered_labels_001"
    data_dir = _write_case_artifacts(tmp_path, run_id)
    gold_path = tmp_path / "gold_annotations.csv"
    shared_labels_path = tmp_path / "shared_labels.parquet"
    _write_gold_csv(gold_path)

    run_evaluation(
        data_dir=data_dir,
        run_id=run_id,
        gold_path=gold_path,
        shared_labels_path=shared_labels_path,
        shared_labels_allowed_doc_ids=[_hex32(1)],
    )

    full_labels = pq.read_table(get_evaluation_labels_path(data_dir, run_id)).to_pylist()
    shared_labels = pq.read_table(shared_labels_path).to_pylist()

    assert len(full_labels) == 6
    assert len(shared_labels) == 1
    assert shared_labels[0]["entity_id_a"] == _hex32(21)
    assert shared_labels[0]["entity_id_b"] == _hex32(22)
    assert shared_labels[0]["label"] == 0


def test_run_evaluation_realigns_gold_offsets_before_scoring(tmp_path: Path) -> None:
    run_id = "eval_run_offset_realign_001"
    data_dir = _write_case_artifacts(tmp_path, run_id)
    gold_path = tmp_path / "gold_annotations_shifted.csv"
    _write_shifted_gold_csv(gold_path)

    report = run_evaluation(
        data_dir=data_dir,
        run_id=run_id,
        gold_path=gold_path,
    )

    assert report["alignment"]["gold_mentions_remapped"] == 2
    assert report["alignment"]["gold_mentions_already_aligned"] == 2
    assert report["alignment"]["gold_mentions_unresolved"] == 0
    assert report["stage_metrics"]["extraction"]["f1"] == pytest.approx(1.0)
    assert report["alignment"]["matched_mentions_by_method"]["exact_span"] == 4


def test_run_evaluation_excludes_unresolved_gold_mentions_from_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "eval_run_unresolved_001"
    data_dir = _write_case_artifacts(tmp_path, run_id)
    gold_path = tmp_path / "gold_annotations.csv"
    _write_gold_csv(gold_path)

    original = _remap_gold_offsets_to_run_text

    def mark_one_row_unresolved(
        gold_mentions: pl.DataFrame,
        chunks: pl.DataFrame,
    ) -> tuple[pl.DataFrame, dict[str, int]]:
        remapped, summary = original(gold_mentions, chunks)
        first = remapped.row(0, named=True)
        forced_row = {
            **first,
            "char_start": int(first["char_start"]) + 20,
            "char_end": int(first["char_end"]) + 20,
            "_offset_resolved": False,
        }
        forced = pl.DataFrame([forced_row])
        patched = pl.concat([forced, remapped.slice(1)], how="vertical")
        return patched, {
            **summary,
            "gold_mentions_already_aligned": summary["gold_mentions_already_aligned"] - 1,
            "gold_mentions_unresolved": summary["gold_mentions_unresolved"] + 1,
        }

    monkeypatch.setattr(
        "src.evaluation.run._remap_gold_offsets_to_run_text",
        mark_one_row_unresolved,
    )

    report = run_evaluation(
        data_dir=data_dir,
        run_id=run_id,
        gold_path=gold_path,
    )

    assert report["counts"]["gold_mentions"] == 4
    assert report["counts"]["trusted_gold_mentions"] == 3
    assert report["alignment"]["gold_mentions_unresolved"] == 1
    assert report["alignment"]["trusted_gold_mentions"] == 3
    assert report["alignment"]["matched_mention_rate"] == pytest.approx(1.0)
    assert report["stage_metrics"]["extraction"]["precision"] == pytest.approx(0.75)
    assert report["stage_metrics"]["extraction"]["recall"] == pytest.approx(1.0)
    assert report["alignment"]["matched_mentions_by_method"]["exact_span"] == 3


def test_run_evaluation_scores_only_confidently_bridged_subset(
    tmp_path: Path,
) -> None:
    run_id = "eval_run_subset_001"
    data_dir = _write_subset_metric_case_artifacts(tmp_path, run_id)
    gold_path = tmp_path / "subset_gold_annotations.csv"
    _write_subset_metric_gold_csv(gold_path)

    report = run_evaluation(
        data_dir=data_dir,
        run_id=run_id,
        gold_path=gold_path,
    )

    assert report["alignment"]["entities_with_gold_group"] == 2
    assert report["alignment"]["entities_without_gold_match"] == 1
    assert report["alignment"]["candidate_pairs_excluded_from_labels"] == 2
    assert report["metric_scope"]["evaluation_entity_count"] == 2
    assert report["metric_scope"]["excluded_unmatched_entity_count"] == 1
    assert report["stage_metrics"]["matching"]["evaluated_candidate_pair_count"] == 1
    assert report["stage_metrics"]["matching"]["precision"] == pytest.approx(1.0)
    assert report["stage_metrics"]["matching"]["recall"] == pytest.approx(1.0)
    assert report["metrics"]["pairwise_f1"] == pytest.approx(1.0)
    assert report["metrics"]["bcubed_f1"] == pytest.approx(1.0)


def test_run_evaluation_empty_confident_subset_reports_zero_metrics_and_fails_checks(
    tmp_path: Path,
) -> None:
    run_id = "eval_run_no_confident_001"
    data_dir = _write_subset_metric_case_artifacts(tmp_path, run_id)
    gold_path = tmp_path / "unmatched_gold_annotations.csv"
    _write_unmatched_gold_csv(gold_path)

    report = run_evaluation(
        data_dir=data_dir,
        run_id=run_id,
        gold_path=gold_path,
    )

    assert report["metric_scope"]["evaluation_entity_count"] == 0
    assert report["metric_scope"]["evaluation_candidate_pair_count"] == 0
    assert report["metrics"]["pairwise_f1"] == pytest.approx(0.0)
    assert report["metrics"]["bcubed_f1"] == pytest.approx(0.0)
    assert report["metrics"]["ari"] == pytest.approx(0.0)
    assert report["metrics"]["nmi"] == pytest.approx(0.0)
    assert report["regression_checks"]["passed"] is False
    assert any(
        check["check"] == "evaluation_entity_count_nonzero"
        and check["passed"] is False
        for check in report["regression_checks"]["checks"]
    )
    assert any(
        check["check"] == "evaluation_candidate_pair_count_nonzero"
        and check["passed"] is False
        for check in report["regression_checks"]["checks"]
    )


def test_build_predicted_mentions_handles_empty_entities() -> None:
    entities = pl.from_arrow(pa.Table.from_pylist([], schema=schemas.ENTITIES_SCHEMA))
    chunks = pl.DataFrame(
        {
            "chunk_id": [_hex32(11)],
            "doc_id": [_hex32(1)],
            "chunk_index": [0],
            "text": ["Alpha"],
            "global_start": [0],
        }
    )

    predicted = _build_predicted_mentions(entities, chunks)

    assert predicted.is_empty()
    assert predicted.columns == [
        "entity_id",
        "doc_id",
        "mention_id",
        "char_start",
        "char_end",
        "text",
        "entity_type",
        "chunk_id",
        "chunk_index",
    ]


def test_match_gold_to_predicted_handles_no_matches() -> None:
    gold_mentions = pl.DataFrame(
        {
            "doc_id": [_hex32(1)],
            "doc_name": ["subset_case.docx"],
            "mention_id": ["m1"],
            "char_start": [0],
            "char_end": [5],
            "text": ["Alpha"],
            "entity_type": ["PER"],
            "group_id": ["g1"],
        }
    )
    predicted_mentions = pl.DataFrame(
        {
            "entity_id": [_hex32(21)],
            "doc_id": [_hex32(1)],
            "mention_id": ["p1"],
            "char_start": [10],
            "char_end": [15],
            "text": ["Beta"],
            "entity_type": ["PER"],
            "chunk_id": [_hex32(11)],
            "chunk_index": [0],
        }
    )

    matched = _match_gold_to_predicted(gold_mentions, predicted_mentions)

    assert matched.is_empty()
    assert matched.columns == [
        "doc_id",
        "doc_name",
        "mention_id",
        "char_start",
        "char_end",
        "text",
        "entity_type",
        "group_id",
        "entity_id",
        "match_method",
    ]


def test_build_regression_checks_flags_metric_drift(tmp_path: Path) -> None:
    report_path = tmp_path / "baseline_report.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "pairwise_f1": 0.8,
                    "bcubed_f1": 0.8,
                    "ari": 0.8,
                    "nmi": 0.8,
                }
            }
        ),
        encoding="utf-8",
    )
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")

    checks = build_regression_checks(
        metrics={
            "pairwise_f1": 0.4,
            "bcubed_f1": 0.5,
            "ari": 0.5,
            "nmi": 0.5,
        },
        input_paths={"report_path": existing},
        baseline_report_path=report_path,
    )

    assert checks["passed"] is False
    assert any(
        check["check"] == "pairwise_f1_drift" and check["passed"] is False
        for check in checks["checks"]
    )


def test_build_regression_checks_skips_missing_baseline_metrics(tmp_path: Path) -> None:
    report_path = tmp_path / "baseline_report.json"
    report_path.write_text(
        json.dumps({"metrics": {"pairwise_f1": 0.8}}),
        encoding="utf-8",
    )
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")

    checks = build_regression_checks(
        metrics={
            "pairwise_f1": 0.8,
            "bcubed_f1": 0.8,
            "ari": 0.8,
            "nmi": 0.8,
        },
        input_paths={"report_path": existing},
        baseline_report_path=report_path,
        metric_scope={
            "evaluation_entity_count": 1,
            "evaluation_candidate_pair_count": 1,
        },
    )

    assert checks["passed"] is True
    assert any(
        check["check"] == "bcubed_f1_drift"
        and check["passed"] is True
        and check.get("skipped") is True
        for check in checks["checks"]
    )


def test_build_regression_checks_fails_empty_metric_scope(tmp_path: Path) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")

    checks = build_regression_checks(
        metrics={
            "pairwise_f1": 0.0,
            "bcubed_f1": 0.0,
            "ari": 0.0,
            "nmi": 0.0,
        },
        input_paths={"report_path": existing},
        metric_scope={
            "evaluation_entity_count": 0,
            "evaluation_candidate_pair_count": 0,
        },
    )

    assert checks["passed"] is False
    assert any(
        check["check"] == "evaluation_entity_count_nonzero"
        and check["passed"] is False
        for check in checks["checks"]
    )
    assert any(
        check["check"] == "evaluation_candidate_pair_count_nonzero"
        and check["passed"] is False
        for check in checks["checks"]
    )
