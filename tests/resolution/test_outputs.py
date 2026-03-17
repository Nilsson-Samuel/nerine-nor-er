"""Final resolution output and canonicalization tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.matching.writer import get_scored_pairs_output_path
from src.resolution.canonicalization import choose_canonical_name, choose_canonical_type
from src.resolution import writer as resolution_writer
from src.resolution.run import run_resolution
from src.resolution.writer import (
    get_clusters_output_path,
    get_resolved_entities_output_path,
)
from src.shared import schemas


def _hex32(number: int) -> str:
    """Build a deterministic lowercase 32-char hex identifier."""
    return f"{number:032x}"


def _build_entities_table(run_id: str, rows: list[dict[str, Any]]) -> pa.Table:
    """Build a contract-valid entities table for resolution output tests."""
    return pa.table(
        {
            "run_id": pa.array([run_id] * len(rows), type=pa.string()),
            "entity_id": pa.array([row["entity_id"] for row in rows], type=pa.string()),
            "doc_id": pa.array([row["doc_id"] for row in rows], type=pa.string()),
            "chunk_id": pa.array([row["chunk_id"] for row in rows], type=pa.string()),
            "text": pa.array([row["text"] for row in rows], type=pa.string()),
            "normalized": pa.array([row["normalized"] for row in rows], type=pa.string()),
            "type": pa.array([row["type"] for row in rows], type=pa.string()),
            "char_start": pa.array([row["char_start"] for row in rows], type=pa.int32()),
            "char_end": pa.array([row["char_end"] for row in rows], type=pa.int32()),
            "context": pa.array([row["context"] for row in rows], type=pa.string()),
            "count": pa.array([row["count"] for row in rows], type=pa.int32()),
            "positions": pa.array(
                [row["positions"] for row in rows],
                type=schemas.ENTITIES_SCHEMA.field("positions").type,
            ),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )


def _build_scored_pairs_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a contract-valid scored pair table for resolution output tests."""
    scored_at = datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc)
    row_count = len(rows)
    return pa.table(
        {
            "run_id": pa.array([row["run_id"] for row in rows], type=pa.string()),
            "entity_id_a": pa.array([row["entity_id_a"] for row in rows], type=pa.string()),
            "entity_id_b": pa.array([row["entity_id_b"] for row in rows], type=pa.string()),
            "score": pa.array([row["score"] for row in rows], type=pa.float32()),
            "model_version": pa.array(["resolution_fixture"] * row_count, type=pa.string()),
            "scored_at": pa.array([scored_at] * row_count, type=pa.timestamp("us", tz="UTC")),
            "blocking_methods": pa.array(
                [row.get("blocking_methods", ["faiss"]) for row in rows],
                type=pa.list_(pa.string()),
            ),
            "blocking_source": pa.array(
                [row.get("blocking_source", "faiss") for row in rows],
                type=pa.string(),
            ),
            "blocking_method_count": pa.array(
                [row.get("blocking_method_count", 1) for row in rows],
                type=pa.int8(),
            ),
            "shap_top5": pa.array(
                [row.get("shap_top5", []) for row in rows],
                type=schemas.SCORED_PAIRS_SCHEMA.field("shap_top5").type,
            ),
        },
        schema=schemas.SCORED_PAIRS_SCHEMA,
    )


def _attribute_map(value: object) -> dict[str, list[str]]:
    """Normalize Arrow map values for simpler assertions."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        normalized: dict[str, list[str]] = {}
        for item in value:
            if isinstance(item, dict):
                normalized[str(item["key"])] = list(item["value"])
            else:
                key, values = item
                normalized[str(key)] = list(values)
        return normalized
    raise TypeError(f"unsupported attribute value: {value!r}")


def _build_resolved_entities_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a strict-schema resolved entities table for writer/validator tests."""
    return pa.Table.from_pylist(rows, schema=schemas.RESOLVED_ENTITIES_SCHEMA)


def test_choose_canonical_name_prefers_aggregated_count_after_length_tie() -> None:
    rows = [
        {
            "normalized": "Alpha Beta",
            "count": 1,
            "chunk_id": _hex32(11),
            "char_start": 40,
        },
        {
            "normalized": "Alpha Gama",
            "count": 2,
            "chunk_id": _hex32(12),
            "char_start": 0,
        },
        {
            "normalized": "Alpha Beta",
            "count": 3,
            "chunk_id": _hex32(13),
            "char_start": 5,
        },
    ]

    assert choose_canonical_name(rows) == "Alpha Beta"


def test_choose_canonical_name_prefers_earliest_span_after_count_tie() -> None:
    rows = [
        {
            "normalized": "Alpha Beta",
            "count": 2,
            "chunk_id": _hex32(20),
            "char_start": 9,
        },
        {
            "normalized": "Alpha Gama",
            "count": 2,
            "chunk_id": _hex32(19),
            "char_start": 4,
        },
    ]

    assert choose_canonical_name(rows) == "Alpha Gama"


def test_choose_canonical_type_rejects_mixed_types() -> None:
    with pytest.raises(ValueError, match="mixed entity types"):
        choose_canonical_type([{"type": "ORG"}, {"type": "PER"}])


def test_run_resolution_writes_resolved_entities_and_clusters_json(tmp_path: Path) -> None:
    run_id = "run_resolution_outputs"
    entity_rows = [
        {
            "entity_id": _hex32(1),
            "doc_id": _hex32(101),
            "chunk_id": _hex32(201),
            "text": "DNB",
            "normalized": "Dnb",
            "type": "ORG",
            "char_start": 4,
            "char_end": 7,
            "context": "DNB inngikk avtalen.",
            "count": 1,
            "positions": [
                {
                    "chunk_id": _hex32(201),
                    "char_start": 4,
                    "char_end": 7,
                    "page_num": 0,
                    "source_unit_kind": "pdf_page",
                }
            ],
        },
        {
            "entity_id": _hex32(2),
            "doc_id": _hex32(102),
            "chunk_id": _hex32(202),
            "text": "Den Norske Bank",
            "normalized": "Den Norske Bank",
            "type": "ORG",
            "char_start": 3,
            "char_end": 19,
            "context": "Den Norske Bank er omtalt her.",
            "count": 1,
            "positions": [
                {
                    "chunk_id": _hex32(202),
                    "char_start": 3,
                    "char_end": 19,
                    "page_num": 0,
                    "source_unit_kind": "pdf_page",
                }
            ],
        },
        {
            "entity_id": _hex32(3),
            "doc_id": _hex32(101),
            "chunk_id": _hex32(203),
            "text": "Den Norske Bank",
            "normalized": "Den Norske Bank",
            "type": "ORG",
            "char_start": 9,
            "char_end": 25,
            "context": "Vitnet nevnte Den Norske Bank igjen.",
            "count": 2,
            "positions": [
                {
                    "chunk_id": _hex32(203),
                    "char_start": 9,
                    "char_end": 25,
                    "page_num": 0,
                    "source_unit_kind": "pdf_page",
                },
                {
                    "chunk_id": _hex32(203),
                    "char_start": 40,
                    "char_end": 56,
                    "page_num": 0,
                    "source_unit_kind": "pdf_page",
                },
            ],
        },
        {
            "entity_id": _hex32(4),
            "doc_id": _hex32(103),
            "chunk_id": _hex32(204),
            "text": "Sparebank 1",
            "normalized": "Sparebank 1",
            "type": "ORG",
            "char_start": 0,
            "char_end": 11,
            "context": "Sparebank 1 er et annet selskap.",
            "count": 1,
            "positions": [
                {
                    "chunk_id": _hex32(204),
                    "char_start": 0,
                    "char_end": 11,
                    "page_num": 0,
                    "source_unit_kind": "pdf_page",
                }
            ],
        },
    ]
    pq.write_table(_build_entities_table(run_id, entity_rows), tmp_path / "entities.parquet")

    scored_pairs_path = get_scored_pairs_output_path(tmp_path, run_id)
    scored_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_scored_pairs_table(
            [
                {
                    "run_id": run_id,
                    "entity_id_a": _hex32(1),
                    "entity_id_b": _hex32(2),
                    "score": 0.97,
                    "shap_top5": [{"feature": "name_similarity", "value": 0.31}],
                },
                {
                    "run_id": run_id,
                    "entity_id_a": _hex32(1),
                    "entity_id_b": _hex32(3),
                    "score": 0.96,
                    "shap_top5": [{"feature": "embedding_cosine", "value": 0.29}],
                },
                {
                    "run_id": run_id,
                    "entity_id_a": _hex32(2),
                    "entity_id_b": _hex32(3),
                    "score": 0.95,
                    "shap_top5": [{"feature": "shared_doc_count", "value": 0.2}],
                },
            ]
        ),
        scored_pairs_path,
    )

    diagnostics = run_resolution(tmp_path, run_id)

    resolved_entities_path = get_resolved_entities_output_path(tmp_path, run_id)
    clusters_path = get_clusters_output_path(tmp_path, run_id)
    resolved_table = pq.read_table(resolved_entities_path)
    cluster_payload = json.loads(clusters_path.read_text())
    resolved_rows = sorted(resolved_table.to_pylist(), key=lambda row: row["entity_id"])

    assert diagnostics["cluster_count"] == 2
    assert not (tmp_path / "resolved_entities.parquet").exists()
    assert not (tmp_path / "clusters.json").exists()
    assert schemas.validate_contract_rules(resolved_table, "resolved_entities") == []

    merged_rows = [row for row in resolved_rows if row["cluster_size"] == 3]
    assert len(merged_rows) == 3
    assert {row["cluster_id"] for row in merged_rows} == {merged_rows[0]["cluster_id"]}
    assert {row["route_action"] for row in merged_rows} == {"auto_merge"}
    assert {row["canonical_name"] for row in merged_rows} == {"Den Norske Bank"}
    assert {row["canonical_type"] for row in merged_rows} == {"ORG"}
    assert {row["confidence"] > 0.85 for row in merged_rows} == {True}
    assert {row["needs_review"] for row in merged_rows} == {False}
    assert {tuple(row["doc_ids"]) for row in merged_rows} == {(_hex32(101), _hex32(102))}

    attributes = _attribute_map(merged_rows[0]["attributes"])
    assert attributes["normalized_variants"] == ["Den Norske Bank", "Dnb"]
    assert attributes["text_variants"] == ["DNB", "Den Norske Bank"]

    assert cluster_payload["run_id"] == run_id
    assert cluster_payload["cluster_count"] == 2
    merged_cluster = next(
        cluster for cluster in cluster_payload["clusters"] if cluster["cluster_size"] == 3
    )
    assert merged_cluster["canonical_name"] == "Den Norske Bank"
    assert merged_cluster["doc_ids"] == [_hex32(101), _hex32(102)]
    assert len(merged_cluster["members"]) == 3
    assert len(merged_cluster["edges"]) == 3
    assert merged_cluster["edges"][0]["shap_top5"][0]["feature"] in {
        "name_similarity",
        "embedding_cosine",
        "shared_doc_count",
    }


def test_run_resolution_defer_cluster_does_not_set_needs_review(tmp_path: Path) -> None:
    run_id = "run_resolution_defer"
    entity_ids = [_hex32(index) for index in range(1, 7)]
    entity_rows = []
    for index, entity_id in enumerate(entity_ids, start=1):
        entity_rows.append(
            {
                "entity_id": entity_id,
                "doc_id": _hex32(100 + index),
                "chunk_id": _hex32(200 + index),
                "text": f"entity_{index}",
                "normalized": f"entity_{index}",
                "type": "ORG",
                "char_start": 0,
                "char_end": 8,
                "context": "fixture context",
                "count": 1,
                "positions": [
                    {
                        "chunk_id": _hex32(200 + index),
                        "char_start": 0,
                        "char_end": 8,
                        "page_num": 0,
                        "source_unit_kind": "pdf_page",
                    }
                ],
            }
        )

    pq.write_table(_build_entities_table(run_id, entity_rows), tmp_path / "entities.parquet")

    scored_pairs_path = get_scored_pairs_output_path(tmp_path, run_id)
    scored_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_scored_pairs_table(
            [
                {"run_id": run_id, "entity_id_a": _hex32(1), "entity_id_b": _hex32(2), "score": 0.95},
                {"run_id": run_id, "entity_id_a": _hex32(1), "entity_id_b": _hex32(3), "score": 0.95},
                {"run_id": run_id, "entity_id_a": _hex32(4), "entity_id_b": _hex32(5), "score": 0.95},
                {"run_id": run_id, "entity_id_a": _hex32(4), "entity_id_b": _hex32(6), "score": 0.95},
            ]
        ),
        scored_pairs_path,
    )

    run_resolution(tmp_path, run_id)

    resolved_rows = pq.read_table(get_resolved_entities_output_path(tmp_path, run_id)).to_pylist()
    defer_rows = [row for row in resolved_rows if row["route_action"] == "defer"]

    assert len(defer_rows) == 6
    assert {row["cluster_size"] for row in defer_rows} == {3}
    assert {row["needs_review"] for row in defer_rows} == {False}


def test_resolved_entities_contract_rejects_route_action_mismatch() -> None:
    rows = [
        {
            "run_id": "run_resolution_contract",
            "cluster_id": _hex32(900),
            "entity_id": _hex32(901),
            "doc_id": _hex32(902),
            "entity_type": "ORG",
            "canonical_name": "Acme",
            "canonical_type": "ORG",
            "cluster_size": 1,
            "confidence": 0.10,
            "needs_review": False,
            "route_action": "auto_merge",
            "clustering_method": "singleton",
            "component_id": _hex32(903),
            "doc_ids": [_hex32(902)],
            "most_recent_doc_id": None,
            "most_recent_doc_date": None,
            "attributes": {
                "chunk_ids": [_hex32(904)],
                "normalized_variants": ["Acme"],
                "text_variants": ["Acme"],
            },
        }
    ]

    errors = schemas.validate_contract_rules(
        _build_resolved_entities_table(rows),
        "resolved_entities",
    )

    assert any(
        "route_action must match the configured routing outcome" in error for error in errors
    )


def test_write_resolved_entities_preserves_existing_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "resolved_entities.parquet"
    original_table = _build_resolved_entities_table(
        [
            {
                "run_id": "run_resolution_writer",
                "cluster_id": _hex32(950),
                "entity_id": _hex32(951),
                "doc_id": _hex32(952),
                "entity_type": "ORG",
                "canonical_name": "Acme",
                "canonical_type": "ORG",
                "cluster_size": 1,
                "confidence": 0.10,
                "needs_review": False,
                "route_action": "keep_separate",
                "clustering_method": "singleton",
                "component_id": _hex32(953),
                "doc_ids": [_hex32(952)],
                "most_recent_doc_id": None,
                "most_recent_doc_date": None,
                "attributes": {
                    "chunk_ids": [_hex32(954)],
                    "normalized_variants": ["Acme"],
                    "text_variants": ["Acme"],
                },
            }
        ]
    )
    resolution_writer.write_resolved_entities(original_table, path)

    replacement_table = _build_resolved_entities_table(
        [
            {
                "run_id": "run_resolution_writer",
                "cluster_id": _hex32(960),
                "entity_id": _hex32(961),
                "doc_id": _hex32(962),
                "entity_type": "ORG",
                "canonical_name": "Beta",
                "canonical_type": "ORG",
                "cluster_size": 1,
                "confidence": 0.10,
                "needs_review": False,
                "route_action": "keep_separate",
                "clustering_method": "singleton",
                "component_id": _hex32(963),
                "doc_ids": [_hex32(962)],
                "most_recent_doc_id": None,
                "most_recent_doc_date": None,
                "attributes": {
                    "chunk_ids": [_hex32(964)],
                    "normalized_variants": ["Beta"],
                    "text_variants": ["Beta"],
                },
            }
        ]
    )

    def fail_after_partial_write(table: pa.Table, out_path: Path | str) -> None:
        Path(out_path).write_bytes(b"partial parquet bytes")
        raise RuntimeError("simulated parquet write failure")

    monkeypatch.setattr(resolution_writer.pq, "write_table", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated parquet write failure"):
        resolution_writer.write_resolved_entities(replacement_table, path)

    assert pq.read_table(path).to_pylist() == original_table.to_pylist()
    assert not path.with_suffix(".parquet.tmp").exists()
