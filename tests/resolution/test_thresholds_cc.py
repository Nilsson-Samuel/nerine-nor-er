"""Resolution threshold, component-building, and diagnostics tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.matching.writer import get_scored_pairs_output_path
from src.resolution.clustering import (
    build_phase1_components,
    include_edge,
    make_component_id,
)
from src.resolution.run import (
    get_resolution_components_path,
    get_resolution_diagnostics_path,
    run_resolution,
)
from src.shared import schemas


PAIR_COLUMNS = ["run_id", "entity_id_a", "entity_id_b", "score"]


def _hex32(number: int) -> str:
    """Build a deterministic lowercase 32-char hex identifier."""
    return f"{number:032x}"


def _build_scored_pairs_table(rows: list[tuple[str, str, str, float]]) -> pa.Table:
    """Build a contract-valid scored pair table for resolution tests."""
    scored_at = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)
    row_count = len(rows)
    return pa.table(
        {
            "run_id": pa.array([row[0] for row in rows], type=pa.string()),
            "entity_id_a": pa.array([row[1] for row in rows], type=pa.string()),
            "entity_id_b": pa.array([row[2] for row in rows], type=pa.string()),
            "score": pa.array([row[3] for row in rows], type=pa.float32()),
            "model_version": pa.array(["resolution_fixture"] * row_count, type=pa.string()),
            "scored_at": pa.array([scored_at] * row_count, type=pa.timestamp("us", tz="UTC")),
            "blocking_methods": pa.array([["faiss"]] * row_count, type=pa.list_(pa.string())),
            "blocking_source": pa.array(["faiss"] * row_count, type=pa.string()),
            "blocking_method_count": pa.array([1] * row_count, type=pa.int8()),
            "shap_top5": pa.array(
                [[] for _ in rows],
                type=schemas.SCORED_PAIRS_SCHEMA.field("shap_top5").type,
            ),
        },
        schema=schemas.SCORED_PAIRS_SCHEMA,
    )


def _build_entities_table(run_id: str, entity_ids: list[str]) -> pa.Table:
    """Build a contract-valid entities table for resolution tests."""
    row_count = len(entity_ids)
    positions = []
    for index in range(row_count):
        chunk_id = _hex32(200 + index)
        positions.append([
            {
                "chunk_id": chunk_id,
                "char_start": 0,
                "char_end": 8,
                "page_num": 0,
                "source_unit_kind": "pdf_page",
            }
        ])

    return pa.table(
        {
            "run_id": pa.array([run_id] * row_count, type=pa.string()),
            "entity_id": pa.array(entity_ids, type=pa.string()),
            "doc_id": pa.array(
                [_hex32(100 + index) for index in range(row_count)],
                type=pa.string(),
            ),
            "chunk_id": pa.array(
                [_hex32(200 + index) for index in range(row_count)],
                type=pa.string(),
            ),
            "text": pa.array([f"entity_{index}" for index in range(row_count)], type=pa.string()),
            "normalized": pa.array(
                [f"entity_{index}" for index in range(row_count)],
                type=pa.string(),
            ),
            "type": pa.array(["ORG"] * row_count, type=pa.string()),
            "char_start": pa.array([0] * row_count, type=pa.int32()),
            "char_end": pa.array([8] * row_count, type=pa.int32()),
            "context": pa.array(["fixture context"] * row_count, type=pa.string()),
            "count": pa.array([1] * row_count, type=pa.int32()),
            "positions": pa.array(
                positions,
                type=schemas.ENTITIES_SCHEMA.field("positions").type,
            ),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )


def test_include_edge_keeps_boundary_value() -> None:
    assert include_edge(0.60) is True
    assert include_edge(0.5999) is False


def test_build_phase1_components_splits_toy_graph_deterministically() -> None:
    a, b, c, d, e, f = (_hex32(index) for index in range(1, 7))
    scored_pairs = _build_scored_pairs_table(
        [
            ("run_resolution", a, b, 0.60),
            ("run_resolution", a, c, 0.59),
            ("run_resolution", b, c, 0.90),
            ("run_resolution", c, d, 0.20),
            ("run_resolution", d, e, 0.95),
            ("run_resolution", e, f, 0.10),
        ]
    )

    components, diagnostics = build_phase1_components(
        "run_resolution",
        pl.from_arrow(scored_pairs).select(PAIR_COLUMNS),
        [a, b, c, d, e, f],
    )

    assert [list(component.entity_ids) for component in components] == [
        [a, b, c],
        [d, e],
        [f],
    ]
    assert components[0].retained_edge_count == 2
    assert components[1].retained_edge_count == 1
    assert components[2].retained_edge_count == 0
    assert diagnostics["retained_edge_count"] == 3
    assert diagnostics["component_count"] == 3
    assert diagnostics["component_size_distribution"] == {"1": 1, "2": 1, "3": 1}
    assert diagnostics["singleton_rate"] == pytest.approx(1 / 6, abs=1e-6)
    assert diagnostics["giant_component_warnings"] == []


def test_make_component_id_is_order_independent() -> None:
    entity_ids = [_hex32(3), _hex32(1), _hex32(2)]
    assert make_component_id(entity_ids) == make_component_id(list(reversed(entity_ids)))


def test_build_phase1_components_keeps_entities_without_scored_pairs() -> None:
    a, b, c, d = (_hex32(index) for index in range(1, 5))
    scored_pairs = _build_scored_pairs_table([("run_resolution", a, b, 0.85)])

    components, diagnostics = build_phase1_components(
        "run_resolution",
        pl.from_arrow(scored_pairs).select(PAIR_COLUMNS),
        [a, b, c, d],
    )

    assert [list(component.entity_ids) for component in components] == [
        [a, b],
        [c],
        [d],
    ]
    assert diagnostics["total_node_count"] == 4
    assert diagnostics["singleton_node_count"] == 2
    assert diagnostics["singleton_rate"] == pytest.approx(0.5, abs=1e-6)


def test_run_resolution_writes_component_and_diagnostic_artifacts(tmp_path: Path) -> None:
    a, b, c, d, e = (_hex32(index) for index in range(1, 6))
    pq.write_table(
        _build_entities_table("run_resolution", [a, b, c, d, e]),
        tmp_path / "entities.parquet",
    )
    table = _build_scored_pairs_table(
        [
            ("run_resolution", a, b, 0.80),
            ("run_resolution", b, c, 0.10),
            ("run_resolution", c, d, 0.92),
        ]
    )
    scored_pairs_path = get_scored_pairs_output_path(tmp_path, "run_resolution")
    scored_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, scored_pairs_path)

    diagnostics = run_resolution(tmp_path, "run_resolution")

    components_payload = json.loads(
        get_resolution_components_path(tmp_path, "run_resolution").read_text()
    )
    diagnostics_payload = json.loads(
        get_resolution_diagnostics_path(tmp_path, "run_resolution").read_text()
    )

    assert diagnostics == diagnostics_payload
    assert not (tmp_path / "resolution_components.json").exists()
    assert not (tmp_path / "resolution_diagnostics.json").exists()
    assert components_payload["run_id"] == "run_resolution"
    assert components_payload["component_count"] == 3
    assert [component["entity_ids"] for component in components_payload["components"]] == [
        [a, b],
        [c, d],
        [e],
    ]
    assert diagnostics_payload["retained_edge_count"] == 2
    assert diagnostics_payload["singleton_node_count"] == 1
    assert diagnostics_payload["thresholds"]["keep_score_threshold"] == 0.6


def test_run_resolution_raises_for_missing_run_id(tmp_path: Path) -> None:
    a, b = (_hex32(index) for index in range(1, 3))
    pq.write_table(
        _build_entities_table("run_resolution", [a, b]),
        tmp_path / "entities.parquet",
    )
    scored_pairs_path = get_scored_pairs_output_path(tmp_path, "run_resolution")
    scored_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_scored_pairs_table([("run_resolution", a, b, 0.90)]),
        scored_pairs_path,
    )

    with pytest.raises(ValueError, match="run_id not found in entities.parquet"):
        run_resolution(tmp_path, "run_missing")


def test_build_phase1_components_flags_giant_component_warning() -> None:
    nodes = [_hex32(index) for index in range(1, 31)]
    rows = [("run_big_component", nodes[index], nodes[index + 1], 0.90) for index in range(29)]
    rows.append(("run_big_component", _hex32(100), _hex32(101), 0.10))
    table = _build_scored_pairs_table(rows)

    components, diagnostics = build_phase1_components(
        "run_big_component",
        pl.from_arrow(table).select(PAIR_COLUMNS),
        nodes + [_hex32(100), _hex32(101)],
    )

    assert len(components) == 3
    assert {warning["code"] for warning in diagnostics["giant_component_warnings"]} == {
        "largest_component_node_count",
        "largest_component_share",
    }
