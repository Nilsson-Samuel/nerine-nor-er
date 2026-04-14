"""Correlation clustering, confidence scoring, and routing diagnostics tests."""

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
    PAIR_COLUMNS,
    SolvedComponent,
    build_phase1_components,
    score_to_objective_weight,
    solve_component_with_pivot,
)
from src.resolution.confidence import (
    compute_base_confidence,
    route_cluster,
    routing_actions_by_profile,
)
from src.resolution.run import (
    _make_cluster_rows,
    _suspicious_missed_merge_examples,
    get_resolution_components_path,
    get_resolution_diagnostics_path,
    run_resolution,
)
from src.shared import schemas
from src.shared.paths import get_extraction_run_output_dir


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
        positions.append(
            [
                {
                    "chunk_id": chunk_id,
                    "char_start": 0,
                    "char_end": 8,
                    "page_num": 0,
                    "source_unit_kind": "pdf_page",
                }
            ]
        )

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


def test_score_to_objective_weight_uses_neutral_threshold_exactly() -> None:
    assert score_to_objective_weight(0.80) == 0.0
    assert score_to_objective_weight(0.95) == pytest.approx(0.15)
    assert score_to_objective_weight(0.60) == pytest.approx(-0.20)


def test_solve_component_with_pivot_splits_contradictory_triangle_deterministically() -> None:
    a, b, c = (_hex32(index) for index in range(1, 4))
    table = _build_scored_pairs_table(
        [
            ("run_triangle", a, b, 0.95),
            ("run_triangle", a, c, 0.95),
            ("run_triangle", b, c, 0.60),
        ]
    )
    components, _ = build_phase1_components(
        "run_triangle",
        pl.from_arrow(table).select(PAIR_COLUMNS),
        [a, b, c],
    )

    solved_a = solve_component_with_pivot(components[0], base_seed=7)
    solved_b = solve_component_with_pivot(components[0], base_seed=7)

    assert solved_a.clusters in {((a, b), (c,)), ((a, c), (b,))}
    assert solved_b.clusters == solved_a.clusters
    assert solved_b.objective_score == solved_a.objective_score
    assert solved_a.clustering_method == "pivot_correlation"


def test_solve_component_with_pivot_splits_two_node_component_below_neutral() -> None:
    a, b = (_hex32(index) for index in range(1, 3))
    table = _build_scored_pairs_table([("run_pair", a, b, 0.60)])
    components, _ = build_phase1_components(
        "run_pair",
        pl.from_arrow(table).select(PAIR_COLUMNS),
        [a, b],
    )

    solved = solve_component_with_pivot(components[0])

    assert solved.clusters == ((a,), (b,))
    assert solved.objective_score == pytest.approx(0.20)
    assert solved.clustering_method == "trivial_objective"


def test_compute_base_confidence_and_routing_profiles() -> None:
    a, b, c = (_hex32(index) for index in range(1, 4))
    evidence = compute_base_confidence(
        (a, b, c),
        {
            tuple(sorted((a, b))): 0.99,
            tuple(sorted((a, c))): 0.99,
            tuple(sorted((b, c))): 0.51,
        },
    )

    assert evidence.base_confidence == pytest.approx(0.638)
    assert route_cluster(evidence.base_confidence, evidence.cluster_size) == "defer"
    assert routing_actions_by_profile(evidence.base_confidence, evidence.cluster_size) == {
        "quick_low_hitl": "defer",
        "balanced_hitl": "review",
    }


def test_compute_base_confidence_does_not_treat_singletons_as_perfect() -> None:
    evidence = compute_base_confidence((_hex32(1),), {})

    assert evidence.min_edge_score == 0.0
    assert evidence.avg_edge_score == 0.0
    assert evidence.base_confidence == 0.0
    assert route_cluster(evidence.base_confidence, evidence.cluster_size) == "keep_separate"


def test_run_resolution_emits_cluster_confidence_and_suspicion_diagnostics(
    tmp_path: Path,
) -> None:
    a, b, c, d, e, f = (_hex32(index) for index in range(1, 7))
    entities_path = get_extraction_run_output_dir(tmp_path, "run_resolution") / "entities.parquet"
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_entities_table("run_resolution", [a, b, c, d, e, f]),
        entities_path,
    )
    scored_pairs_path = get_scored_pairs_output_path(tmp_path, "run_resolution")
    scored_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_scored_pairs_table(
            [
                ("run_resolution", a, b, 0.95),
                ("run_resolution", a, c, 0.95),
                ("run_resolution", b, c, 0.60),
                ("run_resolution", d, e, 0.95),
                ("run_resolution", d, f, 0.95),
            ]
        ),
        scored_pairs_path,
    )

    diagnostics = run_resolution(tmp_path, "run_resolution")
    components_payload = json.loads(
        get_resolution_components_path(tmp_path, "run_resolution").read_text()
    )
    diagnostics_payload = json.loads(
        get_resolution_diagnostics_path(tmp_path, "run_resolution").read_text()
    )

    assert diagnostics == diagnostics_payload
    assert components_payload["cluster_count"] == 3
    assert [cluster["entity_ids"] for cluster in components_payload["clusters"]] == [
        [a, b],
        [c],
        [d, e, f],
    ]
    assert diagnostics_payload["cluster_size_distribution"] == {"1": 1, "2": 1, "3": 1}
    assert diagnostics_payload["routing_action_counts_by_profile"] == {
        "balanced_hitl": {"auto_merge": 1, "keep_separate": 1, "review": 1},
        "quick_low_hitl": {"auto_merge": 1, "defer": 1, "keep_separate": 1},
    }
    assert diagnostics_payload["selected_route_action_counts"] == {
        "auto_merge": 1,
        "defer": 1,
        "keep_separate": 1,
    }
    assert diagnostics_payload["merged_edges_above_neutral_threshold"] == 3
    assert diagnostics_payload["suspicious_merges"][0]["entity_ids"] == [d, e, f]
    assert diagnostics_payload["suspicious_missed_merges"][0]["entity_id_a"] == a
    assert diagnostics_payload["suspicious_missed_merges"][0]["entity_id_b"] == c
    assert diagnostics_payload["timing_by_size_bucket"]["3-5"]["component_count"] == 2


def test_suspicious_missed_merge_examples_excludes_neutral_edges() -> None:
    a, b = (_hex32(index) for index in range(1, 3))
    table = _build_scored_pairs_table([("run_neutral", a, b, 0.80)])
    components, _ = build_phase1_components(
        "run_neutral",
        pl.from_arrow(table).select(PAIR_COLUMNS),
        [a, b],
    )
    solved_component = SolvedComponent(
        component_id=components[0].component_id,
        entity_ids=components[0].entity_ids,
        clusters=((a,), (b,)),
        clustering_method="test_fixture",
        objective_score=0.0,
        run_count=1,
        early_stopped=False,
        elapsed_ms=0.0,
    )

    cluster_rows = _make_cluster_rows(components, [solved_component])
    examples = _suspicious_missed_merge_examples(
        components,
        [solved_component],
        cluster_rows,
    )

    assert examples == []
