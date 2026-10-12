"""Tests for cross-cluster merge candidate queries used to review kept-apart entities."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.hitl.queries import (
    UNMERGED_PRUNED,
    UNMERGED_SPLIT,
    UNMERGED_WEAK,
    build_candidate_table,
    load_candidate_table,
    load_cross_cluster_pairs,
    load_entities_by_id,
    resolve_run_thresholds,
    top_candidate_scores,
)
from src.matching.writer import get_matching_run_output_dir
from src.resolution.writer import get_resolution_run_output_dir
from src.shared.config import KEEP_SCORE_THRESHOLD, OBJECTIVE_NEUTRAL_THRESHOLD
from src.shared.paths import get_extraction_run_output_dir
from src.shared.schemas import (
    ENTITIES_SCHEMA,
    RESOLVED_ENTITIES_SCHEMA,
    SCORED_PAIRS_SCHEMA,
)


RUN_ID = "candidate_test_run"
KEEP, NEUTRAL = 0.60, 0.80

# cluster_a = {e1, e2}, cluster_b = {e3} (singleton), cluster_c = {e4} (singleton)
CLUSTER_OF = {"e1": "cluster_a", "e2": "cluster_a", "e3": "cluster_b", "e4": "cluster_c"}
TEXT_OF = {"e1": "Ola Nordmann", "e2": "O. Nordmann", "e3": "Ola Nordman", "e4": "Oslo"}

# (entity_id_a, entity_id_b, score)
PAIRS = [
    ("e1", "e2", 0.95),  # inside cluster_a, never a candidate
    ("e1", "e3", 0.90),  # voted to merge, split by clustering
    ("e2", "e3", 0.70),  # retained but below neutral
    ("e3", "e4", 0.40),  # pruned below keep
    ("e1", "e4", 0.10),  # below the candidate floor used in tests
]


def _entities_table() -> pa.Table:
    rows = [
        {
            "run_id": RUN_ID, "entity_id": eid, "doc_id": f"doc_{eid}",
            "chunk_id": f"ch_{eid}", "text": TEXT_OF[eid],
            "normalized": TEXT_OF[eid].lower(),
            "type": "LOC" if eid == "e4" else "PER",
            "char_start": 0, "char_end": len(TEXT_OF[eid]),
            "context": f"...{TEXT_OF[eid]}...", "count": 1, "positions": [],
        }
        for eid in CLUSTER_OF
    ]
    return pa.Table.from_pylist(rows, schema=ENTITIES_SCHEMA)


def _resolved_table() -> pa.Table:
    sizes = {"cluster_a": 2, "cluster_b": 1, "cluster_c": 1}
    rows = [
        {
            "run_id": RUN_ID, "cluster_id": cid, "entity_id": eid,
            "doc_id": f"doc_{eid}", "entity_type": "PER",
            "canonical_name": TEXT_OF[eid], "canonical_type": "PER",
            "cluster_size": sizes[cid], "confidence": 0.9 if cid == "cluster_a" else 0.0,
            "needs_review": False,
            "route_action": "auto_merge" if cid == "cluster_a" else "keep_separate",
            "clustering_method": "correlation", "component_id": "comp_1",
            "doc_ids": [f"doc_{eid}"], "most_recent_doc_id": f"doc_{eid}",
            "most_recent_doc_date": date(2026, 1, 1), "attributes": [],
        }
        for eid, cid in CLUSTER_OF.items()
    ]
    return pa.Table.from_pylist(rows, schema=RESOLVED_ENTITIES_SCHEMA)


def _scored_pairs_table() -> pa.Table:
    scored_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "run_id": RUN_ID, "entity_id_a": a, "entity_id_b": b, "score": score,
            "model_version": "lgbm_v1", "scored_at": scored_at,
            "blocking_methods": ["faiss"], "blocking_source": "faiss",
            "blocking_method_count": 1, "shap_top5": [],
        }
        for a, b, score in PAIRS
    ]
    return pa.Table.from_pylist(rows, schema=SCORED_PAIRS_SCHEMA)


def _cluster_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "cluster_id": ["cluster_a", "cluster_b", "cluster_c"],
        "canonical_name": ["Ola Nordmann", "Ola Nordman", "Oslo"],
        "cluster_size": [2, 1, 1],
        "route_balanced_hitl": ["auto_merge", "keep_separate", "keep_separate"],
        "route_quick_low_hitl": ["auto_merge", "keep_separate", "keep_separate"],
    })


@pytest.fixture()
def candidate_data_dir(tmp_path: Path) -> Path:
    """Write entities, resolved entities and scored pairs for one run."""
    extraction_dir = get_extraction_run_output_dir(tmp_path, RUN_ID)
    extraction_dir.mkdir(parents=True)
    pq.write_table(_entities_table(), extraction_dir / "entities.parquet")

    resolution_dir = get_resolution_run_output_dir(tmp_path, RUN_ID)
    resolution_dir.mkdir(parents=True)
    pq.write_table(_resolved_table(), resolution_dir / "resolved_entities.parquet")

    matching_dir = get_matching_run_output_dir(tmp_path, RUN_ID)
    matching_dir.mkdir(parents=True)
    pq.write_table(_scored_pairs_table(), matching_dir / "scored_pairs.parquet")

    return tmp_path


def _candidate_table(data_dir: Path, min_score: float = 0.30) -> pl.DataFrame:
    return load_candidate_table(
        data_dir, RUN_ID, _cluster_frame(), "balanced_hitl", min_score, (KEEP, NEUTRAL)
    )


# ── resolve_run_thresholds ───────────────────────────────────────────────────


def test_resolve_run_thresholds_prefers_run_diagnostics() -> None:
    diagnostics = {
        "thresholds": {"keep_score_threshold": 0.5, "objective_neutral_threshold": 0.7}
    }
    assert resolve_run_thresholds(diagnostics) == (0.5, 0.7)


@pytest.mark.parametrize("diagnostics", [None, {}, {"thresholds": None}])
def test_resolve_run_thresholds_falls_back_to_config(diagnostics) -> None:
    assert resolve_run_thresholds(diagnostics) == (
        KEEP_SCORE_THRESHOLD,
        OBJECTIVE_NEUTRAL_THRESHOLD,
    )


# ── load_cross_cluster_pairs ─────────────────────────────────────────────────


def test_cross_cluster_pairs_exclude_pairs_inside_one_cluster(
    candidate_data_dir: Path,
) -> None:
    pairs = load_cross_cluster_pairs(candidate_data_dir, RUN_ID, 0.0)
    assert not pairs.filter(pl.col("cluster_id") == pl.col("candidate_cluster_id")).height
    assert pairs.filter(
        (pl.col("entity_id") == "e1") & (pl.col("candidate_entity_id") == "e2")
    ).is_empty()


def test_cross_cluster_pairs_appear_from_both_sides(candidate_data_dir: Path) -> None:
    pairs = load_cross_cluster_pairs(candidate_data_dir, RUN_ID, 0.0)
    # 4 cross-cluster pairs, each returned once per side
    assert pairs.height == 8
    singleton_view = pairs.filter(pl.col("cluster_id") == "cluster_b")
    assert sorted(singleton_view["candidate_entity_id"].to_list()) == ["e1", "e2", "e4"]


def test_cross_cluster_pairs_respect_min_score(candidate_data_dir: Path) -> None:
    pairs = load_cross_cluster_pairs(candidate_data_dir, RUN_ID, 0.30)
    assert pairs["score"].min() >= 0.30
    assert "e4" not in pairs.filter(pl.col("entity_id") == "e1")["candidate_entity_id"]


def test_cross_cluster_pairs_empty_when_artifacts_missing(tmp_path: Path) -> None:
    assert load_cross_cluster_pairs(tmp_path, RUN_ID, 0.0).is_empty()


# ── build_candidate_table / load_candidate_table ─────────────────────────────


def test_candidate_table_explains_why_pairs_were_not_merged(
    candidate_data_dir: Path,
) -> None:
    table = _candidate_table(candidate_data_dir)
    reason_by_pair = {
        tuple(sorted((row["entity_id"], row["candidate_entity_id"]))): row["unmerged_reason"]
        for row in table.iter_rows(named=True)
    }
    assert reason_by_pair == {
        ("e1", "e3"): UNMERGED_SPLIT,
        ("e2", "e3"): UNMERGED_WEAK,
        ("e3", "e4"): UNMERGED_PRUNED,
    }


def test_candidate_table_carries_candidate_cluster_routing(
    candidate_data_dir: Path,
) -> None:
    table = _candidate_table(candidate_data_dir)
    row = table.filter(
        (pl.col("cluster_id") == "cluster_b") & (pl.col("candidate_entity_id") == "e1")
    ).row(0, named=True)
    assert row["text"] == "Ola Nordman"
    assert row["candidate_text"] == "Ola Nordmann"
    assert row["candidate_canonical_name"] == "Ola Nordmann"
    assert row["candidate_cluster_size"] == 2
    assert row["candidate_route"] == "auto_merge"


def test_candidate_table_sorted_strongest_first_with_rank_per_cluster(
    candidate_data_dir: Path,
) -> None:
    table = _candidate_table(candidate_data_dir)
    assert table["score"].to_list() == sorted(table["score"].to_list(), reverse=True)

    singleton = table.filter(pl.col("cluster_id") == "cluster_b").sort("candidate_rank")
    assert singleton["candidate_entity_id"].to_list() == ["e1", "e2", "e4"]
    assert singleton["candidate_rank"].to_list() == [1, 2, 3]


def test_candidate_table_empty_without_pairs(tmp_path: Path) -> None:
    assert _candidate_table(tmp_path).is_empty()


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (KEEP - 0.01, UNMERGED_PRUNED),
        (KEEP, UNMERGED_WEAK),
        (NEUTRAL, UNMERGED_WEAK),
        (NEUTRAL + 0.01, UNMERGED_SPLIT),
    ],
)
def test_unmerged_reason_threshold_boundaries(score: float, expected: str) -> None:
    pairs = pl.DataFrame({
        "cluster_id": ["cluster_b"],
        "entity_id": ["e3"],
        "candidate_cluster_id": ["cluster_a"],
        "candidate_entity_id": ["e1"],
        "score": [score],
        "blocking_source": ["faiss"],
    })
    entities = pl.from_arrow(_entities_table())
    table = build_candidate_table(
        pairs, entities, _cluster_frame(), "route_balanced_hitl", KEEP, NEUTRAL
    )
    assert table["unmerged_reason"].to_list() == [expected]


# ── top_candidate_scores ─────────────────────────────────────────────────────


def test_top_candidate_scores_takes_max_per_cluster(candidate_data_dir: Path) -> None:
    top = dict(top_candidate_scores(_candidate_table(candidate_data_dir)).iter_rows())
    assert top == pytest.approx({"cluster_a": 0.90, "cluster_b": 0.90, "cluster_c": 0.40})


def test_top_candidate_scores_empty_table_keeps_schema() -> None:
    top = top_candidate_scores(pl.DataFrame())
    assert top.is_empty()
    assert top.columns == ["cluster_id", "top_candidate_score"]


# ── load_entities_by_id ──────────────────────────────────────────────────────


def test_load_entities_by_id_returns_requested_rows(candidate_data_dir: Path) -> None:
    entities = load_entities_by_id(candidate_data_dir, RUN_ID, ["e4", "e1"])
    assert entities["entity_id"].to_list() == ["e1", "e4"]
    assert "positions" in entities.columns


def test_load_entities_by_id_empty_for_no_ids(candidate_data_dir: Path) -> None:
    assert load_entities_by_id(candidate_data_dir, RUN_ID, []).is_empty()
