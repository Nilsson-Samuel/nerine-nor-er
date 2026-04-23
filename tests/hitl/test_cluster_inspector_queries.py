"""Tests for cluster inspector queries: members, edges, aliases, weakest-link."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.hitl.queries import (
    build_alias_table,
    find_weakest_edge,
    format_shap_reasons,
    load_cluster_edges,
    load_cluster_member_ids,
    load_cluster_members,
)
from src.matching.writer import get_matching_run_output_dir
from src.shared.paths import _encode_run_id_path_segment
from src.resolution.writer import get_resolution_run_output_dir
from src.shared.paths import get_extraction_run_output_dir
from src.shared.schemas import (
    ENTITIES_SCHEMA,
    RESOLVED_ENTITIES_SCHEMA,
    SCORED_PAIRS_SCHEMA,
)


RUN_ID = "inspector_test_run"
CLUSTER_A = "cluster_a"   # 3 members
CLUSTER_B = "cluster_b"   # 1 member (singleton)


def _build_entities_table() -> pa.Table:
    """Three entities in cluster_a, one in cluster_b."""
    rows = [
        {
            "run_id": RUN_ID,
            "entity_id": "e1",
            "doc_id": "doc_001",
            "chunk_id": "ch_01",
            "text": "DNB ASA",
            "normalized": "dnb asa",
            "type": "ORG",
            "char_start": 10,
            "char_end": 17,
            "context": "...kjøpte DNB ASA aksjer...",
            "count": 3,
            "positions": [
                {"chunk_id": "ch_01", "char_start": 10, "char_end": 17,
                 "page_num": 1, "source_unit_kind": "pdf_page"},
            ],
        },
        {
            "run_id": RUN_ID,
            "entity_id": "e2",
            "doc_id": "doc_001",
            "chunk_id": "ch_02",
            "text": "Den Norske Bank",
            "normalized": "dnb asa",
            "type": "ORG",
            "char_start": 50,
            "char_end": 65,
            "context": "...via Den Norske Bank konto...",
            "count": 1,
            "positions": [
                {"chunk_id": "ch_02", "char_start": 50, "char_end": 65,
                 "page_num": 2, "source_unit_kind": "pdf_page"},
            ],
        },
        {
            "run_id": RUN_ID,
            "entity_id": "e3",
            "doc_id": "doc_002",
            "chunk_id": "ch_03",
            "text": "DNB",
            "normalized": "dnb",
            "type": "ORG",
            "char_start": 0,
            "char_end": 3,
            "context": "DNB overførte beløpet...",
            "count": 2,
            "positions": [
                {"chunk_id": "ch_03", "char_start": 0, "char_end": 3,
                 "page_num": 1, "source_unit_kind": "pdf_page"},
            ],
        },
        {
            "run_id": RUN_ID,
            "entity_id": "e4",
            "doc_id": "doc_003",
            "chunk_id": "ch_04",
            "text": "Oslo",
            "normalized": "oslo",
            "type": "LOC",
            "char_start": 20,
            "char_end": 24,
            "context": "...i Oslo sentrum...",
            "count": 1,
            "positions": [
                {"chunk_id": "ch_04", "char_start": 20, "char_end": 24,
                 "page_num": 1, "source_unit_kind": "pdf_page"},
            ],
        },
    ]
    return pa.Table.from_pylist(rows, schema=ENTITIES_SCHEMA)


def _build_resolved_entities_table() -> pa.Table:
    """Map e1, e2, e3 -> cluster_a; e4 -> cluster_b."""
    from datetime import date

    rows = [
        {
            "run_id": RUN_ID, "cluster_id": CLUSTER_A, "entity_id": "e1",
            "doc_id": "doc_001", "entity_type": "ORG",
            "canonical_name": "DNB ASA", "canonical_type": "ORG",
            "cluster_size": 3, "confidence": 0.85, "needs_review": False,
            "route_action": "auto_merge", "clustering_method": "correlation",
            "component_id": "comp_1", "doc_ids": ["doc_001", "doc_002"],
            "most_recent_doc_id": "doc_002",
            "most_recent_doc_date": date(2026, 1, 15),
            "attributes": [("alias", ["DNB ASA", "Den Norske Bank", "DNB"])],
        },
        {
            "run_id": RUN_ID, "cluster_id": CLUSTER_A, "entity_id": "e2",
            "doc_id": "doc_001", "entity_type": "ORG",
            "canonical_name": "DNB ASA", "canonical_type": "ORG",
            "cluster_size": 3, "confidence": 0.85, "needs_review": False,
            "route_action": "auto_merge", "clustering_method": "correlation",
            "component_id": "comp_1", "doc_ids": ["doc_001", "doc_002"],
            "most_recent_doc_id": "doc_002",
            "most_recent_doc_date": date(2026, 1, 15),
            "attributes": [("alias", ["DNB ASA", "Den Norske Bank", "DNB"])],
        },
        {
            "run_id": RUN_ID, "cluster_id": CLUSTER_A, "entity_id": "e3",
            "doc_id": "doc_002", "entity_type": "ORG",
            "canonical_name": "DNB ASA", "canonical_type": "ORG",
            "cluster_size": 3, "confidence": 0.85, "needs_review": False,
            "route_action": "auto_merge", "clustering_method": "correlation",
            "component_id": "comp_1", "doc_ids": ["doc_001", "doc_002"],
            "most_recent_doc_id": "doc_002",
            "most_recent_doc_date": date(2026, 1, 15),
            "attributes": [("alias", ["DNB ASA", "Den Norske Bank", "DNB"])],
        },
        {
            "run_id": RUN_ID, "cluster_id": CLUSTER_B, "entity_id": "e4",
            "doc_id": "doc_003", "entity_type": "LOC",
            "canonical_name": "Oslo", "canonical_type": "LOC",
            "cluster_size": 1, "confidence": 0.0, "needs_review": False,
            "route_action": "keep_separate", "clustering_method": "correlation",
            "component_id": "comp_2", "doc_ids": ["doc_003"],
            "most_recent_doc_id": "doc_003",
            "most_recent_doc_date": date(2026, 1, 10),
            "attributes": [],
        },
    ]
    return pa.Table.from_pylist(rows, schema=RESOLVED_ENTITIES_SCHEMA)


def _build_scored_pairs_table() -> pa.Table:
    """Three intra-cluster edges for cluster_a: e1-e2, e1-e3, e2-e3."""
    scored_at = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    shap_type = SCORED_PAIRS_SCHEMA.field("shap_top5").type

    return pa.Table.from_arrays(
        [
            pa.array([RUN_ID] * 3, type=pa.string()),
            pa.array(["e1", "e1", "e2"], type=pa.string()),
            pa.array(["e2", "e3", "e3"], type=pa.string()),
            pa.array([0.91, 0.72, 0.85], type=pa.float32()),
            pa.array(["lgbm_v1"] * 3, type=pa.string()),
            pa.array([scored_at] * 3, type=pa.timestamp("us", tz="UTC")),
            pa.array([["faiss"], ["faiss", "phonetic"], ["faiss"]], type=pa.list_(pa.string())),
            pa.array(["faiss", "multi", "faiss"], type=pa.string()),
            pa.array([1, 2, 1], type=pa.int8()),
            pa.array(
                [
                    [{"feature": "embedding_sim", "value": 0.12}],
                    [{"feature": "name_jaro", "value": -0.03},
                     {"feature": "embedding_sim", "value": 0.08}],
                    [],
                ],
                type=shap_type,
            ),
        ],
        schema=SCORED_PAIRS_SCHEMA,
    )


@pytest.fixture()
def inspector_data_dir(tmp_path: Path) -> Path:
    """Write all three parquet artifacts needed for the inspector."""
    # entities.parquet in per-run extraction dir
    extraction_dir = get_extraction_run_output_dir(tmp_path, RUN_ID)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(_build_entities_table(), extraction_dir / "entities.parquet")

    # resolved_entities.parquet in per-run resolution dir
    resolution_dir = get_resolution_run_output_dir(tmp_path, RUN_ID)
    resolution_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_resolved_entities_table(),
        resolution_dir / "resolved_entities.parquet",
    )

    # scored_pairs.parquet in per-run matching dir
    matching_dir = get_matching_run_output_dir(tmp_path, RUN_ID)
    matching_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_scored_pairs_table(),
        matching_dir / "scored_pairs.parquet",
    )

    return tmp_path


# ── load_cluster_members ─────────────────────────────────────────────────────


def test_load_cluster_member_ids_returns_cluster_members(
    inspector_data_dir: Path,
) -> None:
    member_ids = sorted(load_cluster_member_ids(inspector_data_dir, RUN_ID, CLUSTER_A))
    assert member_ids == ["e1", "e2", "e3"]


def test_load_cluster_member_ids_empty_for_unknown_cluster(
    inspector_data_dir: Path,
) -> None:
    assert load_cluster_member_ids(inspector_data_dir, RUN_ID, "no_such_cluster") == []


def test_load_cluster_members_returns_correct_count(
    inspector_data_dir: Path,
) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_A)
    assert members.height == 3


def test_load_cluster_members_has_expected_ids(
    inspector_data_dir: Path,
) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_A)
    ids = sorted(members["entity_id"].to_list())
    assert ids == ["e1", "e2", "e3"]


def test_load_cluster_members_singleton(inspector_data_dir: Path) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_B)
    assert members.height == 1
    assert members["entity_id"][0] == "e4"


def test_load_cluster_members_includes_provenance_columns(
    inspector_data_dir: Path,
) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_A)
    for col in ("context", "chunk_id", "char_start", "char_end"):
        assert col in members.columns


def test_load_cluster_members_empty_for_unknown_cluster(
    inspector_data_dir: Path,
) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, "no_such_cluster")
    assert members.is_empty()


def test_load_cluster_members_empty_for_missing_files(tmp_path: Path) -> None:
    members = load_cluster_members(tmp_path, "no_run", "no_cluster")
    assert members.is_empty()


def test_load_cluster_members_tolerates_missing_optional_columns(
    tmp_path: Path,
) -> None:
    extraction_dir = get_extraction_run_output_dir(tmp_path, RUN_ID)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_entities_table().drop_columns(["context", "char_end"]),
        extraction_dir / "entities.parquet",
    )

    resolution_dir = get_resolution_run_output_dir(tmp_path, RUN_ID)
    resolution_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        _build_resolved_entities_table(),
        resolution_dir / "resolved_entities.parquet",
    )

    members = load_cluster_members(tmp_path, RUN_ID, CLUSTER_A)

    assert members.height == 3
    assert "context" not in members.columns
    assert "char_end" not in members.columns


# ── load_cluster_edges ───────────────────────────────────────────────────────


def test_load_cluster_edges_returns_all_intra_edges(
    inspector_data_dir: Path,
) -> None:
    edges = load_cluster_edges(inspector_data_dir, RUN_ID, CLUSTER_A)
    assert edges.height == 3


def test_load_cluster_edges_sorted_ascending_by_score(
    inspector_data_dir: Path,
) -> None:
    edges = load_cluster_edges(inspector_data_dir, RUN_ID, CLUSTER_A)
    scores = edges["score"].to_list()
    assert scores == sorted(scores)


def test_load_cluster_edges_empty_for_singleton(
    inspector_data_dir: Path,
) -> None:
    """Singleton cluster has no intra-cluster edges."""
    edges = load_cluster_edges(inspector_data_dir, RUN_ID, CLUSTER_B)
    assert edges.is_empty()


def test_load_cluster_edges_includes_shap(inspector_data_dir: Path) -> None:
    edges = load_cluster_edges(inspector_data_dir, RUN_ID, CLUSTER_A)
    assert "shap_top5" in edges.columns


def test_load_cluster_edges_empty_for_missing_files(tmp_path: Path) -> None:
    edges = load_cluster_edges(tmp_path, "no_run", "no_cluster")
    assert edges.is_empty()


# ── find_weakest_edge ────────────────────────────────────────────────────────


def test_weakest_edge_is_minimum_score(inspector_data_dir: Path) -> None:
    edges = load_cluster_edges(inspector_data_dir, RUN_ID, CLUSTER_A)
    weakest = find_weakest_edge(edges)
    assert weakest is not None
    # e1-e3 has score 0.72, the lowest
    assert weakest["entity_id_a"] == "e1"
    assert weakest["entity_id_b"] == "e3"
    assert weakest["score"] == pytest.approx(0.72, abs=0.01)


def test_weakest_edge_none_for_empty_frame(inspector_data_dir: Path) -> None:
    edges = load_cluster_edges(inspector_data_dir, RUN_ID, CLUSTER_B)
    assert find_weakest_edge(edges) is None


# ── build_alias_table ────────────────────────────────────────────────────────


def test_alias_table_groups_by_normalized(inspector_data_dir: Path) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_A)
    aliases = build_alias_table(members)
    # Two normalized forms: "dnb asa" (e1, e2) and "dnb" (e3)
    assert aliases.height == 2


def test_alias_table_surface_forms_correct(inspector_data_dir: Path) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_A)
    aliases = build_alias_table(members)

    dnb_asa_row = aliases.filter(aliases["normalized"] == "dnb asa")
    forms = sorted(dnb_asa_row["surface_forms"][0])
    assert forms == ["DNB ASA", "Den Norske Bank"]


def test_alias_table_mention_count_is_sum(inspector_data_dir: Path) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_A)
    aliases = build_alias_table(members)

    # "dnb asa" normalized: e1 (count=3) + e2 (count=1) = 4
    dnb_asa_row = aliases.filter(aliases["normalized"] == "dnb asa")
    assert dnb_asa_row["mention_count"][0] == 4


def test_alias_table_sorted_by_mention_count_desc(
    inspector_data_dir: Path,
) -> None:
    members = load_cluster_members(inspector_data_dir, RUN_ID, CLUSTER_A)
    aliases = build_alias_table(members)
    counts = aliases["mention_count"].to_list()
    assert counts == sorted(counts, reverse=True)


def test_alias_table_empty_for_empty_members() -> None:
    import polars as pl
    aliases = build_alias_table(pl.DataFrame())
    assert aliases.is_empty()


# ── format_shap_reasons ──────────────────────────────────────────────────────


def test_format_shap_reasons_with_values() -> None:
    shap = [{"feature": "embedding_sim", "value": 0.12},
            {"feature": "name_jaro", "value": -0.03}]
    result = format_shap_reasons(shap)
    assert "embedding_sim +0.120" in result
    assert "name_jaro -0.030" in result


def test_format_shap_reasons_empty() -> None:
    assert format_shap_reasons([]) == "-"
    assert format_shap_reasons(None) == "-"
