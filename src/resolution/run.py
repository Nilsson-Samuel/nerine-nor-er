"""Phase-1 resolution orchestration for clustering, canonicalization, and final outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from src.matching.writer import get_scored_pairs_output_path
from src.resolution.canonicalization import (
    build_cluster_records,
    build_resolved_entity_rows,
)
from src.resolution.clustering import (
    ComponentState,
    SolvedComponent,
    _giant_component_warnings,
    _size_distribution,
    build_phase1_components,
    component_timing_rows,
    solve_retained_components,
    summarize_components,
    summarize_timing_by_size_bucket,
)
from src.resolution.confidence import (
    compute_base_confidence,
    count_routing_actions,
    route_cluster,
    routing_actions_by_profile,
)
from src.shared.config import OBJECTIVE_NEUTRAL_THRESHOLD, ROUTING_PROFILE
from src.shared import schemas
from src.resolution.writer import (
    build_resolved_entities_table,
    get_clusters_output_path,
    get_resolution_components_path,
    get_resolution_diagnostics_path,
    get_resolved_entities_output_path,
    write_resolution_json,
    write_resolved_entities,
)


def _load_entities_frame(data_dir: Path | str, run_id: str) -> pl.DataFrame:
    """Load one run from entities.parquet for clustering and lineage joins."""
    table = pq.read_table(Path(data_dir) / "entities.parquet")
    errors = schemas.validate_contract_rules(table, "entities")
    if errors:
        raise ValueError(f"entities failed contract validation: {errors}")

    frame = pl.from_arrow(table).filter(pl.col("run_id") == run_id).sort("entity_id")
    if frame.is_empty():
        raise ValueError(f"run_id not found in entities.parquet: {run_id}")
    return frame


def _load_scored_pairs_frame(data_dir: Path | str, run_id: str) -> pl.DataFrame:
    """Load one run from scored_pairs.parquet with full edge lineage columns."""
    scored_pairs_path = get_scored_pairs_output_path(data_dir, run_id)
    if not scored_pairs_path.exists():
        raise ValueError(
            f"missing scored pairs for run_id={run_id} at {scored_pairs_path}; "
            "rerun matching scoring for this run"
        )

    table = pq.read_table(scored_pairs_path)
    errors = schemas.validate_contract_rules(table, "scored_pairs")
    if errors:
        raise ValueError(f"scored_pairs failed contract validation: {errors}")

    return (
        pl.from_arrow(table)
        .filter(pl.col("run_id") == run_id)
        .sort(["entity_id_a", "entity_id_b"])
    )


def _make_cluster_rows(
    components: list[ComponentState],
    solved_components: list[SolvedComponent],
) -> list[dict[str, Any]]:
    """Build JSON-safe cluster rows with evidence and routing metadata."""
    component_by_id = {component.component_id: component for component in components}
    rows: list[dict[str, Any]] = []

    for solved_component in solved_components:
        component = component_by_id[solved_component.component_id]
        for cluster in solved_component.clusters:
            evidence = compute_base_confidence(cluster, component.edge_scores)
            route_by_profile = routing_actions_by_profile(
                evidence.base_confidence,
                evidence.cluster_size,
            )
            rows.append(
                {
                    "cluster_id": evidence.cluster_id,
                    "component_id": component.component_id,
                    "entity_ids": list(evidence.entity_ids),
                    "cluster_size": evidence.cluster_size,
                    "clustering_method": solved_component.clustering_method,
                    "component_objective_score": round(solved_component.objective_score, 6),
                    "component_solve_elapsed_ms": round(solved_component.elapsed_ms, 3),
                    "pivot_run_count": solved_component.run_count,
                    "actual_edge_count": evidence.actual_edge_count,
                    "possible_edge_count": evidence.possible_edge_count,
                    "density": round(evidence.density, 6),
                    "min_edge_score": round(evidence.min_edge_score, 6),
                    "avg_edge_score": round(evidence.avg_edge_score, 6),
                    "base_confidence": round(evidence.base_confidence, 6),
                    "route_action": route_cluster(
                        evidence.base_confidence,
                        evidence.cluster_size,
                        profile=ROUTING_PROFILE,
                    ),
                    "routing_actions_by_profile": route_by_profile,
                    "suspicious_merge": (
                        evidence.cluster_size > 1
                        and (
                            evidence.min_edge_score < OBJECTIVE_NEUTRAL_THRESHOLD
                            or evidence.density < 1.0
                        )
                    ),
                }
            )

    return sorted(rows, key=lambda row: (row["component_id"], row["entity_ids"]))


def _suspicious_merge_examples(
    cluster_rows: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return compact suspicious merge examples for manual inspection."""
    suspicious_rows = [row for row in cluster_rows if row["suspicious_merge"]]
    suspicious_rows.sort(
        key=lambda row: (
            float(row["base_confidence"]),
            float(row["min_edge_score"]),
            -int(row["cluster_size"]),
        )
    )
    return [
        {
            "cluster_id": row["cluster_id"],
            "component_id": row["component_id"],
            "entity_ids": row["entity_ids"],
            "cluster_size": row["cluster_size"],
            "base_confidence": row["base_confidence"],
            "min_edge_score": row["min_edge_score"],
            "density": row["density"],
            "route_action": row["route_action"],
        }
        for row in suspicious_rows[:limit]
    ]


def _suspicious_missed_merge_examples(
    components: list[ComponentState],
    solved_components: list[SolvedComponent],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return high-score retained edges that the clustering still split apart."""
    component_by_id = {component.component_id: component for component in components}
    examples: list[dict[str, Any]] = []

    for solved_component in solved_components:
        component = component_by_id[solved_component.component_id]
        cluster_index: dict[str, int] = {}
        cluster_ids: list[str] = []
        for cluster_number, cluster in enumerate(solved_component.clusters):
            cluster_id = compute_base_confidence(cluster, component.edge_scores).cluster_id
            cluster_ids.append(cluster_id)
            for entity_id in cluster:
                cluster_index[entity_id] = cluster_number

        for edge_key, score in component.edge_scores.items():
            if component.objective_weights[edge_key] <= 0.0:
                continue
            entity_id_a, entity_id_b = edge_key
            if cluster_index[entity_id_a] == cluster_index[entity_id_b]:
                continue
            examples.append(
                {
                    "component_id": component.component_id,
                    "entity_id_a": entity_id_a,
                    "entity_id_b": entity_id_b,
                    "score": round(score, 6),
                    "cluster_id_a": cluster_ids[cluster_index[entity_id_a]],
                    "cluster_id_b": cluster_ids[cluster_index[entity_id_b]],
                }
            )

    examples.sort(key=lambda row: (-float(row["score"]), row["component_id"]))
    return examples[:limit]


def _base_confidence_summary(cluster_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Build a compact numeric summary of the cluster confidence distribution."""
    if not cluster_rows:
        return {"min": 0.0, "avg": 0.0, "max": 0.0}
    values = [float(row["base_confidence"]) for row in cluster_rows]
    return {
        "min": round(min(values), 6),
        "avg": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
    }


def _count_merged_edges_above_neutral(
    components: list[ComponentState],
    solved_components: list[SolvedComponent],
) -> int:
    """Count retained merge-evidence edges that ended up inside one cluster."""
    component_by_id = {component.component_id: component for component in components}
    merged_edge_count = 0

    for solved_component in solved_components:
        component = component_by_id[solved_component.component_id]
        cluster_index: dict[str, int] = {}
        for cluster_number, cluster in enumerate(solved_component.clusters):
            for entity_id in cluster:
                cluster_index[entity_id] = cluster_number

        for edge_key, score in component.edge_scores.items():
            if (
                component.objective_weights[edge_key] > 0.0
                and cluster_index[edge_key[0]] == cluster_index[edge_key[1]]
            ):
                merged_edge_count += 1

    return merged_edge_count


def _build_enriched_diagnostics(
    phase1_diagnostics: dict[str, Any],
    components: list[ComponentState],
    solved_components: list[SolvedComponent],
    cluster_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach clustering, confidence, suspicious-example, and timing diagnostics."""
    cluster_sizes = [int(row["cluster_size"]) for row in cluster_rows]
    cluster_singleton_count = sum(size == 1 for size in cluster_sizes)
    timing_rows = component_timing_rows(components, solved_components)
    routing_counts = count_routing_actions(cluster_rows)
    selected_route_action_counts = routing_counts[ROUTING_PROFILE]

    diagnostics = dict(phase1_diagnostics)
    diagnostics.update(
        {
            "cluster_count": len(cluster_rows),
            "cluster_singleton_count": cluster_singleton_count,
            "cluster_singleton_rate": (
                round(cluster_singleton_count / len(cluster_rows), 6) if cluster_rows else 0.0
            ),
            "cluster_size_distribution": _size_distribution(cluster_sizes),
            "giant_cluster_warnings": _giant_component_warnings(
                cluster_sizes,
                diagnostics["total_node_count"],
            ),
            "selected_routing_profile": ROUTING_PROFILE,
            "selected_route_action_counts": selected_route_action_counts,
            "routing_action_counts_by_profile": routing_counts,
            "base_confidence_summary": _base_confidence_summary(cluster_rows),
            "merged_edges_above_neutral_threshold": _count_merged_edges_above_neutral(
                components,
                solved_components,
            ),
            "suspicious_merges": _suspicious_merge_examples(cluster_rows),
            "suspicious_missed_merges": _suspicious_missed_merge_examples(
                components,
                solved_components,
            ),
            "component_timings": timing_rows,
            "timing_by_size_bucket": summarize_timing_by_size_bucket(timing_rows),
        }
    )
    return diagnostics


def run_resolution(data_dir: Path | str, run_id: str) -> dict[str, Any]:
    """Run retained-component clustering and persist final resolution artifacts."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    components_path = get_resolution_components_path(data_dir, run_id)
    diagnostics_path = get_resolution_diagnostics_path(data_dir, run_id)
    resolved_entities_path = get_resolved_entities_output_path(data_dir, run_id)
    clusters_path = get_clusters_output_path(data_dir, run_id)

    entities = _load_entities_frame(data_dir, run_id)
    scored_pairs = _load_scored_pairs_frame(data_dir, run_id)
    entity_ids = entities.get_column("entity_id").to_list()
    components, phase1_diagnostics = build_phase1_components(run_id, scored_pairs, entity_ids)
    solved_components = solve_retained_components(components)
    cluster_rows = _make_cluster_rows(components, solved_components)
    cluster_records = build_cluster_records(cluster_rows, entities, scored_pairs)
    resolved_rows = build_resolved_entity_rows(cluster_records)
    resolved_entities_table = build_resolved_entities_table(resolved_rows)
    resolved_errors = schemas.validate_contract_rules(
        resolved_entities_table,
        "resolved_entities",
    )
    if resolved_errors:
        raise ValueError(f"resolved_entities failed contract validation: {resolved_errors}")

    diagnostics = _build_enriched_diagnostics(
        phase1_diagnostics,
        components,
        solved_components,
        cluster_rows,
    )
    diagnostics["resolved_entity_row_count"] = len(resolved_rows)
    component_payload = {
        "run_id": run_id,
        "component_count": len(components),
        "cluster_count": len(cluster_rows),
        "components": summarize_components(components, solved_components),
        "clusters": cluster_rows,
    }
    clusters_payload = {
        "run_id": run_id,
        "cluster_count": len(cluster_records),
        "clusters": cluster_records,
    }

    write_resolved_entities(resolved_entities_table, resolved_entities_path)
    write_resolution_json(component_payload, components_path)
    write_resolution_json(diagnostics, diagnostics_path)
    write_resolution_json(clusters_payload, clusters_path)
    return diagnostics
