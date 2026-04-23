"""Correlation-clustering helpers for the resolution stage.

This module consumes same-entity probabilities from the matching stage
(`scored_pairs.parquet`). Those scores come from the LightGBM reranker and are
treated here as merge evidence between entity mentions.

Resolution happens in two steps:
1. Keep only pair scores at or above `KEEP_SCORE_THRESHOLD` and build retained
   connected components. This keeps clearly implausible edges out of later
   clustering work.
2. Solve each retained component with Pivot correlation clustering. For that
   objective, each retained edge score is shifted by
   `OBJECTIVE_NEUTRAL_THRESHOLD`, so higher scores vote for merging and lower
   retained scores vote for splitting.

The outputs from this module are deterministic component states, clustering
results, and diagnostics that explain how the threshold policy behaved.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, sqrt
from time import perf_counter
from typing import Any, Callable

import networkx as nx
import polars as pl
from src.shared.config import (
    BASE_CONFIDENCE_REVIEW_THRESHOLD,
    KEEP_SCORE_THRESHOLD,
    OBJECTIVE_NEUTRAL_THRESHOLD,
)

# Skip Pivot when a retained component is too small to express a real contradiction.
TRIVIAL_NODE_LIMIT = 2
TRIVIAL_EDGE_LIMIT = 1

# Conservative restart budget for the multi-start Pivot solver.
PIVOT_RUN_CAP = 40
# Stop early when repeated Pivot restarts stop improving.
PIVOT_PATIENCE = 5
# Ignore tiny float noise when comparing objective gains across restarts.
MIN_OBJECTIVE_GAIN = 1e-6

# Warn when one retained component becomes large enough to deserve inspection.
GIANT_COMPONENT_NODE_WARNING_SIZE = 25

# Warn when one component absorbs a large share of the retained graph.
GIANT_COMPONENT_SHARE_WARNING = 0.15

# The neutral point should leave enough retained weak edges to act as split evidence.
MIN_OBJECTIVE_NEUTRAL_GAP = 0.10

PAIR_COLUMNS = ["run_id", "entity_id_a", "entity_id_b", "score"]


@dataclass(frozen=True, slots=True)
class ComponentState:
    """Deterministic local view of one retained connected component.

    The component stores retained edge scores plus the shifted
    correlation-clustering weights used by Pivot. `positive_neighbors` means
    neighbors whose retained edge is above the neutral objective threshold and
    therefore votes in favor of a merge.
    """

    component_id: str
    entity_ids: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]
    neighbor_sets: dict[str, frozenset[str]]
    positive_neighbors: dict[str, tuple[str, ...]]
    edge_scores: dict[tuple[str, str], float]
    objective_weights: dict[tuple[str, str], float]
    retained_edge_count: int

    @property
    def is_trivial(self) -> bool:
        """Return whether this component can bypass multi-start Pivot."""
        return (
            len(self.entity_ids) <= TRIVIAL_NODE_LIMIT
            and self.retained_edge_count <= TRIVIAL_EDGE_LIMIT
        )


@dataclass(frozen=True, slots=True)
class SolvedComponent:
    """Resolved clustering output for one retained component."""

    component_id: str
    entity_ids: tuple[str, ...]
    clusters: tuple[tuple[str, ...], ...]
    clustering_method: str
    objective_score: float
    run_count: int
    early_stopped: bool
    elapsed_ms: float


def include_edge(score: float, threshold: float = KEEP_SCORE_THRESHOLD) -> bool:
    """Return whether a scored pair should stay in the retained graph."""
    return float(score) >= float(threshold)


def validate_resolution_thresholds(
    keep_threshold: float,
    neutral_threshold: float,
) -> tuple[float, float]:
    """Validate retained-graph and objective thresholds as one policy."""
    keep = float(keep_threshold)
    neutral = float(neutral_threshold)
    if not 0.0 <= keep <= 1.0:
        raise ValueError("keep_score_threshold must be in [0, 1]")
    if not 0.0 <= neutral <= 1.0:
        raise ValueError("objective_neutral_threshold must be in [0, 1]")
    if neutral + MIN_OBJECTIVE_GAIN < keep + MIN_OBJECTIVE_NEUTRAL_GAP:
        raise ValueError(
            "objective_neutral_threshold must be at least 0.10 above "
            "keep_score_threshold"
        )
    return keep, neutral


def score_to_objective_weight(
    score: float,
    neutral_threshold: float = OBJECTIVE_NEUTRAL_THRESHOLD,
) -> float:
    """Convert a reranker probability into a correlation-clustering weight."""
    weight = float(score) - float(neutral_threshold)
    if abs(weight) <= MIN_OBJECTIVE_GAIN:
        return 0.0
    return weight


def make_component_id(entity_ids: tuple[str, ...] | list[str]) -> str:
    """Build a stable component identifier from sorted member entity IDs."""
    ordered_ids = tuple(sorted(entity_ids))
    payload = "|".join(ordered_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def build_retained_graph(
    scored_pairs: pl.DataFrame,
    entity_ids: list[str] | tuple[str, ...],
    keep_threshold: float = KEEP_SCORE_THRESHOLD,
) -> nx.Graph:
    """Build one graph over all run entities, keeping only plausible merge edges."""
    graph = nx.Graph()
    graph.add_nodes_from(entity_ids)

    if scored_pairs.is_empty():
        return graph

    known_entity_ids = set(entity_ids)
    pair_entity_ids = {
        *scored_pairs["entity_id_a"].to_list(),
        *scored_pairs["entity_id_b"].to_list(),
    }
    unknown_entity_ids = sorted(pair_entity_ids - known_entity_ids)
    if unknown_entity_ids:
        raise ValueError(
            "scored_pairs contains entity IDs missing from entities.parquet: "
            f"{unknown_entity_ids[:5]}"
        )

    retained_pairs = scored_pairs.filter(
        pl.col("score") >= float(keep_threshold)
    ).select(["entity_id_a", "entity_id_b", "score"])
    if retained_pairs.is_empty():
        return graph

    graph.add_weighted_edges_from(retained_pairs.iter_rows())
    return graph


def build_component_state(
    graph: nx.Graph,
    entity_ids: tuple[str, ...],
    neutral_threshold: float = OBJECTIVE_NEUTRAL_THRESHOLD,
) -> ComponentState:
    """Convert one retained component into local structures for repeated Pivot runs."""
    adjacency: dict[str, tuple[str, ...]] = {}
    neighbor_sets: dict[str, frozenset[str]] = {}
    positive_neighbors: dict[str, tuple[str, ...]] = {}
    edge_scores: dict[tuple[str, str], float] = {}
    objective_weights: dict[tuple[str, str], float] = {}

    subgraph = graph.subgraph(entity_ids)
    edge_rows = sorted(
        subgraph.edges(data="weight"),
        key=lambda row: tuple(sorted((row[0], row[1]))),
    )
    for entity_id_a, entity_id_b, score in edge_rows:
        edge_key: tuple[str, str]
        if entity_id_a <= entity_id_b:
            edge_key = (entity_id_a, entity_id_b)
        else:
            edge_key = (entity_id_b, entity_id_a)
        edge_score = float(score)
        edge_scores[edge_key] = edge_score
        objective_weights[edge_key] = score_to_objective_weight(
            edge_score, neutral_threshold
        )

    for entity_id in entity_ids:
        neighbors = tuple(sorted(subgraph.neighbors(entity_id)))
        adjacency[entity_id] = neighbors
        neighbor_sets[entity_id] = frozenset(neighbors)
        positive_neighbors[entity_id] = tuple(
            neighbor
            for neighbor in neighbors
            if objective_weights[tuple(sorted((entity_id, neighbor)))] > 0.0
        )

    return ComponentState(
        component_id=make_component_id(entity_ids),
        entity_ids=entity_ids,
        adjacency=adjacency,
        neighbor_sets=neighbor_sets,
        positive_neighbors=positive_neighbors,
        edge_scores=edge_scores,
        objective_weights=objective_weights,
        retained_edge_count=len(edge_scores),
    )


def build_component_states(
    retained_graph: nx.Graph,
    neutral_threshold: float = OBJECTIVE_NEUTRAL_THRESHOLD,
) -> list[ComponentState]:
    """Extract deterministic connected components and convert them to local state."""
    components: list[ComponentState] = []
    for nodes in nx.connected_components(retained_graph):
        entity_ids = tuple(sorted(nodes))
        components.append(
            build_component_state(
                retained_graph,
                entity_ids,
                neutral_threshold=neutral_threshold,
            )
        )
    return sorted(components, key=lambda component: component.entity_ids)


def _cluster_signature(
    clusters: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    """Return a deterministic comparable representation of cluster assignments."""
    return tuple(
        sorted(
            (tuple(sorted(cluster)) for cluster in clusters),
            key=lambda cluster: cluster,
        )
    )


def correlation_objective(
    component: ComponentState,
    clusters: tuple[tuple[str, ...], ...],
) -> float:
    """Score one clustering over retained edges using split-vs-merge evidence.

    Positive weights reward placing both entities in the same cluster.
    Negative weights reward keeping them apart.
    """
    cluster_index: dict[str, int] = {}
    for index, cluster in enumerate(clusters):
        for entity_id in cluster:
            cluster_index[entity_id] = index

    objective = 0.0
    for (entity_id_a, entity_id_b), weight in component.objective_weights.items():
        if cluster_index[entity_id_a] == cluster_index[entity_id_b]:
            objective += weight
        else:
            objective -= weight
    return objective


def _pivot_once(
    component: ComponentState,
    rng: random.Random,
) -> tuple[tuple[str, ...], ...]:
    """Run one randomized Pivot pass for correlation clustering.

    Pivot repeatedly picks one unassigned entity as the pivot node, then groups
    it with the still-unassigned neighbors whose shifted objective weight is
    positive. Different pivot orders can lead to different clusterings, which is
    why the solver runs multiple deterministic restarts and keeps the best
    objective value.
    """
    remaining = list(component.entity_ids)
    remaining_set = set(remaining)
    clusters: list[tuple[str, ...]] = []

    while remaining:
        pivot_entity_id = remaining[rng.randrange(len(remaining))]
        cluster_members = {pivot_entity_id}
        for neighbor in component.positive_neighbors[pivot_entity_id]:
            if neighbor in remaining_set:
                cluster_members.add(neighbor)

        remaining_set.difference_update(cluster_members)
        remaining = [entity_id for entity_id in remaining if entity_id in remaining_set]
        clusters.append(tuple(sorted(cluster_members)))

    return _cluster_signature(tuple(clusters))


def _pivot_once_deterministic(component: ComponentState) -> tuple[tuple[str, ...], ...]:
    """Run one deterministic Pivot pass using stable entity order as pivots."""
    remaining = list(component.entity_ids)
    remaining_set = set(remaining)
    clusters: list[tuple[str, ...]] = []

    while remaining:
        pivot_entity_id = remaining[0]
        cluster_members = {pivot_entity_id}
        for neighbor in component.positive_neighbors[pivot_entity_id]:
            if neighbor in remaining_set:
                cluster_members.add(neighbor)

        remaining_set.difference_update(cluster_members)
        remaining = [entity_id for entity_id in remaining if entity_id in remaining_set]
        clusters.append(tuple(sorted(cluster_members)))

    return _cluster_signature(tuple(clusters))


def _solve_trivial_component(component: ComponentState) -> SolvedComponent:
    """Return the only pragmatic clustering for a singleton or one-edge pair."""
    start = perf_counter()
    if len(component.entity_ids) == 1:
        clusters = ((component.entity_ids[0],),)
        clustering_method = "connected_component"
    else:
        merged_clusters = (component.entity_ids,)
        split_clusters = tuple((entity_id,) for entity_id in component.entity_ids)
        merged_score = correlation_objective(component, merged_clusters)
        split_score = correlation_objective(component, split_clusters)
        if merged_score >= split_score:
            clusters = merged_clusters
        else:
            clusters = split_clusters
        clustering_method = "trivial_objective"
    return SolvedComponent(
        component_id=component.component_id,
        entity_ids=component.entity_ids,
        clusters=clusters,
        clustering_method=clustering_method,
        objective_score=correlation_objective(component, clusters),
        run_count=1,
        early_stopped=False,
        elapsed_ms=(perf_counter() - start) * 1000,
    )


def solve_component_with_pivot(
    component: ComponentState,
    cap: int = PIVOT_RUN_CAP,
    patience: int = PIVOT_PATIENCE,
    min_gain: float = MIN_OBJECTIVE_GAIN,
    base_seed: int = 0,
) -> SolvedComponent:
    """Solve one retained component with deterministic multi-start Pivot.

    This is explicitly Pivot correlation clustering, not a generic "pivot"
    pattern. The solver compares several pivot orders and keeps the clustering
    with the best correlation-clustering objective score.
    """
    if component.is_trivial:
        return _solve_trivial_component(component)

    start = perf_counter()
    max_runs = min(cap, 2 + ceil(sqrt(component.retained_edge_count)))
    seed_prefix = int(component.component_id[:16], 16) ^ base_seed
    best_clusters: tuple[tuple[str, ...], ...] | None = None
    best_signature: tuple[tuple[str, ...], ...] | None = None
    best_score = float("-inf")
    no_improve = 0
    run_count = 0
    early_stopped = False

    for run_index in range(max_runs):
        if run_index == 0:
            clusters = _pivot_once_deterministic(component)
        else:
            clusters = _pivot_once(component, random.Random(seed_prefix + run_index))
        score = correlation_objective(component, clusters)
        signature = _cluster_signature(clusters)
        improved = score > best_score + min_gain
        tie_break = (
            abs(score - best_score) <= min_gain
            and best_signature is not None
            and signature < best_signature
        )

        if best_clusters is None or improved or tie_break:
            best_clusters = clusters
            best_signature = signature
            best_score = score
            no_improve = 0
        else:
            no_improve += 1

        run_count = run_index + 1
        if no_improve >= patience:
            early_stopped = True
            break

    elapsed_ms = (perf_counter() - start) * 1000
    return SolvedComponent(
        component_id=component.component_id,
        entity_ids=component.entity_ids,
        clusters=best_clusters or ((component.entity_ids[0],),),
        clustering_method="pivot_correlation",
        objective_score=best_score,
        run_count=run_count,
        early_stopped=early_stopped,
        elapsed_ms=elapsed_ms,
    )


def solve_retained_components(
    components: list[ComponentState],
    base_seed: int = 0,
    progress_every: int | None = None,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
) -> list[SolvedComponent]:
    """Solve each retained component independently in deterministic order."""
    solved_components: list[SolvedComponent] = []
    max_component_ms = 0.0
    solve_start = perf_counter()
    component_count = len(components)

    for solved_count, component in enumerate(components, start=1):
        solved_component = solve_component_with_pivot(component, base_seed=base_seed)
        solved_components.append(solved_component)
        max_component_ms = max(max_component_ms, solved_component.elapsed_ms)
        if (
            progress_callback is not None
            and progress_every is not None
            and progress_every > 0
            and solved_count % progress_every == 0
            and solved_count < component_count
        ):
            progress_callback(
                solved_count,
                component_count,
                perf_counter() - solve_start,
                max_component_ms,
            )

    return solved_components


def summarize_components(
    components: list[ComponentState],
    solved_components: list[SolvedComponent] | None = None,
) -> list[dict[str, Any]]:
    """Build a compact JSON-safe component summary for resolution artifacts."""
    solved_by_id = {}
    if solved_components is not None:
        solved_by_id = {
            solved_component.component_id: solved_component
            for solved_component in solved_components
        }

    summaries: list[dict[str, Any]] = []
    for component in components:
        row = {
            "component_id": component.component_id,
            "entity_ids": list(component.entity_ids),
            "node_count": len(component.entity_ids),
            "retained_edge_count": component.retained_edge_count,
            "is_trivial": component.is_trivial,
        }
        solved_component = solved_by_id.get(component.component_id)
        if solved_component is not None:
            row.update(
                {
                    "cluster_count": len(solved_component.clusters),
                    "clusters": [
                        list(cluster) for cluster in solved_component.clusters
                    ],
                    "clustering_method": solved_component.clustering_method,
                    "objective_score": round(solved_component.objective_score, 6),
                    "pivot_run_count": solved_component.run_count,
                    "early_stopped": solved_component.early_stopped,
                    "solve_elapsed_ms": round(solved_component.elapsed_ms, 3),
                }
            )
        summaries.append(row)
    return summaries


def _size_distribution(sizes: list[int]) -> dict[str, int]:
    """Count how many components or clusters have each size."""
    counts = Counter(sizes)
    return {str(size): counts[size] for size in sorted(counts)}


def _giant_size_warnings(
    sizes: list[int],
    total_node_count: int,
    kind: str,
) -> list[dict[str, Any]]:
    """Return warning records when one size bucket dominates the retained graph."""
    if not sizes or total_node_count == 0:
        return []

    largest_size = max(sizes)
    largest_share = largest_size / total_node_count
    warnings: list[dict[str, Any]] = []

    if largest_size >= GIANT_COMPONENT_NODE_WARNING_SIZE:
        warnings.append(
            {
                "code": f"largest_{kind}_node_count",
                "value": largest_size,
                "threshold": GIANT_COMPONENT_NODE_WARNING_SIZE,
            }
        )
    if (
        total_node_count >= GIANT_COMPONENT_NODE_WARNING_SIZE
        and largest_share >= GIANT_COMPONENT_SHARE_WARNING
    ):
        warnings.append(
            {
                "code": f"largest_{kind}_share",
                "value": round(largest_share, 6),
                "threshold": GIANT_COMPONENT_SHARE_WARNING,
            }
        )
    return warnings


def _giant_component_warnings(
    sizes: list[int],
    total_node_count: int,
) -> list[dict[str, Any]]:
    """Return retained-component warning records for phase-1 diagnostics."""
    return _giant_size_warnings(sizes, total_node_count, kind="component")


def _giant_cluster_warnings(
    sizes: list[int],
    total_node_count: int,
) -> list[dict[str, Any]]:
    """Return resolved-cluster warning records for enriched diagnostics."""
    return _giant_size_warnings(sizes, total_node_count, kind="cluster")


def component_timing_rows(
    components: list[ComponentState],
    solved_components: list[SolvedComponent],
) -> list[dict[str, Any]]:
    """Build per-component timing rows for debugging slow or odd components."""
    component_by_id = {component.component_id: component for component in components}
    rows: list[dict[str, Any]] = []
    for solved_component in solved_components:
        component = component_by_id[solved_component.component_id]
        rows.append(
            {
                "component_id": solved_component.component_id,
                "node_count": len(component.entity_ids),
                "retained_edge_count": component.retained_edge_count,
                "cluster_count": len(solved_component.clusters),
                "clustering_method": solved_component.clustering_method,
                "pivot_run_count": solved_component.run_count,
                "early_stopped": solved_component.early_stopped,
                "elapsed_ms": round(solved_component.elapsed_ms, 3),
            }
        )
    return rows


def _size_bucket(node_count: int) -> str:
    """Bucket component sizes for compact timing summaries."""
    if node_count <= 1:
        return "1"
    if node_count == 2:
        return "2"
    if node_count <= 5:
        return "3-5"
    if node_count <= 10:
        return "6-10"
    return "11+"


def summarize_timing_by_size_bucket(
    timing_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate component timing by retained-component size bucket."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in timing_rows:
        grouped[_size_bucket(int(row["node_count"]))].append(float(row["elapsed_ms"]))

    summary: dict[str, dict[str, float | int]] = {}
    for bucket in sorted(grouped):
        elapsed_values = grouped[bucket]
        summary[bucket] = {
            "component_count": len(elapsed_values),
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 3),
            "max_elapsed_ms": round(max(elapsed_values), 3),
            "total_elapsed_ms": round(sum(elapsed_values), 3),
        }
    return summary


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Return one percentile using linear interpolation over sorted values."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = quantile * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    share = rank - lower_index
    return lower_value + ((upper_value - lower_value) * share)


def summarize_component_timing(
    timing_rows: list[dict[str, Any]],
) -> dict[str, float | int | str | None]:
    """Collapse per-component timings into compact stage-level metrics."""
    if not timing_rows:
        return {
            "component_count": 0,
            "total_elapsed_ms": 0.0,
            "p50_elapsed_ms": 0.0,
            "p95_elapsed_ms": 0.0,
            "max_elapsed_ms": 0.0,
            "slowest_component_id": None,
            "slowest_component_size": 0,
        }

    elapsed_values = sorted(float(row["elapsed_ms"]) for row in timing_rows)
    slowest_row = max(
        timing_rows,
        key=lambda row: (
            float(row["elapsed_ms"]),
            int(row["node_count"]),
            row["component_id"],
        ),
    )
    return {
        "component_count": len(timing_rows),
        "total_elapsed_ms": round(sum(elapsed_values), 3),
        "p50_elapsed_ms": round(_percentile(elapsed_values, 0.50), 3),
        "p95_elapsed_ms": round(_percentile(elapsed_values, 0.95), 3),
        "max_elapsed_ms": round(max(elapsed_values), 3),
        "slowest_component_id": str(slowest_row["component_id"]),
        "slowest_component_size": int(slowest_row["node_count"]),
    }


def build_resolution_diagnostics(
    run_id: str,
    scored_pairs: pl.DataFrame,
    components: list[ComponentState],
    keep_threshold: float = KEEP_SCORE_THRESHOLD,
    neutral_threshold: float = OBJECTIVE_NEUTRAL_THRESHOLD,
) -> dict[str, Any]:
    """Build deterministic retained-graph diagnostics."""
    keep_threshold, neutral_threshold = validate_resolution_thresholds(
        keep_threshold,
        neutral_threshold,
    )
    component_sizes = [len(component.entity_ids) for component in components]
    total_node_count = sum(component_sizes)
    retained_edge_count = sum(component.retained_edge_count for component in components)
    singleton_node_count = sum(size == 1 for size in component_sizes)
    trivial_component_count = sum(component.is_trivial for component in components)
    largest_component_size = max(component_sizes, default=0)

    return {
        "run_id": run_id,
        "scored_pair_count": scored_pairs.height,
        "retained_edge_count": retained_edge_count,
        "component_count": len(components),
        "total_node_count": total_node_count,
        "singleton_node_count": singleton_node_count,
        "singleton_rate": (
            round(singleton_node_count / total_node_count, 6)
            if total_node_count
            else 0.0
        ),
        "trivial_component_count": trivial_component_count,
        "non_trivial_component_count": len(components) - trivial_component_count,
        "largest_component_size": largest_component_size,
        "component_size_distribution": _size_distribution(component_sizes),
        "giant_component_warnings": _giant_component_warnings(
            component_sizes, total_node_count
        ),
        "thresholds": {
            "keep_score_threshold": keep_threshold,
            "objective_neutral_threshold": neutral_threshold,
            "review_confidence_threshold": BASE_CONFIDENCE_REVIEW_THRESHOLD,
        },
        "threshold_note": "provisional placeholder thresholds; phase-1 diagnostics only",
    }


def build_phase1_components(
    run_id: str,
    scored_pairs: pl.DataFrame,
    entity_ids: list[str] | tuple[str, ...],
    keep_threshold: float = KEEP_SCORE_THRESHOLD,
    neutral_threshold: float = OBJECTIVE_NEUTRAL_THRESHOLD,
) -> tuple[list[ComponentState], dict[str, Any]]:
    """Build retained components plus diagnostics from scored pair probabilities."""
    keep_threshold, neutral_threshold = validate_resolution_thresholds(
        keep_threshold,
        neutral_threshold,
    )
    retained_graph = build_retained_graph(scored_pairs, entity_ids, keep_threshold)
    components = build_component_states(
        retained_graph, neutral_threshold=neutral_threshold
    )
    diagnostics = build_resolution_diagnostics(
        run_id,
        scored_pairs,
        components,
        keep_threshold,
        neutral_threshold,
    )
    return components, diagnostics
