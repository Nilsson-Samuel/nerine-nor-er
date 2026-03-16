"""Phase-1 resolution graph building, component extraction, and diagnostics."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl
import pyarrow.parquet as pq

from src.shared import schemas
from src.shared.config import (
    KEEP_SCORE_THRESHOLD,
    OBJECTIVE_NEUTRAL_THRESHOLD,
    REVIEW_CONFIDENCE_THRESHOLD,
)


# Skip Pivot later when a component is too small to contain real contradictions.
TRIVIAL_NODE_LIMIT = 2

# One retained edge cannot express a meaningful split-vs-merge tradeoff on its own.
TRIVIAL_EDGE_LIMIT = 1

# Conservative restart budget for the later multi-start Pivot solver.
PIVOT_RUN_CAP = 40

# Stop early when repeated Pivot restarts stop improving.
PIVOT_PATIENCE = 5

# Ignore tiny float noise when comparing objective gains across restarts.
MIN_OBJECTIVE_GAIN = 1e-6

# Warn when one component becomes large enough to deserve manual inspection.
GIANT_COMPONENT_NODE_WARNING_SIZE = 25

# Warn when one component absorbs a large share of the retained graph.
GIANT_COMPONENT_SHARE_WARNING = 0.15

PAIR_COLUMNS = ["run_id", "entity_id_a", "entity_id_b", "score"]


@dataclass(frozen=True, slots=True)
class ComponentState:
    """Deterministic local view of one retained connected component."""

    component_id: str
    entity_ids: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]
    neighbor_sets: dict[str, frozenset[str]]
    edge_scores: dict[tuple[str, str], float]
    retained_edge_count: int

    @property
    def is_trivial(self) -> bool:
        """Return whether the component can bypass Pivot in the next phase."""
        return (
            len(self.entity_ids) <= TRIVIAL_NODE_LIMIT
            and self.retained_edge_count <= TRIVIAL_EDGE_LIMIT
        )


def include_edge(score: float, threshold: float = KEEP_SCORE_THRESHOLD) -> bool:
    """Return whether a scored pair should stay in the retained graph."""
    return float(score) >= float(threshold)


def make_component_id(entity_ids: tuple[str, ...] | list[str]) -> str:
    """Build a stable component identifier from sorted member entity IDs."""
    ordered_ids = tuple(sorted(entity_ids))
    payload = "|".join(ordered_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def load_scored_pairs(data_dir: Path | str, run_id: str) -> pl.DataFrame:
    """Load one run from scored_pairs.parquet in deterministic pair order."""
    data_dir = Path(data_dir)
    table = pq.read_table(data_dir / "scored_pairs.parquet")
    errors = schemas.validate_contract_rules(table, "scored_pairs")
    if errors:
        raise ValueError(f"scored_pairs failed contract validation: {errors}")

    return (
        pl.from_arrow(table)
        .filter(pl.col("run_id") == run_id)
        .sort(["entity_id_a", "entity_id_b"])
        .select(PAIR_COLUMNS)
    )


def load_entity_ids(data_dir: Path | str, run_id: str) -> list[str]:
    """Load one run from entities.parquet and return deterministic entity IDs."""
    data_dir = Path(data_dir)
    table = pq.read_table(data_dir / "entities.parquet")
    errors = schemas.validate_contract_rules(table, "entities")
    if errors:
        raise ValueError(f"entities failed contract validation: {errors}")

    entity_ids = (
        pl.from_arrow(table)
        .filter(pl.col("run_id") == run_id)
        .select("entity_id")
        .get_column("entity_id")
        .to_list()
    )
    if not entity_ids:
        raise ValueError(f"run_id not found in entities.parquet: {run_id}")
    return sorted(entity_ids)


def build_retained_graph(
    scored_pairs: pl.DataFrame,
    entity_ids: list[str] | tuple[str, ...],
    keep_threshold: float = KEEP_SCORE_THRESHOLD,
) -> nx.Graph:
    """Build one graph over all run entities, keeping only thresholded edges."""
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

    for row in scored_pairs.iter_rows(named=True):
        if include_edge(row["score"], keep_threshold):
            graph.add_edge(
                row["entity_id_a"],
                row["entity_id_b"],
                weight=float(row["score"]),
            )
    return graph


def build_component_state(graph: nx.Graph, entity_ids: tuple[str, ...]) -> ComponentState:
    """Convert one retained component into simple deterministic local structures."""
    adjacency: dict[str, tuple[str, ...]] = {}
    neighbor_sets: dict[str, frozenset[str]] = {}
    edge_scores: dict[tuple[str, str], float] = {}

    for entity_id in entity_ids:
        neighbors = tuple(sorted(graph.neighbors(entity_id)))
        adjacency[entity_id] = neighbors
        neighbor_sets[entity_id] = frozenset(neighbors)

    edge_rows = sorted(
        graph.subgraph(entity_ids).edges(data="weight"),
        key=lambda row: tuple(sorted((row[0], row[1]))),
    )
    for entity_id_a, entity_id_b, score in edge_rows:
        edge_key = tuple(sorted((entity_id_a, entity_id_b)))
        edge_scores[edge_key] = float(score)

    return ComponentState(
        component_id=make_component_id(entity_ids),
        entity_ids=entity_ids,
        adjacency=adjacency,
        neighbor_sets=neighbor_sets,
        edge_scores=edge_scores,
        retained_edge_count=len(edge_scores),
    )


def build_component_states(retained_graph: nx.Graph) -> list[ComponentState]:
    """Extract deterministic connected components and convert them to local state."""
    components: list[ComponentState] = []
    for nodes in nx.connected_components(retained_graph):
        entity_ids = tuple(sorted(nodes))
        components.append(build_component_state(retained_graph, entity_ids))
    return sorted(components, key=lambda component: component.entity_ids)


def summarize_components(components: list[ComponentState]) -> list[dict[str, Any]]:
    """Build a compact JSON-safe summary for later resolution stages."""
    return [
        {
            "component_id": component.component_id,
            "entity_ids": list(component.entity_ids),
            "node_count": len(component.entity_ids),
            "retained_edge_count": component.retained_edge_count,
            "is_trivial": component.is_trivial,
        }
        for component in components
    ]


def _size_distribution(components: list[ComponentState]) -> dict[str, int]:
    """Count how many components have each size."""
    counts = Counter(len(component.entity_ids) for component in components)
    return {str(size): counts[size] for size in sorted(counts)}


def _giant_component_warnings(
    components: list[ComponentState],
    total_node_count: int,
) -> list[dict[str, Any]]:
    """Return giant-component warning records for phase-1 diagnostics."""
    if not components or total_node_count == 0:
        return []

    largest_component = max(components, key=lambda component: len(component.entity_ids))
    largest_size = len(largest_component.entity_ids)
    largest_share = largest_size / total_node_count
    warnings: list[dict[str, Any]] = []

    if largest_size >= GIANT_COMPONENT_NODE_WARNING_SIZE:
        warnings.append(
            {
                "code": "largest_component_node_count",
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
                "code": "largest_component_share",
                "value": round(largest_share, 6),
                "threshold": GIANT_COMPONENT_SHARE_WARNING,
            }
        )
    return warnings


def build_resolution_diagnostics(
    run_id: str,
    scored_pairs: pl.DataFrame,
    components: list[ComponentState],
    keep_threshold: float = KEEP_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Build deterministic graph and component diagnostics for phase 1."""
    total_node_count = sum(len(component.entity_ids) for component in components)
    retained_edge_count = sum(component.retained_edge_count for component in components)
    singleton_node_count = sum(
        1 for component in components if len(component.entity_ids) == 1
    )
    trivial_component_count = sum(component.is_trivial for component in components)
    size_distribution = _size_distribution(components)
    largest_component_size = max((len(component.entity_ids) for component in components), default=0)

    return {
        "run_id": run_id,
        "scored_pair_count": scored_pairs.height,
        "retained_edge_count": retained_edge_count,
        "component_count": len(components),
        "total_node_count": total_node_count,
        "singleton_node_count": singleton_node_count,
        "singleton_rate": (
            round(singleton_node_count / total_node_count, 6) if total_node_count else 0.0
        ),
        "trivial_component_count": trivial_component_count,
        "non_trivial_component_count": len(components) - trivial_component_count,
        "largest_component_size": largest_component_size,
        "component_size_distribution": size_distribution,
        "giant_component_warnings": _giant_component_warnings(components, total_node_count),
        "thresholds": {
            "keep_score_threshold": keep_threshold,
            "objective_neutral_threshold": OBJECTIVE_NEUTRAL_THRESHOLD,
            "review_confidence_threshold": REVIEW_CONFIDENCE_THRESHOLD,
        },
        "threshold_note": "provisional placeholder thresholds; phase-1 diagnostics only",
    }


def build_phase1_components(
    run_id: str,
    scored_pairs: pl.DataFrame,
    entity_ids: list[str] | tuple[str, ...],
    keep_threshold: float = KEEP_SCORE_THRESHOLD,
) -> tuple[list[ComponentState], dict[str, Any]]:
    """Build retained components plus diagnostics from scored pair probabilities."""
    retained_graph = build_retained_graph(scored_pairs, entity_ids, keep_threshold)
    components = build_component_states(retained_graph)
    diagnostics = build_resolution_diagnostics(
        run_id,
        scored_pairs,
        components,
        keep_threshold,
    )
    return components, diagnostics
