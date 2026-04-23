"""Cluster evidence scoring and routing helpers for resolution outputs."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from src.shared.config import (
    BASE_CONFIDENCE_AUTO_MERGE_THRESHOLD,
    BASE_CONFIDENCE_REVIEW_THRESHOLD,
    ROUTING_PROFILE,
)


@dataclass(frozen=True, slots=True)
class ClusterEvidence:
    """Compact evidence summary for one resolved cluster."""

    cluster_id: str
    entity_ids: tuple[str, ...]
    cluster_size: int
    actual_edge_count: int
    possible_edge_count: int
    density: float
    min_edge_score: float
    avg_edge_score: float
    base_confidence: float


def make_cluster_id(entity_ids: tuple[str, ...] | list[str]) -> str:
    """Build a stable cluster identifier from sorted member entity IDs."""
    ordered_ids = tuple(sorted(entity_ids))
    payload = "|".join(ordered_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def compute_base_confidence(
    entity_ids: tuple[str, ...] | list[str],
    edge_scores: dict[tuple[str, str], float],
) -> ClusterEvidence:
    """Compute pragmatic evidence quality signals for one cluster."""
    cluster_entity_ids = tuple(sorted(entity_ids))
    cluster_size = len(cluster_entity_ids)
    possible_edge_count = cluster_size * (cluster_size - 1) // 2
    cluster_scores: list[float] = []

    if possible_edge_count:
        for index, entity_id_a in enumerate(cluster_entity_ids):
            for entity_id_b in cluster_entity_ids[index + 1 :]:
                edge_key = tuple(sorted((entity_id_a, entity_id_b)))
                if edge_key in edge_scores:
                    cluster_scores.append(float(edge_scores[edge_key]))

    actual_edge_count = len(cluster_scores)
    density = actual_edge_count / possible_edge_count if possible_edge_count else 1.0
    min_edge_score = min(cluster_scores, default=0.0)
    avg_edge_score = sum(cluster_scores) / actual_edge_count if actual_edge_count else 0.0
    raw_score = 0.6 * min_edge_score + 0.4 * avg_edge_score
    base_confidence = max(0.0, min(1.0, raw_score * density))

    return ClusterEvidence(
        cluster_id=make_cluster_id(cluster_entity_ids),
        entity_ids=cluster_entity_ids,
        cluster_size=cluster_size,
        actual_edge_count=actual_edge_count,
        possible_edge_count=possible_edge_count,
        density=density,
        min_edge_score=min_edge_score,
        avg_edge_score=avg_edge_score,
        base_confidence=base_confidence,
    )


def route_cluster(
    base_confidence: float,
    cluster_size: int,
    profile: str = ROUTING_PROFILE,
    auto_merge_threshold: float = BASE_CONFIDENCE_AUTO_MERGE_THRESHOLD,
    review_threshold: float = BASE_CONFIDENCE_REVIEW_THRESHOLD,
) -> str:
    """Map one evidence score to a routing action for the selected profile."""
    if cluster_size <= 1:
        return "keep_separate"
    if base_confidence > auto_merge_threshold:
        return "auto_merge"
    if base_confidence >= review_threshold:
        if profile == "balanced_hitl":
            return "review"
        return "defer"
    return "keep_separate"


def routing_actions_by_profile(
    base_confidence: float,
    cluster_size: int,
) -> dict[str, str]:
    """Return routing actions for the currently supported operating profiles."""
    return {
        "quick_low_hitl": route_cluster(
            base_confidence,
            cluster_size,
            profile="quick_low_hitl",
        ),
        "balanced_hitl": route_cluster(
            base_confidence,
            cluster_size,
            profile="balanced_hitl",
        ),
    }


def count_routing_actions(
    cluster_rows: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    """Count routing outcomes per profile for diagnostics."""
    counts: dict[str, Counter[str]] = {
        "quick_low_hitl": Counter(),
        "balanced_hitl": Counter(),
    }
    for cluster_row in cluster_rows:
        routing_actions = cluster_row["routing_actions_by_profile"]
        for profile, action in dict(routing_actions).items():
            counts[profile][str(action)] += 1

    return {
        profile: dict(sorted(profile_counts.items()))
        for profile, profile_counts in counts.items()
    }
