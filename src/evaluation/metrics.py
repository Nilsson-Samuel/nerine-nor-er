"""Pure metric helpers for mention, pairwise, and cluster evaluation."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from typing import Any, Iterable, Mapping

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MetricDict = dict[str, float | int]
PairKey = tuple[str, str]
MentionKey = tuple[str, str, int, int]


def mention_metrics(
    predicted_mentions: Iterable[MentionKey],
    gold_mentions: Iterable[MentionKey],
) -> MetricDict:
    """Compute exact-match mention precision/recall/F1."""
    predicted = set(predicted_mentions)
    gold = set(gold_mentions)
    true_positives = len(predicted & gold)
    precision = _safe_divide(true_positives, len(predicted))
    recall = _safe_divide(true_positives, len(gold))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_positive_count": true_positives,
        "predicted_count": len(predicted),
        "gold_count": len(gold),
    }


def pairwise_metrics(
    predicted_positive_pairs: Iterable[PairKey],
    gold_positive_pairs: Iterable[PairKey],
) -> MetricDict:
    """Compute pairwise precision/recall/F1 from positive-pair sets."""
    predicted = {_normalize_pair(pair) for pair in predicted_positive_pairs}
    gold = {_normalize_pair(pair) for pair in gold_positive_pairs}
    true_positives = len(predicted & gold)
    precision = _safe_divide(true_positives, len(predicted))
    recall = _safe_divide(true_positives, len(gold))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_positive_count": true_positives,
        "predicted_positive_count": len(predicted),
        "gold_positive_count": len(gold),
    }


def positive_pairs_from_memberships(
    membership_by_entity: Mapping[str, str],
) -> set[PairKey]:
    """Expand cluster memberships into canonical positive pair keys."""
    members_by_group: dict[str, list[str]] = defaultdict(list)
    for entity_id, group_id in membership_by_entity.items():
        members_by_group[str(group_id)].append(str(entity_id))

    pairs: set[PairKey] = set()
    for entity_ids in members_by_group.values():
        for entity_id_a, entity_id_b in combinations(sorted(entity_ids), 2):
            pairs.add((entity_id_a, entity_id_b))
    return pairs


def clustering_metrics(
    gold_membership_by_entity: Mapping[str, str],
    predicted_membership_by_entity: Mapping[str, str],
) -> MetricDict:
    """Compute pairwise, ARI, NMI, and B-cubed metrics for one entity set."""
    entity_ids = _require_matching_entity_sets(
        gold_membership_by_entity,
        predicted_membership_by_entity,
    )
    if not entity_ids:
        return {
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "pairwise_f1": 0.0,
            "pairwise_f0_5": 0.0,
            "ari": 0.0,
            "nmi": 0.0,
            "bcubed_precision": 0.0,
            "bcubed_recall": 0.0,
            "bcubed_f1": 0.0,
            "bcubed_f0_5": 0.0,
        }
    gold_labels = [
        str(gold_membership_by_entity[entity_id]) for entity_id in entity_ids
    ]
    predicted_labels = [
        str(predicted_membership_by_entity[entity_id]) for entity_id in entity_ids
    ]

    pairwise = pairwise_metrics(
        positive_pairs_from_memberships(predicted_membership_by_entity),
        positive_pairs_from_memberships(gold_membership_by_entity),
    )
    bcubed = bcubed_metrics(gold_membership_by_entity, predicted_membership_by_entity)
    return {
        "pairwise_precision": pairwise["precision"],
        "pairwise_recall": pairwise["recall"],
        "pairwise_f1": pairwise["f1"],
        "pairwise_f0_5": _fbeta(pairwise["precision"], pairwise["recall"], beta=0.5),
        "ari": float(adjusted_rand_score(gold_labels, predicted_labels)),
        "nmi": float(normalized_mutual_info_score(gold_labels, predicted_labels)),
        "bcubed_precision": bcubed["precision"],
        "bcubed_recall": bcubed["recall"],
        "bcubed_f1": bcubed["f1"],
        "bcubed_f0_5": bcubed["f0_5"],
    }


def pairwise_f_beta_from_metrics(metrics: Mapping[str, Any], beta: float) -> float:
    """Compute pairwise F-beta from a metric row with pairwise precision/recall."""
    return _fbeta(
        float(metrics["pairwise_precision"]),
        float(metrics["pairwise_recall"]),
        beta=beta,
    )


def bcubed_metrics(
    gold_membership_by_entity: Mapping[str, str],
    predicted_membership_by_entity: Mapping[str, str],
) -> MetricDict:
    """Compute B-cubed precision/recall plus F1 and F0.5 scores."""
    entity_ids = _require_matching_entity_sets(
        gold_membership_by_entity,
        predicted_membership_by_entity,
    )
    gold_members = _members_by_group(gold_membership_by_entity)
    predicted_members = _members_by_group(predicted_membership_by_entity)

    precision_total = 0.0
    recall_total = 0.0
    for entity_id in entity_ids:
        gold_cluster = gold_members[str(gold_membership_by_entity[entity_id])]
        predicted_cluster = predicted_members[
            str(predicted_membership_by_entity[entity_id])
        ]
        intersection_size = len(gold_cluster & predicted_cluster)
        precision_total += intersection_size / len(predicted_cluster)
        recall_total += intersection_size / len(gold_cluster)

    precision = precision_total / len(entity_ids) if entity_ids else 0.0
    recall = recall_total / len(entity_ids) if entity_ids else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": _fbeta(precision, recall, beta=1.0),
        "f0_5": _fbeta(precision, recall, beta=0.5),
    }


def _members_by_group(membership_by_entity: Mapping[str, str]) -> dict[str, set[str]]:
    """Invert entity-to-group membership into group-to-members sets."""
    members: dict[str, set[str]] = defaultdict(set)
    for entity_id, group_id in membership_by_entity.items():
        members[str(group_id)].add(str(entity_id))
    return members


def _normalize_pair(pair: PairKey) -> PairKey:
    """Return a canonical entity_id pair with a < b ordering."""
    entity_id_a, entity_id_b = pair
    if entity_id_a == entity_id_b:
        raise ValueError("pairwise metrics do not support self-pairs")
    return (
        (entity_id_a, entity_id_b)
        if entity_id_a < entity_id_b
        else (entity_id_b, entity_id_a)
    )


def _require_matching_entity_sets(
    gold_membership_by_entity: Mapping[str, str],
    predicted_membership_by_entity: Mapping[str, str],
) -> list[str]:
    """Require both cluster labelings to cover exactly the same entity IDs."""
    gold_ids = set(gold_membership_by_entity)
    predicted_ids = set(predicted_membership_by_entity)
    if gold_ids != predicted_ids:
        missing_in_gold = sorted(predicted_ids - gold_ids)
        missing_in_predicted = sorted(gold_ids - predicted_ids)
        raise ValueError(
            "gold/predicted entity sets differ: "
            f"missing_in_gold={missing_in_gold[:5]}, "
            f"missing_in_predicted={missing_in_predicted[:5]}"
        )
    return sorted(gold_ids)


def _safe_divide(numerator: int, denominator: int) -> float:
    """Return a float division result, defaulting to 0 for empty denominators."""
    if denominator == 0:
        return 0.0
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError("metric result must be finite")
    return float(value)


def _f1(precision: float, recall: float) -> float:
    """Compute the harmonic mean of precision and recall."""
    return _fbeta(precision, recall, beta=1.0)


def _fbeta(precision: float, recall: float, beta: float) -> float:
    """Compute F-beta so precision-weighted cluster scores reuse one formula."""
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0.0:
        return 0.0
    return float((1.0 + beta_squared) * precision * recall / denominator)
