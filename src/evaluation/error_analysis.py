"""Compact error-analysis helpers for extraction and clustering output."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping


def summarize_extraction_errors(
    gold_rows: list[dict[str, Any]],
    predicted_rows: list[dict[str, Any]],
    limit: int = 10,
) -> dict[str, Any]:
    """Summarize unmatched gold and predicted mentions by type and example."""
    gold_keys = {_mention_key(row) for row in gold_rows}
    predicted_keys = {_mention_key(row) for row in predicted_rows}
    gold_by_key = {_mention_key(row): row for row in gold_rows}
    predicted_by_key = {_mention_key(row): row for row in predicted_rows}

    missing = [gold_by_key[key] for key in sorted(gold_keys - predicted_keys)]
    spurious = [predicted_by_key[key] for key in sorted(predicted_keys - gold_keys)]
    return {
        "missing_gold_mentions_by_type": dict(Counter(row["entity_type"] for row in missing)),
        "spurious_predicted_mentions_by_type": dict(
            Counter(row["entity_type"] for row in spurious)
        ),
        "missing_gold_examples": [_compact_mention(row) for row in missing[:limit]],
        "spurious_predicted_examples": [_compact_mention(row) for row in spurious[:limit]],
    }


def summarize_false_merges(
    gold_group_by_entity: Mapping[str, str],
    predicted_cluster_by_entity: Mapping[str, str],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    limit: int = 10,
) -> dict[str, Any]:
    """Summarize pairs that the predicted clusters merged incorrectly."""
    pairs_by_cluster: dict[str, list[tuple[str, str]]] = defaultdict(list)
    entity_ids = sorted(predicted_cluster_by_entity)
    for index, entity_id_a in enumerate(entity_ids):
        for entity_id_b in entity_ids[index + 1:]:
            if predicted_cluster_by_entity[entity_id_a] != predicted_cluster_by_entity[entity_id_b]:
                continue
            if gold_group_by_entity[entity_id_a] == gold_group_by_entity[entity_id_b]:
                continue
            pairs_by_cluster[predicted_cluster_by_entity[entity_id_a]].append((entity_id_a, entity_id_b))

    ordered = sorted(
        pairs_by_cluster.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return {
        "cluster_count_with_false_merge": len(ordered),
        "pair_count": sum(len(pairs) for _, pairs in ordered),
        "examples": [
            {
                "cluster_id": cluster_id,
                "pair_count": len(pairs),
                "pairs": [_compact_pair(entity_id_a, entity_id_b, gold_group_by_entity, entity_metadata)
                          for entity_id_a, entity_id_b in pairs[:3]],
            }
            for cluster_id, pairs in ordered[:limit]
        ],
    }


def summarize_false_splits(
    gold_group_by_entity: Mapping[str, str],
    predicted_cluster_by_entity: Mapping[str, str],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    limit: int = 10,
) -> dict[str, Any]:
    """Summarize gold groups that the predicted clusters split apart."""
    entities_by_gold_group: dict[str, list[str]] = defaultdict(list)
    for entity_id, gold_group_id in gold_group_by_entity.items():
        entities_by_gold_group[str(gold_group_id)].append(str(entity_id))

    split_groups: list[tuple[str, list[str]]] = []
    for gold_group_id, entity_ids in entities_by_gold_group.items():
        predicted_clusters = {predicted_cluster_by_entity[entity_id] for entity_id in entity_ids}
        if len(predicted_clusters) > 1:
            split_groups.append((gold_group_id, sorted(entity_ids)))

    split_groups.sort(key=lambda item: (-len(item[1]), item[0]))
    return {
        "gold_group_count_split": len(split_groups),
        "examples": [
            {
                "gold_group_id": gold_group_id,
                "member_count": len(entity_ids),
                "predicted_cluster_ids": sorted(
                    {predicted_cluster_by_entity[entity_id] for entity_id in entity_ids}
                ),
                "members": [_compact_entity(entity_id, entity_metadata) for entity_id in entity_ids[:5]],
            }
            for gold_group_id, entity_ids in split_groups[:limit]
        ],
    }


def _mention_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    """Build one exact-match mention key."""
    return (
        str(row["doc_id"]),
        str(row["entity_type"]),
        int(row["char_start"]),
        int(row["char_end"]),
    )


def _compact_mention(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the minimum fields needed to inspect one mention error."""
    return {
        "doc_id": str(row["doc_id"]),
        "entity_type": str(row["entity_type"]),
        "text": str(row["text"]),
        "char_start": int(row["char_start"]),
        "char_end": int(row["char_end"]),
    }


def _compact_pair(
    entity_id_a: str,
    entity_id_b: str,
    gold_group_by_entity: Mapping[str, str],
    entity_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one compact false-merge pair row."""
    return {
        "entity_id_a": entity_id_a,
        "entity_id_b": entity_id_b,
        "gold_group_a": str(gold_group_by_entity[entity_id_a]),
        "gold_group_b": str(gold_group_by_entity[entity_id_b]),
        "entity_a": _compact_entity(entity_id_a, entity_metadata),
        "entity_b": _compact_entity(entity_id_b, entity_metadata),
    }


def _compact_entity(
    entity_id: str,
    entity_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep the minimum fields needed to inspect one entity-level error."""
    row = entity_metadata[entity_id]
    return {
        "entity_id": entity_id,
        "doc_id": str(row["doc_id"]),
        "entity_type": str(row["entity_type"]),
        "text": str(row["text"]),
        "normalized": str(row["normalized"]),
    }
