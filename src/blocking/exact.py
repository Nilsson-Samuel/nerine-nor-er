"""Exact-match blocking — normalized name and structured identifier matches.

Two complementary strategies that are cheap to compute and high-value:

1. **Exact normalized match**: entities with identical normalized text and
   same type become candidate pairs. Catches trivial matches that FAISS
   would also find, but guarantees them without embedding distance noise.

2. **Structured identifier match**: entities of "strong" types (FIN, COMM,
   VEH) where the normalized text is an exact identifier (phone number,
   account number, license plate). An exact match here is near-certain
   identity, so these pairs are always generated regardless of type mixing.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Types where exact normalized match is a near-certain identity signal.
# These bypass the same-type constraint because a phone number is a phone
# number regardless of how NER typed the surrounding entity.
STRONG_IDENTIFIER_TYPES = {"FIN", "COMM", "VEH"}
MAX_BUCKET_SIZE = 500


def build_exact_name_pairs(
    entity_ids: list[str],
    names: list[str],
    types: list[str],
) -> list[tuple[str, str]]:
    """Generate candidate pairs from entities sharing the same normalized name and type.

    Args:
        entity_ids: Entity ID strings.
        names: Normalized entity name strings (aligned with entity_ids).
        types: Entity type strings (aligned with entity_ids).

    Returns:
        List of (entity_id_a, entity_id_b) pairs with canonical ordering.
    """
    # Group entity IDs by (type, normalized_name)
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for eid, name, etype in zip(entity_ids, names, types):
        buckets[(etype, name)].append(eid)

    pairs: list[tuple[str, str]] = []
    skipped = 0
    for (_etype, _name), ids in buckets.items():
        if len(ids) < 2:
            continue
        if len(ids) > MAX_BUCKET_SIZE:
            skipped += 1
            logger.warning(
                "Exact name bucket too large (%d entities, cap=%d), skipping: %s",
                len(ids), MAX_BUCKET_SIZE, _name[:40],
            )
            continue
        ids_sorted = sorted(ids)
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                pairs.append((ids_sorted[i], ids_sorted[j]))

    logger.info(
        "Exact name blocking: %d pairs from shared (type, normalized), %d buckets skipped",
        len(pairs), skipped,
    )
    return pairs


def build_structured_id_pairs(
    entity_ids: list[str],
    names: list[str],
    types: list[str],
) -> list[tuple[str, str]]:
    """Generate candidate pairs from strong identifier types with exact match.

    For FIN/COMM/VEH entities, an exact normalized text match is a very strong
    identity signal. These pairs are generated across any type combination
    (a phone number tagged as COMM in one doc and FIN in another is still
    the same identifier).

    Args:
        entity_ids: Entity ID strings.
        names: Normalized entity name strings (aligned with entity_ids).
        types: Entity type strings (aligned with entity_ids).

    Returns:
        List of (entity_id_a, entity_id_b) pairs with canonical ordering.
    """
    # Only consider entities of strong identifier types
    buckets: dict[str, list[str]] = defaultdict(list)
    for eid, name, etype in zip(entity_ids, names, types):
        if etype in STRONG_IDENTIFIER_TYPES:
            buckets[name].append(eid)

    pairs: list[tuple[str, str]] = []
    skipped = 0
    for _name, ids in buckets.items():
        if len(ids) < 2:
            continue
        if len(ids) > MAX_BUCKET_SIZE:
            skipped += 1
            logger.warning(
                "Structured ID bucket too large (%d entities, cap=%d), skipping: %s",
                len(ids), MAX_BUCKET_SIZE, _name[:40],
            )
            continue
        ids_sorted = sorted(ids)
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                pairs.append((ids_sorted[i], ids_sorted[j]))

    logger.info(
        "Structured ID blocking: %d pairs from strong identifiers, %d buckets skipped",
        len(pairs), skipped,
    )
    return pairs
