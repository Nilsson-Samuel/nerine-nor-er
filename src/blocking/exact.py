"""Exact-match blocking — normalized name and structured identifier matches.

Two complementary strategies that are cheap to compute and high-value:

1. **Exact normalized match**: entities with identical normalized text and
   same type become candidate pairs. Catches trivial matches that FAISS
   would also find, but guarantees them without embedding distance noise.

2. **Structured identifier match**: entities of "strong" types (FIN, COMM,
   VEH) where the normalized text is an exact identifier (phone number,
   account number, license plate). Same-type pairs are emitted as candidates;
   cross-type collisions are logged separately for analysis but not emitted,
   preserving the blocking contract until a design change is justified.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Types where exact normalized match is a near-certain identity signal.
STRONG_IDENTIFIER_TYPES = {"FIN", "COMM", "VEH"}
MAX_BUCKET_SIZE = 500  # Applied to exact name buckets only (not structured)


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
    identity signal. Same-type pairs are emitted as candidates. Cross-type
    collisions (e.g. a phone number tagged COMM in one doc and FIN in another)
    are logged for analysis but not emitted, to preserve the same-type blocking
    contract.

    No bucket size cap is applied — structured IDs are deterministic high-value
    matches that should not be silently dropped.

    Args:
        entity_ids: Entity ID strings.
        names: Normalized entity name strings (aligned with entity_ids).
        types: Entity type strings (aligned with entity_ids).

    Returns:
        List of (entity_id_a, entity_id_b) pairs with canonical ordering.
    """
    # Build entity_id → type lookup for the entities we care about
    id_to_type: dict[str, str] = {}
    # Group by (type, normalized_name) for same-type pairs
    same_type_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    # Group by normalized_name only for cross-type collision detection
    cross_type_buckets: dict[str, set[str]] = defaultdict(set)

    for eid, name, etype in zip(entity_ids, names, types):
        if etype in STRONG_IDENTIFIER_TYPES:
            id_to_type[eid] = etype
            same_type_buckets[(etype, name)].append(eid)
            cross_type_buckets[name].add(etype)

    # Emit same-type pairs (no bucket cap for structured IDs)
    pairs: list[tuple[str, str]] = []
    for (_etype, _name), ids in same_type_buckets.items():
        if len(ids) < 2:
            continue
        ids_sorted = sorted(ids)
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                pairs.append((ids_sorted[i], ids_sorted[j]))

    # Log cross-type collisions for analysis
    cross_type_collisions = sum(
        1 for types_set in cross_type_buckets.values() if len(types_set) > 1
    )
    if cross_type_collisions:
        logger.info(
            "Structured ID blocking: %d identifiers have cross-type collisions "
            "(not emitted as pairs, logged for analysis)",
            cross_type_collisions,
        )

    logger.info(
        "Structured ID blocking: %d same-type pairs from strong identifiers",
        len(pairs),
    )
    return pairs
