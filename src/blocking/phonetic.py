"""Double Metaphone phonetic blocking for person-name matching.

Builds a phonetic index from entity names using Double Metaphone codes.
Norwegian characters are pre-normalized (æ→ae, ø→o, å→a, etc.) before
encoding. Both per-token and full-name codes are indexed so that
"Hansen" and "Hanssen" land in the same bucket.
"""

import logging
from collections import defaultdict

from metaphone import doublemetaphone

logger = logging.getLogger(__name__)

MAX_BUCKET_SIZE = 500


def _nor_pre_normalize(text: str) -> str:
    """Normalize Norwegian characters to ASCII-friendly phonetic input."""
    return (
        text.lower()
        .replace("æ", "ae")
        .replace("ø", "o")
        .replace("å", "a")
        .replace("kj", "k")
        .replace("skj", "sk")
        .replace("hj", "j")
        .replace("gj", "j")
    )


def build_phonetic_index(
    entity_ids: list[str],
    names: list[str],
    types: list[str],
) -> dict[str, set[str]]:
    """Build a phonetic code → entity_id mapping for PER entities.

    Args:
        entity_ids: Entity ID strings.
        names: Normalized entity name strings (aligned with entity_ids).
        types: Entity type strings (aligned with entity_ids).

    Returns:
        Dict mapping phonetic codes to sets of entity IDs that share that code.
    """
    index: dict[str, set[str]] = defaultdict(set)
    indexed_count = 0

    for eid, name, etype in zip(entity_ids, names, types):
        if etype != "PER":
            continue

        ascii_name = _nor_pre_normalize(name)
        indexed_count += 1

        # Index each token separately (catches partial name matches)
        for part in ascii_name.split():
            if len(part) < 2:
                continue
            primary, secondary = doublemetaphone(part)
            if primary:
                index[primary].add(eid)
            if secondary and secondary != primary:
                index[secondary].add(eid)

        # Also index the full name concatenated (catches full-name similarity)
        full_primary, full_secondary = doublemetaphone(ascii_name.replace(" ", ""))
        if full_primary:
            index[f"FULL_{full_primary}"].add(eid)
        if full_secondary and full_secondary != full_primary:
            index[f"FULL_{full_secondary}"].add(eid)

    logger.info(
        "Phonetic index: %d PER entities indexed, %d distinct codes",
        indexed_count, len(index),
    )
    return dict(index)


def query_phonetic_pairs(
    phonetic_index: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """Extract candidate pairs from entities sharing phonetic codes.

    Entities that share at least one phonetic code become a candidate pair.
    All returned pairs are PER↔PER by construction (only PER entities indexed).

    Args:
        phonetic_index: Output of build_phonetic_index().

    Returns:
        List of (entity_id_a, entity_id_b) pairs (may contain duplicates).
    """
    pairs: list[tuple[str, str]] = []
    skipped = 0
    for bucket in phonetic_index.values():
        if len(bucket) > MAX_BUCKET_SIZE:
            skipped += 1
            continue
        ids = sorted(bucket)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))

    if skipped:
        logger.warning(
            "Phonetic blocking: %d buckets skipped (size > %d)",
            skipped, MAX_BUCKET_SIZE,
        )
    logger.info("Phonetic blocking: %d raw pairs from shared codes", len(pairs))
    return pairs
