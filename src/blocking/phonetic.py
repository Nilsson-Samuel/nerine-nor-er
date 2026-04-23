"""Double Metaphone phonetic blocking for person-name matching.

Builds a phonetic index from entity names using Double Metaphone codes.
Norwegian characters are pre-normalized (æ→ae, ø→o, å→a, etc.) before
encoding. Both per-token and full-name codes are indexed so that
"Hansen" and "Hanssen" land in the same bucket.

Oversized-bucket strategy: a pair is only emitted when the two entities
share at least MIN_SHARED_CODES distinct phonetic codes. Common tokens
(e.g. "HSN" for every Hansen variant) therefore cannot alone produce pairs
— a second signal (the full-name code or another token) is required.
Buckets above ENUMERATION_CAP are skipped for pair counting, but genuinely
matching entities in those buckets will almost always share additional codes
via smaller buckets.
"""

import logging
from collections import defaultdict

from metaphone import doublemetaphone

logger = logging.getLogger(__name__)

ENUMERATION_CAP = 5_000  # skip pair-counting in buckets larger than this
MIN_SHARED_CODES = 2      # pairs must share at least this many phonetic codes


def _nor_pre_normalize(text: str) -> str:
    """Normalize Norwegian characters to ASCII-friendly phonetic input."""
    return (
        text.lower()
        .replace("æ", "ae")
        .replace("ø", "o")
        .replace("å", "a")
        .replace("skj", "sk")
        .replace("kj", "k")
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
    min_shared_codes: int = MIN_SHARED_CODES,
) -> list[tuple[str, str]]:
    """Extract candidate pairs from entities sharing multiple phonetic codes.

    A pair is emitted only when the two entities share at least
    ``min_shared_codes`` distinct phonetic codes. This replaces the hard
    bucket-size skip: a single very common token code (e.g. "HSN") is no
    longer sufficient to produce a candidate pair — the entities must also
    agree on at least one other code (full-name code or another token).

    Buckets above ENUMERATION_CAP are skipped for pair counting. Entities
    from those buckets are still reachable via smaller shared-code buckets.

    Args:
        phonetic_index: Output of build_phonetic_index().
        min_shared_codes: Minimum shared codes required to emit a pair.

    Returns:
        List of deduplicated (entity_id_a, entity_id_b) pairs in canonical order.
    """
    pair_code_count: dict[tuple[str, str], int] = defaultdict(int)
    skipped_large = 0

    for code, eids in phonetic_index.items():
        ids = sorted(eids)
        if len(ids) > ENUMERATION_CAP:
            skipped_large += 1
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                key = (a, b) if a < b else (b, a)
                pair_code_count[key] += 1

    pairs = [
        pair for pair, count in pair_code_count.items() if count >= min_shared_codes
    ]

    if skipped_large:
        logger.warning(
            "Phonetic blocking: %d buckets exceeded enumeration cap (%d) and were "
            "skipped; matching entities are still reachable via other shared codes",
            skipped_large, ENUMERATION_CAP,
        )
    logger.info(
        "Phonetic blocking: %d pairs with >= %d shared codes (%d candidates before filter)",
        len(pairs), min_shared_codes, len(pair_code_count),
    )
    return pairs
