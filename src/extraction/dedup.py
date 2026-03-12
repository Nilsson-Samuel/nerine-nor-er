"""Within-document entity deduplication — exact then fuzzy merge.

Groups mentions by (doc_id, type), merges exact normalized matches,
then fuzzy-merges remaining mentions within each group.  Each merged
entity preserves full position provenance and a deterministic primary
mention (longest normalized, then earliest span as tie-breaker).
"""

import logging
from itertools import groupby

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Fuzzy threshold: rapidfuzz.fuzz.ratio returns 0-100.
# Structured types (COMM, FIN, VEH) use exact-only (no fuzzy) to avoid
# accidental merges of distinct phone numbers or account numbers.
_FUZZY_THRESHOLD = 90
_EXACT_ONLY_TYPES = {"COMM", "FIN", "VEH"}


def dedup_mentions(mentions: list[dict]) -> list[dict]:
    """Deduplicate mentions within each (doc_id, type) group.

    Two-pass approach:
    1. Exact merge: identical `normalized` strings collapse into one entity.
    2. Fuzzy merge: remaining entities within the same group merge when
       rapidfuzz.fuzz.ratio > 90 (skipped for structured types).

    Each merged entity gets a deterministic primary mention: the one with
    the longest normalized text; ties broken by earliest (chunk_id, char_start).

    Args:
        mentions: List of mention dicts, each containing at minimum:
            doc_id, chunk_id, text, normalized, type, char_start, char_end,
            page_num, source_unit_kind, source.

    Returns:
        List of deduped entity dicts, each with keys: doc_id, text, normalized,
        type, chunk_id, char_start, char_end, count, positions.
    """
    if not mentions:
        return []

    # Sort for deterministic grouping
    mentions.sort(key=lambda m: (m["doc_id"], m["type"], m["normalized"]))

    deduped: list[dict] = []
    before_count = len(mentions)

    for (doc_id, entity_type), group in groupby(
        mentions, key=lambda m: (m["doc_id"], m["type"]),
    ):
        group_list = list(group)

        # Pass 1: exact merge on normalized text
        buckets = _exact_merge(group_list)

        # Pass 2: fuzzy merge (skip for structured types)
        if entity_type not in _EXACT_ONLY_TYPES:
            buckets = _fuzzy_merge(buckets)

        # Convert buckets → entity records
        for bucket in buckets:
            deduped.append(_build_entity_record(doc_id, entity_type, bucket))

    logger.info(
        "Dedup: %d mentions → %d entities (%.0f%% reduction)",
        before_count, len(deduped),
        (1 - len(deduped) / before_count) * 100 if before_count else 0,
    )

    return deduped


def _exact_merge(mentions: list[dict]) -> list[list[dict]]:
    """Group mentions by exact normalized text into buckets."""
    buckets: dict[str, list[dict]] = {}
    for m in mentions:
        key = m["normalized"]
        buckets.setdefault(key, []).append(m)
    return list(buckets.values())


def _fuzzy_merge(buckets: list[list[dict]]) -> list[list[dict]]:
    """Merge buckets whose representative normalized texts are similar."""
    if len(buckets) <= 1:
        return buckets

    # Representative = longest normalized in bucket (deterministic)
    reps = [_representative_normalized(b) for b in buckets]
    merged_flags = [False] * len(buckets)
    result: list[list[dict]] = []

    for i in range(len(buckets)):
        if merged_flags[i]:
            continue
        current = list(buckets[i])
        for j in range(i + 1, len(buckets)):
            if merged_flags[j]:
                continue
            if fuzz.ratio(reps[i], reps[j]) > _FUZZY_THRESHOLD:
                current.extend(buckets[j])
                merged_flags[j] = True
        result.append(current)

    return result


def _representative_normalized(bucket: list[dict]) -> str:
    """Pick the representative normalized text from a bucket (longest first)."""
    return max(bucket, key=lambda m: (len(m["normalized"]), m["normalized"]))["normalized"]


def _build_entity_record(
    doc_id: str, entity_type: str, bucket: list[dict],
) -> dict:
    """Build a merged entity record from a bucket of mentions.

    Primary mention is selected deterministically: longest normalized text,
    then earliest (chunk_id, char_start) as tie-breaker.
    """
    # Sort bucket deterministically for stable primary selection
    bucket.sort(key=lambda m: (-len(m["normalized"]), m["chunk_id"], m["char_start"]))
    primary = bucket[0]

    positions = [
        {
            "chunk_id": m["chunk_id"],
            "char_start": m["char_start"],
            "char_end": m["char_end"],
            "page_num": m["page_num"],
            "source_unit_kind": m["source_unit_kind"],
        }
        for m in bucket
    ]
    # Sort positions deterministically
    positions.sort(key=lambda p: (p["chunk_id"], p["char_start"]))

    return {
        "doc_id": doc_id,
        "text": primary["text"],
        "normalized": primary["normalized"],
        "type": entity_type,
        "chunk_id": primary["chunk_id"],
        "char_start": primary["char_start"],
        "char_end": primary["char_end"],
        "count": len(positions),
        "positions": positions,
    }
