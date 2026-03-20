"""Within-document entity deduplication — exact normalized match only.

Groups mentions by (doc_id, type) and merges those with identical
normalized text.  Each merged entity preserves full position provenance
and a deterministic primary mention (longest normalized, then earliest
span as tie-breaker).

Only exact matches are collapsed — near-miss variants are left as separate
entities so downstream candidate generation + matching can decide whether
they are the same identity.  This avoids irreversible false merges early
in the pipeline where recall matters more than compactness.
"""

import logging
from itertools import groupby

logger = logging.getLogger(__name__)


def dedup_mentions(mentions: list[dict]) -> list[dict]:
    """Deduplicate mentions within each (doc_id, type) group.

    Mentions with identical `normalized` text within the same document and
    entity type are collapsed into a single entity row.  Each merged entity
    gets a deterministic primary mention: the one with the longest normalized
    text; ties broken by earliest (chunk_id, char_start).

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

        # Exact merge on normalized text
        buckets = _exact_merge(group_list)

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
