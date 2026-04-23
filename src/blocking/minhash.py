"""MinHash LSH blocking — token and character n-gram similarity.

Builds MinHash signatures from word tokens + character 3-grams of each
entity's normalized name, then queries a MinHashLSH index (threshold=0.3)
to find similar entities. The low threshold intentionally catches
abbreviations and partial overlaps (e.g. "Statsbygg" ↔ "Statsbygget").
"""

import logging

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

LSH_THRESHOLD = 0.3
NUM_PERM = 128
_MINHASH_PROGRESS_LOG_INTERVAL = 1000


def _build_minhash(name: str) -> MinHash:
    """Create a MinHash signature from word tokens + character 3-grams."""
    m = MinHash(num_perm=NUM_PERM)
    for tok in name.split():
        m.update(tok.encode("utf-8"))
    for i in range(len(name) - 2):
        m.update(name[i : i + 3].encode("utf-8"))
    return m


def build_minhash_index(
    entity_ids: list[str],
    names: list[str],
) -> tuple[MinHashLSH, dict[str, MinHash]]:
    """Build a MinHash LSH index over all entity names.

    Args:
        entity_ids: Entity ID strings.
        names: Normalized entity name strings (aligned with entity_ids).

    Returns:
        Tuple of (lsh_index, signatures_dict) where signatures_dict maps
        entity_id → MinHash for query lookups.
    """
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    signatures: dict[str, MinHash] = {}
    total_entities = len(entity_ids)

    for index, (eid, name) in enumerate(zip(entity_ids, names), start=1):
        m = _build_minhash(name)
        signatures[eid] = m
        try:
            lsh.insert(eid, m)
        except ValueError:
            # Duplicate key — skip (can happen if entity_id already inserted)
            pass
        _log_progress("MinHash index", index, total_entities)

    logger.info(
        "MinHash LSH: %d entities indexed (threshold=%.2f, num_perm=%d)",
        len(signatures), LSH_THRESHOLD, NUM_PERM,
    )
    return lsh, signatures


def query_minhash_pairs(
    lsh: MinHashLSH,
    signatures: dict[str, MinHash],
    entity_ids: list[str],
) -> list[tuple[str, str]]:
    """Query the MinHash LSH index and return candidate pairs.

    Args:
        lsh: Built MinHash LSH index.
        signatures: Entity ID → MinHash mapping.
        entity_ids: All entity IDs to query.

    Returns:
        List of (entity_id_a, entity_id_b) pairs (may contain duplicates).
    """
    pairs: list[tuple[str, str]] = []
    total_entities = len(entity_ids)
    for index, eid in enumerate(entity_ids, start=1):
        m = signatures[eid]
        results = lsh.query(m)
        for neighbor_id in results:
            if neighbor_id != eid:
                pairs.append((eid, neighbor_id))
        _log_progress("MinHash query", index, total_entities, raw_pairs=len(pairs))

    logger.info("MinHash blocking: %d raw pairs", len(pairs))
    return pairs


def _log_progress(
    stage_name: str,
    completed: int,
    total: int,
    *,
    raw_pairs: int | None = None,
) -> None:
    """Emit low-noise MinHash progress only for large workloads."""
    if total < _MINHASH_PROGRESS_LOG_INTERVAL:
        return
    if completed % _MINHASH_PROGRESS_LOG_INTERVAL != 0 and completed != total:
        return
    if raw_pairs is None:
        logger.info("%s progress: %d/%d entities", stage_name, completed, total)
        return
    logger.info(
        "%s progress: %d/%d entities, %d raw pairs so far",
        stage_name,
        completed,
        total,
        raw_pairs,
    )
