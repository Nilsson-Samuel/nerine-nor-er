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

    for eid, name in zip(entity_ids, names):
        m = _build_minhash(name)
        signatures[eid] = m
        try:
            lsh.insert(eid, m)
        except ValueError:
            # Duplicate key — skip (can happen if entity_id already inserted)
            pass

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
    for eid in entity_ids:
        m = signatures[eid]
        results = lsh.query(m)
        for neighbor_id in results:
            if neighbor_id != eid:
                pairs.append((eid, neighbor_id))

    logger.info("MinHash blocking: %d raw pairs", len(pairs))
    return pairs
