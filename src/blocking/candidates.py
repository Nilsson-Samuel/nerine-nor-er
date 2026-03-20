"""Candidate union — merge pairs from all blocking methods with source tracking.

Takes raw pair lists from all blocking strategies (exact, structured, FAISS,
phonetic, MinHash), enforces same-type pairing, canonical ID ordering (a < b),
deduplication, and produces final candidate records with blocking_methods /
blocking_source metadata.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def union_candidates(
    faiss_pairs: list[tuple[str, str]],
    phonetic_pairs: list[tuple[str, str]],
    minhash_pairs: list[tuple[str, str]],
    exact_pairs: list[tuple[str, str]],
    structured_pairs: list[tuple[str, str]],
    entity_types: dict[str, str],
) -> list[dict]:
    """Merge candidate pairs from all blocking methods into deduplicated records.

    Enforces:
    - Same-type constraint: only pairs where both entities share the same type.
      (Structured pairs bypass this — a phone number is the same identifier
      regardless of how NER typed the surrounding entity.)
    - Canonical ordering: entity_id_a < entity_id_b.
    - No self-pairs.
    - Source tracking: which methods produced each pair.

    Args:
        faiss_pairs: Raw pairs from FAISS neighbor search.
        phonetic_pairs: Raw pairs from phonetic blocking (PER only).
        minhash_pairs: Raw pairs from MinHash LSH.
        exact_pairs: Raw pairs from exact normalized name match.
        structured_pairs: Raw pairs from strong identifier match (FIN/COMM/VEH).
        entity_types: Mapping of entity_id → entity type string.

    Returns:
        List of candidate dicts with keys:
        (entity_id_a, entity_id_b, blocking_methods, blocking_source,
         blocking_method_count).
    """
    # Accumulate methods per canonical pair
    pair_methods: dict[tuple[str, str], set[str]] = defaultdict(set)

    def _add_pairs(
        raw: list[tuple[str, str]],
        method: str,
        *,
        skip_type_check: bool = False,
    ) -> int:
        added = 0
        for a, b in raw:
            if a == b:
                continue
            # Same-type constraint (structured pairs bypass this)
            if not skip_type_check:
                type_a = entity_types.get(a)
                type_b = entity_types.get(b)
                if type_a != type_b:
                    continue
            # Canonical ordering
            key = (a, b) if a < b else (b, a)
            pair_methods[key].add(method)
            added += 1
        return added

    n_exact = _add_pairs(exact_pairs, "exact")
    n_structured = _add_pairs(structured_pairs, "structured", skip_type_check=True)
    n_faiss = _add_pairs(faiss_pairs, "faiss")
    n_phonetic = _add_pairs(phonetic_pairs, "phonetic")
    n_minhash = _add_pairs(minhash_pairs, "minhash")

    logger.info(
        "Candidate union input: exact=%d, structured=%d, faiss=%d, "
        "phonetic=%d, minhash=%d",
        n_exact, n_structured, n_faiss, n_phonetic, n_minhash,
    )

    # Build final records
    candidates: list[dict] = []
    for (a, b), methods in pair_methods.items():
        sorted_methods = sorted(methods)
        count = len(sorted_methods)
        candidates.append({
            "entity_id_a": a,
            "entity_id_b": b,
            "blocking_methods": sorted_methods,
            "blocking_source": "multi" if count > 1 else sorted_methods[0],
            "blocking_method_count": count,
        })

    multi_count = sum(1 for c in candidates if c["blocking_source"] == "multi")
    logger.info(
        "Candidate union output: %d unique pairs (%d multi-method)",
        len(candidates), multi_count,
    )
    return candidates
