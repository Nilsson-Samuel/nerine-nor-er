"""Tests for candidate union logic.

Covers:
- Same-type constraint filters cross-type pairs
- Structured pairs bypass same-type constraint
- Self-pairs are excluded
- Canonical ordering (a < b) enforced regardless of input order
- Multi-method pairs get blocking_source="multi"
- Single-method pairs get blocking_source equal to the method name
- blocking_methods is sorted and distinct
- blocking_method_count matches len(blocking_methods)
- Deduplication: same pair from multiple sources counted once
"""

import pytest

from src.blocking.candidates import union_candidates


ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32
ID_D = "d" * 32

TYPES = {ID_A: "PER", ID_B: "PER", ID_C: "ORG", ID_D: "FIN"}


def _make_empty():
    return []


class TestSameTypeConstraint:
    def test_same_type_pair_kept(self):
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_B)],
            phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert len(result) == 1
        assert result[0]["entity_id_a"] == ID_A
        assert result[0]["entity_id_b"] == ID_B

    def test_cross_type_pair_filtered(self):
        # PER + ORG should be rejected
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_C)],
            phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert len(result) == 0

    def test_structured_bypasses_type_check(self):
        # Structured pairs skip the same-type constraint
        result = union_candidates(
            faiss_pairs=[], phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[],
            structured_pairs=[(ID_A, ID_C)],  # PER + ORG
            entity_types=TYPES,
        )
        assert len(result) == 1


class TestSelfPairFiltering:
    def test_self_pair_excluded(self):
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_A)],
            phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert len(result) == 0


class TestCanonicalOrdering:
    def test_reversed_input_still_ordered(self):
        result = union_candidates(
            faiss_pairs=[(ID_B, ID_A)],  # reversed
            phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert len(result) == 1
        assert result[0]["entity_id_a"] == ID_A
        assert result[0]["entity_id_b"] == ID_B


class TestSourceTracking:
    def test_single_method_source(self):
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_B)],
            phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert result[0]["blocking_source"] == "faiss"
        assert result[0]["blocking_methods"] == ["faiss"]
        assert result[0]["blocking_method_count"] == 1

    def test_multi_method_source(self):
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_B)],
            phonetic_pairs=[(ID_A, ID_B)],
            minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert result[0]["blocking_source"] == "multi"
        assert result[0]["blocking_methods"] == ["faiss", "phonetic"]
        assert result[0]["blocking_method_count"] == 2

    def test_three_methods_multi(self):
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_B)],
            phonetic_pairs=[(ID_A, ID_B)],
            minhash_pairs=[(ID_A, ID_B)],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert result[0]["blocking_source"] == "multi"
        assert result[0]["blocking_methods"] == ["faiss", "minhash", "phonetic"]
        assert result[0]["blocking_method_count"] == 3

    def test_methods_sorted_and_distinct(self):
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_B), (ID_A, ID_B)],  # duplicate
            phonetic_pairs=[(ID_B, ID_A)],  # reversed duplicate
            minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert len(result) == 1
        assert result[0]["blocking_methods"] == ["faiss", "phonetic"]


class TestDeduplication:
    def test_same_pair_from_same_method_counted_once(self):
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_B), (ID_B, ID_A), (ID_A, ID_B)],
            phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert len(result) == 1

    def test_different_pairs_kept_separate(self):
        types = {ID_A: "PER", ID_B: "PER", ID_C: "PER"}
        result = union_candidates(
            faiss_pairs=[(ID_A, ID_B), (ID_A, ID_C)],
            phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=types,
        )
        assert len(result) == 2


class TestEmptyInputs:
    def test_all_empty_produces_no_candidates(self):
        result = union_candidates(
            faiss_pairs=[], phonetic_pairs=[], minhash_pairs=[],
            exact_pairs=[], structured_pairs=[],
            entity_types=TYPES,
        )
        assert len(result) == 0
