"""Tests for exact normalized name and structured identifier blocking.

Covers:
- Same (type, normalized) pairs are generated
- Different types with same name produce no pairs
- Singletons produce no pairs
- Structured ID pairs are generated across type boundaries (FIN/COMM/VEH)
- Non-strong types (PER, ORG, LOC) are excluded from structured blocking
- Canonical ordering (a < b) on all output pairs
"""

import pytest

from src.blocking.exact import build_exact_name_pairs, build_structured_id_pairs


# Synthetic entity IDs (32-char hex, lexicographically ordered)
ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32
ID_D = "d" * 32


# ---------------------------------------------------------------------------
# Exact normalized name match
# ---------------------------------------------------------------------------

class TestBuildExactNamePairs:
    def test_same_type_same_name_produces_pair(self):
        pairs = build_exact_name_pairs(
            [ID_A, ID_B], ["per hansen", "per hansen"], ["PER", "PER"],
        )
        assert (ID_A, ID_B) in pairs

    def test_different_types_same_name_no_pair(self):
        pairs = build_exact_name_pairs(
            [ID_A, ID_B], ["oslo", "oslo"], ["LOC", "ORG"],
        )
        assert len(pairs) == 0

    def test_different_names_same_type_no_pair(self):
        pairs = build_exact_name_pairs(
            [ID_A, ID_B], ["per hansen", "kari nordmann"], ["PER", "PER"],
        )
        assert len(pairs) == 0

    def test_singleton_no_pair(self):
        pairs = build_exact_name_pairs([ID_A], ["per hansen"], ["PER"])
        assert len(pairs) == 0

    def test_three_way_produces_three_pairs(self):
        pairs = build_exact_name_pairs(
            [ID_A, ID_B, ID_C],
            ["dnb", "dnb", "dnb"],
            ["ORG", "ORG", "ORG"],
        )
        assert len(pairs) == 3
        assert (ID_A, ID_B) in pairs
        assert (ID_A, ID_C) in pairs
        assert (ID_B, ID_C) in pairs

    def test_canonical_ordering(self):
        # Pass IDs in reverse order — output should still be (a, b) where a < b
        pairs = build_exact_name_pairs(
            [ID_B, ID_A], ["oslo", "oslo"], ["LOC", "LOC"],
        )
        assert all(a < b for a, b in pairs)


# ---------------------------------------------------------------------------
# Structured identifier match
# ---------------------------------------------------------------------------

class TestBuildStructuredIdPairs:
    def test_fin_exact_match(self):
        pairs = build_structured_id_pairs(
            [ID_A, ID_B], ["1234567890", "1234567890"], ["FIN", "FIN"],
        )
        assert (ID_A, ID_B) in pairs

    def test_comm_exact_match(self):
        pairs = build_structured_id_pairs(
            [ID_A, ID_B], ["+4712345678", "+4712345678"], ["COMM", "COMM"],
        )
        assert (ID_A, ID_B) in pairs

    def test_veh_exact_match(self):
        pairs = build_structured_id_pairs(
            [ID_A, ID_B], ["ab12345", "ab12345"], ["VEH", "VEH"],
        )
        assert (ID_A, ID_B) in pairs

    def test_cross_type_strong_ids_produces_pair(self):
        # Same identifier found as FIN in one doc and COMM in another
        pairs = build_structured_id_pairs(
            [ID_A, ID_B], ["1234567890", "1234567890"], ["FIN", "COMM"],
        )
        assert (ID_A, ID_B) in pairs

    def test_per_excluded_from_structured(self):
        pairs = build_structured_id_pairs(
            [ID_A, ID_B], ["per hansen", "per hansen"], ["PER", "PER"],
        )
        assert len(pairs) == 0

    def test_org_excluded_from_structured(self):
        pairs = build_structured_id_pairs(
            [ID_A, ID_B], ["dnb", "dnb"], ["ORG", "ORG"],
        )
        assert len(pairs) == 0

    def test_different_values_no_pair(self):
        pairs = build_structured_id_pairs(
            [ID_A, ID_B], ["1234567890", "9876543210"], ["FIN", "FIN"],
        )
        assert len(pairs) == 0

    def test_singleton_no_pair(self):
        pairs = build_structured_id_pairs(
            [ID_A], ["1234567890"], ["FIN"],
        )
        assert len(pairs) == 0
