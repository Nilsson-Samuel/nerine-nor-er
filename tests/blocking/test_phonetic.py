"""Tests for Double Metaphone phonetic blocking.

Covers:
- Norwegian pre-normalization (æ, ø, å, kj, skj, hj, gj)
- Only PER entities are indexed
- Spelling variants land in same bucket ("Hansen" / "Hanssen")
- Non-PER entities produce no pairs
- Singleton names produce no pairs
- All output pairs have canonical ordering
"""

import pytest

from src.blocking.phonetic import (
    _nor_pre_normalize,
    build_phonetic_index,
    query_phonetic_pairs,
)


ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32


# ---------------------------------------------------------------------------
# Norwegian pre-normalization
# ---------------------------------------------------------------------------

class TestNorPreNormalize:
    def test_ae_replacement(self):
        assert _nor_pre_normalize("Ærlig") == "aerlig"

    def test_oe_replacement(self):
        assert _nor_pre_normalize("Ødegård") == "odegard"

    def test_aa_replacement(self):
        assert _nor_pre_normalize("Åsen") == "asen"

    def test_kj_replacement(self):
        assert _nor_pre_normalize("kjøkken") == "kokken"

    def test_skj_replacement(self):
        assert _nor_pre_normalize("skjerm") == "skerm"

    def test_hj_replacement(self):
        assert _nor_pre_normalize("hjemme") == "jemme"

    def test_gj_replacement(self):
        assert _nor_pre_normalize("gjerde") == "jerde"

    def test_combined(self):
        assert _nor_pre_normalize("Kjærlig Ås") == "kaerlig as"


# ---------------------------------------------------------------------------
# Phonetic index building
# ---------------------------------------------------------------------------

class TestBuildPhoneticIndex:
    def test_per_entities_indexed(self):
        index = build_phonetic_index(
            [ID_A, ID_B], ["per hansen", "kari nordmann"], ["PER", "PER"],
        )
        assert len(index) > 0

    def test_non_per_entities_excluded(self):
        index = build_phonetic_index(
            [ID_A, ID_B], ["oslo", "bergen"], ["LOC", "LOC"],
        )
        assert len(index) == 0

    def test_spelling_variants_share_bucket(self):
        # Hansen and Hanssen should produce at least one shared phonetic code
        index = build_phonetic_index(
            [ID_A, ID_B], ["hansen", "hanssen"], ["PER", "PER"],
        )
        shared = [ids for ids in index.values() if ID_A in ids and ID_B in ids]
        assert len(shared) > 0, "Hansen/Hanssen should share a phonetic bucket"


# ---------------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------------

class TestQueryPhoneticPairs:
    def test_shared_bucket_produces_pair(self):
        index = build_phonetic_index(
            [ID_A, ID_B], ["hansen", "hanssen"], ["PER", "PER"],
        )
        pairs = query_phonetic_pairs(index)
        assert len(pairs) > 0
        # All pairs should be canonically ordered
        assert all(a < b for a, b in pairs)

    def test_unrelated_names_no_pair(self):
        index = build_phonetic_index(
            [ID_A, ID_B], ["per", "oslo"], ["PER", "PER"],
        )
        pairs = query_phonetic_pairs(index)
        # "per" and "oslo" have completely different phonetic codes
        shared_pairs = [(a, b) for a, b in pairs if {a, b} == {ID_A, ID_B}]
        assert len(shared_pairs) == 0

    def test_singleton_no_pair(self):
        index = build_phonetic_index([ID_A], ["hansen"], ["PER"])
        pairs = query_phonetic_pairs(index)
        assert len(pairs) == 0
