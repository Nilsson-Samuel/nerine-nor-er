"""Tests for Double Metaphone phonetic blocking.

Covers:
- Norwegian pre-normalization (æ, ø, å, kj, skj, hj, gj)
- Only PER entities are indexed
- Spelling variants land in same bucket ("Hansen" / "Hanssen")
- Non-PER entities produce no pairs
- Singleton names produce no pairs
- All output pairs have canonical ordering
- Intersection strategy: single shared code is not enough to emit a pair
- Oversized-bucket enumeration cap: buckets > ENUMERATION_CAP are skipped
  but entities in them are still reachable via smaller shared-code buckets
- min_shared_codes parameter is respected
"""

import pytest

from src.blocking.phonetic import (
    ENUMERATION_CAP,
    MIN_SHARED_CODES,
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

    def test_output_deduplicated(self):
        # Hansen/Hanssen share multiple codes; the pair must appear exactly once.
        index = build_phonetic_index(
            [ID_A, ID_B], ["per hansen", "per hanssen"], ["PER", "PER"],
        )
        pairs = query_phonetic_pairs(index)
        assert len(pairs) == len(set(pairs)), "output contains duplicate pairs"

    def test_canonical_ordering_all_pairs(self):
        index = build_phonetic_index(
            [ID_A, ID_B, ID_C],
            ["per hansen", "per hanssen", "petter hannsen"],
            ["PER", "PER", "PER"],
        )
        pairs = query_phonetic_pairs(index)
        assert all(a < b for a, b in pairs), "not all pairs are canonically ordered"


# ---------------------------------------------------------------------------
# Intersection strategy: MIN_SHARED_CODES threshold
# ---------------------------------------------------------------------------

class TestMinSharedCodes:
    def test_single_shared_code_suppressed_by_default(self):
        # Two entities that share exactly one phonetic code should NOT be paired
        # because MIN_SHARED_CODES == 2.  We manufacture this by injecting a
        # synthetic index directly rather than going through build_phonetic_index,
        # so the test is independent of Double Metaphone output stability.
        synthetic_index = {
            "XYZ": {ID_A, ID_B},   # only one shared code
        }
        pairs = query_phonetic_pairs(synthetic_index)
        assert (ID_A, ID_B) not in pairs and (ID_B, ID_A) not in pairs

    def test_two_shared_codes_emits_pair(self):
        synthetic_index = {
            "XYZ": {ID_A, ID_B},
            "ABC": {ID_A, ID_B},   # second shared code → should emit
        }
        pairs = query_phonetic_pairs(synthetic_index)
        canonical = (ID_A, ID_B) if ID_A < ID_B else (ID_B, ID_A)
        assert canonical in pairs

    def test_min_shared_codes_parameter_respected(self):
        # With min_shared_codes=1 the single-code pair should be emitted.
        synthetic_index = {"XYZ": {ID_A, ID_B}}
        pairs = query_phonetic_pairs(synthetic_index, min_shared_codes=1)
        canonical = (ID_A, ID_B) if ID_A < ID_B else (ID_B, ID_A)
        assert canonical in pairs

    def test_min_shared_codes_three_filters_two(self):
        # With min_shared_codes=3, sharing only 2 codes must NOT emit.
        synthetic_index = {
            "XYZ": {ID_A, ID_B},
            "ABC": {ID_A, ID_B},
        }
        pairs = query_phonetic_pairs(synthetic_index, min_shared_codes=3)
        assert len(pairs) == 0

    def test_spelling_variants_pass_default_threshold(self):
        # "Per Hansen" / "Per Hanssen" are genuine phonetic siblings and must
        # survive the default MIN_SHARED_CODES=2 filter.
        index = build_phonetic_index(
            [ID_A, ID_B], ["per hansen", "per hanssen"], ["PER", "PER"],
        )
        pairs = query_phonetic_pairs(index)
        canonical = (ID_A, ID_B) if ID_A < ID_B else (ID_B, ID_A)
        assert canonical in pairs, (
            "Genuine spelling variant (Per Hansen / Per Hanssen) was filtered out "
            "by the MIN_SHARED_CODES threshold — check that both token and "
            "full-name codes are being counted"
        )

    def test_unrelated_names_suppressed_by_threshold(self):
        # "Erik Andersen" and "Nils Bakke" share no tokens phonetically;
        # they must not appear as a pair regardless of threshold.
        index = build_phonetic_index(
            [ID_A, ID_B], ["erik andersen", "nils bakke"], ["PER", "PER"],
        )
        pairs = query_phonetic_pairs(index)
        assert not any({a, b} == {ID_A, ID_B} for a, b in pairs)

    def test_constants_are_sane(self):
        assert MIN_SHARED_CODES >= 2, "threshold below 2 defeats the whole point"
        assert ENUMERATION_CAP > MIN_SHARED_CODES


# ---------------------------------------------------------------------------
# Oversized-bucket enumeration cap
# ---------------------------------------------------------------------------

class TestEnumerationCap:
    def _make_ids(self, n: int) -> list[str]:
        return [f"{i:032x}" for i in range(n)]

    def test_oversized_bucket_skipped_for_enumeration(self):
        # A bucket with ENUMERATION_CAP+1 entities must not produce pairs
        # through that bucket alone (needs a second shared code).
        big_ids = self._make_ids(ENUMERATION_CAP + 1)
        # Only one code → all IDs share exactly one code → no pairs expected.
        synthetic_index = {"HUGE": set(big_ids)}
        pairs = query_phonetic_pairs(synthetic_index)
        assert len(pairs) == 0

    def test_large_bucket_entities_reachable_via_second_code(self):
        # Even when two entities sit in a huge bucket (skipped), they must
        # still be paired if they also share a second, smaller-bucket code.
        big_ids = self._make_ids(ENUMERATION_CAP + 1)
        id_x, id_y = big_ids[0], big_ids[1]
        synthetic_index = {
            "HUGE": set(big_ids),          # skipped — too large
            "SMALL": {id_x, id_y},         # small bucket, counted
        }
        # They now share HUGE (skipped) + SMALL (counted) = 1 counted code.
        # With min_shared_codes=1 they should surface; default (2) they won't.
        pairs_1 = query_phonetic_pairs(synthetic_index, min_shared_codes=1)
        canonical = (id_x, id_y) if id_x < id_y else (id_y, id_x)
        assert canonical in pairs_1, "entity reachable via small bucket with threshold=1"

    def test_entities_with_two_small_shared_codes_survive_cap(self):
        # Two entities in a huge bucket but also sharing two small-bucket codes
        # must be emitted at the default threshold.
        big_ids = self._make_ids(ENUMERATION_CAP + 1)
        id_x, id_y = big_ids[0], big_ids[1]
        synthetic_index = {
            "HUGE": set(big_ids),
            "SM1": {id_x, id_y},
            "SM2": {id_x, id_y},
        }
        pairs = query_phonetic_pairs(synthetic_index)
        canonical = (id_x, id_y) if id_x < id_y else (id_y, id_x)
        assert canonical in pairs
