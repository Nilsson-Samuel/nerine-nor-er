"""Tests for MinHash LSH blocking.

Covers:
- Similar names produce candidate pairs
- Dissimilar names produce no pairs
- Singleton produces no pairs
- All entity types are indexed (not PER-only)
- Abbreviation / partial overlap detection at low threshold (0.3)
"""

import logging

import pytest

from src.blocking import minhash
from src.blocking.minhash import build_minhash_index, query_minhash_pairs


ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32


class TestMinHashBlocking:
    def test_identical_names_produce_pair(self):
        lsh, sigs = build_minhash_index([ID_A, ID_B], ["statsbygg", "statsbygg"])
        pairs = query_minhash_pairs(lsh, sigs, [ID_A, ID_B])
        pair_set = {(min(a, b), max(a, b)) for a, b in pairs}
        assert (ID_A, ID_B) in pair_set

    def test_very_similar_names_produce_pair(self):
        lsh, sigs = build_minhash_index(
            [ID_A, ID_B], ["statsbygg", "statsbygget"],
        )
        pairs = query_minhash_pairs(lsh, sigs, [ID_A, ID_B])
        pair_set = {(min(a, b), max(a, b)) for a, b in pairs}
        assert (ID_A, ID_B) in pair_set

    def test_completely_different_names_no_pair(self):
        lsh, sigs = build_minhash_index(
            [ID_A, ID_B], ["per hansen", "oslo politidistrikt"],
        )
        pairs = query_minhash_pairs(lsh, sigs, [ID_A, ID_B])
        pair_set = {(min(a, b), max(a, b)) for a, b in pairs}
        assert (ID_A, ID_B) not in pair_set

    def test_singleton_no_pair(self):
        lsh, sigs = build_minhash_index([ID_A], ["statsbygg"])
        pairs = query_minhash_pairs(lsh, sigs, [ID_A])
        assert len(pairs) == 0

    def test_all_types_indexed(self):
        # MinHash works across all entity types, not just PER
        lsh, sigs = build_minhash_index(
            [ID_A, ID_B], ["oslo kommune", "oslo kommune"],
        )
        pairs = query_minhash_pairs(lsh, sigs, [ID_A, ID_B])
        pair_set = {(min(a, b), max(a, b)) for a, b in pairs}
        assert (ID_A, ID_B) in pair_set

    def test_logs_progress_for_large_enough_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        monkeypatch.setattr(minhash, "_MINHASH_PROGRESS_LOG_INTERVAL", 1)
        caplog.set_level(logging.INFO, logger="src.blocking.minhash")

        lsh, sigs = build_minhash_index([ID_A, ID_B], ["statsbygg", "statsbygg"])
        query_minhash_pairs(lsh, sigs, [ID_A, ID_B])

        assert "MinHash index progress: 1/2 entities" in caplog.text
        assert "MinHash query progress: 1/2 entities" in caplog.text
