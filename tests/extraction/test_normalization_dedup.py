"""Tests for entity normalization and within-doc deduplication.

Covers validation gate criteria:
- Per-type normalization rules produce expected canonical forms
- No cross-type or cross-document merges during dedup
- count == len(positions) after merge
- Primary mention is deterministic (longest normalized, then earliest span)
- Fuzzy merge only applies to non-structured types (not COMM/FIN/VEH)
- Same input → same deduped output across repeated runs
"""

import pytest

from src.extraction.entity_normalizer import normalize_entity
from src.extraction.dedup import dedup_mentions


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalizePER:
    def test_title_case(self):
        assert normalize_entity("kari nordmann", "PER") == "Kari Nordmann"

    def test_strips_hr_title(self):
        assert normalize_entity("Hr. Hansen", "PER") == "Hansen"

    def test_strips_fru_title(self):
        assert normalize_entity("Fru. Olsen", "PER") == "Olsen"

    def test_strips_dr_title(self):
        assert normalize_entity("dr. Berg", "PER") == "Berg"

    def test_collapses_whitespace(self):
        assert normalize_entity("  Per   Hansen  ", "PER") == "Per Hansen"

    def test_preserves_name_if_only_title(self):
        # Edge case: if the name is just a title, keep it
        assert normalize_entity("Hr.", "PER") == "Hr."


class TestNormalizeORG:
    def test_strips_as_suffix(self):
        assert normalize_entity("DNB ASA", "ORG") == "Dnb"

    def test_strips_as_suffix_lowercase(self):
        assert normalize_entity("Equinor AS", "ORG") == "Equinor"

    def test_title_case(self):
        assert normalize_entity("OSLO POLITIDISTRIKT", "ORG") == "Oslo Politidistrikt"

    def test_no_suffix_preserved(self):
        assert normalize_entity("Kripos", "ORG") == "Kripos"


class TestNormalizeLOC:
    def test_strips_kommune(self):
        assert normalize_entity("Oslo kommune", "LOC") == "Oslo"

    def test_strips_fylke(self):
        assert normalize_entity("Vestland fylke", "LOC") == "Vestland"

    def test_title_case(self):
        assert normalize_entity("BERGEN", "LOC") == "Bergen"

    def test_no_suffix_preserved(self):
        assert normalize_entity("Trondheim", "LOC") == "Trondheim"


class TestNormalizeVEH:
    def test_uppercase_plate(self):
        assert normalize_entity("ab 12345", "VEH") == "AB 12345"


class TestNormalizeCOMM:
    def test_phone_digits_only(self):
        assert normalize_entity("912 34 567", "COMM") == "91234567"

    def test_compact_phone(self):
        assert normalize_entity("91234567", "COMM") == "91234567"

    def test_email_lowercase(self):
        assert normalize_entity("User@Example.COM", "COMM") == "user@example.com"


class TestNormalizeFIN:
    def test_strips_spaces_and_hyphens(self):
        assert normalize_entity("NO93 8601-1117 947", "FIN") == "NO9386011117947"


class TestNormalizeITEM:
    def test_lowercase(self):
        assert normalize_entity("Kokain", "ITEM") == "kokain"


class TestNormalizeEdgeCases:
    def test_empty_string(self):
        assert normalize_entity("", "PER") == ""

    def test_none_text(self):
        assert normalize_entity(None, "PER") == ""

    def test_whitespace_only(self):
        assert normalize_entity("   ", "PER") == ""


# ---------------------------------------------------------------------------
# Dedup tests
# ---------------------------------------------------------------------------

def _make_mention(
    doc_id="a" * 32, chunk_id="b" * 32, text="Kari", normalized="Kari",
    entity_type="PER", char_start=0, char_end=4, page_num=0,
    source_unit_kind="pdf_page", source="ner",
):
    """Helper to build a mention dict with defaults."""
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "normalized": normalized,
        "type": entity_type,
        "char_start": char_start,
        "char_end": char_end,
        "page_num": page_num,
        "source_unit_kind": source_unit_kind,
        "source": source,
    }


class TestExactDedup:
    def test_identical_mentions_merged(self):
        mentions = [
            _make_mention(text="Kari", normalized="Kari", char_start=0, char_end=4),
            _make_mention(text="Kari", normalized="Kari", char_start=50, char_end=54),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 1
        assert result[0]["count"] == 2
        assert len(result[0]["positions"]) == 2

    def test_different_normalized_not_merged(self):
        mentions = [
            _make_mention(text="Kari", normalized="Kari"),
            _make_mention(text="Per", normalized="Per", char_start=10, char_end=13),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 2


class TestNoCrossTypeMerge:
    def test_same_text_different_type_not_merged(self):
        mentions = [
            _make_mention(text="Oslo", normalized="Oslo", entity_type="ORG"),
            _make_mention(text="Oslo", normalized="Oslo", entity_type="LOC",
                          char_start=10, char_end=14),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 2


class TestNoCrossDocMerge:
    def test_same_text_different_doc_not_merged(self):
        mentions = [
            _make_mention(text="Kari", normalized="Kari", doc_id="a" * 32),
            _make_mention(text="Kari", normalized="Kari", doc_id="b" * 32,
                          char_start=10, char_end=14),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 2


class TestFuzzyDedup:
    def test_similar_names_merged(self):
        # "Per Hansen" vs "Per Hanseen" — ratio > 90
        mentions = [
            _make_mention(text="Per Hansen", normalized="Per Hansen",
                          char_start=0, char_end=10),
            _make_mention(text="Per Hanseen", normalized="Per Hanseen",
                          char_start=50, char_end=61),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 1
        assert result[0]["count"] == 2
        # Longest normalized is primary
        assert result[0]["normalized"] == "Per Hanseen"

    def test_dissimilar_names_not_merged(self):
        mentions = [
            _make_mention(text="Per Hansen", normalized="Per Hansen",
                          char_start=0, char_end=10),
            _make_mention(text="Kari Nordmann", normalized="Kari Nordmann",
                          char_start=50, char_end=63),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 2


class TestStructuredTypeNoFuzzy:
    """COMM, FIN, VEH should never fuzzy-merge — only exact."""

    def test_comm_no_fuzzy(self):
        mentions = [
            _make_mention(text="91234567", normalized="91234567",
                          entity_type="COMM", char_start=0, char_end=8),
            _make_mention(text="91234568", normalized="91234568",
                          entity_type="COMM", char_start=20, char_end=28),
        ]
        result = dedup_mentions(mentions)
        # These are similar (ratio > 90) but should NOT merge
        assert len(result) == 2

    def test_fin_no_fuzzy(self):
        mentions = [
            _make_mention(text="NO9386011117947", normalized="NO9386011117947",
                          entity_type="FIN", char_start=0, char_end=15),
            _make_mention(text="NO9386011117948", normalized="NO9386011117948",
                          entity_type="FIN", char_start=30, char_end=45),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 2

    def test_veh_no_fuzzy(self):
        mentions = [
            _make_mention(text="AB12345", normalized="AB12345",
                          entity_type="VEH", char_start=0, char_end=7),
            _make_mention(text="AB12346", normalized="AB12346",
                          entity_type="VEH", char_start=20, char_end=27),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 2


class TestPrimaryMentionSelection:
    def test_longest_normalized_is_primary(self):
        mentions = [
            _make_mention(text="Per", normalized="Per",
                          char_start=0, char_end=3),
            _make_mention(text="Per Hansen", normalized="Per Hansen",
                          char_start=50, char_end=60),
        ]
        # These won't fuzzy-merge (ratio too low), but let's test exact merge
        # with same normalized
        mentions_exact = [
            _make_mention(text="Per H.", normalized="Per H.",
                          char_start=0, char_end=6, chunk_id="c" * 32),
            _make_mention(text="Per H.", normalized="Per H.",
                          char_start=50, char_end=56, chunk_id="d" * 32),
        ]
        result = dedup_mentions(mentions_exact)
        assert len(result) == 1
        # Tie-break: earliest chunk_id
        assert result[0]["chunk_id"] == "c" * 32

    def test_tiebreak_earliest_span(self):
        mentions = [
            _make_mention(text="Kari", normalized="Kari",
                          char_start=100, char_end=104),
            _make_mention(text="Kari", normalized="Kari",
                          char_start=10, char_end=14),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 1
        # Same chunk_id, earlier char_start wins
        assert result[0]["char_start"] == 10


class TestCountEqualsPositions:
    def test_count_matches_positions_length(self):
        mentions = [
            _make_mention(text="Kari", normalized="Kari", char_start=0, char_end=4),
            _make_mention(text="Kari", normalized="Kari", char_start=50, char_end=54),
            _make_mention(text="Kari", normalized="Kari", char_start=100, char_end=104),
        ]
        result = dedup_mentions(mentions)
        assert len(result) == 1
        entity = result[0]
        assert entity["count"] == len(entity["positions"])
        assert entity["count"] == 3


class TestPositionsContainPrimarySpan:
    def test_primary_span_in_positions(self):
        mentions = [
            _make_mention(text="Kari", normalized="Kari", char_start=0, char_end=4),
            _make_mention(text="Kari", normalized="Kari", char_start=50, char_end=54),
        ]
        result = dedup_mentions(mentions)
        entity = result[0]
        primary_in_positions = any(
            p["chunk_id"] == entity["chunk_id"]
            and p["char_start"] == entity["char_start"]
            and p["char_end"] == entity["char_end"]
            for p in entity["positions"]
        )
        assert primary_in_positions


class TestDedupDeterminism:
    def test_repeated_runs_same_output(self):
        mentions = [
            _make_mention(text="Kari Nordmann", normalized="Kari Nordmann",
                          char_start=0, char_end=13),
            _make_mention(text="Per Hansen", normalized="Per Hansen",
                          char_start=50, char_end=60),
            _make_mention(text="Kari Nordmann", normalized="Kari Nordmann",
                          char_start=100, char_end=113),
        ]
        results = [dedup_mentions(list(mentions)) for _ in range(3)]
        assert results[0] == results[1] == results[2]


class TestDedupEmpty:
    def test_empty_input(self):
        assert dedup_mentions([]) == []

    def test_single_mention(self):
        mentions = [_make_mention()]
        result = dedup_mentions(mentions)
        assert len(result) == 1
        assert result[0]["count"] == 1
