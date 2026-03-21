"""Tests for NER extraction and regex supplement modules.

Covers important (validation gate) criteria:
- NER produces only the currently supported PER, ORG, LOC mentions
- Unsupported NER labels like DRV/PROD are dropped explicitly
- Regex supplements produce COMM, PER (fnr), VEH, FIN hits
- All mentions have valid char_start/char_end provenance
- chunk_text[char_start:char_end] == mention["text"]
- Regex mentions do not overlap with NER mentions
- Overlap removal prefers NER spans over regex spans
- Mention stream is deterministic across repeated runs
"""

import pytest

from src.extraction.ner import extract_ner_mentions
from src.extraction.regex_supplements import (
    extract_regex_mentions,
    filter_overlapping_with_ner,
    _remove_overlaps,
)


# ---------------------------------------------------------------------------
# Fixtures — shared chunk metadata
# ---------------------------------------------------------------------------

_CHUNK_META = {
    "doc_id": "a" * 32,
    "chunk_id": "b" * 32,
    "page_num": 0,
    "source_unit_kind": "pdf_page",
}


# ---------------------------------------------------------------------------
# NER mapping tests
# ---------------------------------------------------------------------------

class TestNerLabelMapping:
    """Base NER only emits supported PER/ORG/LOC labels."""

    def test_supported_labels_are_mapped(self):
        text = "Ola jobber i DNB i Oslo."
        ner_pipe = lambda _: [
            {"entity_group": "PER", "start": 0, "end": 3},
            {"entity_group": "ORG", "start": 14, "end": 17},
            {"entity_group": "LOC", "start": 20, "end": 24},
        ]

        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)

        assert [m["type"] for m in mentions] == ["PER", "ORG", "LOC"]
        assert [m["text"] for m in mentions] == ["Ola", "DNB", "Oslo"]
        assert all(m["source"] == "ner" for m in mentions)

    def test_unsupported_labels_are_dropped(self):
        text = "Norwegian leste VG i Bergen."
        ner_pipe = lambda _: [
            {"entity_group": "DRV", "start": 0, "end": 9},
            {"entity_group": "PROD", "start": 16, "end": 18},
            {"entity_group": "GPE_LOC", "start": 21, "end": 27},
        ]

        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)

        assert len(mentions) == 1
        assert mentions[0]["type"] == "LOC"
        assert mentions[0]["text"] == "Bergen"

    def test_empty_or_whitespace_text_skips_ner(self):
        ner_pipe = pytest.fail
        assert extract_ner_mentions("   \n\t", ner_pipe=ner_pipe, **_CHUNK_META) == []


# ---------------------------------------------------------------------------
# Regex supplement tests
# ---------------------------------------------------------------------------

class TestRegexPhone:
    """Norwegian phone number patterns (COMM)."""

    def test_compact_8_digits(self):
        text = "Ring oss på 91234567 for mer info."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        phone_mentions = [m for m in mentions if m["type"] == "COMM"]
        assert len(phone_mentions) == 1
        m = phone_mentions[0]
        assert m["text"] == "91234567"
        assert text[m["char_start"]:m["char_end"]] == m["text"]
        assert m["source"] == "regex"

    def test_spaced_3_2_3(self):
        text = "Telefon: 912 34 567 er registrert."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        phone_mentions = [m for m in mentions if m["type"] == "COMM"]
        assert len(phone_mentions) == 1
        assert phone_mentions[0]["text"] == "912 34 567"


class TestRegexFnr:
    """Fødselsnummer patterns (PER)."""

    def test_compact_fnr(self):
        text = "Fødselsnummer 01019912345 tilhører mistenkte."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        fnr_mentions = [m for m in mentions if m["type"] == "PER"]
        assert len(fnr_mentions) == 1
        m = fnr_mentions[0]
        assert m["text"] == "01019912345"
        assert text[m["char_start"]:m["char_end"]] == m["text"]

    def test_hyphenated_fnr(self):
        text = "ID: 010199-12345."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        fnr_mentions = [m for m in mentions if m["type"] == "PER"]
        assert len(fnr_mentions) == 1
        assert fnr_mentions[0]["text"] == "010199-12345"


class TestRegexPlate:
    """Norwegian license plate patterns (VEH)."""

    def test_compact_plate(self):
        text = "Bilen med kjennemerke AB12345 ble observert."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        plate_mentions = [m for m in mentions if m["type"] == "VEH"]
        assert len(plate_mentions) == 1
        assert plate_mentions[0]["text"] == "AB12345"

    def test_spaced_plate(self):
        text = "Registrert: AB 12345."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        plate_mentions = [m for m in mentions if m["type"] == "VEH"]
        assert len(plate_mentions) == 1
        assert plate_mentions[0]["text"] == "AB 12345"


class TestRegexBankAccount:
    """Norwegian bank account number patterns (FIN)."""

    def test_dot_separated_11_digits(self):
        text = "Konto 1234.56.78901 ble brukt."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        fin_mentions = [m for m in mentions if m["type"] == "FIN"]
        assert len(fin_mentions) == 1
        assert fin_mentions[0]["text"] == "1234.56.78901"

    def test_dot_separated_10_digits(self):
        text = "Overført til 8601.11.17947."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        fin_mentions = [m for m in mentions if m["type"] == "FIN"]
        assert len(fin_mentions) == 1
        assert fin_mentions[0]["text"] == "8601.11.17947"

    def test_provenance_valid(self):
        text = "Konto 1234.56.78901 registrert."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        fin_mentions = [m for m in mentions if m["type"] == "FIN"]
        assert len(fin_mentions) == 1
        m = fin_mentions[0]
        assert text[m["char_start"]:m["char_end"]] == m["text"]


class TestRegexIban:
    """Norwegian IBAN patterns (FIN)."""

    def test_compact_iban(self):
        text = "Konto NO9386011117947 ble brukt."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        fin_mentions = [m for m in mentions if m["type"] == "FIN"]
        assert len(fin_mentions) == 1
        assert fin_mentions[0]["text"] == "NO9386011117947"

    def test_spaced_iban(self):
        text = "IBAN: NO93 8601 1117 947 registrert."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        fin_mentions = [m for m in mentions if m["type"] == "FIN"]
        assert len(fin_mentions) == 1
        assert fin_mentions[0]["text"] == "NO93 8601 1117 947"


class TestRegexProvenance:
    """Span provenance: char_start/char_end point to correct text."""

    def test_all_mentions_have_valid_spans(self):
        text = (
            "Ring 91234567. Fnr 010199-12345. "
            "Bil AB12345. IBAN NO9386011117947."
        )
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        assert len(mentions) >= 4
        for m in mentions:
            assert m["char_end"] > m["char_start"]
            assert text[m["char_start"]:m["char_end"]] == m["text"]

    def test_metadata_fields_propagated(self):
        text = "Ring 91234567."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        assert len(mentions) == 1
        m = mentions[0]
        assert m["doc_id"] == _CHUNK_META["doc_id"]
        assert m["chunk_id"] == _CHUNK_META["chunk_id"]
        assert m["page_num"] == _CHUNK_META["page_num"]
        assert m["source_unit_kind"] == _CHUNK_META["source_unit_kind"]


class TestOverlapFiltering:
    """Regex mentions are dropped when they overlap NER spans."""

    def test_overlapping_regex_removed(self):
        # NER found "01019912345" as part of a PER mention at [10, 21)
        ner_mentions = [{
            "doc_id": "a" * 32, "chunk_id": "b" * 32,
            "text": "01019912345", "type": "PER",
            "char_start": 10, "char_end": 21,
            "page_num": 0, "source_unit_kind": "pdf_page", "source": "ner",
        }]
        # Regex also found the same span
        regex_mentions = [{
            "doc_id": "a" * 32, "chunk_id": "b" * 32,
            "text": "01019912345", "type": "PER",
            "char_start": 10, "char_end": 21,
            "page_num": 0, "source_unit_kind": "pdf_page", "source": "regex",
        }]

        filtered = filter_overlapping_with_ner(regex_mentions, ner_mentions)
        assert len(filtered) == 0

    def test_non_overlapping_regex_kept(self):
        ner_mentions = [{
            "doc_id": "a" * 32, "chunk_id": "b" * 32,
            "text": "Kari Nordmann", "type": "PER",
            "char_start": 0, "char_end": 13,
            "page_num": 0, "source_unit_kind": "pdf_page", "source": "ner",
        }]
        regex_mentions = [{
            "doc_id": "a" * 32, "chunk_id": "b" * 32,
            "text": "91234567", "type": "COMM",
            "char_start": 20, "char_end": 28,
            "page_num": 0, "source_unit_kind": "pdf_page", "source": "regex",
        }]

        filtered = filter_overlapping_with_ner(regex_mentions, ner_mentions)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "91234567"

    def test_partial_overlap_removed(self):
        ner_mentions = [{
            "doc_id": "a" * 32, "chunk_id": "b" * 32,
            "text": "overlapping", "type": "PER",
            "char_start": 5, "char_end": 15,
            "page_num": 0, "source_unit_kind": "pdf_page", "source": "ner",
        }]
        regex_mentions = [{
            "doc_id": "a" * 32, "chunk_id": "b" * 32,
            "text": "partial", "type": "COMM",
            "char_start": 10, "char_end": 20,
            "page_num": 0, "source_unit_kind": "pdf_page", "source": "regex",
        }]

        filtered = filter_overlapping_with_ner(regex_mentions, ner_mentions)
        assert len(filtered) == 0

    def test_empty_ner_keeps_all_regex(self):
        regex_mentions = [{
            "doc_id": "a" * 32, "chunk_id": "b" * 32,
            "text": "91234567", "type": "COMM",
            "char_start": 0, "char_end": 8,
            "page_num": 0, "source_unit_kind": "pdf_page", "source": "regex",
        }]

        filtered = filter_overlapping_with_ner(regex_mentions, [])
        assert len(filtered) == 1


class TestInternalOverlapRemoval:
    """Internal overlap removal within regex results."""

    def test_overlapping_regex_hits_keep_longest(self):
        # Two overlapping mentions, sorted by (start, -end)
        mentions = [
            {"char_start": 0, "char_end": 10, "text": "0123456789"},
            {"char_start": 5, "char_end": 13, "text": "56789abc"},
        ]
        result = _remove_overlaps(mentions)
        assert len(result) == 1
        assert result[0]["char_start"] == 0

    def test_non_overlapping_both_kept(self):
        mentions = [
            {"char_start": 0, "char_end": 8, "text": "first"},
            {"char_start": 10, "char_end": 18, "text": "second"},
        ]
        result = _remove_overlaps(mentions)
        assert len(result) == 2


class TestRegexDeterminism:
    """Same input → same output across repeated runs."""

    def test_deterministic_output(self):
        text = (
            "Ring 91234567. Fnr 010199-12345. "
            "Bil AB12345. IBAN NO9386011117947."
        )
        results = [extract_regex_mentions(text, **_CHUNK_META) for _ in range(3)]
        assert results[0] == results[1] == results[2]


class TestRegexNoText:
    """Edge case: empty or whitespace-only text."""

    def test_empty_string(self):
        assert extract_regex_mentions("", **_CHUNK_META) == []

    def test_whitespace_only(self):
        assert extract_regex_mentions("   \n\t  ", **_CHUNK_META) == []

    def test_no_matches(self):
        text = "Ingen strukturerte data her."
        assert extract_regex_mentions(text, **_CHUNK_META) == []


# ---------------------------------------------------------------------------
# NER label mapping tests
# ---------------------------------------------------------------------------

class TestNerLabelMapping:
    """Base NER only emits supported PER/ORG/LOC labels."""

    def test_supported_labels_are_mapped(self):
        text = "Ola jobber i DNB i Oslo."
        ner_pipe = lambda _: [
            {"entity_group": "PER", "start": 0, "end": 3},
            {"entity_group": "ORG", "start": 13, "end": 16},
            {"entity_group": "LOC", "start": 19, "end": 23},
        ]

        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)

        assert [m["type"] for m in mentions] == ["PER", "ORG", "LOC"]
        assert [m["text"] for m in mentions] == ["Ola", "DNB", "Oslo"]
        assert all(m["source"] == "ner" for m in mentions)

    def test_unsupported_labels_are_dropped(self):
        text = "Norwegian leste VG i Bergen."
        ner_pipe = lambda _: [
            {"entity_group": "DRV", "start": 0, "end": 9},
            {"entity_group": "PROD", "start": 16, "end": 18},
            {"entity_group": "GPE_LOC", "start": 21, "end": 27},
        ]

        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)

        assert len(mentions) == 1
        assert mentions[0]["type"] == "LOC"
        assert mentions[0]["text"] == "Bergen"

    def test_empty_text_returns_empty(self):
        ner_pipe = lambda _: []
        assert extract_ner_mentions("   \n\t", ner_pipe=ner_pipe, **_CHUNK_META) == []
