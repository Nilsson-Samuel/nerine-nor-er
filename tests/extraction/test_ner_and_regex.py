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

from src.extraction.ner import (
    extract_ner_mentions,
    _repair_span_boundaries,
    _merge_comma_separated_spans,
    _correct_entity_type,
    _filter_short_fragments,
)
from src.extraction.regex_supplements import (
    extract_regex_mentions,
    filter_overlapping_with_ner,
    merge_regex_with_ner,
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


class TestRegexInternationalPhone:
    """International phone number patterns (COMM)."""

    def test_compact_international(self):
        text = "Kontakt: +66878767665 ble ringt."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 1
        assert comm[0]["text"] == "+66878767665"
        assert text[comm[0]["char_start"]:comm[0]["char_end"]] == comm[0]["text"]

    def test_segmented_international(self):
        text = "Ring +1 800 555 1234 for hjelp."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 1
        assert comm[0]["text"] == "+1 800 555 1234"

    def test_not_matched_without_plus(self):
        # A 11-digit number without + should not match international pattern
        text = "Nummer 66878767665 ble brukt."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        # May match domestic 8-digit sub-pattern but not the international one
        comm = [m for m in mentions if m["type"] == "COMM" and m["text"].startswith("+")]
        assert len(comm) == 0


class TestRegexEmail:
    """Email address patterns (COMM)."""

    def test_simple_email(self):
        text = "Send til arh_hast@gmail.com for mer info."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 1
        assert comm[0]["text"] == "arh_hast@gmail.com"
        assert text[comm[0]["char_start"]:comm[0]["char_end"]] == comm[0]["text"]

    def test_email_with_dots_in_local(self):
        text = "Epost: per.hansen@example.no er brukt."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 1
        assert comm[0]["text"] == "per.hansen@example.no"

    def test_no_false_positive_on_plain_text(self):
        text = "Ingen epost her."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 0


class TestRegexUsername:
    """Underscore-username patterns (COMM)."""

    def test_alphanumeric_underscore(self):
        text = "Brukernavn: alfred_1212 er registrert."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 1
        assert comm[0]["text"] == "alfred_1212"
        assert text[comm[0]["char_start"]:comm[0]["char_end"]] == comm[0]["text"]

    def test_plain_name_not_matched(self):
        # "Alfred" has no underscore — should not be a username
        text = "Alfred møtte Kari i dag."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 0

    def test_version_string_not_matched(self):
        # "3_2" has no letters — the pattern requires at least one letter
        text = "Versjon 3_2 av programmet."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        comm = [m for m in mentions if m["type"] == "COMM"]
        assert len(comm) == 0


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


# ---------------------------------------------------------------------------
# Span boundary repair tests
# ---------------------------------------------------------------------------

class TestSpanBoundaryRepair:
    """Adjacent same-type fragments are merged into one span."""

    def test_adjacent_same_type_merged(self):
        # "Dor" + "cas Manning" should merge into "Dorcas Manning"
        spans = [
            {"start": 0, "end": 3, "type": "PER"},
            {"start": 3, "end": 14, "type": "PER"},
        ]
        result = _repair_span_boundaries(spans)
        assert len(result) == 1
        assert result[0] == {"start": 0, "end": 14, "type": "PER"}

    def test_small_gap_same_type_merged(self):
        # "Scot" + " " + "land Yard" — gap of 1 char
        spans = [
            {"start": 0, "end": 4, "type": "LOC"},
            {"start": 5, "end": 14, "type": "LOC"},
        ]
        result = _repair_span_boundaries(spans)
        assert len(result) == 1
        assert result[0] == {"start": 0, "end": 14, "type": "LOC"}

    def test_different_types_not_merged(self):
        spans = [
            {"start": 0, "end": 5, "type": "PER"},
            {"start": 5, "end": 10, "type": "ORG"},
        ]
        result = _repair_span_boundaries(spans)
        assert len(result) == 2

    def test_comma_space_gap_not_merged(self):
        # Gap of 2 chars (", ") — two entities joined by comma should stay separate
        # e.g. "St Mary stasjon" + ", " + "Essex Constabulary"
        spans = [
            {"start": 0, "end": 15, "type": "ORG"},
            {"start": 17, "end": 35, "type": "ORG"},
        ]
        result = _repair_span_boundaries(spans)
        assert len(result) == 2

    def test_large_gap_not_merged(self):
        # Gap of 10 chars — too far apart to be a split token
        spans = [
            {"start": 0, "end": 5, "type": "PER"},
            {"start": 15, "end": 20, "type": "PER"},
        ]
        result = _repair_span_boundaries(spans)
        assert len(result) == 2

    def test_three_fragments_merged(self):
        # "Sty" + "les" + " Court 12" — three adjacent PER fragments
        spans = [
            {"start": 0, "end": 3, "type": "LOC"},
            {"start": 3, "end": 6, "type": "LOC"},
            {"start": 7, "end": 15, "type": "LOC"},
        ]
        result = _repair_span_boundaries(spans)
        assert len(result) == 1
        assert result[0] == {"start": 0, "end": 15, "type": "LOC"}

    def test_empty_input(self):
        assert _repair_span_boundaries([]) == []

    def test_single_span_unchanged(self):
        spans = [{"start": 0, "end": 5, "type": "PER"}]
        result = _repair_span_boundaries(spans)
        assert result == spans

    def test_does_not_mutate_input(self):
        spans = [
            {"start": 0, "end": 3, "type": "PER"},
            {"start": 3, "end": 10, "type": "PER"},
        ]
        original = [s.copy() for s in spans]
        _repair_span_boundaries(spans)
        assert spans == original


# ---------------------------------------------------------------------------
# Type correction heuristic tests
# ---------------------------------------------------------------------------

class TestTypeCorrection:
    """Suffix-based heuristics fix common NER type misclassifications."""

    def test_constabulary_loc_to_org(self):
        assert _correct_entity_type("Essex Constabulary", "LOC") == "ORG"

    def test_police_loc_to_org(self):
        assert _correct_entity_type("Oslo Police", "LOC") == "ORG"

    def test_court_stays_loc(self):
        # "Court" in this domain typically means a building, not a judicial institution
        assert _correct_entity_type("Styles Court", "LOC") == "LOC"

    def test_yard_stays_loc(self):
        # "Yard" in this domain typically means a building/location
        assert _correct_entity_type("Scotland Yard", "LOC") == "LOC"

    def test_tingrett_loc_to_org(self):
        assert _correct_entity_type("Oslo tingrett", "LOC") == "ORG"

    def test_politidistrikt_loc_to_org(self):
        assert _correct_entity_type("Øst politidistrikt", "LOC") == "ORG"

    def test_street_org_to_loc(self):
        assert _correct_entity_type("Storgata 12", "ORG") == "LOC"

    def test_vei_org_to_loc(self):
        assert _correct_entity_type("Parkveien 31", "ORG") == "LOC"

    def test_per_type_unchanged(self):
        # Type correction only applies to LOC↔ORG, not PER
        assert _correct_entity_type("Essex Constabulary", "PER") == "PER"

    def test_plain_loc_unchanged(self):
        assert _correct_entity_type("Oslo", "LOC") == "LOC"

    def test_plain_org_unchanged(self):
        assert _correct_entity_type("DNB", "ORG") == "ORG"

    def test_integration_with_extract(self):
        # "Essex Constabulary" labeled LOC by model should become ORG after post-processing
        text = "Kontakt Essex Constabulary for detaljer."
        ner_pipe = lambda _: [
            {"entity_group": "LOC", "start": 8, "end": 26},
        ]
        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)
        assert len(mentions) == 1
        assert mentions[0]["type"] == "ORG"
        assert mentions[0]["text"] == "Essex Constabulary"


# ---------------------------------------------------------------------------
# Short fragment filter tests
# ---------------------------------------------------------------------------

class TestShortFragmentFilter:
    """Pathologically short fragments are dropped."""

    def test_single_char_dropped(self):
        mentions = [{"text": "D", "type": "PER", "source": "ner"}]
        assert _filter_short_fragments(mentions) == []

    def test_two_char_kept(self):
        mentions = [{"text": "Li", "type": "PER", "source": "ner"}]
        assert len(_filter_short_fragments(mentions)) == 1

    def test_normal_mention_kept(self):
        mentions = [{"text": "Kari Nordmann", "type": "PER", "source": "ner"}]
        assert len(_filter_short_fragments(mentions)) == 1

    def test_mixed_short_and_normal(self):
        mentions = [
            {"text": "D", "type": "PER", "source": "ner"},
            {"text": "Kari", "type": "PER", "source": "ner"},
            {"text": "S", "type": "LOC", "source": "ner"},
            {"text": "Oslo", "type": "LOC", "source": "ner"},
        ]
        result = _filter_short_fragments(mentions)
        assert len(result) == 2
        assert [m["text"] for m in result] == ["Kari", "Oslo"]

    def test_whitespace_only_text_dropped(self):
        mentions = [{"text": " ", "type": "PER", "source": "ner"}]
        assert _filter_short_fragments(mentions) == []

    def test_empty_list(self):
        assert _filter_short_fragments([]) == []

    def test_lowercase_single_token_org_dropped(self):
        # "det", "gruppe" — lowercase single-word ORG spans are model noise
        for word in ("det", "gruppe", "kriminal"):
            mentions = [{"text": word, "type": "ORG", "source": "ner"}]
            assert _filter_short_fragments(mentions) == [], f"Expected {word!r} to be dropped"

    def test_uppercase_abbreviation_org_kept(self):
        # "DNB", "FBI" — uppercase abbreviations are legitimate single-token ORGs
        for word in ("DNB", "FBI", "Kripos"):
            mentions = [{"text": word, "type": "ORG", "source": "ner"}]
            assert len(_filter_short_fragments(mentions)) == 1, f"Expected {word!r} to be kept"

    def test_multi_token_org_kept_regardless_of_case(self):
        # Multi-word ORGs are never dropped by the capitalization rule
        mentions = [{"text": "kriminalteknisk gruppe", "type": "ORG", "source": "ner"}]
        assert len(_filter_short_fragments(mentions)) == 1

    def test_lowercase_single_token_per_and_loc_unaffected(self):
        # Capitalization rule applies only to ORG, not other types
        mentions = [
            {"text": "oslo", "type": "LOC", "source": "ner"},
            {"text": "kb", "type": "PER", "source": "ner"},  # too short, but by length rule
        ]
        result = _filter_short_fragments(mentions)
        assert any(m["text"] == "oslo" for m in result)


# ---------------------------------------------------------------------------
# Norwegian address regex tests
# ---------------------------------------------------------------------------

class TestRegexNorwegianAddresses:
    """Norwegian street address patterns (LOC)."""

    def test_storgata_with_number(self):
        text = "Adressen er Storgata 12 i sentrum."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Storgata 12"
        assert text[loc[0]["char_start"]:loc[0]["char_end"]] == loc[0]["text"]

    def test_multi_word_street(self):
        text = "Karl Johans gate 5 ligger sentralt."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Karl Johans gate 5"

    def test_veien_suffix(self):
        text = "Han bor i Parkveien 31."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Parkveien 31"

    def test_plass_suffix_without_number(self):
        text = "Møtet var på Rådhusplassen i Oslo."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Rådhusplassen"

    def test_allé_suffix(self):
        text = "Konferanse på Drammensveien allé 4."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) >= 1

    def test_house_number_with_letter(self):
        text = "Besøk oss på Storgata 12B for mer."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert "12B" in loc[0]["text"]

    def test_no_match_for_plain_text(self):
        text = "Ingen adresser her."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 0

    def test_provenance_valid(self):
        text = "Adressen er Storgata 12 i sentrum."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        m = loc[0]
        assert text[m["char_start"]:m["char_end"]] == m["text"]
        assert m["source"] == "regex"

    def test_compound_with_postal_code(self):
        text = "Adressen er Klokkersvingen 1, 1580 Rygge i kommunen."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Klokkersvingen 1, 1580 Rygge"
        assert text[loc[0]["char_start"]:loc[0]["char_end"]] == loc[0]["text"]

    def test_multiword_with_postal_code(self):
        text = "Bodde på Fridtjof Nansens vei 14, 0369 Oslo."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Fridtjof Nansens vei 14, 0369 Oslo"

    def test_address_without_postal_code_still_works(self):
        # Postal tail is optional — existing behaviour must not regress
        text = "Bodde på Storgata 12 i sentrum."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Storgata 12"


# ---------------------------------------------------------------------------
# End-to-end boundary repair integration test
# ---------------------------------------------------------------------------

class TestBoundaryRepairIntegration:
    """Boundary repair works correctly through extract_ner_mentions."""

    def test_fragmented_name_merged(self):
        # Model splits "Dorcas Manning" into "Dor" + "cas Manning"
        text = "Vitnet Dorcas Manning forklarte seg."
        ner_pipe = lambda _: [
            {"entity_group": "PER", "start": 7, "end": 10},   # "Dor"
            {"entity_group": "PER", "start": 10, "end": 21},  # "cas Manning"
        ]
        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)
        assert len(mentions) == 1
        assert mentions[0]["text"] == "Dorcas Manning"
        assert mentions[0]["char_start"] == 7
        assert mentions[0]["char_end"] == 21

    def test_fragmented_loc_merged(self):
        # "Scotland Yard" split into "Scot" + "land Yard"
        text = "Informasjon fra Scotland Yard bekreftet saken."
        ner_pipe = lambda _: [
            {"entity_group": "LOC", "start": 16, "end": 20},  # "Scot"
            {"entity_group": "LOC", "start": 20, "end": 29},  # "land Yard"
        ]
        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)
        assert len(mentions) == 1
        # "Yard" is kept as LOC — in this domain it means a location, not an org
        assert mentions[0]["type"] == "LOC"
        assert mentions[0]["text"] == "Scotland Yard"

    def test_mixed_types_not_merged(self):
        # PER span followed immediately by ORG span — should stay separate
        text = "Kari jobber i DNB bank."
        ner_pipe = lambda _: [
            {"entity_group": "PER", "start": 0, "end": 4},
            {"entity_group": "ORG", "start": 14, "end": 17},
        ]
        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)
        assert len(mentions) == 2
        assert mentions[0]["type"] == "PER"
        assert mentions[1]["type"] == "ORG"


# ---------------------------------------------------------------------------
# Comma-separated span merge tests
# ---------------------------------------------------------------------------

class TestCommaSpanMerge:
    """_merge_comma_separated_spans handles 'Place, City' compound LOC forms."""

    def test_loc_comma_city_merged(self):
        # "Skogsstua, Hammerfest" — two adjacent LOC spans with ", " between
        text = "Han bodde på Skogsstua, Hammerfest."
        spans = [
            {"start": 13, "end": 22, "type": "LOC"},  # "Skogsstua"
            {"start": 24, "end": 34, "type": "LOC"},  # "Hammerfest"
        ]
        result = _merge_comma_separated_spans(spans, text)
        assert len(result) == 1
        assert result[0] == {"start": 13, "end": 34, "type": "LOC"}

    def test_org_suffix_in_merged_span_prevents_merge(self):
        # "St Mary stasjon, Essex Constabulary" — constabulary suffix blocks merge
        text = "St Mary stasjon, Essex Constabulary mottok meldingen."
        spans = [
            {"start": 0, "end": 15, "type": "LOC"},   # "St Mary stasjon"
            {"start": 17, "end": 35, "type": "LOC"},  # "Essex Constabulary"
        ]
        result = _merge_comma_separated_spans(spans, text)
        assert len(result) == 2

    def test_different_types_not_merged(self):
        text = "Oslo, DNB-gruppen er involvert."
        spans = [
            {"start": 0, "end": 4, "type": "LOC"},
            {"start": 6, "end": 15, "type": "ORG"},
        ]
        result = _merge_comma_separated_spans(spans, text)
        assert len(result) == 2

    def test_non_comma_separator_not_merged(self):
        # Gap of 2 chars but not ", " — should not merge
        text = "Oslo  Bergen er to byer."
        spans = [
            {"start": 0, "end": 4, "type": "LOC"},
            {"start": 6, "end": 12, "type": "LOC"},
        ]
        result = _merge_comma_separated_spans(spans, text)
        assert len(result) == 2

    def test_empty_input(self):
        assert _merge_comma_separated_spans([], "any text") == []

    def test_single_span_unchanged(self):
        text = "Hammerfest er en by."
        spans = [{"start": 0, "end": 10, "type": "LOC"}]
        result = _merge_comma_separated_spans(spans, text)
        assert result == spans


class TestCommaSpanMergeIntegration:
    """_merge_comma_separated_spans integrates correctly with extract_ner_mentions."""

    def test_skogsstua_hammerfest_merged(self):
        text = "Han bodde på Skogsstua, Hammerfest om sommeren."
        ner_pipe = lambda _: [
            {"entity_group": "LOC", "start": 13, "end": 22},  # "Skogsstua"
            {"entity_group": "LOC", "start": 24, "end": 34},  # "Hammerfest"
        ]
        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)
        assert len(mentions) == 1
        assert mentions[0]["text"] == "Skogsstua, Hammerfest"
        assert mentions[0]["type"] == "LOC"

    def test_st_mary_constabulary_not_merged(self):
        # After comma-merge the merged text has constabulary suffix → not merged
        text = "St Mary stasjon, Essex Constabulary mottok meldingen."
        ner_pipe = lambda _: [
            {"entity_group": "LOC", "start": 0, "end": 15},   # "St Mary stasjon"
            {"entity_group": "LOC", "start": 17, "end": 35},  # "Essex Constabulary"
        ]
        mentions = extract_ner_mentions(text, ner_pipe=ner_pipe, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        org = [m for m in mentions if m["type"] == "ORG"]
        assert len(loc) == 1
        assert loc[0]["text"] == "St Mary stasjon"
        assert len(org) == 1
        assert org[0]["text"] == "Essex Constabulary"


# ---------------------------------------------------------------------------
# Postal tail all-caps city tests
# ---------------------------------------------------------------------------

class TestRegexPostalTailAllCaps:
    """Postal tail regex matches all-caps city names (e.g. RYGGE, OSLO)."""

    def test_compound_with_allcaps_postal_city(self):
        text = "Adressen er Klokkersvingen 1, 1580 RYGGE i kommunen."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Klokkersvingen 1, 1580 RYGGE"
        assert text[loc[0]["char_start"]:loc[0]["char_end"]] == loc[0]["text"]

    def test_multiword_with_allcaps_postal_city(self):
        text = "Bodde på Fridtjof Nansens vei 14, 0369 OSLO."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Fridtjof Nansens vei 14, 0369 OSLO"

    def test_title_case_still_works(self):
        # Original Title-case behaviour must not regress
        text = "Bodde på Fridtjof Nansens vei 14, 0369 Oslo."
        mentions = extract_regex_mentions(text, **_CHUNK_META)
        loc = [m for m in mentions if m["type"] == "LOC"]
        assert len(loc) == 1
        assert loc[0]["text"] == "Fridtjof Nansens vei 14, 0369 Oslo"


# ---------------------------------------------------------------------------
# merge_regex_with_ner tests
# ---------------------------------------------------------------------------

class TestMergeRegexWithNer:
    """Regex supersets of same-type NER fragments replace the fragments."""

    def _make(self, start, end, etype, source="ner"):
        return {
            "doc_id": "a" * 32, "chunk_id": "b" * 32, "page_num": 0,
            "source_unit_kind": "pdf_page", "source": source,
            "text": "x", "char_start": start, "char_end": end, "type": etype,
        }

    def test_regex_superset_wins_drops_ner_subspans(self):
        # Regex [0,28) contains NER [0,15) and NER [17,28), both LOC
        ner = [self._make(0, 15, "LOC"), self._make(17, 28, "LOC")]
        regex = [self._make(0, 28, "LOC", "regex")]
        kept_regex, kept_ner = merge_regex_with_ner(regex, ner)
        assert len(kept_regex) == 1
        assert kept_regex[0]["char_start"] == 0
        assert kept_regex[0]["char_end"] == 28
        assert kept_ner == []

    def test_gap_filler_kept_no_overlap(self):
        # Regex span in a gap between NER spans — kept, NER unchanged
        ner = [self._make(0, 5, "PER"), self._make(20, 30, "PER")]
        regex = [self._make(8, 15, "COMM", "regex")]
        kept_regex, kept_ner = merge_regex_with_ner(regex, ner)
        assert len(kept_regex) == 1
        assert len(kept_ner) == 2

    def test_partial_overlap_drops_regex(self):
        # Regex [0,20) only partially contains NER [10,25) — regex dropped
        ner = [self._make(10, 25, "LOC")]
        regex = [self._make(0, 20, "LOC", "regex")]
        kept_regex, kept_ner = merge_regex_with_ner(regex, ner)
        assert kept_regex == []
        assert len(kept_ner) == 1

    def test_type_mismatch_drops_regex(self):
        # Regex LOC fully contains NER ORG — type mismatch, regex dropped
        ner = [self._make(0, 15, "ORG")]
        regex = [self._make(0, 20, "LOC", "regex")]
        kept_regex, kept_ner = merge_regex_with_ner(regex, ner)
        assert kept_regex == []
        assert len(kept_ner) == 1

    def test_empty_ner_returns_all_regex(self):
        regex = [self._make(0, 10, "LOC", "regex")]
        kept_regex, kept_ner = merge_regex_with_ner(regex, [])
        assert len(kept_regex) == 1
        assert kept_ner == []

    def test_only_overlapping_subspans_dropped_others_kept(self):
        # NER has 3 spans; regex covers only the first two
        ner = [self._make(0, 10, "LOC"), self._make(12, 20, "LOC"), self._make(30, 40, "PER")]
        regex = [self._make(0, 20, "LOC", "regex")]
        kept_regex, kept_ner = merge_regex_with_ner(regex, ner)
        assert len(kept_regex) == 1
        assert len(kept_ner) == 1
        assert kept_ner[0]["char_start"] == 30
