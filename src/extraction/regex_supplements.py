"""Regex supplement extraction — structured patterns for identifiers and addresses.

Augments NER model output with deterministic pattern matches for structured
identifiers that transformer models typically miss, plus Norwegian street
addresses that the model often fragments or skips entirely.  Each match
becomes a typed mention with char-level provenance, same shape as NER mentions.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Compiled patterns grouped by target entity type.
# Order within each type list matters: first match wins for a given span.
PATTERNS: dict[str, list[re.Pattern[str]]] = {
    # Norwegian phone: 8 digits (compact or spaced 3-2-3)
    # International phone: + country code (1–3 digits) + 7–12 digits, optional spaces/dashes.
    #   Anchored on left by \b so we don't eat into longer numbers.
    #   The Norwegian domestic patterns must come first so they win on overlap.
    # Email: standard RFC-ish local@domain.tld — covers handles like arh_hast@gmail.com.
    # Username/handle: word with at least one underscore or digit run, e.g. alfred_1212.
    #   Requires ≥3 chars total and at least one letter to avoid matching bare numbers.
    #   Must not look like a plain name (heuristic: contains _ or digit-then-letter).
    "COMM": [
        re.compile(r"\b\d{3}\s\d{2}\s\d{3}\b"),
        re.compile(r"\b\d{8}\b"),
        # International phone: + followed by 7–15 digits (compact or spaced/dashed).
        # The compact variant handles numbers like +66878767665 with no separators.
        # The segmented variant handles +66 878 767 665 or +1-800-555-1234.
        re.compile(r"(?<!\d)\+\d{7,15}\b"),
        re.compile(r"(?<!\d)\+\d{1,3}[\s\-]\d{2,5}(?:[\s\-]\d{2,5}){1,4}\b"),
        re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
        re.compile(r"\b(?=[a-zA-Z0-9_]*[a-zA-Z])(?=[a-zA-Z0-9_]*_)[a-zA-Z0-9_]{3,}\b"),
    ],
    # Fødselsnummer: 6 digits (DOB) + optional separator + 5 digits
    "PER": [
        re.compile(r"\b\d{6}[-\s]?\d{5}\b"),
    ],
    # Norwegian license plate: 2 uppercase letters + optional space + 4-5 digits
    "VEH": [
        re.compile(r"\b[A-Z]{2}\s?\d{4,5}\b"),
    ],
    # Norwegian bank account: 4.2.5 or 4.2.4 digits with dot separators
    # This is the most common format investigators use (e.g. "1234.56.78901").
    # Norwegian IBAN: NO + 2 check digits + 11 digits (spaced or compact).
    "FIN": [
        re.compile(r"\b\d{4}\.\d{2}\.\d{4,5}\b"),
        re.compile(r"\bNO\d{2}\s?\d{4}\s?\d{4}\s?\d{3}\b"),
    ],
    # Norwegian street addresses — two sub-patterns, both optionally followed
    # by a Norwegian postal code + city: ", NNNN <City>" (e.g. ", 1580 Rygge").
    # This covers full address forms like "Klokkersvingen 1, 1580 Rygge" and
    # "Fridtjof Nansens vei 14, 0369 Oslo" that investigators commonly write.
    #
    # 1) Compound names where the suffix is fused into the word:
    #    "Storgata 12", "Parkveien 31", "Rådhusplassen"
    # 2) Multi-word names with a standalone suffix word:
    #    "Karl Johans gate 5", "Kongens gate 10"
    "LOC": [
        # Compound: capitalized word ending in a street suffix + optional number
        #           + optional ", NNNN City" postal tail
        re.compile(
            r"\b[A-ZÆØÅ][a-zæøå]*"
            r"(?:gata|gaten|gate|veien|vei|allé[en]?"
            r"|plassen|plass|torget|torg|stien|sti"
            r"|bakken|løkka|svingen|tunet|kroken)"
            r"(?:\s\d{1,4}[A-Za-z]?)?"
            r"(?:,\s\d{4}\s[A-ZÆØÅ][A-ZÆØÅa-zæøå]+)?\b",
        ),
        # Multi-word: one or more capitalized words + space + street suffix
        #             + optional number + optional ", NNNN City" postal tail
        re.compile(
            r"\b[A-ZÆØÅ][a-zæøå]+"
            r"(?:\s[A-ZÆØÅ][a-zæøå]+)*"
            r"\s(?:gate|gata|gaten|vei|veien|allé"
            r"|plass|plassen|torg|torget|stien|sti"
            r"|bakken|løkka|svingen|tunet|kroken)"
            r"(?:\s\d{1,4}[A-Za-z]?)?"
            r"(?:,\s\d{4}\s[A-ZÆØÅ][A-ZÆØÅa-zæøå]+)?\b",
        ),
    ],
}


def extract_regex_mentions(
    chunk_text: str,
    doc_id: str,
    chunk_id: str,
    page_num: int,
    source_unit_kind: str,
) -> list[dict]:
    """Run all regex patterns on chunk text and return non-overlapping mentions.

    Args:
        chunk_text: The text content of the chunk.
        doc_id: Document identifier.
        chunk_id: Chunk identifier.
        page_num: Page/paragraph index.
        source_unit_kind: "pdf_page" or "docx_paragraph".

    Returns:
        List of mention dicts (same shape as NER mentions, source="regex").
    """
    raw_hits: list[dict] = []

    for entity_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(chunk_text):
                raw_hits.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": match.group(),
                    "type": entity_type,
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "page_num": page_num,
                    "source_unit_kind": source_unit_kind,
                    "source": "regex",
                })

    # Sort by position for deterministic output, then remove overlaps
    raw_hits.sort(key=lambda m: (m["char_start"], -m["char_end"]))
    return _remove_overlaps(raw_hits)


def filter_overlapping_with_ner(
    regex_mentions: list[dict],
    ner_mentions: list[dict],
) -> list[dict]:
    """Remove regex mentions that overlap with any NER mention.

    NER model spans take priority — regex only fills gaps the model missed.

    Args:
        regex_mentions: Mentions from regex extraction.
        ner_mentions: Mentions from NER extraction.

    Returns:
        Filtered regex mentions with no NER overlap.
    """
    if not ner_mentions:
        return regex_mentions

    # Build set of occupied char ranges from NER
    ner_spans = [(m["char_start"], m["char_end"]) for m in ner_mentions]

    return [
        m for m in regex_mentions
        if not _overlaps_any(m["char_start"], m["char_end"], ner_spans)
    ]


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """Check if [start, end) overlaps with any span in the list."""
    return any(start < s_end and end > s_start for s_start, s_end in spans)


def _remove_overlaps(mentions: list[dict]) -> list[dict]:
    """Remove overlapping mentions via greedy first-match.

    Input must be sorted by (char_start, -char_end).  At each position the
    first (i.e. earliest-starting, longest) non-overlapping mention wins.
    """
    result: list[dict] = []
    last_end = -1

    for m in mentions:
        if m["char_start"] >= last_end:
            result.append(m)
            last_end = m["char_end"]

    return result
