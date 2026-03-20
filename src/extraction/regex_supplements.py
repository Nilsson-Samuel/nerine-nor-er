"""Regex supplement extraction — structured patterns for phone, fnr, plates, IBAN.

Augments NER model output with deterministic pattern matches for structured
identifiers that transformer models typically miss.  Each match becomes a
typed mention with char-level provenance, same shape as NER mentions.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Compiled patterns grouped by target entity type.
# Order within each type list matters: first match wins for a given span.
PATTERNS: dict[str, list[re.Pattern[str]]] = {
    # Norwegian phone: 8 digits (compact or spaced 3-2-3)
    "COMM": [
        re.compile(r"\b\d{3}\s\d{2}\s\d{3}\b"),
        re.compile(r"\b\d{8}\b"),
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
