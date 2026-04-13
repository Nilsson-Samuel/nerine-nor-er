"""Regex supplement extraction — structured patterns for identifiers and addresses.

Augments NER model output with deterministic pattern matches for structured
identifiers that transformer models typically miss, plus Norwegian street
addresses that the model often fragments or skips entirely.  Each match
becomes a typed mention with char-level provenance, same shape as NER mentions.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Shared Norwegian address suffix vocabulary used by both regex extraction
# and NER post-processing, so the two layers stay aligned on address forms.
NORWEGIAN_ADDRESS_SUFFIXES: tuple[str, ...] = (
    "gate",
    "gata",
    "gaten",
    "vei",
    "veien",
    "allé",
    "alléen",
    "plass",
    "plassen",
    "torg",
    "torget",
    "stien",
    "sti",
    "bakken",
    "løkka",
    "svingen",
    "tunet",
    "kroken",
)
_NORWEGIAN_ADDRESS_SUFFIX_PATTERN = "|".join(
    re.escape(suffix) for suffix in NORWEGIAN_ADDRESS_SUFFIXES
)
_POSTAL_TAIL_PATTERN = r"(?:,\s\d{4}\s[A-ZÆØÅ][A-ZÆØÅa-zæøå]+)?"

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
            rf"\b[A-ZÆØÅ][A-ZÆØÅa-zæøå]*"
            rf"(?:{_NORWEGIAN_ADDRESS_SUFFIX_PATTERN})"
            rf"(?:\s\d{{1,4}}[A-Za-z]?)?"
            rf"{_POSTAL_TAIL_PATTERN}\b",
        ),
        # Multi-word: one or more capitalized words + space + street suffix
        #             + optional number + optional ", NNNN City" postal tail
        re.compile(
            rf"\b[A-ZÆØÅ][a-zæøå]+"
            rf"(?:\s[A-ZÆØÅ][a-zæøå]+)*"
            rf"\s(?:{_NORWEGIAN_ADDRESS_SUFFIX_PATTERN})"
            rf"(?:\s\d{{1,4}}[A-Za-z]?)?"
            rf"{_POSTAL_TAIL_PATTERN}\b",
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


def merge_regex_with_ner(
    regex_mentions: list[dict],
    ner_mentions: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge regex and NER mentions, letting regex supersets win over NER subspans.

    Normal rule: NER spans take priority — regex only fills gaps the model missed.
    Exception: when a regex span completely contains all overlapping NER spans of
    the same type, the regex span wins and the covered NER subspans are dropped.
    This handles cases like "Klokkersvingen 1, 1580 RYGGE" where NER returns two
    separate LOC fragments but the regex produces the correct full-address span.

    The guard is tight: regex only supersedes NER when:
    - The regex span fully contains every overlapping NER span (no partial overlaps).
    - All overlapping NER spans share the same type as the regex span.
    A regex span that partially overlaps even one NER span of a different type is
    dropped as before.

    Args:
        regex_mentions: Mentions from regex extraction.
        ner_mentions: Mentions from NER extraction.

    Returns:
        Tuple of (filtered_regex_mentions, filtered_ner_mentions). The NER list
        has any subspans fully consumed by a winning regex span removed.
    """
    if not ner_mentions:
        return regex_mentions, ner_mentions

    # Index NER mentions for efficient lookup
    ner_spans = [(m["char_start"], m["char_end"]) for m in ner_mentions]

    kept_regex: list[dict] = []
    ner_indices_to_drop: set[int] = set()

    for rm in regex_mentions:
        r_start, r_end, r_type = rm["char_start"], rm["char_end"], rm["type"]

        # Find all NER spans that overlap this regex span
        overlapping = [
            i for i, (ns, ne) in enumerate(ner_spans)
            if r_start < ne and r_end > ns
        ]

        if not overlapping:
            # No overlap — regex fills a gap, keep it
            kept_regex.append(rm)
            continue

        # Check superset condition: regex fully contains every overlapping NER span
        # and all overlapping NER spans share the same type as the regex span
        all_contained = all(
            ner_spans[i][0] >= r_start and ner_spans[i][1] <= r_end
            for i in overlapping
        )
        all_same_type = all(
            ner_mentions[i]["type"] == r_type
            for i in overlapping
        )

        if all_contained and all_same_type:
            # Regex superset wins — keep regex, drop the NER subspans
            kept_regex.append(rm)
            ner_indices_to_drop.update(overlapping)
        # else: partial overlap or type mismatch — drop the regex span (NER wins)

    filtered_ner = [m for i, m in enumerate(ner_mentions) if i not in ner_indices_to_drop]
    return kept_regex, filtered_ner


def filter_overlapping_with_ner(
    regex_mentions: list[dict],
    ner_mentions: list[dict],
) -> list[dict]:
    """Remove regex mentions that overlap with any NER mention.

    NER model spans take priority — regex only fills gaps the model missed.
    Use merge_regex_with_ner instead when regex supersets of NER fragments
    should be preferred over the individual NER subspans.

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
