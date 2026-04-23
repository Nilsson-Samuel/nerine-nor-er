"""Per-type entity normalization — canonical forms for downstream matching.

Transforms raw mention text into a normalized form so that trivially
different surface strings (case, titles, suffixes, whitespace) collapse
to the same canonical key before dedup and blocking.
"""

import re

# Norwegian honorific prefixes to strip from PER mentions.
_TITLE_PREFIXES = {"hr", "hr.", "fru", "fru.", "dr", "dr.", "herr", "frøken"}

# Corporate suffixes to strip from ORG mentions (case-insensitive match).
_ORG_SUFFIXES_RE = re.compile(
    r"\s+\b(?:as|asa|ans|sa|da)\s*$",
    re.IGNORECASE,
)

# Norwegian administrative suffixes to strip from LOC mentions.
_LOC_SUFFIXES_RE = re.compile(
    r"\s+\b(?:kommune|fylke)\s*$",
    re.IGNORECASE,
)

# Keep only digits from phone-like COMM strings.
_DIGITS_ONLY_RE = re.compile(r"[^\d]")

# Strip spaces and hyphens from financial identifiers.
_FIN_STRIP_RE = re.compile(r"[\s\-]")


def normalize_entity(text: str, entity_type: str) -> str:
    """Normalize a raw mention string based on its entity type.

    Args:
        text: Raw mention text from NER or regex extraction.
        entity_type: One of PER, ORG, LOC, ITEM, VEH, COMM, FIN.

    Returns:
        Normalized canonical form of the mention.
    """
    # Collapse whitespace universally
    t = " ".join((text or "").split())
    if not t:
        return t

    if entity_type == "PER":
        return _normalize_per(t)
    if entity_type == "ORG":
        return _normalize_org(t)
    if entity_type == "LOC":
        return _normalize_loc(t)
    if entity_type == "VEH":
        return t.upper()
    if entity_type == "COMM":
        return _normalize_comm(t)
    if entity_type == "FIN":
        return _FIN_STRIP_RE.sub("", t).upper()
    # ITEM and fallback: lowercase
    return t.lower()


def _normalize_per(text: str) -> str:
    """Strip titles, collapse whitespace, title-case."""
    tokens = text.split()
    cleaned = [tok for tok in tokens if tok.lower().rstrip(".") not in _TITLE_PREFIXES
               and tok.lower() not in _TITLE_PREFIXES]
    result = " ".join(cleaned) if cleaned else text
    return result.title()


def _normalize_org(text: str) -> str:
    """Strip corporate suffixes, title-case."""
    stripped = _ORG_SUFFIXES_RE.sub("", text).strip()
    return stripped.title() if stripped else text.title()


def _normalize_loc(text: str) -> str:
    """Strip kommune/fylke suffixes, title-case."""
    stripped = _LOC_SUFFIXES_RE.sub("", text).strip()
    return stripped.title() if stripped else text.title()


def _normalize_comm(text: str) -> str:
    """Phone numbers → digits only; emails/usernames → lowercase."""
    digits = _DIGITS_ONLY_RE.sub("", text)
    # If mostly digits, treat as phone number
    if len(digits) >= len(text) * 0.5:
        return digits
    # Otherwise lowercase (email, username, etc.)
    return text.lower()
