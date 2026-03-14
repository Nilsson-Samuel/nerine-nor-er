"""Text normalization — deterministic cleanup for extracted text units.

Pipeline: ftfy fix → NFC normalization → control-char removal → whitespace collapse.
Preserves Norwegian characters (æøå) and repairs common mojibake (Ã¸ → ø, etc.).
"""

import re
import unicodedata

import ftfy

# Control chars except newline (\x0a) — remove tabs, carriage returns, etc.
_CTRL_EXCEPT_NL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")

# Collapse horizontal whitespace (spaces, tabs, etc.) but not newlines
_MULTISPACE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(text: str) -> str:
    """Normalize a raw text unit for downstream NER and chunking.

    Args:
        text: Raw extracted text (may contain mojibake, control chars, etc.).

    Returns:
        Cleaned text with stable Unicode (NFC), no control chars, collapsed whitespace.
    """
    text = ftfy.fix_text(text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\t", " ")  # Preserve word boundaries before ctrl strip
    text = _CTRL_EXCEPT_NL.sub("", text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()
