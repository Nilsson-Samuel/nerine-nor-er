"""Context window extraction — ±50 char slices around entity mentions.

Extracts a surrounding text window from the source chunk for each entity's
primary mention.  Used for HITL display and context embedding in blocking.
"""


def extract_context(
    chunk_text: str, start: int, end: int, window: int = 50,
) -> str:
    """Extract a context window around a mention span.

    Args:
        chunk_text: Full text of the source chunk.
        start: Char offset where the mention starts (inclusive).
        end: Char offset where the mention ends (exclusive).
        window: Number of chars to include on each side.

    Returns:
        Substring of chunk_text centered on [start, end) with ±window padding.
    """
    left = max(0, start - window)
    right = min(len(chunk_text), end + window)
    return chunk_text[left:right]
