"""NER pipeline wrapper — runs NbAiLab/nb-bert-base-ner on chunk text.

Loads the HuggingFace NER pipeline once, runs inference per chunk, and maps
supported NorNE labels into the current extraction entity types. Today that
means only base NER support for PER, ORG, and LOC. Structured types are added
by regex supplements, while ITEM remains unsupported until a dedicated
extractor or fine-tuned model is introduced.

Post-processing steps applied after raw NER output:
1. Span boundary repair — merges adjacent same-type fragments split by the
   tokenizer (e.g. "Dor" + "cas Manning" → "Dorcas Manning").
2. Type correction heuristics — re-labels common misclassifications using
   suffix-based rules (e.g. "Essex Constabulary" LOC → ORG).
3. Short fragment filter — drops pathologically short spans that are almost
   certainly tokenizer artifacts rather than real entities.
"""

import logging
import re
from typing import Any

from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

# NorNE label → pipeline entity type. Unmapped labels are skipped.
# DRV (derived names, e.g. "Norwegian") and PROD (software, newspapers) are
# dropped — they don't map cleanly to the extraction ontology and inject noise.
# Regex supplements currently add structured PER/COMM/VEH/FIN mentions only.
# ITEM is intentionally unsupported in this stage until we have a more
# defensible extractor or a fine-tuned model.
_LABEL_MAP: dict[str, str] = {
    "PER": "PER",
    "ORG": "ORG",
    "GPE_ORG": "ORG",   # geopolitical-as-org
    "LOC": "LOC",
    "GPE_LOC": "LOC",   # geopolitical-as-loc
}

_DEFAULT_MODEL = "NbAiLab/nb-bert-base-ner"

# Singleton cache for the pipeline (heavy to load, reuse across calls).
# NOTE: only caches one model — if called with a different model name after
# the first load, the cached (first) model is returned silently.  Fine today
# since only _DEFAULT_MODEL is used; revisit if fine-tuned model swap is added.
_cached_pipeline: Any = None

# Maximum gap (in chars) between two adjacent same-type spans for boundary
# repair to merge them. Gap=0 covers direct continuations ("Dor"+"cas Manning"),
# gap=1 allows a single whitespace between fragments. Keeping this at 1 avoids
# merging spans that are genuinely separate entities joined by ", " (gap=2).
_MERGE_GAP_MAX = 1

# Minimum length (stripped) for a mention to survive the short-fragment filter.
# Per-type overrides below; this is the fallback for unlisted types.
_MIN_MENTION_LEN: dict[str, int] = {
    "PER": 2,
    "ORG": 2,
    "LOC": 2,
}
_MIN_MENTION_LEN_DEFAULT = 2

# Suffix patterns that indicate an entity is an ORG even when the NER model
# labels it as LOC. Case-insensitive match against the end of the mention text.
# NOTE: "court" and "yard" are intentionally excluded — in Norwegian criminal
# investigation documents they typically refer to residential buildings
# (e.g. "Styles Court") rather than judicial institutions.
_ORG_SUFFIX_RE = re.compile(
    r"\b(?:constabulary|police|department|agency|authority"
    r"|kommissariat|politidistrikt|tingrett|lagmannsrett)\s*$",
    re.IGNORECASE,
)

# Suffix patterns that indicate an entity is a LOC even when the NER model
# labels it as ORG. No leading \b — Norwegian compound street names fuse the
# suffix into the word (e.g. "Storgata", "Parkveien").
_LOC_SUFFIX_RE = re.compile(
    r"(?:gate|gata|gaten|vei|veien|allé|plass|plassen|torg|torget"
    r"|street|road|avenue|square|lane|drive)\s*\d*\s*$",
    re.IGNORECASE,
)


def build_ner(model: str = _DEFAULT_MODEL) -> Any:
    """Build (or return cached) HuggingFace NER pipeline.

    Args:
        model: HuggingFace model identifier.

    Returns:
        A transformers NER pipeline with aggregation_strategy="simple".
    """
    global _cached_pipeline
    if _cached_pipeline is not None:
        return _cached_pipeline

    logger.info("Loading NER model: %s", model)
    _cached_pipeline = hf_pipeline(
        "ner",
        model=model,
        aggregation_strategy="simple",
    )
    logger.info("NER model loaded.")
    return _cached_pipeline


def extract_ner_mentions(
    chunk_text: str,
    doc_id: str,
    chunk_id: str,
    page_num: int,
    source_unit_kind: str,
    ner_pipe: Any | None = None,
) -> list[dict]:
    """Run NER on a single chunk and return mapped mention dicts.

    Applies boundary repair, type correction, and short-fragment filtering
    on the raw model output before returning.

    Args:
        chunk_text: The text content of the chunk.
        doc_id: Document identifier (FK to docs.parquet).
        chunk_id: Chunk identifier (FK to chunks.parquet).
        page_num: Page/paragraph index where the chunk starts.
        source_unit_kind: "pdf_page" or "docx_paragraph".
        ner_pipe: Optional pre-built NER pipeline (uses cached singleton if None).

    Returns:
        List of mention dicts with keys: doc_id, chunk_id, text, type,
        char_start, char_end, page_num, source_unit_kind, source. Only
        PER/ORG/LOC are emitted from the base NER model in the current setup.
    """
    if not chunk_text.strip():
        return []

    if ner_pipe is None:
        ner_pipe = build_ner()

    raw_entities = ner_pipe(chunk_text)

    # Map to pipeline types, dropping unsupported labels
    mapped: list[dict] = []
    for ent in raw_entities:
        entity_type = _LABEL_MAP.get(ent["entity_group"])
        if entity_type is None:
            continue
        mapped.append({"start": ent["start"], "end": ent["end"], "type": entity_type})

    # Merge adjacent same-type fragments the tokenizer split apart
    merged = _repair_span_boundaries(mapped)
    # Second pass: merge same-type spans separated by ", " where the result
    # won't be reclassified (e.g. "Skogsstua, Hammerfest" → one LOC span)
    merged = _merge_comma_separated_spans(merged, chunk_text)

    mentions: list[dict] = []
    for span in merged:
        start = span["start"]
        end = span["end"]
        span_text = chunk_text[start:end]

        # Skip empty or whitespace-only spans
        if not span_text.strip():
            continue

        # Apply type correction heuristics
        corrected_type = _correct_entity_type(span_text, span["type"])

        mentions.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": span_text,
            "type": corrected_type,
            "char_start": start,
            "char_end": end,
            "page_num": page_num,
            "source_unit_kind": source_unit_kind,
            "source": "ner",
        })

    # Drop pathologically short fragments (tokenizer artifacts)
    return _filter_short_fragments(mentions)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _repair_span_boundaries(spans: list[dict]) -> list[dict]:
    """Merge adjacent same-type spans separated by a small gap.

    The NER model sometimes splits a single entity across adjacent tokens,
    producing fragments like ("Dor", PER) + ("cas Manning", PER). When two
    consecutive spans share the same type and the gap between them is at most
    _MERGE_GAP_MAX characters, they are merged into one span.

    Args:
        spans: Sorted list of {"start", "end", "type"} dicts from raw NER.

    Returns:
        Merged span list (same dict shape, potentially fewer entries).
    """
    if not spans:
        return []

    merged: list[dict] = [spans[0].copy()]

    for span in spans[1:]:
        prev = merged[-1]
        gap = span["start"] - prev["end"]

        if span["type"] == prev["type"] and 0 <= gap <= _MERGE_GAP_MAX:
            # Extend previous span to cover the new one
            prev["end"] = span["end"]
        else:
            merged.append(span.copy())

    return merged


def _merge_comma_separated_spans(spans: list[dict], chunk_text: str) -> list[dict]:
    """Merge adjacent same-type spans separated by exactly ', ' into one span.

    Handles compound location forms like "Skogsstua, Hammerfest" where the NER
    model returns two adjacent LOC spans with a comma-space gap (2 chars) that
    _repair_span_boundaries intentionally leaves unmerged to avoid incorrect
    cross-type merges like "St Mary stasjon, Essex Constabulary".

    The guard check ensures the merged text would not trigger ORG type
    correction — so "St Mary stasjon, Essex Constabulary" is never merged here
    (the constabulary suffix would fire), but "Skogsstua, Hammerfest" is.

    Args:
        spans: Span list after _repair_span_boundaries.
        chunk_text: The original chunk text (needed to read gap characters).

    Returns:
        Span list with eligible comma-separated pairs merged.
    """
    if not spans:
        return []

    result: list[dict] = [spans[0].copy()]

    for span in spans[1:]:
        prev = result[-1]
        gap = span["start"] - prev["end"]

        if (
            gap == 2
            and span["type"] == prev["type"]
            and chunk_text[prev["end"]: span["start"]] == ", "
            and not _ORG_SUFFIX_RE.search(chunk_text[prev["start"]: span["end"]])
        ):
            prev["end"] = span["end"]
        else:
            result.append(span.copy())

    return result


def _correct_entity_type(text: str, current_type: str) -> str:
    """Apply heuristic type corrections for common NER misclassifications.

    Rules applied:
    - LOC with org-indicating suffix (Constabulary, Court, etc.) → ORG
    - ORG with street/address suffix (gate, vei, etc.) → LOC

    Args:
        text: The raw mention text.
        current_type: The entity type assigned by the NER model.

    Returns:
        Corrected entity type string.
    """
    if current_type == "LOC" and _ORG_SUFFIX_RE.search(text):
        return "ORG"
    if current_type == "ORG" and _LOC_SUFFIX_RE.search(text):
        return "LOC"
    return current_type


def _filter_short_fragments(mentions: list[dict]) -> list[dict]:
    """Remove implausible mentions by length and ORG-specific capitalization rules.

    Two checks are applied:
    1. Length filter — drops spans shorter than a per-type minimum, removing
       tokenizer artifacts like "Dor", "Scot", "Sty".
    2. Single-token ORG capitalization filter — drops single-word ORG mentions
       whose first character is not uppercase. Legitimate single-token ORGs are
       always proper nouns ("DNB", "Kripos") or uppercase abbreviations.
       Lowercase single-token spans ("det", "gruppe") and job-title abbreviations
       ("Kb") are model noise from fragmented compound words.

    Args:
        mentions: Post-processed mention dicts.

    Returns:
        Filtered list with implausible mentions removed.
    """
    kept: list[dict] = []
    dropped = 0

    for m in mentions:
        text = m["text"].strip()

        # Length check
        min_len = _MIN_MENTION_LEN.get(m["type"], _MIN_MENTION_LEN_DEFAULT)
        if len(text) < min_len:
            dropped += 1
            continue

        # Single-token ORG must start with an uppercase letter
        if m["type"] == "ORG" and " " not in text and not text[0].isupper():
            dropped += 1
            continue

        kept.append(m)

    if dropped:
        logger.debug("Fragment filter dropped %d mentions", dropped)

    return kept
