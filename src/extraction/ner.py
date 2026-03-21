"""NER pipeline wrapper — runs NbAiLab/nb-bert-base-ner on chunk text.

Loads the HuggingFace NER pipeline once, runs inference per chunk, and maps
supported NorNE labels into the current extraction entity types. Today that
means only base NER support for PER, ORG, and LOC. Structured types are added
by regex supplements, while ITEM remains unsupported until a dedicated
extractor or fine-tuned model is introduced.
"""

import logging
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
    mentions: list[dict] = []

    for ent in raw_entities:
        entity_type = _LABEL_MAP.get(ent["entity_group"])
        if entity_type is None:
            continue

        start = ent["start"]
        end = ent["end"]
        span_text = chunk_text[start:end]

        # Skip empty or whitespace-only spans
        if not span_text.strip():
            continue

        mentions.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": span_text,
            "type": entity_type,
            "char_start": start,
            "char_end": end,
            "page_num": page_num,
            "source_unit_kind": source_unit_kind,
            "source": "ner",
        })

    return mentions
