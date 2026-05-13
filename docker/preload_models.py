"""Pre-download HuggingFace models into the image during `docker build`.

Bakes the two models referenced by the pipeline into HF_HOME so the runtime
container has no network dependency on first use:

  - NbAiLab/nb-bert-base-ner   (src/extraction/ner.py)
  - NbAiLab/nb-sbert-base      (src/blocking/embeddings.py)
"""

from __future__ import annotations

import os
import sys

NER_MODEL = "NbAiLab/nb-bert-base-ner"
SBERT_MODEL = "NbAiLab/nb-sbert-base"


def main() -> int:
    cache_dir = os.environ.get("HF_HOME", "/opt/hf-cache")
    print(f"[preload] HF_HOME={cache_dir}", flush=True)

    print(f"[preload] downloading NER model: {NER_MODEL}", flush=True)
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(NER_MODEL)
    AutoModelForTokenClassification.from_pretrained(NER_MODEL)

    print(f"[preload] downloading SBERT model: {SBERT_MODEL}", flush=True)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(SBERT_MODEL)

    print("[preload] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
