# NERINE - Entity Resolution in Norwegian criminal investigations

Resolves person, organization, and other entity mentions across large collections of Norwegian investigative documents. Combines high-recall NER and blocking with ML-based matching, designed for offline deployment in law-enforcement contexts where explainability and cautious resolution matter.

<p align="left">
  <img src="assets/Nerine-logo.png" alt="NERINE logo" width="180"/>
</p>

## Problem

Investigative document collections can be large, with up to 5 000 documents per case and documents ranging from 1 to 50 pages. The same person, organization, location, vehicle, or account may appear under many surface forms across PDFs and Word files, making manual linking slow and error-prone. An example is `DNB ASA`, `DNB`, and `Den Norske Bank` - all referring to the same entity.


## Solution

This project addresses the problem with an entity resolution pipeline for Norwegian case documents. It is a thesis prototype intended to support analysis workflows by producing a structured draft for human review.

NERINE uses a multi-stage pipeline:

- `Ingestion`: extract text from PDF/DOCX, clean it, and split it into chunks
- `Extraction`: detect entity mentions, normalize variants, deduplicate within documents, and attach context
- `Blocking`: generate candidate pairs with semantic, phonetic, and token-based retrieval
- `Matching / reranking`: score candidate pairs with engineered features, LightGBM, and SHAP explanations
- `Resolution`: merge high-confidence matches into clusters
- `HITL`: expose uncertain pairs and resolved clusters for review in Streamlit

## Target Architecture

<p align="left">
  <img src="assets/architecture-diagram.png" alt="Target architecture diagram for the NERINE pipeline" width="860"/>
</p>


## Tech Stack

| Component | Tool |
|-----------|------|
| Text extraction | PyMuPDF, `python-docx` |
| NER | `NbAiLab/nb-bert-base-ner` |
| Embeddings | `NbAiLab/nb-sbert-base` |
| Blocking | FAISS, Double Metaphone, MinHash LSH |
| Matching / reranking | LightGBM, SHAP |
| Resolution | NetworkX, OR-Tools |
| Storage | Parquet + DuckDB + Polars |
| HITL UI | Streamlit |
| Tuning | Optuna |


## Project Structure

```
src/
  ingestion/       extraction/      blocking/
  matching/        resolution/      hitl/
  evaluation/      synthetic/       shared/
  pipeline.py
tests/
data/
models/
documents/
```

Each stage has a `run.py` orchestrator. `pipeline.py` calls them in order.
