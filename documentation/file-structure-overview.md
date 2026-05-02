# Nerine - File Structure Overview

This document provides an overview of the current repository layout.
It explains the purpose of the main files and folders, while omitting some less important files.

---

## Reading Order

If someone is new to the repository, this order is usually the fastest:

1. `README.md` for the project goal and pipeline idea
2. `documentation/` for practical run guides, evaluation notes, and HITL usage
3. `src/pipeline.py` for the overall execution flow
4. `src/<stage>/run.py` files for stage boundaries
5. Stage helper modules for the details inside each phase
6. `tests/` for expected behavior and contract assumptions


## Top Level

| Path | Purpose |
|------|---------|
| `src/` | Main pipeline source code, split by stage plus shared support modules |
| `tests/` | Automated test suite for pipeline stages, contracts, integration flows, and smoke checks |
| `documentation/` | Reader-facing guides for evaluation, annotation tooling, and the HITL app |
| `data/` | Runtime data area for raw inputs, processed outputs, synthetic artifacts, and handoff material |
| `scripts/` | Focused operational scripts for packaging, flattening, or evaluation runs |
| `README.md` and root-level CI/repo config files | Project overview plus repository automation, formatting defaults, ignore rules, and CI dependency setup |

## Pipeline Entry Files

These are the fastest code entry points for understanding how the repository runs end to end.

| Path | Purpose |
|------|---------|
| `src/pipeline.py` | End-to-end CLI orchestrator that runs the pipeline stages in sequence |
| `src/pipeline_support.py` | Shared helpers for stage metadata, artifact summaries, and pipeline run summaries |

## Supporting Folders And Files

These are important operational and reader-facing files outside the stage packages.

| Path | Purpose |
|------|---------|
| `scripts/flatten_label_studio_export.py` | Normalizes reviewed Label Studio exports into a single flat CSV |
| `scripts/pick_case_doc_plan.py` | Helper for selecting case documents to package |
| `scripts/prepare_label_studio_case.py` | Builds Label Studio inputs and config for a selected case |
| `scripts/run_case_fold_eval.py` | Runs explicit held-out case-fold evaluation with isolated outputs |
| `scripts/run_case_fold_tuning.py` | Thin CLI wrapper for case-fold Optuna tuning |
| `documentation/README-evaluation.md` | Guide for evaluation workflow, metrics, and held-out case runs |
| `documentation/README-hitl.md` | Guide for launching and using the Streamlit review interface |
| `documentation/README-annotation-tooling.md` | Guide for annotation preparation and review tooling |


## `src/`

### `src/shared/`

Shared contracts and utilities used across multiple pipeline stages.

| Path | Purpose |
|------|---------|
| `src/shared/__init__.py` | Package marker for shared utilities |
| `src/shared/config.py` | Pipeline-wide thresholds, routing profiles, and other shared constants |
| `src/shared/schemas.py` | Core parquet schemas plus validation helpers for required columns and contracts |
| `src/shared/validators.py` | Validation helpers for cross-artifact consistency, including embedding alignment |
| `src/shared/fixtures.py` | Reusable deterministic fixtures for tests and synthetic stage setup |

### `src/ingestion/`

Turns source documents into cleaned, deduplicated, chunked text artifacts.

| Path | Purpose |
|------|---------|
| `src/ingestion/__init__.py` | Package marker for ingestion |
| `src/ingestion/run.py` | Stage orchestrator for discovery, registration, extraction, normalization, and chunking |
| `src/ingestion/discovery.py` | Finds supported input documents using `pathlib`-based traversal |
| `src/ingestion/registration.py` | Builds stable document identifiers and deduplicates repeated inputs |
| `src/ingestion/extraction.py` | Extracts text from PDF and DOCX sources |
| `src/ingestion/normalization.py` | Cleans text using Unicode normalization and whitespace/control-character cleanup |
| `src/ingestion/chunking.py` | Splits normalized text into stable chunks for downstream processing |

### `src/extraction/`

Finds entity mentions and prepares normalized mention records for downstream matching.

| Path | Purpose |
|------|---------|
| `src/extraction/__init__.py` | Package marker for extraction |
| `src/extraction/run.py` | Stage orchestrator for NER, normalization, deduplication, context building, and writing |
| `src/extraction/ner.py` | Wrapper around the Norwegian NER model and label mapping |
| `src/extraction/regex_supplements.py` | Pattern-based extraction for structured entities such as identifiers and plates |
| `src/extraction/entity_normalizer.py` | Per-entity-type normalization rules before matching |
| `src/extraction/dedup.py` | Within-document mention deduplication logic |
| `src/extraction/context.py` | Extracts local text windows around each mention |
| `src/extraction/writer.py` | Writes entity outputs with stable IDs and mention-position payloads |

### `src/blocking/`

Builds high-recall candidate pairs so later stages only score plausible matches.

| Path | Purpose |
|------|---------|
| `src/blocking/__init__.py` | Package marker for blocking |
| `src/blocking/run.py` | Stage orchestrator for exact, semantic, phonetic, and MinHash candidate generation |
| `src/blocking/embeddings.py` | Encodes entity names into normalized embedding vectors and stores artifacts |
| `src/blocking/faiss_index.py` | Builds and queries the FAISS index used for semantic neighbor search |
| `src/blocking/exact.py` | Generates exact-match candidates from normalized names and structured identifiers |
| `src/blocking/phonetic.py` | Generates phonetic candidates using Norwegian-aware normalization |
| `src/blocking/minhash.py` | Generates token and n-gram overlap candidates via MinHash LSH |
| `src/blocking/candidates.py` | Merges candidate sources, filters invalid pairs, and tracks provenance |
| `src/blocking/writer.py` | Writes candidate-pair outputs and handoff metadata |

### `src/matching/`

Converts candidate pairs into features and model-based match scores with explanations.

| Path | Purpose |
|------|---------|
| `src/matching/__init__.py` | Package marker for matching |
| `src/matching/run.py` | Main entrypoint for feature generation and pair scoring |
| `src/matching/features.py` | Builds the engineered pairwise feature set from names, embeddings, and metadata |
| `src/matching/fold_training.py` | Trains and summarizes fold-based matching models across prepared runs |
| `src/matching/fold_preparation.py` | Prepares case artifacts and labels for held-out fold workflows |
| `src/matching/fold_tuning.py` | Runs case-held-out Optuna tuning across folds |
| `src/matching/reranker.py` | LightGBM training, inference, persistence, and evaluation helpers |
| `src/matching/shap_explain.py` | SHAP-based feature attribution formatting for scored pairs |
| `src/matching/tuning.py` | Shared Optuna study and objective plumbing outside the fold driver |
| `src/matching/writer.py` | Writers for feature artifacts, scored outputs, and associated metadata |

### `src/resolution/`

Turns pairwise match signals into final entity clusters and canonical outputs.

| Path | Purpose |
|------|---------|
| `src/resolution/__init__.py` | Package marker and stage-boundary export |
| `src/resolution/run.py` | Stage orchestrator for thresholding, clustering, canonicalization, and output writing |
| `src/resolution/clustering.py` | Connected-component preparation plus Pivot-style correlation clustering |
| `src/resolution/confidence.py` | Confidence scoring and routing-profile logic for resolved clusters |
| `src/resolution/canonicalization.py` | Chooses canonical names and assembles cluster lineage output |
| `src/resolution/writer.py` | Writers and path helpers for final resolution artifacts |

### `src/hitl/`

Provides the human-review interface and query helpers for inspecting uncertain results.

| Path | Purpose |
|------|---------|
| `src/hitl/__init__.py` | Package marker for HITL tooling |
| `src/hitl/queries.py` | DuckDB-facing query helpers for cluster and run inspection |
| `src/hitl/status.py` | Diagnostics loading and sidebar summary helpers |
| `src/hitl/cluster_view.py` | Rendering logic for cluster inspection in the Streamlit app |
| `src/hitl/streamlit_app.py` | Main Streamlit interface for review and cluster triage |

### `src/evaluation/`

Measures pipeline quality and summarizes failures or regressions after a run.

| Path | Purpose |
|------|---------|
| `src/evaluation/__init__.py` | Package marker and lazy export for the evaluation entrypoint |
| `src/evaluation/metrics.py` | Metric helpers for mention, pairwise, and cluster-level quality checks |
| `src/evaluation/error_analysis.py` | Compact summaries for false merges, false splits, and extraction mistakes |
| `src/evaluation/run.py` | CLI runner for end-to-end evaluation and regression-style checks |

### `src/synthetic/`

Outdated scripts that generated and validated basic synthetic data used for early matching development and checks while upstream stages were in development.

| Path | Purpose |
|------|---------|
| `src/synthetic/__init__.py` | Package marker and lazy exports for synthetic tooling |
| `src/synthetic/build_matching_dataset.py` | Builds synthetic entities, candidates, labels, and embeddings for matching work |
| `src/synthetic/validate_identity_groups.py` | Validates synthetic identity-group payloads and quotas |
| `src/synthetic/validate.py` | Validates generated synthetic artifacts and downstream feature compatibility |

### `src/annotation/`

Holds annotation-related helpers used to prepare or flatten labeling artifacts.

| Path | Purpose |
|------|---------|
| `src/annotation/__init__.py` | Package marker for annotation helpers |
| `src/annotation/label_studio_flatten.py` | Shared Label Studio export flattening logic used by annotation tooling |

## `tests/`

Tests are summarized here instead of listed file by file.

| Path | Purpose |
|------|---------|
| `tests/ingestion/` | Validates discovery, extraction, normalization, chunking, and ingestion-stage contracts |
| `tests/extraction/` | Covers NER, regex supplements, normalization, deduplication, context, and writer behavior |
| `tests/matching/` | Covers feature loading, feature generation, training, tuning, scoring, and SHAP output |
| `tests/resolution/` | Covers thresholds, clustering behavior, confidence routing, and final outputs |
| `tests/hitl/` | Covers query helpers, diagnostics loading, filters, and cluster-inspector support logic |
| `tests/synthetic/` | Covers synthetic dataset building, validators, and downstream feature integration |
| `tests/evaluation/` | Covers evaluation metrics, runners, and regression-style checks |
| `tests/test_pipeline.py` | Smoke tests for pipeline stage order, run propagation, and summary generation |
| `tests/test_mock_handoff_fixtures.py` | Shared contract checks for mock handoff fixture artifacts |
| `tests/test_mock_embedding_alignment.py` | Shared embedding alignment and failure-case checks |
