"""Parquet schema contracts for the entity resolution pipeline.

Defines PyArrow schemas for all stage-output parquet files and provides
shape/type validation (`validate`) and strict value-level contract
validation (`validate_contract_rules`).
"""

from __future__ import annotations

import math
from datetime import date, datetime

import polars as pl
import pyarrow as pa

from src.shared.config import (
    BASE_CONFIDENCE_AUTO_MERGE_THRESHOLD,
    BASE_CONFIDENCE_REVIEW_THRESHOLD,
    ROUTING_PROFILE,
)
from src.shared.validators import is_hex32, is_lower_trimmed_non_empty


POSITION_STRUCT_TYPE = pa.struct([
    ("chunk_id", pa.string()),
    ("char_start", pa.int32()),
    ("char_end", pa.int32()),
    ("page_num", pa.int32()),
    ("source_unit_kind", pa.string()),
])

# ---------------------------------------------------------------------------
# Ingestion schemas
# ---------------------------------------------------------------------------

DOCS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("doc_id", pa.string()),  # 32-char lowercase hex (truncated SHA-256)
    ("path", pa.string()),  # Relative POSIX path from case root
    ("mime_type", pa.string()),  # application/pdf or application/vnd.openxml...
    ("source_unit_kind", pa.string()),  # pdf_page | docx_paragraph
    ("page_count", pa.int32()),  # Nullable; filled during extraction
    ("file_size", pa.int64()),  # > 0 bytes
    ("extracted_at", pa.timestamp("us", tz="UTC")),
])

CHUNKS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("chunk_id", pa.string()),  # 32-char lowercase hex (hash of doc_id:chunk_index)
    ("doc_id", pa.string()),  # FK to docs(run_id, doc_id)
    ("chunk_index", pa.int32()),  # >= 0
    ("text", pa.string()),  # Non-empty chunk text
    ("source_unit_kind", pa.string()),  # pdf_page | docx_paragraph
    ("page_num", pa.int32()),  # Page/paragraph index where chunk starts
])

VALID_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
VALID_SOURCE_UNIT_KINDS = {"pdf_page", "docx_paragraph"}


# ---------------------------------------------------------------------------
# Handoff schemas (Developer A -> Developer B boundary)
# ---------------------------------------------------------------------------

ENTITIES_SCHEMA = pa.schema([
    ("run_id", pa.string()),  # Example: "run_2026_02_20_a1b2"
    ("entity_id", pa.string()),  # Example: 32-char lowercase hex
    ("doc_id", pa.string()),  # Example: "9f0e7f5bc2a9405db5a6e11e3b0ec5b3"
    ("chunk_id", pa.string()),  # Example: "ab0f3a8e2e4c4ec5a140bd7f970f3e95"
    ("text", pa.string()),  # Raw mention text, example: "DNB ASA"
    ("normalized", pa.string()),  # Canonical text, example: "dnb asa"
    ("type", pa.string()),  # Enum in practice: PER, ORG, LOC, ITEM, VEH, COMM, FIN
    ("char_start", pa.int32()),  # Zero-based span start, example: 124
    ("char_end", pa.int32()),  # Half-open span end, must be > char_start
    ("context", pa.string()),  # Context window text around mention
    ("count", pa.int32()),  # Number of merged mentions represented by this row
    ("positions", pa.list_(POSITION_STRUCT_TYPE)),  # Provenance list of merged mentions
])

CANDIDATE_PAIRS_SCHEMA = pa.schema([
    ("run_id", pa.string()),  # Same run_id as entities table
    ("entity_id_a", pa.string()),  # Canonical rule: entity_id_a < entity_id_b
    ("entity_id_b", pa.string()),  # Canonical rule: no self-pairs
    ("blocking_methods", pa.list_(pa.string())),  # Distinct sorted methods
    ("blocking_source", pa.string()),  # "multi" for 2+ methods, else sole method
    ("blocking_method_count", pa.int8()),  # Int8 count of methods (1..3)
])

SHAP_VALUE_SCHEMA = pa.struct([
    ("feature", pa.string()),
    ("value", pa.float32()),
])

RESOLVED_ATTRIBUTES_TYPE = pa.map_(pa.string(), pa.list_(pa.string()))

SCORED_PAIRS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("entity_id_a", pa.string()),
    ("entity_id_b", pa.string()),
    ("score", pa.float32()),
    ("model_version", pa.string()),
    ("scored_at", pa.timestamp("us", tz="UTC")),
    ("blocking_methods", pa.list_(pa.string())),
    ("blocking_source", pa.string()),
    ("blocking_method_count", pa.int8()),
    ("shap_top5", pa.list_(SHAP_VALUE_SCHEMA)),
])

RESOLVED_ENTITIES_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("cluster_id", pa.string()),
    ("entity_id", pa.string()),
    ("doc_id", pa.string()),
    ("entity_type", pa.string()),
    ("canonical_name", pa.string()),
    ("canonical_type", pa.string()),
    ("cluster_size", pa.int32()),
    ("confidence", pa.float32()),
    ("needs_review", pa.bool_()),
    ("route_action", pa.string()),
    ("clustering_method", pa.string()),
    ("component_id", pa.string()),
    ("doc_ids", pa.list_(pa.string())),
    ("most_recent_doc_id", pa.string()),
    ("most_recent_doc_date", pa.date32()),
    ("attributes", RESOLVED_ATTRIBUTES_TYPE),
])

# Expected keys in handoff_manifest.json
HANDOFF_MANIFEST_KEYS = {
    "schema_version",  # Example: "1.1"
    "run_id",  # Example: "run_2026_02_20_a1b2"
    "created_at",  # Example: "2026-02-20T14:52:31Z"
    "mention_count",  # Number of rows in entities.parquet
    "candidate_count",  # Number of rows in candidate_pairs.parquet
    "embedding_dim",  # Example: 768
    "entity_types_present",  # Example: ["PER", "ORG", "LOC"]
    "k",  # Top-k neighbors used for FAISS blocking (example: 100)
}

# Contract enums used by strict contract validation helpers below.
VALID_ENTITY_TYPES = {"PER", "ORG", "LOC", "ITEM", "VEH", "COMM", "FIN"}
VALID_BLOCKING_METHODS = {"faiss", "phonetic", "minhash"}
VALID_BLOCKING_SOURCES = {"faiss", "phonetic", "minhash", "multi"}
VALID_ROUTE_ACTIONS = {"auto_merge", "review", "defer", "keep_separate"}


# Validate only column presence and exact PyArrow types against a schema contract.
def validate(table: pa.Table, schema: pa.Schema) -> list[str]:
    """Validate a table against an expected schema using shape/type checks only.

    Args:
        table: Loaded Parquet table.
        schema: Expected schema contract.

    Returns:
        List[str]: Validation errors. Empty list means schema-valid for required
        column presence and exact PyArrow type equality.
    """
    errors = []
    for field in schema:
        if field.name not in table.schema.names:
            errors.append(f"missing column: {field.name}")
            continue
        actual = table.schema.field(field.name).type
        if actual != field.type:
            errors.append(f"{field.name}: expected {field.type}, got {actual}")
    return errors


# Enforce invariant rules after schema/type validation succeeds.
def validate_contract_rules(
    table: pa.Table,
    contract_name: str,
    candidate_pairs_table: pa.Table | None = None,
) -> list[str]:
    """Validate strict value-level contract rules for a known table contract.
   
    Args:
    table: Loaded Parquet table.
    contract_name: One of `docs`, `docs.parquet`, `entities`,
        `entities.parquet`, `candidate_pairs`, `candidate_pairs.parquet`,
        `scored_pairs`, `scored_pairs.parquet`, `resolved_entities`, or
        `resolved_entities.parquet`.
    candidate_pairs_table: Candidate pairs used to verify scored-pair coverage.
            
    Returns:
        List[str]: Validation errors. Empty list means contract-valid.

    Raises:
        ValueError: If `contract_name` is not one of the supported contracts.
    """
    if contract_name in {"docs", "docs.parquet"}:
        schema_errors = validate(table, DOCS_SCHEMA)
        if schema_errors:
            return schema_errors
        return _validate_docs_rules(table)

    if contract_name in {"chunks", "chunks.parquet"}:
        schema_errors = validate(table, CHUNKS_SCHEMA)
        if schema_errors:
            return schema_errors
        return _validate_chunks_rules(table)

    if contract_name in {"entities", "entities.parquet"}:
        schema_errors = validate(table, ENTITIES_SCHEMA)
        if schema_errors:
            return schema_errors
        return _validate_entities_rules(table)

    if contract_name in {"candidate_pairs", "candidate_pairs.parquet"}:
        schema_errors = validate(table, CANDIDATE_PAIRS_SCHEMA)
        if schema_errors:
            return schema_errors
        return _validate_candidate_pair_rules(table)

    if contract_name in {"scored_pairs", "scored_pairs.parquet"}:
        schema_errors = validate(table, SCORED_PAIRS_SCHEMA)
        if schema_errors:
            return schema_errors
        return _validate_scored_pair_rules(table, candidate_pairs_table)

    if contract_name in {"resolved_entities", "resolved_entities.parquet"}:
        schema_errors = validate(table, RESOLVED_ENTITIES_SCHEMA)
        if schema_errors:
            return schema_errors
        return _validate_resolved_entity_rules(table)

    raise ValueError(f"unsupported contract_name: {contract_name}")


# Validate row-level doc invariants and uniqueness for docs.parquet.
def _validate_docs_rules(table: pa.Table) -> list[str]:
    errors: list[str] = []
    seen_doc_keys: set[tuple[str, str]] = set()
    seen_path_keys: set[tuple[str, str]] = set()

    for row_index, row in enumerate(table.to_pylist(), start=1):
        run_id = row["run_id"]
        doc_id = row["doc_id"]
        path = row["path"]
        mime_type = row["mime_type"]
        source_unit_kind = row["source_unit_kind"]
        page_count = row["page_count"]
        file_size = row["file_size"]

        if not is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not is_hex32(doc_id):
            errors.append(f"row {row_index}: doc_id must be 32-char lowercase hex")
        if not _is_non_empty_string(path):
            errors.append(f"row {row_index}: path must be non-empty")
        if mime_type not in VALID_MIME_TYPES:
            errors.append(
                f"row {row_index}: mime_type must be one of {sorted(VALID_MIME_TYPES)}"
            )
        if source_unit_kind not in VALID_SOURCE_UNIT_KINDS:
            errors.append(
                f"row {row_index}: source_unit_kind must be one of "
                f"{sorted(VALID_SOURCE_UNIT_KINDS)}"
            )
        if page_count is not None and (not isinstance(page_count, int) or page_count < 0):
            errors.append(f"row {row_index}: page_count must be >= 0 or null")
        if not isinstance(file_size, int) or file_size <= 0:
            errors.append(f"row {row_index}: file_size must be > 0")

        # Uniqueness: (run_id, doc_id)
        doc_key = (run_id, doc_id)
        if doc_key in seen_doc_keys:
            errors.append(f"row {row_index}: duplicate (run_id, doc_id) key")
        seen_doc_keys.add(doc_key)

        # Uniqueness: (run_id, path)
        path_key = (run_id, path)
        if path_key in seen_path_keys:
            errors.append(f"row {row_index}: duplicate (run_id, path) key")
        seen_path_keys.add(path_key)

    return errors


# Validate row-level chunk invariants and uniqueness for chunks.parquet.
def _validate_chunks_rules(table: pa.Table) -> list[str]:
    errors: list[str] = []
    seen_chunk_keys: set[tuple[str, str]] = set()
    seen_index_keys: set[tuple[str, str, int]] = set()

    for row_index, row in enumerate(table.to_pylist(), start=1):
        run_id = row["run_id"]
        chunk_id = row["chunk_id"]
        doc_id = row["doc_id"]
        chunk_idx = row["chunk_index"]
        text = row["text"]
        source_unit_kind = row["source_unit_kind"]
        page_num = row["page_num"]

        if not is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not is_hex32(chunk_id):
            errors.append(f"row {row_index}: chunk_id must be 32-char lowercase hex")
        if not is_hex32(doc_id):
            errors.append(f"row {row_index}: doc_id must be 32-char lowercase hex")
        if not isinstance(chunk_idx, int) or chunk_idx < 0:
            errors.append(f"row {row_index}: chunk_index must be >= 0")
        if not _is_non_empty_string(text):
            errors.append(f"row {row_index}: text must be non-empty")
        if source_unit_kind not in VALID_SOURCE_UNIT_KINDS:
            errors.append(
                f"row {row_index}: source_unit_kind must be one of "
                f"{sorted(VALID_SOURCE_UNIT_KINDS)}"
            )
        if not isinstance(page_num, int) or page_num < 0:
            errors.append(f"row {row_index}: page_num must be >= 0")

        # Uniqueness: (run_id, chunk_id)
        chunk_key = (run_id, chunk_id)
        if chunk_key in seen_chunk_keys:
            errors.append(f"row {row_index}: duplicate (run_id, chunk_id) key")
        seen_chunk_keys.add(chunk_key)

        # Uniqueness: (run_id, doc_id, chunk_index)
        index_key = (run_id, doc_id, chunk_idx)
        if index_key in seen_index_keys:
            errors.append(
                f"row {row_index}: duplicate (run_id, doc_id, chunk_index) key"
            )
        seen_index_keys.add(index_key)

    return errors


# Validate row-level entity invariants and uniqueness for entities.parquet.
def _validate_entities_rules(table: pa.Table) -> list[str]:
    errors: list[str] = []
    seen_keys: set[tuple[str, str]] = set()

    for row_index, row in enumerate(table.to_pylist(), start=1):
        run_id = row["run_id"]
        entity_id = row["entity_id"]
        doc_id = row["doc_id"]
        chunk_id = row["chunk_id"]
        text = row["text"]
        normalized = row["normalized"]
        entity_type = row["type"]
        char_start = row["char_start"]
        char_end = row["char_end"]
        context = row["context"]
        count = row["count"]
        positions = row["positions"]

        if not is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not is_hex32(entity_id):
            errors.append(f"row {row_index}: entity_id must be 32-char lowercase hex")
        if not is_hex32(doc_id):
            errors.append(f"row {row_index}: doc_id must be 32-char lowercase hex")
        if not is_hex32(chunk_id):
            errors.append(f"row {row_index}: chunk_id must be 32-char lowercase hex")
        if not _is_non_empty_string(text):
            errors.append(f"row {row_index}: text must be non-empty")
        if not _is_non_empty_string(normalized):
            errors.append(f"row {row_index}: normalized must be non-empty")
        if entity_type not in VALID_ENTITY_TYPES:
            errors.append(
                f"row {row_index}: type must be one of {sorted(VALID_ENTITY_TYPES)}"
            )
        if not isinstance(char_start, int) or char_start < 0:
            errors.append(f"row {row_index}: char_start must be >= 0")
        if not isinstance(char_end, int):
            errors.append(f"row {row_index}: char_end must be an int")
        elif isinstance(char_start, int) and char_end <= char_start:
            errors.append(f"row {row_index}: char_end must be > char_start")
        if not _is_non_empty_string(context):
            errors.append(f"row {row_index}: context must be non-empty")
        if not isinstance(count, int) or count < 1:
            errors.append(f"row {row_index}: count must be >= 1")
        if not isinstance(positions, list) or len(positions) == 0:
            errors.append(f"row {row_index}: positions must be a non-empty list")
        elif not isinstance(count, int) or count != len(positions):
            errors.append(f"row {row_index}: count must equal len(positions)")

        key = (run_id, entity_id)
        if key in seen_keys:
            errors.append(f"row {row_index}: duplicate (run_id, entity_id) key")
        seen_keys.add(key)

    return errors


# Validate pair canonicalization, blocking consistency, and uniqueness invariants.
def _validate_candidate_pair_rules(table: pa.Table) -> list[str]:
    errors: list[str] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for row_index, row in enumerate(table.to_pylist(), start=1):
        run_id = row["run_id"]
        entity_id_a = row["entity_id_a"]
        entity_id_b = row["entity_id_b"]
        blocking_methods = row["blocking_methods"]
        blocking_source = row["blocking_source"]
        blocking_method_count = row["blocking_method_count"]

        if not is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not is_hex32(entity_id_a):
            errors.append(f"row {row_index}: entity_id_a must be 32-char lowercase hex")
        if not is_hex32(entity_id_b):
            errors.append(f"row {row_index}: entity_id_b must be 32-char lowercase hex")
        if isinstance(entity_id_a, str) and isinstance(entity_id_b, str):
            if entity_id_a == entity_id_b:
                errors.append(f"row {row_index}: pair cannot be a self-pair")
            elif entity_id_a > entity_id_b:
                errors.append(f"row {row_index}: entity_id_a must be < entity_id_b")

        if not isinstance(blocking_methods, list) or not blocking_methods:
            errors.append(f"row {row_index}: blocking_methods must be a non-empty list")
        else:
            if blocking_methods != sorted(set(blocking_methods)):
                errors.append(
                    f"row {row_index}: blocking_methods must be sorted and distinct"
                )
            unknown_methods = set(blocking_methods) - VALID_BLOCKING_METHODS
            if unknown_methods:
                errors.append(
                    f"row {row_index}: invalid blocking methods {sorted(unknown_methods)}"
                )

        if blocking_source not in VALID_BLOCKING_SOURCES:
            errors.append(
                f"row {row_index}: blocking_source must be one of "
                f"{sorted(VALID_BLOCKING_SOURCES)}"
            )
        if not isinstance(blocking_method_count, int) or not (
            1 <= blocking_method_count <= 3
        ):
            errors.append(f"row {row_index}: blocking_method_count must be in 1..3")
        elif isinstance(blocking_methods, list) and (
            blocking_method_count != len(blocking_methods)
        ):
            errors.append(
                f"row {row_index}: blocking_method_count must equal len(blocking_methods)"
            )
        elif isinstance(blocking_methods, list):
            if blocking_method_count > 1 and blocking_source != "multi":
                errors.append(
                    f"row {row_index}: blocking_source must be 'multi' when count > 1"
                )
            if blocking_method_count == 1:
                sole_method = blocking_methods[0]
                if blocking_source != sole_method:
                    errors.append(
                        f"row {row_index}: blocking_source must equal sole blocking method"
                    )

        key = (run_id, entity_id_a, entity_id_b)
        if key in seen_keys:
            errors.append(
                f"row {row_index}: duplicate (run_id, entity_id_a, entity_id_b) key"
            )
        seen_keys.add(key)

    return errors


def _validate_scored_pair_rules(
    table: pa.Table,
    candidate_pairs_table: pa.Table | None,
) -> list[str]:
    errors: list[str] = []
    frame = _prepare_scored_pairs_frame(table)

    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("run_id").is_null()
            | (pl.col("run_id").str.len_chars() == 0).fill_null(True)
            | (pl.col("run_id") != pl.col("run_id").str.strip_chars()).fill_null(True)
            | (pl.col("run_id") != pl.col("run_id").str.to_lowercase()).fill_null(True),
        ),
        "run_id must be lowercase, trimmed, non-empty",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("entity_id_a").is_null()
            | (~pl.col("entity_id_a").str.contains(r"^[0-9a-f]{32}$")).fill_null(True),
        ),
        "entity_id_a must be 32-char lowercase hex",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("entity_id_b").is_null()
            | (~pl.col("entity_id_b").str.contains(r"^[0-9a-f]{32}$")).fill_null(True),
        ),
        "entity_id_b must be 32-char lowercase hex",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            (pl.col("entity_id_a") == pl.col("entity_id_b")).fill_null(False),
        ),
        "pair cannot be a self-pair",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            (pl.col("entity_id_a") > pl.col("entity_id_b")).fill_null(False),
        ),
        "entity_id_a must be < entity_id_b",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("score").is_null()
            | (~pl.col("score").is_finite()).fill_null(True)
            | (~pl.col("score").is_between(0.0, 1.0, closed="both")).fill_null(True),
        ),
        "score must be finite and in [0, 1]",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("model_version").is_null()
            | (pl.col("model_version").str.len_chars() == 0).fill_null(True),
        ),
        "model_version must be non-empty",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(frame, pl.col("scored_at").is_null()),
        "scored_at must be a UTC timestamp",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("blocking_methods").is_null() | (pl.col("__blocking_methods_len") == 0),
        ),
        "blocking_methods must be a non-empty list",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            (pl.col("__blocking_methods_len") > 0)
            & (pl.col("blocking_methods") != pl.col("__blocking_methods_sorted_distinct")).fill_null(
                False
            ),
        ),
        "blocking_methods must be sorted and distinct",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            (pl.col("__blocking_methods_len") > 0)
            & (~pl.col("__blocking_methods_known").fill_null(False)),
        ),
        "blocking_methods contain unsupported values",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            ~pl.col("blocking_source").is_in(sorted(VALID_BLOCKING_SOURCES)).fill_null(False),
        ),
        f"blocking_source must be one of {sorted(VALID_BLOCKING_SOURCES)}",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("blocking_method_count").is_null()
            | (~pl.col("blocking_method_count").is_between(1, 3, closed="both")).fill_null(True),
        ),
        "blocking_method_count must be in 1..3",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            pl.col("blocking_method_count").is_between(1, 3, closed="both").fill_null(False)
            & (pl.col("blocking_method_count") != pl.col("__blocking_methods_len")).fill_null(False),
        ),
        "blocking_method_count must equal len(blocking_methods)",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            (pl.col("blocking_method_count") > 1).fill_null(False)
            & (pl.col("blocking_source") != "multi").fill_null(False),
        ),
        "blocking_source must be 'multi' when count > 1",
    )
    _extend_row_errors(
        errors,
        _row_indices_where(
            frame,
            (pl.col("blocking_method_count") == 1).fill_null(False)
            & (pl.col("blocking_source") != pl.col("__blocking_method_sole")).fill_null(False),
        ),
        "blocking_source must equal sole blocking method",
    )
    _extend_row_errors(
        errors,
        _duplicate_key_row_indices(frame),
        "duplicate (run_id, entity_id_a, entity_id_b) key",
    )
    _extend_row_errors(
        errors,
        _row_indices_with_missing_candidate_keys(frame, candidate_pairs_table),
        "pair must reference candidate_pairs for the same run",
    )
    _extend_row_errors(
        errors,
        _candidate_row_indices_missing_scored_keys(frame, candidate_pairs_table),
        "candidate_pairs row is missing from scored_pairs",
    )

    for row_index, shap_top5 in zip(
        frame["__row_index"].to_list(),
        table.column("shap_top5").to_pylist(),
        strict=True,
    ):
        errors.extend(_validate_shap_top5(row_index, shap_top5))

    return errors


def _validate_resolved_entity_rules(table: pa.Table) -> list[str]:
    errors: list[str] = []
    seen_keys: set[tuple[str, str, str]] = set()
    cluster_state: dict[tuple[str, str], dict[str, object]] = {}

    for row_index, row in enumerate(table.to_pylist(), start=1):
        run_id = row["run_id"]
        cluster_id = row["cluster_id"]
        entity_id = row["entity_id"]
        doc_id = row["doc_id"]
        entity_type = row["entity_type"]
        canonical_name = row["canonical_name"]
        canonical_type = row["canonical_type"]
        cluster_size = row["cluster_size"]
        confidence = row["confidence"]
        needs_review = row["needs_review"]
        route_action = row["route_action"]
        clustering_method = row["clustering_method"]
        component_id = row["component_id"]
        doc_ids = row["doc_ids"]
        most_recent_doc_id = row["most_recent_doc_id"]
        most_recent_doc_date = row["most_recent_doc_date"]
        attributes = row["attributes"]

        if not is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not is_hex32(cluster_id):
            errors.append(f"row {row_index}: cluster_id must be 32-char lowercase hex")
        if not is_hex32(entity_id):
            errors.append(f"row {row_index}: entity_id must be 32-char lowercase hex")
        if not is_hex32(doc_id):
            errors.append(f"row {row_index}: doc_id must be 32-char lowercase hex")
        if entity_type not in VALID_ENTITY_TYPES:
            errors.append(
                f"row {row_index}: entity_type must be one of {sorted(VALID_ENTITY_TYPES)}"
            )
        if not _is_non_empty_string(canonical_name):
            errors.append(f"row {row_index}: canonical_name must be non-empty")
        if canonical_type not in VALID_ENTITY_TYPES:
            errors.append(
                f"row {row_index}: canonical_type must be one of {sorted(VALID_ENTITY_TYPES)}"
            )
        if not isinstance(cluster_size, int) or cluster_size < 1:
            errors.append(f"row {row_index}: cluster_size must be >= 1")
        if not _is_finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"row {row_index}: confidence must be finite and in [0, 1]")
        if not isinstance(needs_review, bool):
            errors.append(f"row {row_index}: needs_review must be boolean")
        if route_action not in VALID_ROUTE_ACTIONS:
            errors.append(
                f"row {row_index}: route_action must be one of {sorted(VALID_ROUTE_ACTIONS)}"
            )
        if not _is_non_empty_string(clustering_method):
            errors.append(f"row {row_index}: clustering_method must be non-empty")
        if not is_hex32(component_id):
            errors.append(f"row {row_index}: component_id must be 32-char lowercase hex")
        if not _is_sorted_distinct_string_list(doc_ids):
            errors.append(f"row {row_index}: doc_ids must be a sorted distinct non-empty list")
        elif doc_id not in doc_ids:
            errors.append(f"row {row_index}: doc_id must appear in doc_ids")
        if most_recent_doc_id is not None:
            if not is_hex32(most_recent_doc_id):
                errors.append(
                    f"row {row_index}: most_recent_doc_id must be null or 32-char lowercase hex"
                )
            elif isinstance(doc_ids, list) and most_recent_doc_id not in doc_ids:
                errors.append(f"row {row_index}: most_recent_doc_id must appear in doc_ids")
        if most_recent_doc_date is not None and not isinstance(
            most_recent_doc_date,
            date | datetime,
        ):
            errors.append(
                f"row {row_index}: most_recent_doc_date must be null or a date value"
            )
        if most_recent_doc_date is not None and most_recent_doc_id is None:
            errors.append(
                f"row {row_index}: most_recent_doc_id is required when most_recent_doc_date exists"
            )

        attribute_items = _iter_attribute_items(attributes)
        if attribute_items is None:
            errors.append(f"row {row_index}: attributes must be a string->list[string] map")
        else:
            seen_attribute_keys: set[str] = set()
            for attribute_key, values in attribute_items:
                if not _is_non_empty_string(attribute_key):
                    errors.append(f"row {row_index}: attributes keys must be non-empty strings")
                    continue
                if attribute_key in seen_attribute_keys:
                    errors.append(f"row {row_index}: attributes keys must be unique")
                seen_attribute_keys.add(attribute_key)
                if not _is_sorted_distinct_string_list(values):
                    errors.append(
                        f"row {row_index}: attributes values must be sorted distinct non-empty lists"
                    )

        expected_route_action = _expected_resolved_route_action(cluster_size, confidence)
        if expected_route_action is not None and route_action != expected_route_action:
            errors.append(
                f"row {row_index}: route_action must match the configured routing outcome"
            )

        expected_review = expected_route_action == "review"
        if isinstance(needs_review, bool) and needs_review != expected_review:
            errors.append(
                f"row {row_index}: needs_review must match the selected route_action"
            )

        key = (run_id, cluster_id, entity_id)
        if key in seen_keys:
            errors.append(f"row {row_index}: duplicate (run_id, cluster_id, entity_id) key")
        seen_keys.add(key)

        cluster_key = (run_id, cluster_id)
        state = cluster_state.setdefault(
            cluster_key,
            {
                "entity_ids": set(),
                "entity_types": set(),
                "cluster_size": cluster_size,
                "confidence": confidence,
                "needs_review": needs_review,
                "route_action": route_action,
                "canonical_name": canonical_name,
                "canonical_type": canonical_type,
                "clustering_method": clustering_method,
                "component_id": component_id,
                "doc_ids": doc_ids,
                "most_recent_doc_id": most_recent_doc_id,
                "most_recent_doc_date": most_recent_doc_date,
                "attributes": attributes,
            },
        )
        state["entity_ids"].add(entity_id)
        state["entity_types"].add(entity_type)

        for field_name in (
            "cluster_size",
            "confidence",
            "needs_review",
            "route_action",
            "canonical_name",
            "canonical_type",
            "clustering_method",
            "component_id",
            "doc_ids",
            "most_recent_doc_id",
            "most_recent_doc_date",
            "attributes",
        ):
            if row[field_name] != state[field_name]:
                errors.append(
                    f"row {row_index}: {field_name} must be identical for every row in the cluster"
                )

    for run_id, cluster_id in sorted(cluster_state):
        state = cluster_state[(run_id, cluster_id)]
        member_count = len(state["entity_ids"])
        if member_count != state["cluster_size"]:
            errors.append(
                f"cluster {cluster_id}: cluster_size must equal member row count within the run"
            )

        entity_types = sorted(state["entity_types"])
        if len(entity_types) != 1:
            errors.append(
                f"cluster {cluster_id}: entity_type must be consistent across cluster members"
            )
            continue

        if state["canonical_type"] != entity_types[0]:
            errors.append(
                f"cluster {cluster_id}: canonical_type must match the member entity_type"
            )

    return errors


def _prepare_scored_pairs_frame(table: pa.Table) -> pl.DataFrame:
    """Build a scored-pairs frame with derived validation columns."""
    return pl.from_arrow(table).with_row_index("__row_index", offset=1).with_columns(
        pl.col("blocking_methods").list.len().alias("__blocking_methods_len"),
        pl.col("blocking_methods").list.unique().list.sort().alias(
            "__blocking_methods_sorted_distinct"
        ),
        pl.col("blocking_methods")
        .list.eval(pl.element().is_in(sorted(VALID_BLOCKING_METHODS)))
        .list.all()
        .alias("__blocking_methods_known"),
        pl.col("blocking_methods").list.get(0).alias("__blocking_method_sole"),
    )


def _row_indices_where(frame: pl.DataFrame, invalid_expr: pl.Expr) -> list[int]:
    """Return 1-based row indices matching a validation error expression."""
    return frame.filter(invalid_expr).get_column("__row_index").to_list()


def _extend_row_errors(errors: list[str], row_indices: list[int], message: str) -> None:
    """Append one error per failing row index."""
    errors.extend(f"row {row_index}: {message}" for row_index in row_indices)


def _duplicate_key_row_indices(frame: pl.DataFrame) -> list[int]:
    """Return row indices for duplicate scored-pair keys."""
    key_columns = ["run_id", "entity_id_a", "entity_id_b"]
    duplicate_keys = frame.group_by(key_columns).len().filter(pl.col("len") > 1).select(key_columns)
    if duplicate_keys.is_empty():
        return []
    return (
        frame.join(duplicate_keys, on=key_columns, how="inner")
        .sort("__row_index")
        .get_column("__row_index")
        .to_list()
    )


def _row_indices_with_missing_candidate_keys(
    frame: pl.DataFrame,
    candidate_pairs_table: pa.Table | None,
) -> list[int]:
    """Return scored row indices whose keys are absent from candidate_pairs."""
    if candidate_pairs_table is None:
        return []

    key_columns = ["run_id", "entity_id_a", "entity_id_b"]
    candidate_keys = pl.from_arrow(candidate_pairs_table).select(key_columns).unique()
    return (
        frame.join(candidate_keys, on=key_columns, how="anti")
        .sort("__row_index")
        .get_column("__row_index")
        .to_list()
    )


def _candidate_row_indices_missing_scored_keys(
    frame: pl.DataFrame,
    candidate_pairs_table: pa.Table | None,
) -> list[int]:
    """Return candidate row indices that are missing from scored_pairs."""
    if candidate_pairs_table is None:
        return []

    key_columns = ["run_id", "entity_id_a", "entity_id_b"]
    candidate_frame = pl.from_arrow(candidate_pairs_table).with_row_index(
        "__candidate_row_index",
        offset=1,
    )
    scored_keys = frame.select(key_columns).unique()
    return (
        candidate_frame.join(scored_keys, on=key_columns, how="anti")
        .sort("__candidate_row_index")
        .get_column("__candidate_row_index")
        .to_list()
    )


def _validate_shap_top5(row_index: int, shap_top5: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(shap_top5, list):
        return [f"row {row_index}: shap_top5 must be a list"]
    if len(shap_top5) > 5:
        errors.append(f"row {row_index}: shap_top5 must have length <= 5")

    seen_features: set[str] = set()
    previous_abs_value: float | None = None
    for entry in shap_top5:
        if not isinstance(entry, dict):
            errors.append(f"row {row_index}: shap_top5 entries must be structs")
            continue
        feature = entry.get("feature")
        value = entry.get("value")
        if not _is_non_empty_string(feature):
            errors.append(f"row {row_index}: shap_top5 feature must be non-empty")
            continue
        if feature in seen_features:
            errors.append(f"row {row_index}: shap_top5 feature names must be unique")
        seen_features.add(feature)

        if not _is_finite_number(value):
            errors.append(f"row {row_index}: shap_top5 values must be finite")
            continue
        abs_value = abs(float(value))
        if previous_abs_value is not None and abs_value > previous_abs_value:
            errors.append(
                f"row {row_index}: shap_top5 must be ordered by absolute value descending"
            )
        previous_abs_value = abs_value

    return errors


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _is_sorted_distinct_string_list(value: object) -> bool:
    return isinstance(value, list) and value != [] and value == sorted(set(value)) and all(
        _is_non_empty_string(entry) for entry in value
    )


def _iter_attribute_items(value: object) -> list[tuple[object, object]] | None:
    if isinstance(value, dict):
        return list(value.items())
    if isinstance(value, list):
        items: list[tuple[object, object]] = []
        for entry in value:
            if isinstance(entry, tuple) and len(entry) == 2:
                items.append(entry)
                continue
            if isinstance(entry, list) and len(entry) == 2:
                items.append((entry[0], entry[1]))
                continue
            if isinstance(entry, dict) and {"key", "value"} <= set(entry):
                items.append((entry["key"], entry["value"]))
                continue
            return None
        return items
    return None


def _expected_resolved_route_action(
    cluster_size: object,
    confidence: object,
) -> str | None:
    if not isinstance(cluster_size, int):
        return None
    if not _is_finite_number(confidence):
        return None

    confidence_value = float(confidence)
    if cluster_size <= 1:
        return "keep_separate"
    if confidence_value > BASE_CONFIDENCE_AUTO_MERGE_THRESHOLD:
        return "auto_merge"
    if confidence_value >= BASE_CONFIDENCE_REVIEW_THRESHOLD:
        if ROUTING_PROFILE == "balanced_hitl":
            return "review"
        return "defer"
    return "keep_separate"


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
