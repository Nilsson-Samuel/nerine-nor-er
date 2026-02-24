"""Parquet schema contracts for the entity resolution pipeline.

Defines PyArrow schemas for all stage-output parquet files and provides
shape/type validation (`validate`) and strict value-level contract
validation (`validate_contract_rules`).
"""

import pyarrow as pa


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
    ("positions", pa.list_(pa.struct([
        ("chunk_id", pa.string()),
        ("char_start", pa.int32()),
        ("char_end", pa.int32()),
        ("page_num", pa.int32()),
        ("source_unit_kind", pa.string()),
    ]))),  # Provenance list of merged mentions
])

CANDIDATE_PAIRS_SCHEMA = pa.schema([
    ("run_id", pa.string()),  # Same run_id as entities table
    ("entity_id_a", pa.string()),  # Canonical rule: entity_id_a < entity_id_b
    ("entity_id_b", pa.string()),  # Canonical rule: no self-pairs
    ("blocking_methods", pa.list_(pa.string())),  # Distinct sorted methods
    ("blocking_source", pa.string()),  # "multi" for 2+ methods, else sole method
    ("blocking_method_count", pa.int8()),  # Int8 count of methods (1..3)
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
_HEX32_CHARS = set("0123456789abcdef")  # For validating 32-char lowercase hex strings


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
def validate_contract_rules(table: pa.Table, contract_name: str) -> list[str]:
    """Validate strict value-level contract rules for a known table contract.

    Args:
        table: Loaded Parquet table.
        contract_name: One of `docs`, `docs.parquet`, `entities`,
            `entities.parquet`, `candidate_pairs`, or `candidate_pairs.parquet`.

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

        if not _is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not _is_hex32(doc_id):
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

        if not _is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not _is_hex32(entity_id):
            errors.append(f"row {row_index}: entity_id must be 32-char lowercase hex")
        if not _is_hex32(doc_id):
            errors.append(f"row {row_index}: doc_id must be 32-char lowercase hex")
        if not _is_hex32(chunk_id):
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

        if not _is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not _is_hex32(entity_id_a):
            errors.append(f"row {row_index}: entity_id_a must be 32-char lowercase hex")
        if not _is_hex32(entity_id_b):
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


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


# Check ID-like fields follow lowercase + trimmed + non-empty contract rules.
def _is_lower_trimmed_non_empty(value: object) -> bool:
    return (
        isinstance(value, str)
        and value != ""
        and value == value.strip()
        and value == value.lower()
    )


def _is_hex32(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(c in _HEX32_CHARS for c in value)
    )
