"""Parquet schema contracts for the entity resolution pipeline.

Defines the handoff schemas between pipeline stages. Both developers import
these to validate inputs/outputs at stage boundaries.
"""

import pyarrow as pa


# Handoff schemas (Developer A -> Developer B boundary)

ENTITIES_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("entity_id", pa.string()),
    ("doc_id", pa.string()),
    ("chunk_id", pa.string()),
    ("text", pa.string()),
    ("normalized", pa.string()),
    ("type", pa.string()),
    ("char_start", pa.int32()),
    ("char_end", pa.int32()),
    ("context", pa.string()),
    ("count", pa.int32()),
    # positions: list<struct<...>> — added when extraction stage is built
])

CANDIDATE_PAIRS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("entity_id_a", pa.string()),
    ("entity_id_b", pa.string()),
    ("blocking_methods", pa.list_(pa.string())),
    ("blocking_source", pa.string()),
    ("blocking_method_count", pa.int8()),
])

# Expected keys in handoff_manifest.json
HANDOFF_MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "created_at",
    "mention_count",
    "candidate_count",
    "embedding_dim",
    "entity_types_present",
    "k",
}

VALID_ENTITY_TYPES = {"PER", "ORG", "LOC", "ITEM", "VEH", "COMM", "FIN"}
VALID_BLOCKING_METHODS = {"faiss", "phonetic", "minhash"}
VALID_BLOCKING_SOURCES = {"faiss", "phonetic", "minhash", "multi"}


def validate(table: pa.Table, schema: pa.Schema) -> list[str]:
    """Check a PyArrow table against an expected schema.

    Args:
        table: Loaded parquet table.
        schema: Expected schema contract.

    Returns:
        List of error strings. Empty means valid.
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
