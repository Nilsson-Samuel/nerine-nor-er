"""Validate synthetic matching artifacts before feature training or debugging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.shared import schemas
from src.shared.validators import validate_embedding_alignment
from src.synthetic.build_matching_dataset import LABELS_SCHEMA


logger = logging.getLogger(__name__)

EMBEDDING_FILES = (
    "embeddings.npy",
    "context_embeddings.npy",
    "embedding_entity_ids.npy",
)
_HEX32_CHARS = set("0123456789abcdef")


def _prefix_errors(file_name: str, errors: list[str]) -> list[str]:
    """Prefix validation errors with their source artifact."""
    return [f"{file_name}: {error}" for error in errors]


def _read_table(path: Path):
    """Read a required parquet file and return `(table, errors)`."""
    if not path.exists():
        return None, [f"missing file: {path.name}"]

    try:
        return pq.read_table(path), []
    except Exception as exc:  # pragma: no cover - hard to trigger deterministically
        return None, [f"failed to read {path.name}: {exc}"]


def _validate_schema(table, schema, file_name: str) -> tuple[list[str], bool]:
    """Validate required columns/types before deeper checks."""
    errors = _prefix_errors(file_name, schemas.validate(table, schema))
    return errors, not errors


def _validate_labels_table(table) -> tuple[list[str], bool]:
    """Validate labels.parquet schema and value-level invariants."""
    errors, schema_ok = _validate_schema(table, LABELS_SCHEMA, "labels.parquet")
    if not schema_ok:
        return errors, False

    pair_keys: set[tuple[str, str, str]] = set()
    for row_index, row in enumerate(table.to_pylist(), start=1):
        run_id = row["run_id"]
        entity_id_a = row["entity_id_a"]
        entity_id_b = row["entity_id_b"]
        key = (run_id, entity_id_a, entity_id_b)
        if not _is_lower_trimmed_non_empty(run_id):
            errors.append(
                f"labels.parquet: row {row_index}: run_id must be lowercase, trimmed, non-empty"
            )
        if not _is_hex32(entity_id_a):
            errors.append(
                f"labels.parquet: row {row_index}: entity_id_a must be 32-char lowercase hex"
            )
        if not _is_hex32(entity_id_b):
            errors.append(
                f"labels.parquet: row {row_index}: entity_id_b must be 32-char lowercase hex"
            )
        if row["label"] not in {0, 1}:
            errors.append(f"labels.parquet: row {row_index}: label must be 0 or 1")
        if entity_id_a == entity_id_b:
            errors.append(f"labels.parquet: row {row_index}: pair cannot be a self-pair")
        if (
            isinstance(entity_id_a, str)
            and isinstance(entity_id_b, str)
            and entity_id_a > entity_id_b
        ):
            errors.append(f"labels.parquet: row {row_index}: entity_id_a must be < entity_id_b")
        if key in pair_keys:
            errors.append(
                "labels.parquet: "
                f"row {row_index}: duplicate (run_id, entity_id_a, entity_id_b) key"
            )
        pair_keys.add(key)

    return errors, True


def _table_pair_keys(table) -> set[tuple[str, str, str]]:
    """Extract canonical pair keys from a candidate or label table."""
    return {
        (row["run_id"], row["entity_id_a"], row["entity_id_b"])
        for row in table.to_pylist()
    }


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
        and all(char in _HEX32_CHARS for char in value)
    )


def _log_label_distribution(labels_table) -> None:
    """Log positive/negative counts and positive rate."""
    labels = labels_table.column("label").to_pylist()
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    total = len(labels)
    positive_rate = positives / total if total else 0.0
    logger.info(
        "label_distribution positives=%d negatives=%d positive_rate=%.6f",
        positives,
        negatives,
        positive_rate,
    )


def _validate_embedding_files(data_dir: Path, entity_ids: list[str]) -> list[str]:
    """Validate embedding file presence and row alignment."""
    missing = [
        file_name for file_name in EMBEDDING_FILES if not (data_dir / file_name).exists()
    ]
    if missing:
        return [f"missing file: {file_name}" for file_name in missing]

    try:
        embeddings = np.load(data_dir / "embeddings.npy", allow_pickle=False)
        context_embeddings = np.load(data_dir / "context_embeddings.npy", allow_pickle=False)
        embedding_entity_ids = np.load(data_dir / "embedding_entity_ids.npy", allow_pickle=False)
        validate_embedding_alignment(
            embeddings=embeddings,
            context_embeddings=context_embeddings,
            embedding_entity_ids=embedding_entity_ids,
            entity_ids=entity_ids,
        )
    except Exception as exc:
        return [f"embedding artifacts: {exc}"]

    return []


def _validate_pair_entity_references(
    table,
    file_name: str,
    entity_keys: set[tuple[str, str]],
) -> list[str]:
    """Ensure pair rows only reference entity IDs present for the same run."""
    errors: list[str] = []
    for row_index, row in enumerate(table.to_pylist(), start=1):
        run_id = row["run_id"]
        for column in ("entity_id_a", "entity_id_b"):
            key = (run_id, row[column])
            if key not in entity_keys:
                errors.append(
                    f"{file_name}: row {row_index}: {column} must reference entities.parquet"
                )
    return errors


def validate_synthetic_data(data_dir: Path | str) -> list[str]:
    """Validate synthetic matching artifacts and return any discovered issues."""
    data_dir = Path(data_dir)
    issues: list[str] = []

    entities_table, entity_read_errors = _read_table(data_dir / "entities.parquet")
    issues.extend(entity_read_errors)
    entities_schema_ok = False
    if entities_table is not None:
        entity_schema_errors, entities_schema_ok = _validate_schema(
            entities_table,
            schemas.ENTITIES_SCHEMA,
            "entities.parquet",
        )
        issues.extend(entity_schema_errors)
    if entities_table is not None and entities_schema_ok:
        issues.extend(
            _prefix_errors(
                "entities.parquet",
                schemas.validate_contract_rules(entities_table, "entities"),
            )
        )

    candidate_table, candidate_read_errors = _read_table(data_dir / "candidate_pairs.parquet")
    issues.extend(candidate_read_errors)
    candidate_schema_ok = False
    if candidate_table is not None:
        candidate_schema_errors, candidate_schema_ok = _validate_schema(
            candidate_table,
            schemas.CANDIDATE_PAIRS_SCHEMA,
            "candidate_pairs.parquet",
        )
        issues.extend(candidate_schema_errors)
    if candidate_table is not None and candidate_schema_ok:
        issues.extend(
            _prefix_errors(
                "candidate_pairs.parquet",
                schemas.validate_contract_rules(candidate_table, "candidate_pairs"),
            )
        )

    labels_table, label_read_errors = _read_table(data_dir / "labels.parquet")
    issues.extend(label_read_errors)
    labels_schema_ok = False
    if labels_table is not None:
        label_errors, labels_schema_ok = _validate_labels_table(labels_table)
        issues.extend(label_errors)
        if labels_schema_ok:
            _log_label_distribution(labels_table)

    if entities_table is not None and entities_schema_ok:
        entity_ids = entities_table.column("entity_id").to_pylist()
        issues.extend(_validate_embedding_files(data_dir, entity_ids))
        entity_keys = set(
            zip(
                entities_table.column("run_id").to_pylist(),
                entity_ids,
                strict=True,
            )
        )
        if candidate_table is not None and candidate_schema_ok:
            issues.extend(
                _validate_pair_entity_references(
                    candidate_table,
                    "candidate_pairs.parquet",
                    entity_keys,
                )
            )
        if labels_table is not None and labels_schema_ok:
            issues.extend(
                _validate_pair_entity_references(
                    labels_table,
                    "labels.parquet",
                    entity_keys,
                )
            )
    else:
        missing_embeddings = [
            file_name for file_name in EMBEDDING_FILES if not (data_dir / file_name).exists()
        ]
        issues.extend([f"missing file: {file_name}" for file_name in missing_embeddings])

    if (
        candidate_table is not None
        and candidate_schema_ok
        and labels_table is not None
        and labels_schema_ok
    ):
        candidate_keys = _table_pair_keys(candidate_table)
        label_keys = _table_pair_keys(labels_table)
        orphan_label_count = len(label_keys - candidate_keys)
        unlabeled_pair_count = len(candidate_keys - label_keys)
        if orphan_label_count:
            issues.append(f"labels.parquet: found {orphan_label_count} orphan labels")
        if unlabeled_pair_count:
            issues.append(
                f"labels.parquet: found {unlabeled_pair_count} candidate pairs without labels"
            )

    return issues


def _build_cli_parser() -> argparse.ArgumentParser:
    """Create CLI parser for synthetic artifact validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_cli_parser()
    args = parser.parse_args()
    issues = validate_synthetic_data(args.data_dir)
    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print("Synthetic data validation passed.")


if __name__ == "__main__":
    main()
