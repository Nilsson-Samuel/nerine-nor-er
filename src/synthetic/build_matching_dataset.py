"""Build synthetic matching-stage artifacts from curated identity groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.shared import schemas
from src.shared.validators import validate_embedding_alignment


EMBEDDING_DIM = 768
TARGET_POSITIVE_RATE = 0.15
DEFAULT_MAX_PAIRS = 2500
ALLOWED_SYNTHETIC_TYPES = set(schemas.VALID_ENTITY_TYPES)

LABELS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("entity_id_a", pa.string()),
    ("entity_id_b", pa.string()),
    ("label", pa.int8()),
])

_POSITION_TYPE = pa.struct([
    ("chunk_id", pa.string()),
    ("char_start", pa.int32()),
    ("char_end", pa.int32()),
    ("page_num", pa.int32()),
    ("source_unit_kind", pa.string()),
])


class EntityRecord(NamedTuple):
    """Minimal entity metadata needed for pair and embedding generation."""

    entity_id: str
    group_id: str
    entity_type: str


def _stable_hex32(seed: str) -> str:
    """Return a deterministic 32-char lowercase hex ID from a seed string."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / (norms + 1e-12)).astype(np.float32)


def _sample_without_replacement(
    rows: list[tuple[str, str]],
    sample_size: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Sample pair keys deterministically without replacement."""
    if sample_size <= 0:
        return []
    if sample_size >= len(rows):
        return list(rows)
    selected_idx = rng.choice(len(rows), size=sample_size, replace=False)
    return [rows[int(i)] for i in np.sort(selected_idx)]


def _require_non_empty_string(value: object, field_name: str) -> str:
    """Validate and return a non-empty string value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def validate_identity_groups_payload(payload: dict) -> None:
    """Validate required identity-group JSON shape and value constraints."""
    run_id = _require_non_empty_string(payload.get("run_id"), "run_id")
    if run_id != run_id.lower() or run_id != run_id.strip():
        raise ValueError("run_id must be lowercase and trimmed")

    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a non-empty list")

    seen_group_ids: set[str] = set()
    group_id_to_type: dict[str, str] = {}
    for group in groups:
        group_id = _require_non_empty_string(group.get("group_id"), "group_id")
        if group_id in seen_group_ids:
            raise ValueError(f"duplicate group_id: {group_id}")
        seen_group_ids.add(group_id)

        entity_type = _require_non_empty_string(group.get("entity_type"), "entity_type").upper()
        if entity_type not in ALLOWED_SYNTHETIC_TYPES:
            raise ValueError(
                f"entity_type must be one of {sorted(ALLOWED_SYNTHETIC_TYPES)}, got {entity_type}"
            )
        group_id_to_type[group_id] = entity_type

        variants = group.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            raise ValueError(f"group {group_id} must contain at least two variants")
        for variant in variants:
            _require_non_empty_string(variant.get("text"), f"{group_id}.variants.text")
            _require_non_empty_string(variant.get("context"), f"{group_id}.variants.context")

        doc_ids = group.get("doc_ids")
        if not isinstance(doc_ids, list) or not doc_ids:
            raise ValueError(f"group {group_id} must define at least one doc_id")

    hard_negatives = payload.get("hard_negatives", [])
    if not isinstance(hard_negatives, list):
        raise ValueError("hard_negatives must be a list when present")
    for i, item in enumerate(hard_negatives):
        if not isinstance(item, dict):
            raise ValueError(f"hard_negatives[{i}] must be an object")
        group_id_a = _require_non_empty_string(item.get("group_id_a"), "hard_negatives.group_id_a")
        group_id_b = _require_non_empty_string(item.get("group_id_b"), "hard_negatives.group_id_b")
        if group_id_a not in seen_group_ids or group_id_b not in seen_group_ids:
            raise ValueError("hard_negative references unknown group_id")
        if group_id_a == group_id_b:
            raise ValueError("hard_negative group_id_a/group_id_b must differ")
        if group_id_to_type[group_id_a] != group_id_to_type[group_id_b]:
            raise ValueError("hard_negative groups must have the same entity_type")


def _build_entities(
    run_id: str,
    groups: list[dict],
) -> tuple[pa.Table, list[EntityRecord], dict[str, list[str]]]:
    """Build entities.parquet rows and helper maps from group variants."""
    rows: list[dict] = []
    group_to_entity_ids: dict[str, list[str]] = {}

    for group in groups:
        group_id = group["group_id"].strip()
        entity_type = group["entity_type"].strip().upper()
        doc_ids = group["doc_ids"]
        for variant_idx, variant in enumerate(group["variants"]):
            text = _require_non_empty_string(variant.get("text"), f"{group_id}.text")
            normalized = variant.get("normalized")
            if not isinstance(normalized, str) or not normalized.strip():
                normalized = text.lower().strip()
            context = _require_non_empty_string(variant.get("context"), f"{group_id}.context")
            doc_seed = _require_non_empty_string(
                doc_ids[variant_idx % len(doc_ids)],
                f"{group_id}.doc_id",
            )

            entity_id = _stable_hex32(f"entity:{group_id}:{variant_idx}")
            doc_id = _stable_hex32(f"doc:{doc_seed}")
            chunk_id = _stable_hex32(f"chunk:{group_id}:{variant_idx}")
            char_end = max(1, len(text))
            position = {
                "chunk_id": chunk_id,
                "char_start": 0,
                "char_end": char_end,
                "page_num": int(variant_idx),
                "source_unit_kind": "pdf_page",
            }
            rows.append(
                {
                    "run_id": run_id,
                    "entity_id": entity_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "normalized": normalized.strip(),
                    "type": entity_type,
                    "char_start": 0,
                    "char_end": char_end,
                    "context": context,
                    "count": 1,
                    "positions": [position],
                    "group_id": group_id,
                }
            )
            group_to_entity_ids.setdefault(group_id, []).append(entity_id)

    rows.sort(key=lambda row: row["entity_id"])
    records = [
        EntityRecord(
            entity_id=row["entity_id"],
            group_id=row["group_id"],
            entity_type=row["type"],
        )
        for row in rows
    ]

    entities = pa.table(
        {
            "run_id": pa.array([row["run_id"] for row in rows], type=pa.string()),
            "entity_id": pa.array([row["entity_id"] for row in rows], type=pa.string()),
            "doc_id": pa.array([row["doc_id"] for row in rows], type=pa.string()),
            "chunk_id": pa.array([row["chunk_id"] for row in rows], type=pa.string()),
            "text": pa.array([row["text"] for row in rows], type=pa.string()),
            "normalized": pa.array([row["normalized"] for row in rows], type=pa.string()),
            "type": pa.array([row["type"] for row in rows], type=pa.string()),
            "char_start": pa.array([row["char_start"] for row in rows], type=pa.int32()),
            "char_end": pa.array([row["char_end"] for row in rows], type=pa.int32()),
            "context": pa.array([row["context"] for row in rows], type=pa.string()),
            "count": pa.array([row["count"] for row in rows], type=pa.int32()),
            "positions": pa.array([row["positions"] for row in rows], type=pa.list_(_POSITION_TYPE)),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )
    return entities, records, group_to_entity_ids


def _first_pair_between_groups(
    group_id_a: str,
    group_id_b: str,
    group_to_entity_ids: dict[str, list[str]],
) -> tuple[str, str] | None:
    """Return one canonical negative pair key linking two groups."""
    left = sorted(group_to_entity_ids.get(group_id_a, []))
    right = sorted(group_to_entity_ids.get(group_id_b, []))
    if not left or not right:
        return None
    a = left[0]
    b = right[0]
    return (a, b) if a < b else (b, a)


def _choose_blocking_methods(entity_type: str, rng: np.random.Generator) -> list[str]:
    """Choose deterministic synthetic blocking methods with PER-aware weighting."""
    methods: set[str] = set()
    if rng.random() < 0.85:
        methods.add("faiss")
    if rng.random() < 0.55:
        methods.add("minhash")
    if entity_type == "PER" and rng.random() < 0.65:
        methods.add("phonetic")
    if not methods:
        methods.add("phonetic" if entity_type == "PER" else "faiss")
    return sorted(methods)


def _build_pairs_and_labels(
    run_id: str,
    records: list[EntityRecord],
    hard_negatives: list[dict],
    max_pairs: int,
    seed: int,
    target_positive_rate: float = TARGET_POSITIVE_RATE,
) -> tuple[pa.Table, pa.Table]:
    """Build candidate_pairs.parquet and labels.parquet."""
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1")

    id_to_group = {record.entity_id: record.group_id for record in records}
    id_to_type = {record.entity_id: record.entity_type for record in records}

    ids_by_type: dict[str, list[str]] = {}
    group_to_entity_ids: dict[str, list[str]] = {}
    for record in records:
        ids_by_type.setdefault(record.entity_type, []).append(record.entity_id)
        group_to_entity_ids.setdefault(record.group_id, []).append(record.entity_id)
    for entity_ids in ids_by_type.values():
        entity_ids.sort()

    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []
    for entity_type in sorted(ids_by_type):
        for entity_id_a, entity_id_b in combinations(ids_by_type[entity_type], 2):
            if id_to_group[entity_id_a] == id_to_group[entity_id_b]:
                positives.append((entity_id_a, entity_id_b))
            else:
                negatives.append((entity_id_a, entity_id_b))
    if not positives:
        raise ValueError("identity groups must produce at least one positive pair")

    total_pair_count = len(positives) + len(negatives)
    negative_set = set(negatives)
    hard_negative_keys: list[tuple[str, str]] = []
    for item in hard_negatives:
        pair_key = _first_pair_between_groups(
            item["group_id_a"],
            item["group_id_b"],
            group_to_entity_ids,
        )
        if pair_key is not None and pair_key in negative_set:
            hard_negative_keys.append(pair_key)
    hard_negative_keys = list(dict.fromkeys(hard_negative_keys))

    rng = np.random.default_rng(seed)
    target_total = min(max_pairs, total_pair_count)
    target_from_rate = int(round(len(positives) / max(target_positive_rate, 1e-9)))
    target_total = min(target_total, max(len(positives), target_from_rate))

    if len(positives) >= target_total:
        selected_positives = _sample_without_replacement(positives, target_total, rng)
        selected_negatives: list[tuple[str, str]] = []
    else:
        selected_positives = list(positives)
        negative_slots = target_total - len(selected_positives)
        required = [pair for pair in hard_negative_keys if pair in negative_set]
        required = list(dict.fromkeys(required))[:negative_slots]
        required_set = set(required)
        remaining_pool = [pair for pair in negatives if pair not in required_set]
        sampled = _sample_without_replacement(remaining_pool, negative_slots - len(required), rng)
        selected_negatives = required + sampled

    selected_pairs = sorted(selected_positives + selected_negatives)
    missing_hard_negatives = sorted(set(hard_negative_keys) - set(selected_negatives))
    if missing_hard_negatives:
        raise ValueError(
            "max_pairs too small to include all hard_negatives; "
            f"missing {len(missing_hard_negatives)} hard-negative pairs"
        )
    label_by_pair = {pair: 1 for pair in selected_positives}
    label_by_pair.update({pair: 0 for pair in selected_negatives})

    blocking_methods: list[list[str]] = []
    blocking_source: list[str] = []
    blocking_method_count: list[int] = []
    for entity_id_a, entity_id_b in selected_pairs:
        entity_type = id_to_type[entity_id_a]
        methods = _choose_blocking_methods(entity_type, rng)
        blocking_methods.append(methods)
        blocking_source.append("multi" if len(methods) > 1 else methods[0])
        blocking_method_count.append(len(methods))

    candidates = pa.table(
        {
            "run_id": pa.array([run_id] * len(selected_pairs), type=pa.string()),
            "entity_id_a": pa.array([pair[0] for pair in selected_pairs], type=pa.string()),
            "entity_id_b": pa.array([pair[1] for pair in selected_pairs], type=pa.string()),
            "blocking_methods": pa.array(blocking_methods, type=pa.list_(pa.string())),
            "blocking_source": pa.array(blocking_source, type=pa.string()),
            "blocking_method_count": pa.array(blocking_method_count, type=pa.int8()),
        },
        schema=schemas.CANDIDATE_PAIRS_SCHEMA,
    )

    labels = pa.table(
        {
            "run_id": pa.array([run_id] * len(selected_pairs), type=pa.string()),
            "entity_id_a": pa.array([pair[0] for pair in selected_pairs], type=pa.string()),
            "entity_id_b": pa.array([pair[1] for pair in selected_pairs], type=pa.string()),
            "label": pa.array([label_by_pair[pair] for pair in selected_pairs], type=pa.int8()),
        },
        schema=LABELS_SCHEMA,
    )
    return candidates, labels


def _build_embedding_artifacts(
    records: list[EntityRecord],
    seed: int,
    embedding_dim: int = EMBEDDING_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic embedding artifacts aligned to entities.parquet order."""
    rng = np.random.default_rng(seed)
    group_ids = sorted({record.group_id for record in records})
    entity_ids = np.array([record.entity_id for record in records])

    base_name_vectors = {
        group_id: rng.normal(size=(embedding_dim,)).astype(np.float32) for group_id in group_ids
    }
    base_context_vectors = {
        group_id: rng.normal(size=(embedding_dim,)).astype(np.float32) for group_id in group_ids
    }

    entity_rows = []
    context_rows = []
    for record in records:
        entity_rows.append(base_name_vectors[record.group_id] + rng.normal(scale=0.03, size=embedding_dim))
        context_rows.append(
            base_context_vectors[record.group_id] + rng.normal(scale=0.035, size=embedding_dim)
        )
    embeddings = _l2_normalize_rows(np.asarray(entity_rows, dtype=np.float32))
    context_embeddings = _l2_normalize_rows(np.asarray(context_rows, dtype=np.float32))
    return embeddings, context_embeddings, entity_ids


def build_matching_dataset(
    identity_groups_path: Path | str,
    out_dir: Path | str,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    seed: int = 7,
) -> None:
    """Build synthetic matching-stage artifacts from curated identity groups.

    Output files:
    - entities.parquet
    - candidate_pairs.parquet
    - labels.parquet
    - embeddings.npy
    - context_embeddings.npy
    - embedding_entity_ids.npy
    """
    identity_groups_path = Path(identity_groups_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with identity_groups_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_identity_groups_payload(payload)

    run_id = payload["run_id"]
    entities, records, group_to_entity_ids = _build_entities(run_id, payload["groups"])
    candidates, labels = _build_pairs_and_labels(
        run_id=run_id,
        records=records,
        hard_negatives=payload.get("hard_negatives", []),
        max_pairs=max_pairs,
        seed=seed,
    )
    embeddings, context_embeddings, embedding_entity_ids = _build_embedding_artifacts(records, seed)

    entity_errors = schemas.validate_contract_rules(entities, "entities")
    if entity_errors:
        raise ValueError(f"entities.parquet contract violations: {entity_errors}")
    candidate_errors = schemas.validate_contract_rules(candidates, "candidate_pairs")
    if candidate_errors:
        raise ValueError(f"candidate_pairs.parquet contract violations: {candidate_errors}")
    validate_embedding_alignment(
        embeddings=embeddings,
        context_embeddings=context_embeddings,
        embedding_entity_ids=embedding_entity_ids,
        entity_ids=entities.column("entity_id").to_pylist(),
    )

    pq.write_table(entities, out_dir / "entities.parquet")
    pq.write_table(candidates, out_dir / "candidate_pairs.parquet")
    pq.write_table(labels, out_dir / "labels.parquet")
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "context_embeddings.npy", context_embeddings)
    np.save(out_dir / "embedding_entity_ids.npy", embedding_entity_ids)

    expected_group_ids = {group["group_id"].strip() for group in payload["groups"]}
    if set(group_to_entity_ids) != expected_group_ids:
        raise ValueError("not all groups produced entities")


def _build_cli_parser() -> argparse.ArgumentParser:
    """Create CLI parser for manual dataset generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity_groups_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_cli_parser()
    args = parser.parse_args()
    build_matching_dataset(
        identity_groups_path=args.identity_groups_path,
        out_dir=args.out_dir,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
