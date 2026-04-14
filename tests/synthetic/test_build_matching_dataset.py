"""Tests for synthetic matching dataset generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.shared import schemas
from src.shared.paths import get_blocking_run_output_dir, get_extraction_run_output_dir
from src.synthetic.build_matching_dataset import (
    build_matching_dataset,
    validate_identity_groups_payload,
)

_RUN_ID = "synthetic_run_test_001"


def _variant_triplet(
    base: str,
    alias: str,
    context_prefix: str,
    index: int,
    alt: str | None = None,
) -> list[dict]:
    """Build three variants for one synthetic identity group."""
    normalized = base.lower().replace(".", "").replace(",", "").strip()
    return [
        {
            "text": base,
            "normalized": normalized,
            "context": f"{context_prefix} {base} ble omtalt i rapport {index}.",
        },
        {
            "text": alias,
            "normalized": alias.lower().replace(".", "").replace(",", "").strip(),
            "context": f"{context_prefix} alias {alias} ble registrert i rapport {index}.",
        },
        {
            "text": alt if alt is not None else f"{base}, sak {index}",
            "normalized": normalized,
            "context": f"{context_prefix} variant {base} sak {index} ble verifisert.",
        },
    ]


def _identity_groups_fixture() -> dict:
    """Return a compact but realistic synthetic identity-group payload."""
    groups: list[dict] = []
    hard_negatives: list[dict] = []

    per_names = [
        ("Per Hansen", "P. Hansen"),
        ("Kari Nilsen", "K. Nilsen"),
        ("Ola Johansen", "O. Johansen"),
        ("Anne Larsen", "A. Larsen"),
        ("Per Hansen", "Per H."),
    ]
    for i, (base, alias) in enumerate(per_names, start=1):
        group_id = f"per_{i:02d}"
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "PER",
                "variants": _variant_triplet(base, alias, "Avhør", i),
                "doc_ids": [f"doc_per_{i:02d}", f"doc_per_{i + 30:02d}"],
                "metadata": {"fnr": f"1203{i:02d}45226"},
            }
        )
    hard_negatives.append(
        {"group_id_a": "per_01", "group_id_b": "per_05", "reason": "Samme navn, ulikt fnr"}
    )

    item_names = [
        ("Apple iPhone 14", "iPhone 14"),
        ("Samsung Galaxy S21", "Galaxy S21"),
        ("Lenovo ThinkPad X1", "ThinkPad X1"),
    ]
    for i, (base, alias) in enumerate(item_names, start=1):
        group_id = f"item_{i:02d}"
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "ITEM",
                "variants": _variant_triplet(base, alias, "Beslag", i),
                "doc_ids": [f"doc_item_{i:02d}", f"doc_item_{i + 20:02d}"],
                "metadata": {"serial_no": f"ITM-{i:03d}-A"},
            }
        )

    comm_names = [
        ("Telefon 92011234", "Mob 92011234"),
        ("Signal @nordlys77", "@nordlys77"),
    ]
    for i, (base, alias) in enumerate(comm_names, start=1):
        group_id = f"comm_{i:02d}"
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "COMM",
                "variants": _variant_triplet(base, alias, "Kommunikasjon", i),
                "doc_ids": [f"doc_comm_{i:02d}", f"doc_comm_{i + 10:02d}"],
                "metadata": {"identifier": alias},
            }
        )

    veh_names = [
        ("AB12345", "Regnr AB 12345"),
        ("CD67890", "Regnr CD 67890"),
    ]
    for i, (base, alias) in enumerate(veh_names, start=1):
        group_id = f"veh_{i:02d}"
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "VEH",
                "variants": _variant_triplet(base, alias, "Kjøretøy", i),
                "doc_ids": [f"doc_veh_{i:02d}", f"doc_veh_{i + 10:02d}"],
                "metadata": {"regnr": base},
            }
        )
    hard_negatives.append(
        {"group_id_a": "veh_01", "group_id_b": "veh_02", "reason": "Lik format på registrering"}
    )

    loc_names = [
        ("Storgata 12, Oslo", "Storgata 12"),
        ("Bergen sentrum", "Sentrum Bergen"),
    ]
    for i, (base, alias) in enumerate(loc_names, start=1):
        group_id = f"loc_{i:02d}"
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "LOC",
                "variants": _variant_triplet(base, alias, "Sted", i),
                "doc_ids": [f"doc_loc_{i:02d}", f"doc_loc_{i + 10:02d}"],
                "metadata": {"municipality": "Norge"},
            }
        )

    return {
        "run_id": "synthetic_run_test_001",
        "groups": groups,
        "hard_negatives": hard_negatives,
    }


def _write_identity_groups_json(path: Path) -> None:
    """Write test identity-groups payload to JSON."""
    payload = _identity_groups_fixture()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_identity_groups_with_org_fin_json(path: Path) -> None:
    """Write a valid payload that also contains ORG and FIN groups."""
    payload = _identity_groups_fixture()
    payload["groups"].extend(
        [
            {
                "group_id": "org_01",
                "entity_type": "ORG",
                "variants": _variant_triplet(
                    "DNB ASA",
                    "DNB",
                    "Organisasjon",
                    1,
                    alt="Den Norske Bank",
                ),
                "doc_ids": ["doc_org_01", "doc_org_11"],
                "metadata": {"orgnr": "920058817"},
            },
            {
                "group_id": "fin_01",
                "entity_type": "FIN",
                "variants": _variant_triplet(
                    "Kontonummer 15035551234",
                    "Konto 1503.55.51234",
                    "Finans",
                    1,
                    alt="Finansiell konto 15035551234",
                ),
                "doc_ids": ["doc_fin_01", "doc_fin_11"],
                "metadata": {"account_ref": "15035551234"},
            },
        ]
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cosine_similarity_rows(embeddings: np.ndarray, i: int, j: int) -> float:
    """Cosine via dot product for already normalized embeddings."""
    return float(np.dot(embeddings[i], embeddings[j]))


def test_build_matching_dataset_outputs_contract_valid_parquet(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_curated.json"
    out_dir = tmp_path / "synthetic_out"
    _write_identity_groups_json(identity_path)

    build_matching_dataset(identity_path, out_dir, max_pairs=2500, seed=11)

    entities = pq.read_table(get_extraction_run_output_dir(out_dir, _RUN_ID) / "entities.parquet")
    candidates = pq.read_table(get_blocking_run_output_dir(out_dir, _RUN_ID) / "candidate_pairs.parquet")
    labels = pq.read_table(out_dir / "labels.parquet")

    assert schemas.validate_contract_rules(entities, "entities") == []
    assert schemas.validate_contract_rules(candidates, "candidate_pairs") == []
    assert labels.num_rows == candidates.num_rows


def test_label_distribution_is_reasonable(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_curated.json"
    out_dir = tmp_path / "synthetic_out"
    _write_identity_groups_json(identity_path)

    build_matching_dataset(identity_path, out_dir, max_pairs=2500, seed=11)
    labels = pq.read_table(out_dir / "labels.parquet")
    values = np.array(labels.column("label").to_pylist(), dtype=np.float32)

    positive_rate = float(values.mean())
    assert 0.05 <= positive_rate <= 0.30


def test_embeddings_are_l2_normalized(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_curated.json"
    out_dir = tmp_path / "synthetic_out"
    _write_identity_groups_json(identity_path)

    build_matching_dataset(identity_path, out_dir, max_pairs=2500, seed=11)

    blocking_dir = get_blocking_run_output_dir(out_dir, _RUN_ID)
    embeddings = np.load(blocking_dir / "embeddings.npy", allow_pickle=False)
    context_embeddings = np.load(blocking_dir / "context_embeddings.npy", allow_pickle=False)

    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(context_embeddings, axis=1), 1.0, atol=1e-3)


def test_same_identity_pairs_have_high_cosine_similarity(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_curated.json"
    out_dir = tmp_path / "synthetic_out"
    _write_identity_groups_json(identity_path)

    build_matching_dataset(identity_path, out_dir, max_pairs=2500, seed=11)

    blocking_dir = get_blocking_run_output_dir(out_dir, _RUN_ID)
    labels = pq.read_table(out_dir / "labels.parquet").to_pylist()
    embeddings = np.load(blocking_dir / "embeddings.npy", allow_pickle=False)
    entity_ids = np.load(blocking_dir / "embedding_entity_ids.npy", allow_pickle=False)
    id_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids.tolist())}

    positives = []
    for row in labels:
        if row["label"] == 1:
            positives.append(
                _cosine_similarity_rows(
                    embeddings,
                    id_to_idx[row["entity_id_a"]],
                    id_to_idx[row["entity_id_b"]],
                )
            )

    assert positives
    assert min(positives) > 0.8


def test_different_identity_pairs_have_low_average_cosine_similarity(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_curated.json"
    out_dir = tmp_path / "synthetic_out"
    _write_identity_groups_json(identity_path)

    build_matching_dataset(identity_path, out_dir, max_pairs=2500, seed=11)

    blocking_dir = get_blocking_run_output_dir(out_dir, _RUN_ID)
    labels = pq.read_table(out_dir / "labels.parquet").to_pylist()
    embeddings = np.load(blocking_dir / "embeddings.npy", allow_pickle=False)
    entity_ids = np.load(blocking_dir / "embedding_entity_ids.npy", allow_pickle=False)
    id_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids.tolist())}

    negatives = []
    for row in labels:
        if row["label"] == 0:
            negatives.append(
                _cosine_similarity_rows(
                    embeddings,
                    id_to_idx[row["entity_id_a"]],
                    id_to_idx[row["entity_id_b"]],
                )
            )

    assert negatives
    assert float(np.mean(negatives)) < 0.5


def test_same_seed_produces_byte_identical_artifacts(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_curated.json"
    _write_identity_groups_json(identity_path)

    out_dir_a = tmp_path / "out_a"
    out_dir_b = tmp_path / "out_b"
    build_matching_dataset(identity_path, out_dir_a, max_pairs=2500, seed=11)
    build_matching_dataset(identity_path, out_dir_b, max_pairs=2500, seed=11)

    ext_a = get_extraction_run_output_dir(out_dir_a, _RUN_ID)
    ext_b = get_extraction_run_output_dir(out_dir_b, _RUN_ID)
    blk_a = get_blocking_run_output_dir(out_dir_a, _RUN_ID)
    blk_b = get_blocking_run_output_dir(out_dir_b, _RUN_ID)

    assert (ext_a / "entities.parquet").read_bytes() == (ext_b / "entities.parquet").read_bytes()
    assert (out_dir_a / "labels.parquet").read_bytes() == (out_dir_b / "labels.parquet").read_bytes()
    for file_name in (
        "candidate_pairs.parquet",
        "embeddings.npy",
        "context_embeddings.npy",
        "embedding_entity_ids.npy",
    ):
        assert (blk_a / file_name).read_bytes() == (blk_b / file_name).read_bytes()


def test_build_matching_dataset_accepts_org_and_fin_types(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_with_org_fin.json"
    out_dir = tmp_path / "synthetic_out"
    _write_identity_groups_with_org_fin_json(identity_path)

    build_matching_dataset(identity_path, out_dir, max_pairs=2500, seed=11)
    entities = pq.read_table(get_extraction_run_output_dir(out_dir, _RUN_ID) / "entities.parquet")
    entity_types = set(entities.column("type").to_pylist())

    assert "ORG" in entity_types
    assert "FIN" in entity_types


def test_validate_identity_groups_rejects_non_object_hard_negative() -> None:
    payload = _identity_groups_fixture()
    payload["hard_negatives"] = ["not-an-object"]

    with pytest.raises(ValueError, match=r"hard_negatives\[0\] must be an object"):
        validate_identity_groups_payload(payload)


def test_validate_identity_groups_rejects_cross_type_hard_negative() -> None:
    payload = _identity_groups_fixture()
    payload["hard_negatives"] = [
        {"group_id_a": "per_01", "group_id_b": "item_01", "reason": "Cross-type pair"}
    ]

    with pytest.raises(ValueError, match="same entity_type"):
        validate_identity_groups_payload(payload)


def test_build_matching_dataset_fails_if_hard_negatives_cannot_fit(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_groups_curated.json"
    out_dir = tmp_path / "synthetic_out"
    _write_identity_groups_json(identity_path)

    with pytest.raises(ValueError, match="max_pairs too small to include all hard_negatives"):
        build_matching_dataset(identity_path, out_dir, max_pairs=5, seed=11)
