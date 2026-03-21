"""Tests for synthetic identity-group validator utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.synthetic.validate_identity_groups import (
    validate_identity_groups_file,
    validate_plan_distribution,
)


def _group(group_id: str, entity_type: str, n: int) -> dict:
    """Create one valid group with n variants."""
    variants = []
    for i in range(n):
        variants.append(
            {
                "text": f"{group_id} v{i}",
                "normalized": f"{group_id} v{i}",
                "context": f"Kontekst {group_id} v{i}",
            }
        )
    return {
        "group_id": group_id,
        "entity_type": entity_type,
        "variants": variants,
        "doc_ids": [f"{group_id}_doc_a", f"{group_id}_doc_b"],
        "metadata": {},
    }


def _payload_with_plan_counts() -> dict:
    """Build payload that satisfies strict plan distribution ranges."""
    groups = []
    for i in range(24):
        groups.append(_group(f"per_{i:02d}", "PER", 2))
    for i in range(13):
        groups.append(_group(f"item_{i:02d}", "ITEM", 2))
    for i in range(5):
        groups.append(_group(f"comm_{i:02d}", "COMM", 2))
    for i in range(5):
        groups.append(_group(f"veh_{i:02d}", "VEH", 2))
    for i in range(3):
        groups.append(_group(f"loc_{i:02d}", "LOC", 2))
    groups.append(_group("org_00", "ORG", 2))
    groups.append(_group("fin_00", "FIN", 2))

    hard_negatives = [
        {"group_id_a": "per_00", "group_id_b": "per_01", "reason": "x"},
        {"group_id_a": "per_02", "group_id_b": "per_03", "reason": "x"},
        {"group_id_a": "per_04", "group_id_b": "per_05", "reason": "x"},
        {"group_id_a": "per_06", "group_id_b": "per_07", "reason": "x"},
        {"group_id_a": "item_00", "group_id_b": "item_01", "reason": "x"},
        {"group_id_a": "item_02", "group_id_b": "item_03", "reason": "x"},
        {"group_id_a": "veh_00", "group_id_b": "veh_01", "reason": "x"},
        {"group_id_a": "loc_00", "group_id_b": "loc_01", "reason": "x"},
    ]
    return {
        "run_id": "synthetic_run_validator_001",
        "groups": groups,
        "hard_negatives": hard_negatives,
    }


def test_validate_identity_groups_file_returns_summary(tmp_path: Path) -> None:
    payload = _payload_with_plan_counts()
    path = tmp_path / "identity_groups.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = validate_identity_groups_file(path, enforce_plan_distribution=True)

    assert summary["run_id"] == payload["run_id"]
    assert summary["group_count"] == len(payload["groups"])
    assert summary["hard_negative_count"] == len(payload["hard_negatives"])
    assert summary["counts_by_type"]["PER"] == 24
    assert summary["counts_by_type"]["LOC"] == 3
    assert summary["counts_by_type"]["ORG"] == 1
    assert summary["counts_by_type"]["FIN"] == 1


def test_validate_plan_distribution_rejects_out_of_range_counts() -> None:
    payload = _payload_with_plan_counts()
    payload["groups"] = [group for group in payload["groups"] if group["entity_type"] != "LOC"]
    with pytest.raises(ValueError, match="LOC group count"):
        validate_plan_distribution(payload)


def test_validate_identity_groups_file_accepts_org_and_fin(tmp_path: Path) -> None:
    payload = _payload_with_plan_counts()
    payload["groups"].append(_group("org_01", "ORG", 2))
    payload["groups"].append(_group("fin_01", "FIN", 2))
    path = tmp_path / "identity_groups_with_org_fin.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = validate_identity_groups_file(path, enforce_plan_distribution=True)

    assert summary["counts_by_type"]["ORG"] == 2
    assert summary["counts_by_type"]["FIN"] == 2
