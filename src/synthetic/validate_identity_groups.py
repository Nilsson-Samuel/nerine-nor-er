"""Validate synthetic identity-group JSON files before dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.synthetic.build_matching_dataset import (
    ALLOWED_SYNTHETIC_TYPES,
    validate_identity_groups_payload,
)


PLAN_TYPE_RANGES = {
    "PER": (24, 30),
    "ITEM": (13, 17),
    "COMM": (5, 7),
    "VEH": (5, 7),
    "LOC": (3, 4),
    "ORG": (1, 3),
    "FIN": (1, 2),
}

PLAN_HARD_NEGATIVE_RANGE = (8, 12)
PLAN_TOTAL_GROUP_RANGE = (52, 70)


def _count_by_type(groups: list[dict]) -> dict[str, int]:
    """Count groups per entity type."""
    counts = {entity_type: 0 for entity_type in sorted(ALLOWED_SYNTHETIC_TYPES)}
    for group in groups:
        counts[group["entity_type"].strip().upper()] += 1
    return counts


def validate_plan_distribution(payload: dict) -> None:
    """Validate plan-specific type quotas and hard-negative count ranges."""
    groups = payload["groups"]
    counts = _count_by_type(groups)
    for entity_type, (min_count, max_count) in PLAN_TYPE_RANGES.items():
        value = counts.get(entity_type, 0)
        if not (min_count <= value <= max_count):
            raise ValueError(
                f"{entity_type} group count must be in [{min_count}, {max_count}], got {value}"
            )
    group_count = len(groups)
    low, high = PLAN_TOTAL_GROUP_RANGE
    if not (low <= group_count <= high):
        raise ValueError(f"group_count must be in [{low}, {high}], got {group_count}")

    hard_negative_count = len(payload.get("hard_negatives", []))
    low, high = PLAN_HARD_NEGATIVE_RANGE
    if not (low <= hard_negative_count <= high):
        raise ValueError(f"hard_negatives count must be in [{low}, {high}], got {hard_negative_count}")


def summarize_payload(payload: dict) -> dict[str, object]:
    """Return compact summary used for quick validation feedback."""
    groups = payload["groups"]
    variants_total = sum(len(group["variants"]) for group in groups)
    summary = {
        "run_id": payload["run_id"],
        "group_count": len(groups),
        "variants_total": variants_total,
        "hard_negative_count": len(payload.get("hard_negatives", [])),
        "counts_by_type": _count_by_type(groups),
    }
    return summary


def validate_identity_groups_file(
    path: Path | str,
    enforce_plan_distribution: bool = True,
) -> dict[str, object]:
    """Validate a JSON file and return a summary if valid."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    validate_identity_groups_payload(payload)
    if enforce_plan_distribution:
        validate_plan_distribution(payload)
    return summarize_payload(payload)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity_groups_path", type=Path)
    parser.add_argument(
        "--skip-plan-distribution",
        action="store_true",
        help="Skip plan-specific quota checks and validate only schema/value rules.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for script-first validation."""
    parser = _build_cli_parser()
    args = parser.parse_args()
    summary = validate_identity_groups_file(
        args.identity_groups_path,
        enforce_plan_distribution=not args.skip_plan_distribution,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
