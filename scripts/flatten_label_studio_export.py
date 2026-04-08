#!/usr/bin/env python3
"""Validate and flatten one reviewed Label Studio export into gold CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotation.label_studio_flatten import (
    build_mentions_from_label_studio_export,
    convert_label_studio_export_to_csv,
    normalize_label_studio_export,
    summarize_mentions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_path", type=Path, help="Path to the reviewed Label Studio JSON export")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <export dir>/gold_annotations.csv",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the export and print a summary without writing CSV output.",
    )
    parser.add_argument(
        "--drop-empty-zero-length-labels",
        action="store_true",
        help=(
            "Prune structurally invalid Label Studio results where start=end and text=\"\" "
            "before validation/conversion."
        ),
    )
    parser.add_argument(
        "--normalized-export-output",
        type=Path,
        default=None,
        help="Optional path to write a normalized export JSON after pruning invalid zero-length labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_path = args.export_path
    output_path = args.output or export_path.with_name("gold_annotations.csv")
    normalized_summary: dict[str, object] | None = None

    if args.normalized_export_output is not None:
        normalized_summary = normalize_label_studio_export(
            export_path,
            output_path=args.normalized_export_output,
        )

    if args.validate_only:
        mentions = build_mentions_from_label_studio_export(
            export_path,
            drop_empty_zero_length_labels=args.drop_empty_zero_length_labels,
        )
        summary = summarize_mentions(mentions)
    else:
        summary = convert_label_studio_export_to_csv(
            export_path,
            output_path,
            drop_empty_zero_length_labels=args.drop_empty_zero_length_labels,
        )

    if normalized_summary is not None:
        summary["normalized_export"] = normalized_summary

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
