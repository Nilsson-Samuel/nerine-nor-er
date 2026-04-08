#!/usr/bin/env python3
"""Prepare a case folder for Label Studio span annotation.

This script is intentionally dependency-light:
- DOCX text is extracted from OOXML with stdlib zip/xml parsing
- brief.yaml is parsed with a small parser tailored to this repo's format
- optional extra seed patterns are loaded from JSON

Outputs are written under <case_root>/annotation/ by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
TYPE_PRIORITY = {
    "PER": 0,
    "ORG": 1,
    "LOC": 2,
    "ITEM": 3,
    "VEH": 4,
    "COMM": 5,
    "FIN": 6,
}
LABEL_CONFIG = """<View>
  <Text name="text" value="$text"/>
  <Labels name="label" toName="text">
    <Label value="PER" background="#d62828"/>
    <Label value="ORG" background="#1d3557"/>
    <Label value="LOC" background="#2a9d8f"/>
    <Label value="ITEM" background="#bc6c25"/>
    <Label value="VEH" background="#6a4c93"/>
    <Label value="COMM" background="#577590"/>
    <Label value="FIN" background="#4d908e"/>
  </Labels>
</View>
"""


@dataclass
class EntitySeed:
    entity_id: str
    entity_type: str
    canonical: str
    aliases: list[str]


@dataclass
class Match:
    start: int
    end: int
    text: str
    entity_type: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <case_root>/annotation",
    )
    parser.add_argument(
        "--seed-patterns",
        type=Path,
        default=None,
        help="Optional JSON file with extra patterns by entity type.",
    )
    return parser.parse_args()


def parse_inline_list(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw.startswith("[") or not raw.endswith("]"):
        return []
    inner = raw[1:-1].strip()
    if not inner:
        return []
    return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]


def parse_brief_yaml(path: Path) -> tuple[str, list[EntitySeed]]:
    case_id: str | None = None
    entities: list[EntitySeed] = []
    in_core_entities = False
    current: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith("case_id:"):
            case_id = line.split(":", 1)[1].strip()
            continue

        if not in_core_entities:
            if line == "core_entities:":
                in_core_entities = True
            continue

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", line):
            if current is not None:
                entities.append(
                    EntitySeed(
                        entity_id=current["id"],
                        entity_type=current["type"],
                        canonical=current["canonical"],
                        aliases=current["aliases"],
                    )
                )
                current = None
            in_core_entities = False
            continue

        if line.startswith("  - id:"):
            if current is not None:
                entities.append(
                    EntitySeed(
                        entity_id=current["id"],
                        entity_type=current["type"],
                        canonical=current["canonical"],
                        aliases=current["aliases"],
                    )
                )
            current = {
                "id": line.split(":", 1)[1].strip(),
                "type": "",
                "canonical": "",
                "aliases": [],
            }
            continue

        if current is None:
            continue

        if line.startswith("    type:"):
            current["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("    canonical:"):
            current["canonical"] = line.split(":", 1)[1].strip()
        elif line.startswith("    aliases:"):
            current["aliases"] = parse_inline_list(line.split(":", 1)[1].strip())

    if current is not None:
        entities.append(
            EntitySeed(
                entity_id=current["id"],
                entity_type=current["type"],
                canonical=current["canonical"],
                aliases=current["aliases"],
            )
        )

    if not case_id:
        raise ValueError(f"Could not parse case_id from {path}")

    return case_id, entities


def compute_doc_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:32]


def extract_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    tree = ET.fromstring(xml)
    paragraphs: list[str] = []
    for para in tree.findall(".//w:body/w:p", WORD_NS):
        texts = [node.text or "" for node in para.findall(".//w:t", WORD_NS)]
        paragraphs.append("".join(texts))
    return "\n\n".join(paragraphs)


def build_phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    left = r"(?<!\w)" if phrase and phrase[0].isalnum() else ""
    right = r"(?!\w)" if phrase and phrase[-1].isalnum() else ""
    return re.compile(left + escaped + right)


def load_seed_patterns(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            raise ValueError(f"Expected list for {key} in {path}")
        result[key] = [str(value) for value in values]
    return result


def find_matches(text: str, entity_type: str, phrases: list[str], source: str) -> list[Match]:
    matches: list[Match] = []
    seen: set[tuple[int, int, str]] = set()
    for phrase in sorted(set(phrases), key=lambda value: (-len(value), value)):
        pattern = build_phrase_pattern(phrase)
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), entity_type)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                Match(
                    start=match.start(),
                    end=match.end(),
                    text=text[match.start() : match.end()],
                    entity_type=entity_type,
                    source=source,
                )
            )
    return matches


def spans_overlap(left: Match, right: Match) -> bool:
    return left.start < right.end and right.start < left.end


def resolve_overlaps(matches: list[Match]) -> list[Match]:
    ranked = sorted(
        matches,
        key=lambda item: (
            item.start,
            -(item.end - item.start),
            TYPE_PRIORITY.get(item.entity_type, 999),
            item.text,
        ),
    )
    accepted: list[Match] = []
    for candidate in ranked:
        if any(spans_overlap(candidate, kept) for kept in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.start, item.end, item.entity_type))


def make_prediction_result(doc_text: str, matches: list[Match]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, match in enumerate(matches, start=1):
        if doc_text[match.start : match.end] != match.text:
            raise ValueError(
                f"Offset mismatch for {match.entity_type} {match.text!r} at "
                f"{match.start}:{match.end}"
            )
        results.append(
            {
                "id": f"seed-{index}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": match.start,
                    "end": match.end,
                    "text": match.text,
                    "labels": [match.entity_type],
                },
            }
        )
    return results


def main() -> int:
    args = parse_args()
    case_root = args.case_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else case_root / "annotation"
    docs_dir = case_root / "docs"
    brief_path = case_root / "brief.yaml"
    seed_patterns_path = (
        args.seed_patterns.resolve()
        if args.seed_patterns
        else case_root / "annotation_seed_patterns.json"
    )

    case_id, entity_seeds = parse_brief_yaml(brief_path)
    extra_patterns = load_seed_patterns(seed_patterns_path if seed_patterns_path.exists() else None)

    output_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = output_dir / "extracted_text"
    texts_dir.mkdir(parents=True, exist_ok=True)

    import_tasks: list[dict[str, Any]] = []
    prediction_tasks: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    alias_map: dict[str, list[str]] = {}
    for seed in entity_seeds:
        alias_map.setdefault(seed.entity_type, [])
        alias_map[seed.entity_type].append(seed.canonical)
        alias_map[seed.entity_type].extend(seed.aliases)

    for doc_path in sorted(docs_dir.glob("*.docx")):
        doc_id = compute_doc_id(doc_path)
        text = extract_docx_text(doc_path)
        (texts_dir / f"{doc_path.stem}.txt").write_text(text, encoding="utf-8")

        matches: list[Match] = []
        for entity_type, phrases in alias_map.items():
            matches.extend(find_matches(text, entity_type, phrases, source="brief"))
        for entity_type, phrases in extra_patterns.items():
            matches.extend(find_matches(text, entity_type, phrases, source="seed_patterns"))
        matches = resolve_overlaps(matches)

        import_task = {
            "data": {
                "text": text,
                "doc_name": doc_path.name,
                "doc_id": doc_id,
                "case_id": case_id,
            }
        }
        prediction_task = {
            "data": import_task["data"],
            "predictions": [
                {
                    "model_version": "seed-preannotations-v1",
                    "score": 0.5,
                    "result": make_prediction_result(text, matches),
                }
            ],
        }
        import_tasks.append(import_task)
        prediction_tasks.append(prediction_task)
        manifest.append(
            {
                "doc_name": doc_path.name,
                "doc_id": doc_id,
                "char_count": len(text),
                "seed_count": len(matches),
                "seed_counts_by_type": dict(Counter(match.entity_type for match in matches)),
            }
        )

    (output_dir / "label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    (output_dir / "label_studio_import.json").write_text(
        json.dumps(import_tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_studio_predictions.json").write_text(
        json.dumps(prediction_tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "doc_manifest.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "documents": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "case_id": case_id,
        "document_count": len(manifest),
        "seed_total": sum(item["seed_count"] for item in manifest),
        "by_doc": {item["doc_name"]: item["seed_counts_by_type"] for item in manifest},
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
