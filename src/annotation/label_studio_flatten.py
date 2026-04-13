"""Convert reviewed Label Studio span exports into the repo's flat gold CSV."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ALLOWED_ENTITY_TYPES = frozenset({"PER", "ORG", "LOC", "ITEM", "VEH", "COMM", "FIN"})
GOLD_CSV_COLUMNS = [
    "case_id",
    "doc_id",
    "doc_name",
    "mention_id",
    "char_start",
    "char_end",
    "text",
    "entity_type",
    "group_id",
    "canonical_text",
    "notes",
]


@dataclass(frozen=True)
class FlatGoldMention:
    case_id: str
    doc_id: str
    doc_name: str
    mention_id: str
    char_start: int
    char_end: int
    text: str
    entity_type: str
    group_id: str = ""
    canonical_text: str = ""
    notes: str = ""

    def to_row(self) -> dict[str, str | int]:
        return asdict(self)


def make_deterministic_mention_id(
    case_id: str,
    doc_id: str,
    char_start: int,
    char_end: int,
    entity_type: str,
    text: str,
) -> str:
    """Build one stable mention id from intrinsic mention fields, not row order."""
    payload = "\x1f".join(
        [
            case_id,
            doc_id,
            str(char_start),
            str(char_end),
            entity_type,
            text,
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"m_{digest}"


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _active_annotations(task: dict[str, Any], task_label: str) -> list[dict[str, Any]]:
    raw_annotations = _require_list(task.get("annotations"), f"{task_label}.annotations")
    active = [
        _require_dict(annotation, f"{task_label}.annotations[]")
        for annotation in raw_annotations
        if not bool(annotation.get("was_cancelled"))
    ]
    if len(active) != 1:
        raise ValueError(
            f"{task_label} must contain exactly one active annotation; found {len(active)}"
        )
    return active


def _validate_offsets(
    doc_text: str,
    start: int,
    end: int,
    mention_text: str,
    location: str,
) -> None:
    if start < 0 or end < 0:
        raise ValueError(f"{location} start/end must be >= 0")
    if end <= start:
        raise ValueError(f"{location} end must be greater than start")
    if end > len(doc_text):
        raise ValueError(f"{location} end={end} exceeds document length={len(doc_text)}")
    if doc_text[start:end] != mention_text:
        raise ValueError(
            f"{location} text mismatch at {start}:{end}; "
            f"document has {doc_text[start:end]!r}, export has {mention_text!r}"
        )


def _ensure_unique_doc_mapping(
    doc_name_to_id: dict[str, str],
    doc_id_to_name: dict[str, str],
    doc_name: str,
    doc_id: str,
    task_label: str,
) -> None:
    known_doc_id = doc_name_to_id.get(doc_name)
    if known_doc_id is not None and known_doc_id != doc_id:
        raise ValueError(
            f"{task_label} reuses doc_name={doc_name!r} with conflicting doc_id values"
        )
    known_doc_name = doc_id_to_name.get(doc_id)
    if known_doc_name is not None and known_doc_name != doc_name:
        raise ValueError(
            f"{task_label} reuses doc_id={doc_id!r} with conflicting doc_name values"
        )
    doc_name_to_id[doc_name] = doc_id
    doc_id_to_name[doc_id] = doc_name


def _ensure_no_overlaps(mentions: list[FlatGoldMention], doc_name: str) -> None:
    ordered = sorted(
        mentions,
        key=lambda mention: (
            mention.char_start,
            mention.char_end,
            mention.entity_type,
            mention.text,
            mention.mention_id,
        ),
    )
    for previous, current in zip(ordered, ordered[1:]):
        if previous.char_end > current.char_start:
            raise ValueError(
                "overlapping spans detected in "
                f"{doc_name}: {previous.char_start}:{previous.char_end} "
                f"overlaps {current.char_start}:{current.char_end}"
            )


def _is_empty_zero_length_label(result: dict[str, Any]) -> bool:
    if result.get("type") != "labels":
        return False
    value = result.get("value")
    if not isinstance(value, dict):
        return False
    start = value.get("start")
    end = value.get("end")
    text = value.get("text")
    labels = value.get("labels")
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and start == end
        and text == ""
        and isinstance(labels, list)
        and len(labels) == 1
    )


def normalize_label_studio_export(
    export_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Write or summarize one export with empty zero-length labels pruned."""
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    tasks = _require_list(payload, f"{export_path}")
    normalized_tasks = copy.deepcopy(tasks)
    dropped_count = 0

    for task_index, raw_task in enumerate(normalized_tasks, start=1):
        task_label = f"task[{task_index}]"
        task = _require_dict(raw_task, task_label)
        annotations = _require_list(task.get("annotations"), f"{task_label}.annotations")
        for annotation_index, raw_annotation in enumerate(annotations, start=1):
            annotation_label = f"{task_label}.annotations[{annotation_index}]"
            annotation = _require_dict(raw_annotation, annotation_label)
            results = _require_list(annotation.get("result"), f"{annotation_label}.result")
            filtered_results = [
                _require_dict(result, f"{annotation_label}.result[]")
                for result in results
                if not _is_empty_zero_length_label(_require_dict(result, f"{annotation_label}.result[]"))
            ]
            dropped_count += len(results) - len(filtered_results)
            annotation["result"] = filtered_results

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(normalized_tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "input_path": str(export_path),
        "output_path": str(output_path) if output_path is not None else None,
        "dropped_empty_zero_length_results": dropped_count,
    }


def _load_tasks(
    export_path: Path,
    *,
    drop_empty_zero_length_labels: bool,
) -> tuple[list[Any], int]:
    if not drop_empty_zero_length_labels:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        return _require_list(payload, f"{export_path}"), 0

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    tasks = _require_list(payload, f"{export_path}")
    normalized_tasks = copy.deepcopy(tasks)
    dropped_count = 0
    for raw_task in normalized_tasks:
        task = _require_dict(raw_task, "task")
        annotations = _require_list(task.get("annotations"), "task.annotations")
        for raw_annotation in annotations:
            annotation = _require_dict(raw_annotation, "task.annotations[]")
            results = _require_list(annotation.get("result"), "task.annotations[].result")
            filtered_results = []
            for raw_result in results:
                result = _require_dict(raw_result, "task.annotations[].result[]")
                if _is_empty_zero_length_label(result):
                    dropped_count += 1
                    continue
                filtered_results.append(result)
            annotation["result"] = filtered_results
    return normalized_tasks, dropped_count


def build_mentions_from_label_studio_export(
    export_path: Path,
    *,
    drop_empty_zero_length_labels: bool = False,
) -> list[FlatGoldMention]:
    """Parse and validate one reviewed Label Studio JSON export."""
    tasks, _ = _load_tasks(
        export_path,
        drop_empty_zero_length_labels=drop_empty_zero_length_labels,
    )
    if not tasks:
        raise ValueError(f"{export_path} does not contain any tasks")

    mentions: list[FlatGoldMention] = []
    case_ids: set[str] = set()
    doc_name_to_id: dict[str, str] = {}
    doc_id_to_name: dict[str, str] = {}

    for task_index, raw_task in enumerate(tasks, start=1):
        task_label = f"task[{task_index}]"
        task = _require_dict(raw_task, task_label)
        data = _require_dict(task.get("data"), f"{task_label}.data")
        case_id = _require_non_empty_string(data.get("case_id"), f"{task_label}.data.case_id")
        doc_id = _require_non_empty_string(data.get("doc_id"), f"{task_label}.data.doc_id")
        doc_name = _require_non_empty_string(data.get("doc_name"), f"{task_label}.data.doc_name")
        doc_text = _require_non_empty_string(data.get("text"), f"{task_label}.data.text")
        case_ids.add(case_id)
        _ensure_unique_doc_mapping(doc_name_to_id, doc_id_to_name, doc_name, doc_id, task_label)

        annotation = _active_annotations(task, task_label)[0]
        results = _require_list(annotation.get("result"), f"{task_label}.annotations[0].result")
        doc_mentions: list[FlatGoldMention] = []
        seen_spans: set[tuple[int, int, str, str]] = set()

        for result_index, raw_result in enumerate(results, start=1):
            result_label = f"{task_label}.result[{result_index}]"
            result = _require_dict(raw_result, result_label)
            result_type = _require_non_empty_string(result.get("type"), f"{result_label}.type")
            if result_type != "labels":
                raise ValueError(f"{result_label}.type must be 'labels', got {result_type!r}")

            value = _require_dict(result.get("value"), f"{result_label}.value")
            start = _require_int(value.get("start"), f"{result_label}.value.start")
            end = _require_int(value.get("end"), f"{result_label}.value.end")
            mention_text = _require_non_empty_string(
                value.get("text"), f"{result_label}.value.text"
            )
            labels = _require_list(value.get("labels"), f"{result_label}.value.labels")
            if len(labels) != 1:
                raise ValueError(
                    f"{result_label}.value.labels must contain exactly one entity type"
                )
            entity_type = _require_non_empty_string(labels[0], f"{result_label}.value.labels[0]")
            if entity_type not in ALLOWED_ENTITY_TYPES:
                raise ValueError(
                    f"{result_label}.value.labels[0]={entity_type!r} is not a supported entity type"
                )

            _validate_offsets(doc_text, start, end, mention_text, result_label)
            span_key = (start, end, entity_type, mention_text)
            if span_key in seen_spans:
                raise ValueError(
                    f"duplicate span detected in {doc_name}: {start}:{end} {entity_type} {mention_text!r}"
                )
            seen_spans.add(span_key)

            doc_mentions.append(
                FlatGoldMention(
                    case_id=case_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    mention_id=make_deterministic_mention_id(
                        case_id=case_id,
                        doc_id=doc_id,
                        char_start=start,
                        char_end=end,
                        entity_type=entity_type,
                        text=mention_text,
                    ),
                    char_start=start,
                    char_end=end,
                    text=mention_text,
                    entity_type=entity_type,
                )
            )

        _ensure_no_overlaps(doc_mentions, doc_name)
        mentions.extend(
            sorted(
                doc_mentions,
                key=lambda mention: (
                    mention.doc_name,
                    mention.char_start,
                    mention.char_end,
                    mention.entity_type,
                    mention.text,
                    mention.mention_id,
                ),
            )
        )

    if len(case_ids) != 1:
        raise ValueError(f"{export_path} must contain exactly one case_id; found {sorted(case_ids)}")

    return sorted(
        mentions,
        key=lambda mention: (
            mention.doc_name,
            mention.char_start,
            mention.char_end,
            mention.entity_type,
            mention.text,
            mention.mention_id,
        ),
    )


def summarize_mentions(mentions: list[FlatGoldMention]) -> dict[str, Any]:
    """Return a compact summary for validation output."""
    counts_by_type = Counter(mention.entity_type for mention in mentions)
    counts_by_doc = Counter(mention.doc_name for mention in mentions)
    return {
        "case_id": mentions[0].case_id if mentions else None,
        "doc_count": len({mention.doc_id for mention in mentions}),
        "mention_count": len(mentions),
        "entity_type_counts": dict(sorted(counts_by_type.items())),
        "doc_counts": dict(sorted(counts_by_doc.items())),
    }


def write_gold_csv(mentions: list[FlatGoldMention], output_path: Path) -> None:
    """Write the flat gold CSV in the repo's standard column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_CSV_COLUMNS)
        writer.writeheader()
        for mention in mentions:
            writer.writerow(mention.to_row())


def convert_label_studio_export_to_csv(
    export_path: Path,
    output_path: Path,
    *,
    drop_empty_zero_length_labels: bool = False,
) -> dict[str, Any]:
    """Validate one export and write the flat gold CSV."""
    mentions = build_mentions_from_label_studio_export(
        export_path,
        drop_empty_zero_length_labels=drop_empty_zero_length_labels,
    )
    write_gold_csv(mentions, output_path)
    summary = summarize_mentions(mentions)
    if drop_empty_zero_length_labels:
        _, dropped_count = _load_tasks(
            export_path,
            drop_empty_zero_length_labels=drop_empty_zero_length_labels,
        )
        summary["dropped_empty_zero_length_results"] = dropped_count
    summary["output_path"] = str(output_path)
    return summary
