from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.annotation.label_studio_flatten import (
    build_mentions_from_label_studio_export,
    convert_label_studio_export_to_csv,
    make_deterministic_mention_id,
    normalize_label_studio_export,
)


def _task(doc_name: str, doc_id: str, text: str, spans: list[dict[str, object]]) -> dict[str, object]:
    return {
        "data": {
            "case_id": "case_demo_01",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "text": text,
        },
        "annotations": [
            {
                "result": [
                    {
                        "type": "labels",
                        "value": {
                            "start": span["start"],
                            "end": span["end"],
                            "text": text[int(span["start"]) : int(span["end"])],
                            "labels": [span["label"]],
                        },
                    }
                    for span in spans
                ]
            }
        ],
    }


def test_make_deterministic_mention_id_is_stable_for_same_span() -> None:
    mention_id = make_deterministic_mention_id(
        case_id="case_demo_01",
        doc_id="doc_abc",
        char_start=10,
        char_end=15,
        entity_type="PER",
        text="Alice",
    )

    assert mention_id == make_deterministic_mention_id(
        case_id="case_demo_01",
        doc_id="doc_abc",
        char_start=10,
        char_end=15,
        entity_type="PER",
        text="Alice",
    )
    assert mention_id != make_deterministic_mention_id(
        case_id="case_demo_01",
        doc_id="doc_abc",
        char_start=10,
        char_end=15,
        entity_type="ORG",
        text="Alice",
    )


def test_convert_label_studio_export_to_csv_writes_gold_schema(tmp_path: Path) -> None:
    export_path = tmp_path / "label_studio_export.ready.json"
    output_path = tmp_path / "gold_annotations.csv"
    payload = [
        _task(
            "01-note.docx",
            "doc_a",
            "Alice visited Oslo.",
            [
                {"start": 0, "end": 5, "label": "PER"},
                {"start": 14, "end": 18, "label": "LOC"},
            ],
        ),
        _task(
            "02-note.docx",
            "doc_b",
            "Beta AS called Alice.",
            [
                {"start": 0, "end": 7, "label": "ORG"},
                {"start": 15, "end": 20, "label": "PER"},
            ],
        ),
    ]
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = convert_label_studio_export_to_csv(export_path, output_path)

    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    assert summary["case_id"] == "case_demo_01"
    assert summary["mention_count"] == 4
    assert summary["entity_type_counts"] == {"LOC": 1, "ORG": 1, "PER": 2}
    assert [row["doc_name"] for row in rows] == [
        "01-note.docx",
        "01-note.docx",
        "02-note.docx",
        "02-note.docx",
    ]
    assert all(row["group_id"] == "" for row in rows)
    assert all(row["canonical_text"] == "" for row in rows)
    assert all(row["notes"] == "" for row in rows)
    assert rows[0]["mention_id"].startswith("m_")


def test_build_mentions_from_label_studio_export_rejects_offset_mismatch(
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "bad.ready.json"
    payload = [
        {
            "data": {
                "case_id": "case_demo_01",
                "doc_id": "doc_a",
                "doc_name": "01-note.docx",
                "text": "Alice visited Oslo.",
            },
            "annotations": [
                {
                    "result": [
                        {
                            "type": "labels",
                            "value": {
                                "start": 0,
                                "end": 5,
                                "text": "Alicf",
                                "labels": ["PER"],
                            },
                        }
                    ]
                }
            ],
        }
    ]
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="text mismatch"):
        build_mentions_from_label_studio_export(export_path)


def test_build_mentions_from_label_studio_export_rejects_overlaps(tmp_path: Path) -> None:
    export_path = tmp_path / "overlap.ready.json"
    payload = [
        _task(
            "01-note.docx",
            "doc_a",
            "Alice Smith arrived.",
            [
                {"start": 0, "end": 5, "label": "PER"},
                {"start": 0, "end": 11, "label": "PER"},
            ],
        )
    ]
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overlapping spans"):
        build_mentions_from_label_studio_export(export_path)


def test_normalize_label_studio_export_prunes_empty_zero_length_labels(
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "needs-normalization.ready.json"
    normalized_path = tmp_path / "normalized.ready.json"
    payload = [
        {
            "data": {
                "case_id": "case_demo_01",
                "doc_id": "doc_a",
                "doc_name": "01-note.docx",
                "text": "Alice visited Oslo.",
            },
            "annotations": [
                {
                    "result": [
                        {
                            "type": "labels",
                            "value": {
                                "start": 0,
                                "end": 5,
                                "text": "Alice",
                                "labels": ["PER"],
                            },
                        },
                        {
                            "type": "labels",
                            "value": {
                                "start": 10,
                                "end": 10,
                                "text": "",
                                "labels": ["PER"],
                            },
                        },
                    ]
                }
            ],
        }
    ]
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = normalize_label_studio_export(export_path, normalized_path)
    mentions = build_mentions_from_label_studio_export(
        export_path,
        drop_empty_zero_length_labels=True,
    )

    normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert summary["dropped_empty_zero_length_results"] == 1
    assert len(normalized_payload[0]["annotations"][0]["result"]) == 1
    assert len(mentions) == 1
