"""Smoke tests for end-to-end pipeline orchestration and CLI wiring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.pipeline as pipeline
import src.pipeline_support as pipeline_support
from src.matching.writer import (
    get_features_output_path,
    get_scored_pairs_output_path,
    get_scoring_metadata_path,
)
from src.resolution.writer import (
    get_clusters_output_path,
    get_resolution_components_path,
    get_resolution_diagnostics_path,
    get_resolved_entities_output_path,
)
from src.shared import schemas


def _hex32(number: int) -> str:
    """Build one deterministic lowercase 32-char hex identifier."""
    return f"{number:032x}"


def _write_ingestion_outputs(data_dir: Path, run_id: str) -> None:
    """Write minimal docs/chunks parquet outputs for one run."""
    docs_table = pa.table(
        {
            "run_id": pa.array([run_id], type=pa.string()),
            "doc_id": pa.array([_hex32(1)], type=pa.string()),
            "path": pa.array(["case/doc_a.pdf"], type=pa.string()),
            "mime_type": pa.array(["application/pdf"], type=pa.string()),
            "source_unit_kind": pa.array(["pdf_page"], type=pa.string()),
            "page_count": pa.array([1], type=pa.int32()),
            "file_size": pa.array([128], type=pa.int64()),
            "extracted_at": pa.array(
                [datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)],
                type=pa.timestamp("us", tz="UTC"),
            ),
        },
        schema=schemas.DOCS_SCHEMA,
    )
    chunks_table = pa.table(
        {
            "run_id": pa.array([run_id], type=pa.string()),
            "chunk_id": pa.array([_hex32(11)], type=pa.string()),
            "doc_id": pa.array([_hex32(1)], type=pa.string()),
            "chunk_index": pa.array([0], type=pa.int32()),
            "text": pa.array(["Per Hansen forklarte seg."], type=pa.string()),
            "source_unit_kind": pa.array(["pdf_page"], type=pa.string()),
            "page_num": pa.array([0], type=pa.int32()),
        },
        schema=schemas.CHUNKS_SCHEMA,
    )
    pq.write_table(docs_table, data_dir / "docs.parquet")
    pq.write_table(chunks_table, data_dir / "chunks.parquet")


def _write_extraction_outputs(data_dir: Path, run_id: str) -> None:
    """Write a minimal entities parquet output for one run."""
    entities_table = pa.table(
        {
            "run_id": pa.array([run_id], type=pa.string()),
            "entity_id": pa.array([_hex32(21)], type=pa.string()),
            "doc_id": pa.array([_hex32(1)], type=pa.string()),
            "chunk_id": pa.array([_hex32(11)], type=pa.string()),
            "text": pa.array(["Per Hansen"], type=pa.string()),
            "normalized": pa.array(["per hansen"], type=pa.string()),
            "type": pa.array(["PER"], type=pa.string()),
            "char_start": pa.array([0], type=pa.int32()),
            "char_end": pa.array([10], type=pa.int32()),
            "context": pa.array(["Per Hansen forklarte seg."], type=pa.string()),
            "count": pa.array([1], type=pa.int32()),
            "positions": pa.array(
                [[{
                    "chunk_id": _hex32(11),
                    "char_start": 0,
                    "char_end": 10,
                    "page_num": 0,
                    "source_unit_kind": "pdf_page",
                }]],
                type=pa.list_(schemas.POSITION_STRUCT_TYPE),
            ),
        },
        schema=schemas.ENTITIES_SCHEMA,
    )
    pq.write_table(entities_table, data_dir / "entities.parquet")


def _write_empty_entities_output(data_dir: Path) -> None:
    """Write an empty but schema-valid entities parquet file."""
    pq.write_table(schemas.ENTITIES_SCHEMA.empty_table(), data_dir / "entities.parquet")


def _write_blocking_outputs(data_dir: Path, run_id: str) -> None:
    """Write minimal blocking outputs for one run."""
    candidate_pairs_table = pa.table(
        {
            "run_id": pa.array([run_id], type=pa.string()),
            "entity_id_a": pa.array([_hex32(21)], type=pa.string()),
            "entity_id_b": pa.array([_hex32(22)], type=pa.string()),
            "blocking_methods": pa.array([["faiss"]], type=pa.list_(pa.string())),
            "blocking_source": pa.array(["faiss"], type=pa.string()),
            "blocking_method_count": pa.array([1], type=pa.int8()),
        },
        schema=schemas.CANDIDATE_PAIRS_SCHEMA,
    )
    pq.write_table(candidate_pairs_table, data_dir / "candidate_pairs.parquet")
    (data_dir / "handoff_manifest.json").write_text(
        json.dumps({"run_id": run_id, "candidate_count": 1}),
        encoding="utf-8",
    )


def _write_matching_outputs(data_dir: Path, run_id: str) -> None:
    """Write minimal features and scored-pairs outputs for one run."""
    features_path = get_features_output_path(data_dir, run_id)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"run_id": pa.array([run_id], type=pa.string())}),
        features_path,
    )

    scored_pairs_path = get_scored_pairs_output_path(data_dir, run_id)
    pq.write_table(
        pa.table({"run_id": pa.array([run_id], type=pa.string())}),
        scored_pairs_path,
    )
    get_scoring_metadata_path(data_dir, run_id).write_text(
        json.dumps({"run_id": run_id}),
        encoding="utf-8",
    )


def _write_resolution_outputs(data_dir: Path, run_id: str) -> None:
    """Write minimal resolution outputs for one run."""
    resolved_entities_path = get_resolved_entities_output_path(data_dir, run_id)
    resolved_entities_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"run_id": pa.array([run_id], type=pa.string())}),
        resolved_entities_path,
    )
    get_clusters_output_path(data_dir, run_id).write_text(
        json.dumps({"cluster_count": 1, "clusters": [{"cluster_id": "c1"}]}),
        encoding="utf-8",
    )
    get_resolution_diagnostics_path(data_dir, run_id).write_text(
        json.dumps({"cluster_count": 1}),
        encoding="utf-8",
    )
    get_resolution_components_path(data_dir, run_id).write_text(
        json.dumps({"component_count": 1}),
        encoding="utf-8",
    )


def test_run_pipeline_calls_stages_in_order_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    case_root = tmp_path / "case"
    case_root.mkdir()
    data_dir = tmp_path / "data"
    run_id = "pipeline_run_001"
    calls: list[tuple[str, str]] = []
    shap_flags: list[bool] = []

    def fake_ingestion(case_root_arg: Path, data_dir_arg: Path, run_id: str | None = None) -> str:
        calls.append(("ingestion", run_id or ""))
        _write_ingestion_outputs(data_dir_arg, run_id or "")
        return run_id or ""

    def fake_extraction(data_dir_arg: Path, run_id_arg: str) -> str:
        calls.append(("extraction", run_id_arg))
        _write_extraction_outputs(data_dir_arg, run_id_arg)
        return run_id_arg

    def fake_blocking(data_dir_arg: Path, run_id_arg: str) -> str:
        calls.append(("blocking", run_id_arg))
        _write_blocking_outputs(data_dir_arg, run_id_arg)
        return run_id_arg

    def fake_features(data_dir_arg: Path, run_id_arg: str) -> list[dict[str, str]]:
        calls.append(("matching_features", run_id_arg))
        features_path = get_features_output_path(data_dir_arg, run_id_arg)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"run_id": pa.array([run_id_arg], type=pa.string())}), features_path)
        return [{"run_id": run_id_arg}]

    def fake_scoring(data_dir_arg: Path, run_id_arg: str, **kwargs: object) -> list[dict[str, str]]:
        calls.append(("matching_scoring", run_id_arg))
        shap_flags.append(bool(kwargs.get("enable_shap")))
        scored_pairs_path = get_scored_pairs_output_path(data_dir_arg, run_id_arg)
        pq.write_table(pa.table({"run_id": pa.array([run_id_arg], type=pa.string())}), scored_pairs_path)
        get_scoring_metadata_path(data_dir_arg, run_id_arg).write_text(
            json.dumps({"run_id": run_id_arg}),
            encoding="utf-8",
        )
        return [{"run_id": run_id_arg}]

    def fake_resolution(data_dir_arg: Path, run_id_arg: str) -> dict[str, int]:
        calls.append(("resolution", run_id_arg))
        _write_resolution_outputs(data_dir_arg, run_id_arg)
        return {"cluster_count": 1}

    monkeypatch.setattr(pipeline, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(pipeline, "run_blocking", fake_blocking)
    monkeypatch.setattr(pipeline, "run_features", fake_features)
    monkeypatch.setattr(pipeline, "run_scoring", fake_scoring)
    monkeypatch.setattr(pipeline, "run_resolution", fake_resolution)

    summary = pipeline.run_pipeline(case_root, data_dir, run_id=run_id, enable_shap=True)
    summary_path = pipeline.get_pipeline_summary_path(data_dir, run_id)
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [stage_name for stage_name, _ in calls] == pipeline.STAGE_ORDER
    assert all(stage_run_id == run_id for _, stage_run_id in calls)
    assert shap_flags == [True]
    assert summary["status"] == "succeeded"
    assert persisted["status"] == "succeeded"
    assert persisted["run_id"] == run_id
    assert persisted["counts"] == {
        "docs": 1,
        "chunks": 1,
        "entities": 1,
        "candidate_pairs": 1,
        "features": 1,
        "scored_pairs": 1,
        "resolved_entities": 1,
        "clusters": 1,
    }
    assert [stage["stage"] for stage in persisted["stages"]] == pipeline.STAGE_ORDER
    assert all(stage["success"] is True for stage in persisted["stages"])
    assert [stage["status"] for stage in persisted["stages"]] == ["succeeded"] * len(pipeline.STAGE_ORDER)
    assert persisted["artifacts"]["pipeline_summary_path"] == str(summary_path)
    assert Path(persisted["artifacts"]["clusters_path"]).exists()
    assert "Starting stage=ingestion" in caplog.text
    assert "Finished stage=resolution success=true" in caplog.text


def test_run_pipeline_writes_failed_summary_and_stops_on_first_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    data_dir = tmp_path / "data"
    run_id = "pipeline_run_fail"
    calls: list[str] = []

    def fake_ingestion(case_root_arg: Path, data_dir_arg: Path, run_id: str | None = None) -> str:
        calls.append("ingestion")
        _write_ingestion_outputs(data_dir_arg, run_id or "")
        return run_id or ""

    def fake_extraction(_data_dir_arg: Path, run_id_arg: str) -> str:
        calls.append("extraction")
        raise ValueError(f"bad extraction for {run_id_arg}")

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("later stages should not be called after extraction failure")

    monkeypatch.setattr(pipeline, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(pipeline, "run_blocking", unexpected)
    monkeypatch.setattr(pipeline, "run_features", unexpected)
    monkeypatch.setattr(pipeline, "run_scoring", unexpected)
    monkeypatch.setattr(pipeline, "run_resolution", unexpected)

    with pytest.raises(RuntimeError, match="stage 'extraction'"):
        pipeline.run_pipeline(case_root, data_dir, run_id=run_id)

    summary_path = pipeline.get_pipeline_summary_path(data_dir, run_id)
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))

    assert calls == ["ingestion", "extraction"]
    assert persisted["status"] == "failed"
    assert persisted["failed_stage"] == "extraction"
    assert persisted["error"] == f"ValueError: bad extraction for {run_id}"
    assert [stage["stage"] for stage in persisted["stages"]] == ["ingestion", "extraction"]
    assert persisted["stages"][0]["success"] is True
    assert persisted["stages"][1]["success"] is False
    assert persisted["counts"]["docs"] == 1
    assert persisted["counts"]["chunks"] == 1
    assert persisted["counts"]["entities"] is None


def test_run_pipeline_fails_early_for_empty_case_root_and_surfaces_reason(
    tmp_path: Path,
) -> None:
    case_root = tmp_path / "empty_case"
    case_root.mkdir()
    data_dir = tmp_path / "data"
    run_id = "pipeline_empty_input"

    with pytest.raises(
        RuntimeError,
        match="stage 'ingestion'.*No PDF/DOCX files found under case_root",
    ):
        pipeline.run_pipeline(case_root, data_dir, run_id=run_id)

    summary_path = pipeline.get_pipeline_summary_path(data_dir, run_id)
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))

    assert persisted["status"] == "failed"
    assert persisted["failed_stage"] == "ingestion"
    assert persisted["error"] == (
        f"ValueError: No PDF/DOCX files found under case_root: {case_root.resolve()}"
    )
    assert [stage["stage"] for stage in persisted["stages"]] == ["ingestion"]
    assert persisted["stages"][0]["success"] is False
    assert persisted["counts"]["docs"] is None
    assert persisted["counts"]["chunks"] is None


def test_main_forwards_cli_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    case_root = tmp_path / "case"
    case_root.mkdir()
    data_dir = tmp_path / "data"

    def fake_run_pipeline(
        case_root: str,
        data_dir: str,
        run_id: str | None = None,
        *,
        enable_shap: bool = False,
    ) -> dict[str, str]:
        captured["case_root"] = case_root
        captured["data_dir"] = data_dir
        captured["run_id"] = run_id
        captured["enable_shap"] = enable_shap
        return {"run_id": run_id or "generated"}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    exit_code = pipeline.main(
        [
            "--case-root",
            str(case_root),
            "--data-dir",
            str(data_dir),
            "--run-id",
            "cli_run_001",
            "--enable-shap",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "case_root": str(case_root),
        "data_dir": str(data_dir),
        "run_id": "cli_run_001",
        "enable_shap": True,
    }


def test_run_pipeline_stops_cleanly_when_extraction_finds_no_mentions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    data_dir = tmp_path / "data"
    run_id = "pipeline_no_mentions"
    calls: list[str] = []

    def fake_ingestion(case_root_arg: Path, data_dir_arg: Path, run_id: str | None = None) -> str:
        calls.append("ingestion")
        _write_ingestion_outputs(data_dir_arg, run_id or "")
        return run_id or ""

    def fake_extraction(_data_dir_arg: Path, run_id_arg: str) -> str:
        calls.append("extraction")
        return run_id_arg

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("downstream stages should be skipped after no-results extraction")

    monkeypatch.setattr(pipeline, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(pipeline, "run_blocking", unexpected)
    monkeypatch.setattr(pipeline, "run_features", unexpected)
    monkeypatch.setattr(pipeline, "run_scoring", unexpected)
    monkeypatch.setattr(pipeline, "run_resolution", unexpected)

    summary = pipeline.run_pipeline(case_root, data_dir, run_id=run_id)

    assert calls == ["ingestion", "extraction"]
    assert summary["status"] == "succeeded_no_results"
    assert summary["counts"]["docs"] == 1
    assert summary["counts"]["chunks"] == 1
    assert summary["counts"]["entities"] is None
    assert summary["stages"][1]["outcome"] == "no_results"
    assert [stage["status"] for stage in summary["stages"]] == [
        "succeeded",
        "succeeded",
        "skipped",
        "skipped",
        "skipped",
        "skipped",
    ]


def test_run_pipeline_stops_cleanly_when_extraction_writes_zero_entities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    data_dir = tmp_path / "data"
    run_id = "pipeline_zero_entities"

    def fake_ingestion(case_root_arg: Path, data_dir_arg: Path, run_id: str | None = None) -> str:
        _write_ingestion_outputs(data_dir_arg, run_id or "")
        return run_id or ""

    def fake_extraction(data_dir_arg: Path, run_id_arg: str) -> str:
        _write_empty_entities_output(data_dir_arg)
        return run_id_arg

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("downstream stages should be skipped after zero-entity extraction")

    monkeypatch.setattr(pipeline, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(pipeline, "run_blocking", unexpected)
    monkeypatch.setattr(pipeline, "run_features", unexpected)
    monkeypatch.setattr(pipeline, "run_scoring", unexpected)
    monkeypatch.setattr(pipeline, "run_resolution", unexpected)

    summary = pipeline.run_pipeline(case_root, data_dir, run_id=run_id)

    assert summary["status"] == "succeeded_no_results"
    assert summary["counts"]["entities"] == 0
    assert summary["stages"][1]["counts"] == {"entities": 0}
    assert summary["stages"][1]["outcome"] == "no_results"
    assert summary["stages"][2]["status"] == "skipped"


def test_run_pipeline_stops_cleanly_when_blocking_finds_no_candidate_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    data_dir = tmp_path / "data"
    run_id = "pipeline_no_candidates"

    def fake_ingestion(case_root_arg: Path, data_dir_arg: Path, run_id: str | None = None) -> str:
        _write_ingestion_outputs(data_dir_arg, run_id or "")
        return run_id or ""

    def fake_extraction(data_dir_arg: Path, run_id_arg: str) -> str:
        _write_extraction_outputs(data_dir_arg, run_id_arg)
        return run_id_arg

    def fake_blocking(_data_dir_arg: Path, run_id_arg: str) -> str:
        return run_id_arg

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("matching and resolution should be skipped after zero candidates")

    monkeypatch.setattr(pipeline, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(pipeline, "run_blocking", fake_blocking)
    monkeypatch.setattr(pipeline, "run_features", unexpected)
    monkeypatch.setattr(pipeline, "run_scoring", unexpected)
    monkeypatch.setattr(pipeline, "run_resolution", unexpected)

    summary = pipeline.run_pipeline(case_root, data_dir, run_id=run_id)

    assert summary["status"] == "succeeded_no_results"
    assert summary["counts"]["candidate_pairs"] is None
    assert summary["stages"][2]["counts"] == {"candidate_pairs": 0}
    assert summary["stages"][2]["outcome"] == "no_results"
    assert [stage["status"] for stage in summary["stages"][3:]] == [
        "skipped",
        "skipped",
        "skipped",
    ]


def test_summarize_stage_allows_zero_entities_without_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stage_spec = pipeline_support.build_stage_specs(
        case_root=tmp_path / "case",
        data_dir=data_dir,
        run_id="run_zero_entities",
        enable_shap=False,
        run_ingestion=lambda *_args, **_kwargs: "run_zero_entities",
        run_extraction=lambda *_args, **_kwargs: "run_zero_entities",
        run_blocking=lambda *_args, **_kwargs: "run_zero_entities",
        run_features=lambda *_args, **_kwargs: None,
        run_scoring=lambda *_args, **_kwargs: None,
        run_resolution=lambda *_args, **_kwargs: {"cluster_count": 0},
        run_stage_with_run_id=lambda *_args, **_kwargs: "run_zero_entities",
    )[1]

    summary = pipeline_support.summarize_stage(stage_spec, "run_zero_entities")

    assert summary["counts"] == {"entities": 0}
    assert summary["outcome"] == "no_results"
    assert summary["artifacts"]["entities_path"].endswith("entities.parquet")


def test_summarize_stage_reads_zero_row_entities_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_empty_entities_output(data_dir)
    stage_spec = pipeline_support.build_stage_specs(
        case_root=tmp_path / "case",
        data_dir=data_dir,
        run_id="run_zero_file",
        enable_shap=False,
        run_ingestion=lambda *_args, **_kwargs: "run_zero_file",
        run_extraction=lambda *_args, **_kwargs: "run_zero_file",
        run_blocking=lambda *_args, **_kwargs: "run_zero_file",
        run_features=lambda *_args, **_kwargs: None,
        run_scoring=lambda *_args, **_kwargs: None,
        run_resolution=lambda *_args, **_kwargs: {"cluster_count": 0},
        run_stage_with_run_id=lambda *_args, **_kwargs: "run_zero_file",
    )[1]

    summary = pipeline_support.summarize_stage(stage_spec, "run_zero_file")

    assert summary["counts"] == {"entities": 0}
    assert summary["outcome"] == "no_results"
