"""Writers and path helpers for final resolution artifacts."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.matching.writer import get_matching_run_output_dir
from src.shared.schemas import RESOLVED_ENTITIES_SCHEMA


RESOLUTION_STAGE_DIRNAME = "resolution"
RESOLUTION_COMPONENTS_FILENAME = "resolution_components.json"
RESOLUTION_DIAGNOSTICS_FILENAME = "resolution_diagnostics.json"
RESOLVED_ENTITIES_FILENAME = "resolved_entities.parquet"
CLUSTERS_FILENAME = "clusters.json"


def get_resolution_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run resolution output directory."""
    return get_matching_run_output_dir(data_dir, run_id).parent / RESOLUTION_STAGE_DIRNAME


def get_resolution_components_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run resolution components path."""
    return get_resolution_run_output_dir(data_dir, run_id) / RESOLUTION_COMPONENTS_FILENAME


def get_resolution_diagnostics_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run resolution diagnostics path."""
    return get_resolution_run_output_dir(data_dir, run_id) / RESOLUTION_DIAGNOSTICS_FILENAME


def get_resolved_entities_output_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run resolved entities parquet path."""
    return get_resolution_run_output_dir(data_dir, run_id) / RESOLVED_ENTITIES_FILENAME


def get_clusters_output_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run clusters lineage JSON path."""
    return get_resolution_run_output_dir(data_dir, run_id) / CLUSTERS_FILENAME


def build_resolved_entities_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build the strict-schema resolved entities table."""
    return pa.Table.from_pylist(rows, schema=RESOLVED_ENTITIES_SCHEMA)


def write_resolved_entities(table: pa.Table, path: Path | str) -> None:
    """Write resolved entities parquet to its per-run output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_resolution_json(payload: dict[str, Any], path: Path | str) -> None:
    """Write one JSON artifact with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _json_default(value: Any) -> str:
    """Serialize date-like values consistently for JSON artifacts."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
