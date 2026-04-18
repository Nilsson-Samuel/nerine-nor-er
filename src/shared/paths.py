"""Centralized per-run artifact path helpers for all pipeline stages.

Every stage writes artifacts under:
    data_dir / runs / rid_{base64(run_id)} / {stage_name} /

This module is the single source of truth for the run-directory layout,
the run-id encoding scheme, and the per-stage output directory builders.
"""

from __future__ import annotations

import base64
from pathlib import Path


RUN_OUTPUTS_DIRNAME = "runs"

INGESTION_STAGE_DIRNAME = "ingestion"
EXTRACTION_STAGE_DIRNAME = "extraction"
BLOCKING_STAGE_DIRNAME = "blocking"
MATCHING_STAGE_DIRNAME = "matching"
RESOLUTION_STAGE_DIRNAME = "resolution"
EVALUATION_STAGE_DIRNAME = "evaluation"
PIPELINE_STAGE_DIRNAME = "pipeline"

EVALUATION_REPORT_FILENAME = "evaluation_report.json"
EVALUATION_LABELS_FILENAME = "labels.parquet"


def _encode_run_id_path_segment(run_id: str) -> str:
    """Encode run_id into one cross-platform-safe directory name."""
    encoded = base64.urlsafe_b64encode(run_id.encode("utf-8")).decode("ascii").rstrip("=")
    return f"rid_{encoded}"


def get_run_root_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run root directory: data_dir/runs/rid_{encoded}/."""
    return Path(data_dir) / RUN_OUTPUTS_DIRNAME / _encode_run_id_path_segment(run_id)


def get_ingestion_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run ingestion output directory."""
    return get_run_root_dir(data_dir, run_id) / INGESTION_STAGE_DIRNAME


def get_extraction_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run extraction output directory."""
    return get_run_root_dir(data_dir, run_id) / EXTRACTION_STAGE_DIRNAME


def get_blocking_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run blocking output directory."""
    return get_run_root_dir(data_dir, run_id) / BLOCKING_STAGE_DIRNAME


def get_matching_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run matching output directory."""
    return get_run_root_dir(data_dir, run_id) / MATCHING_STAGE_DIRNAME


def get_resolution_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run resolution output directory."""
    return get_run_root_dir(data_dir, run_id) / RESOLUTION_STAGE_DIRNAME


def get_evaluation_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run evaluation output directory."""
    return get_run_root_dir(data_dir, run_id) / EVALUATION_STAGE_DIRNAME


def get_evaluation_report_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run evaluation report path."""
    return get_evaluation_run_output_dir(data_dir, run_id) / EVALUATION_REPORT_FILENAME


def get_evaluation_labels_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run matcher-label bridge path."""
    return get_evaluation_run_output_dir(data_dir, run_id) / EVALUATION_LABELS_FILENAME


def get_pipeline_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run pipeline output directory."""
    return get_run_root_dir(data_dir, run_id) / PIPELINE_STAGE_DIRNAME
