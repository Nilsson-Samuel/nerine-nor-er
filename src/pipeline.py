"""Top-level pipeline CLI for one end-to-end entity-resolution run.

Generates one run ID at the workflow boundary, executes the real stage
entrypoints in order, captures stage timing and output counts, and writes one
per-run pipeline summary JSON under the existing runs/<encoded>/ tree.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from src.pipeline_support import (
    build_stage_specs,
    collect_artifact_paths,
    collect_counts,
    summarize_stage,
    write_summary,
)
from src.matching.writer import (
    RUN_OUTPUTS_DIRNAME,
    _encode_run_id_path_segment,
)


logger = logging.getLogger(__name__)

PIPELINE_STAGE_DIRNAME = "pipeline"
PIPELINE_SUMMARY_FILENAME = "pipeline_summary.json"
STAGE_ORDER = [
    "ingestion",
    "extraction",
    "blocking",
    "matching_features",
    "matching_scoring",
    "resolution",
]


def run_ingestion(*args: Any, **kwargs: Any) -> str:
    """Import and run the ingestion stage on demand."""
    from src.ingestion.run import run_ingestion as _run_ingestion

    return _run_ingestion(*args, **kwargs)


def run_extraction(*args: Any, **kwargs: Any) -> str:
    """Import and run the extraction stage on demand."""
    from src.extraction.run import run_extraction as _run_extraction

    return _run_extraction(*args, **kwargs)


def run_blocking(*args: Any, **kwargs: Any) -> str:
    """Import and run the blocking stage on demand."""
    from src.blocking.run import run_blocking as _run_blocking

    return _run_blocking(*args, **kwargs)


def run_features(*args: Any, **kwargs: Any) -> Any:
    """Import and run the matching feature stage on demand."""
    from src.matching.run import run_features as _run_features

    return _run_features(*args, **kwargs)


def run_scoring(*args: Any, **kwargs: Any) -> Any:
    """Import and run the matching scoring stage on demand."""
    from src.matching.run import run_scoring as _run_scoring

    return _run_scoring(*args, **kwargs)


def run_resolution(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Import and run the resolution stage on demand."""
    from src.resolution.run import run_resolution as _run_resolution

    return _run_resolution(*args, **kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for one end-to-end pipeline run."""
    parser = argparse.ArgumentParser(description="Run the Nerine pipeline end-to-end.")
    parser.add_argument("--case-root", required=True, help="Directory containing input PDF/DOCX files.")
    parser.add_argument("--data-dir", required=True, help="Directory for pipeline outputs.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier to reuse.")
    parser.add_argument(
        "--enable-shap",
        action="store_true",
        help="Generate SHAP top-5 explanations during matching scoring.",
    )
    return parser.parse_args(argv)


def get_pipeline_run_output_dir(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run pipeline output directory."""
    return (
        Path(data_dir)
        / RUN_OUTPUTS_DIRNAME
        / _encode_run_id_path_segment(run_id)
        / PIPELINE_STAGE_DIRNAME
    )


def get_pipeline_summary_path(data_dir: Path | str, run_id: str) -> Path:
    """Build the per-run pipeline summary JSON path."""
    return get_pipeline_run_output_dir(data_dir, run_id) / PIPELINE_SUMMARY_FILENAME


def run_pipeline(
    case_root: Path | str,
    data_dir: Path | str,
    run_id: str | None = None,
    *,
    enable_shap: bool = False,
) -> dict[str, Any]:
    """Run the real stage entrypoints in order and write one summary JSON."""
    case_root = Path(case_root).resolve()
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    effective_run_id = run_id or uuid4().hex[:32]
    started_at = _utc_now()
    summary_path = get_pipeline_summary_path(data_dir, effective_run_id)
    stages: list[dict[str, Any]] = []
    stage_specs = build_stage_specs(
        case_root=case_root,
        data_dir=data_dir,
        run_id=effective_run_id,
        enable_shap=enable_shap,
        run_ingestion=run_ingestion,
        run_extraction=run_extraction,
        run_blocking=run_blocking,
        run_features=run_features,
        run_scoring=run_scoring,
        run_resolution=run_resolution,
        run_stage_with_run_id=_run_stage_with_run_id,
    )
    failed_stage: str | None = None
    error_message: str | None = None
    current_stage = "initialization"
    pipeline_status = "succeeded"

    try:
        if not case_root.exists():
            raise FileNotFoundError(f"case_root does not exist: {case_root}")
        if not case_root.is_dir():
            raise NotADirectoryError(f"case_root is not a directory: {case_root}")

        for index, spec in enumerate(stage_specs):
            current_stage = str(spec["name"])
            record = _execute_stage(
                stage_name=current_stage,
                action=spec["action"],
                summarize=lambda result, spec=spec: summarize_stage(spec, result),
                stages=stages,
            )
            if record["outcome"] == "no_results":
                pipeline_status = "succeeded_no_results"
                _append_skipped_stages(
                    stages=stages,
                    stage_specs=stage_specs[index + 1:],
                    reason=f"upstream stage '{current_stage}' produced no rows",
                )
                break
    except Exception as exc:
        failed_stage = current_stage
        error_message = f"{type(exc).__name__}: {exc}"
        pipeline_status = "failed"
        raise RuntimeError(
            f"Pipeline failed in stage '{current_stage}' for run_id={effective_run_id}: {exc}. "
            f"Summary: {summary_path}"
        ) from exc
    finally:
        finished_at = _utc_now()
        summary = {
            "run_id": effective_run_id,
            "status": pipeline_status,
            "failed_stage": failed_stage,
            "error": error_message,
            "case_root": str(case_root),
            "data_dir": str(data_dir),
            "started_at": _isoformat(started_at),
            "finished_at": _isoformat(finished_at),
            "elapsed_seconds": round((finished_at - started_at).total_seconds(), 3),
            "enable_shap": enable_shap,
            "stage_order": [str(spec["name"]) for spec in stage_specs],
            "stages": stages,
            "counts": collect_counts(data_dir, effective_run_id),
            "artifacts": collect_artifact_paths(data_dir, effective_run_id, summary_path),
        }
        try:
            write_summary(summary_path, summary)
        except Exception:
            logger.exception("Failed to write pipeline summary to %s", summary_path)

    logger.info(
        "Pipeline complete for run_id=%s summary=%s",
        effective_run_id,
        summary_path,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for one pipeline run."""
    args = parse_args(argv)
    _configure_logging()

    try:
        run_pipeline(
            case_root=args.case_root,
            data_dir=args.data_dir,
            run_id=args.run_id,
            enable_shap=args.enable_shap,
        )
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


def _configure_logging() -> None:
    """Install one basic logging handler when none exists."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _execute_stage(
    *,
    stage_name: str,
    action: Callable[[], Any],
    summarize: Callable[[Any], dict[str, Any]],
    stages: list[dict[str, Any]],
) -> Any:
    """Run one stage, capture timing and summary fields, then append the record."""
    started_at = _utc_now()
    t0 = time.monotonic()
    logger.info("Starting stage=%s", stage_name)
    record: dict[str, Any] = {
        "stage": stage_name,
        "started_at": _isoformat(started_at),
    }

    try:
        result = action()
        stage_summary = summarize(result)
        should_stop = bool(stage_summary.pop("_should_stop", False))
        record["success"] = True
        record["status"] = "succeeded"
        record.update(stage_summary)
        logger.info(
            "Finished stage=%s success=true elapsed_seconds=%.3f",
            stage_name,
            time.monotonic() - t0,
        )
        if should_stop:
            logger.info("Stopping pipeline after stage=%s outcome=no_results", stage_name)
        return record
    except Exception as exc:
        record["success"] = False
        record["status"] = "failed"
        record["outcome"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Finished stage=%s success=false", stage_name)
        raise
    finally:
        finished_at = _utc_now()
        record["finished_at"] = _isoformat(finished_at)
        record["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        stages.append(record)


def _append_skipped_stages(
    *,
    stages: list[dict[str, Any]],
    stage_specs: Sequence[dict[str, Any]],
    reason: str,
) -> None:
    """Append skipped stage records after one clean no-results boundary."""
    skipped_at = _utc_now()
    for spec in stage_specs:
        stages.append(
            {
                "stage": str(spec["name"]),
                "status": "skipped",
                "success": None,
                "reason": reason,
                "started_at": _isoformat(skipped_at),
                "finished_at": _isoformat(skipped_at),
                "elapsed_seconds": 0.0,
            }
        )


def _run_stage_with_run_id(
    stage_fn: Callable[..., str],
    *args: Path,
    run_id: str,
) -> str:
    """Run a stage that returns run_id and assert it preserves the boundary run_id."""
    returned_run_id = stage_fn(*args, run_id)
    if returned_run_id != run_id:
        raise RuntimeError(
            f"{stage_fn.__name__} changed run_id from {run_id} to {returned_run_id}"
        )
    return returned_run_id


def _utc_now() -> datetime:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    """Serialize one UTC timestamp consistently for JSON artifacts."""
    return value.isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
