"""Support helpers for pipeline stage specs, summary assembly, and artifact counts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq

from src.matching.writer import (
    get_features_output_path,
    get_scored_pairs_output_path,
    get_scoring_metadata_path,
)
from src.shared.paths import get_run_root_dir
from src.resolution.writer import (
    get_clusters_output_path,
    get_resolution_components_path,
    get_resolution_diagnostics_path,
    get_resolved_entities_output_path,
)


def build_stage_specs(
    *,
    case_root: Path,
    data_dir: Path,
    run_id: str,
    enable_shap: bool,
    run_ingestion: Callable[..., str],
    run_extraction: Callable[..., str],
    run_blocking: Callable[..., str],
    run_features: Callable[..., Any],
    run_scoring: Callable[..., Any],
    run_resolution: Callable[..., dict[str, Any]],
    run_stage_with_run_id: Callable[..., str],
) -> list[dict[str, Any]]:
    """Build one small stage-spec table for the sequential pipeline run."""
    return [
        {
            "name": "ingestion",
            "action": lambda: run_stage_with_run_id(
                run_ingestion,
                case_root,
                data_dir,
                run_id=run_id,
            ),
            "counts": [
                parquet_count("docs", data_dir / "docs.parquet", run_id=run_id),
                parquet_count(
                    "chunks",
                    data_dir / "chunks.parquet",
                    run_id=run_id,
                    allow_zero=True,
                    stop_on_zero=True,
                ),
            ],
            "artifacts": [
                artifact_path("docs_path", data_dir / "docs.parquet", required=True),
                artifact_path(
                    "chunks_path",
                    data_dir / "chunks.parquet",
                    required_if_results=True,
                ),
            ],
        },
        {
            "name": "extraction",
            "action": lambda: run_stage_with_run_id(run_extraction, data_dir, run_id=run_id),
            "counts": [
                parquet_count(
                    "entities",
                    data_dir / "entities.parquet",
                    run_id=run_id,
                    allow_zero=True,
                    stop_on_zero=True,
                )
            ],
            "artifacts": [
                artifact_path(
                    "entities_path",
                    data_dir / "entities.parquet",
                    required_if_results=True,
                )
            ],
        },
        {
            "name": "blocking",
            "action": lambda: run_stage_with_run_id(run_blocking, data_dir, run_id=run_id),
            "counts": [
                parquet_count(
                    "candidate_pairs",
                    data_dir / "candidate_pairs.parquet",
                    run_id=run_id,
                    allow_zero=True,
                    stop_on_zero=True,
                )
            ],
            "artifacts": [
                artifact_path(
                    "candidate_pairs_path",
                    data_dir / "candidate_pairs.parquet",
                    required_if_results=True,
                ),
                artifact_path(
                    "handoff_manifest_path",
                    data_dir / "handoff_manifest.json",
                    required_if_results=True,
                ),
            ],
        },
        {
            "name": "matching_features",
            "action": lambda: run_features(data_dir, run_id),
            "counts": [
                parquet_count(
                    "features",
                    get_features_output_path(data_dir, run_id),
                    allow_zero=True,
                    stop_on_zero=True,
                )
            ],
            "artifacts": [
                artifact_path(
                    "features_path",
                    get_features_output_path(data_dir, run_id),
                    required_if_results=True,
                )
            ],
        },
        {
            "name": "matching_scoring",
            "action": lambda: run_scoring(data_dir, run_id, enable_shap=enable_shap),
            "counts": [
                parquet_count(
                    "scored_pairs",
                    get_scored_pairs_output_path(data_dir, run_id),
                    allow_zero=True,
                    stop_on_zero=True,
                )
            ],
            "artifacts": [
                artifact_path(
                    "scored_pairs_path",
                    get_scored_pairs_output_path(data_dir, run_id),
                    required_if_results=True,
                ),
                artifact_path(
                    "matching_scoring_metadata_path",
                    get_scoring_metadata_path(data_dir, run_id),
                    required_if_results=True,
                ),
            ],
        },
        {
            "name": "resolution",
            "action": lambda: run_resolution(data_dir, run_id),
            "counts": [
                parquet_count(
                    "resolved_entities",
                    get_resolved_entities_output_path(data_dir, run_id),
                ),
                json_count("clusters", get_clusters_output_path(data_dir, run_id)),
            ],
            "artifacts": [
                artifact_path(
                    "resolved_entities_path",
                    get_resolved_entities_output_path(data_dir, run_id),
                    required_if_results=True,
                ),
                artifact_path(
                    "clusters_path",
                    get_clusters_output_path(data_dir, run_id),
                    required_if_results=True,
                ),
                artifact_path(
                    "resolution_diagnostics_path",
                    get_resolution_diagnostics_path(data_dir, run_id),
                    required_if_results=True,
                ),
                artifact_path(
                    "resolution_components_path",
                    get_resolution_components_path(data_dir, run_id),
                    required_if_results=True,
                ),
            ],
            "count_overrides": {
                "clusters": lambda result: int(result.get("cluster_count", 0)),
            },
        },
    ]


def artifact_path(
    name: str,
    path: Path,
    *,
    required: bool = False,
    required_if_results: bool = False,
) -> dict[str, Any]:
    """Build one artifact-path specification."""
    return {
        "name": name,
        "path": path,
        "required": required,
        "required_if_results": required_if_results,
    }


def parquet_count(
    name: str,
    path: Path,
    run_id: str | None = None,
    *,
    allow_zero: bool = False,
    stop_on_zero: bool = False,
) -> dict[str, Any]:
    """Build one parquet-count specification."""
    return {
        "name": name,
        "path": path,
        "kind": "parquet",
        "run_id": run_id,
        "allow_zero": allow_zero,
        "stop_on_zero": stop_on_zero,
    }


def json_count(
    name: str,
    path: Path,
    *,
    allow_zero: bool = False,
    stop_on_zero: bool = False,
) -> dict[str, Any]:
    """Build one JSON-count specification."""
    return {
        "name": name,
        "path": path,
        "kind": "json",
        "allow_zero": allow_zero,
        "stop_on_zero": stop_on_zero,
    }


def summarize_stage(stage_spec: dict[str, Any], result: Any) -> dict[str, Any]:
    """Validate stage artifacts and collect one compact summary payload."""
    stage_name = str(stage_spec["name"])
    count_overrides = stage_spec.get("count_overrides", {})
    counts = {
        str(spec["name"]): resolve_count(
            stage_name=stage_name,
            count_spec=spec,
            override_count=resolve_override(count_overrides.get(str(spec["name"])), result),
        )
        for spec in stage_spec["counts"]
    }
    no_results = any(
        bool(spec.get("stop_on_zero")) and counts[str(spec["name"])] == 0
        for spec in stage_spec["counts"]
    )
    artifacts = stringify_artifacts(
        stage_name=stage_name,
        artifact_specs=stage_spec["artifacts"],
        require_result_artifacts=not no_results,
    )
    return {
        "counts": counts,
        "artifacts": artifacts,
        "outcome": "no_results" if no_results else "completed",
        "_should_stop": no_results,
    }


def resolve_override(
    override_fn: Callable[[Any], int] | None,
    result: Any,
) -> int | None:
    """Resolve one optional in-memory count override from a stage result."""
    if override_fn is None:
        return None
    count = int(override_fn(result))
    return count if count > 0 else None


def resolve_count(
    *,
    stage_name: str,
    count_spec: dict[str, Any],
    override_count: int | None = None,
) -> int:
    """Resolve one artifact count or raise with short stage context."""
    count_name = str(count_spec["name"])
    path = Path(count_spec["path"])
    if override_count is not None:
        return override_count

    if count_spec["kind"] == "parquet":
        count = count_parquet_rows(path, count_spec.get("run_id"))
    else:
        count = count_clusters(path)

    if count is None and count_spec.get("allow_zero"):
        return 0
    if count is None:
        raise RuntimeError(f"{stage_name} did not write readable output at {path}")
    if count < 1 and not count_spec.get("allow_zero"):
        raise RuntimeError(f"{stage_name} produced 0 {count_name} rows at {path}")
    return count


def stringify_artifacts(
    *,
    stage_name: str,
    artifact_specs: list[dict[str, Any]],
    require_result_artifacts: bool,
) -> dict[str, str]:
    """Assert required artifact existence and return stringified paths for JSON output."""
    missing = [
        str(spec["path"])
        for spec in artifact_specs
        if _artifact_is_required(spec, require_result_artifacts)
        and not Path(spec["path"]).exists()
    ]
    if missing:
        raise RuntimeError(f"{stage_name} missing expected artifacts: {', '.join(missing)}")
    return {str(spec["name"]): str(spec["path"]) for spec in artifact_specs}


def _artifact_is_required(
    artifact_spec: dict[str, Any],
    require_result_artifacts: bool,
) -> bool:
    """Return True when one artifact must exist for the current stage outcome."""
    if artifact_spec.get("required"):
        return True
    return require_result_artifacts and bool(artifact_spec.get("required_if_results"))


def count_parquet_rows(path: Path, run_id: str | None = None) -> int | None:
    """Count rows from a parquet artifact, filtering by run_id when present."""
    if not path.exists():
        return None

    try:
        schema_names = set(pq.read_schema(path).names)
        if run_id is not None and "run_id" in schema_names:
            return int(
                pq.read_table(
                    path,
                    columns=["run_id"],
                    filters=[("run_id", "=", run_id)],
                ).num_rows
            )
        return int(pq.read_metadata(path).num_rows)
    except Exception:
        return None


def count_clusters(path: Path) -> int | None:
    """Count cluster rows from one JSON artifact."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "cluster_count" in payload:
            return int(payload["cluster_count"])
        return len(payload.get("clusters", []))
    except Exception:
        return None


def collect_counts(data_dir: Path, run_id: str) -> dict[str, int | None]:
    """Collect one compact end-of-run count snapshot across stage artifacts."""
    return {
        "docs": count_parquet_rows(data_dir / "docs.parquet", run_id),
        "chunks": count_parquet_rows(data_dir / "chunks.parquet", run_id),
        "entities": count_parquet_rows(data_dir / "entities.parquet", run_id),
        "candidate_pairs": count_parquet_rows(data_dir / "candidate_pairs.parquet", run_id),
        "features": count_parquet_rows(get_features_output_path(data_dir, run_id)),
        "scored_pairs": count_parquet_rows(get_scored_pairs_output_path(data_dir, run_id)),
        "resolved_entities": count_parquet_rows(get_resolved_entities_output_path(data_dir, run_id)),
        "clusters": count_clusters(get_clusters_output_path(data_dir, run_id)),
    }


def collect_artifact_paths(
    data_dir: Path,
    run_id: str,
    summary_path: Path,
) -> dict[str, str]:
    """Collect the main per-run artifact paths useful for debugging and HITL."""
    run_output_dir = get_run_root_dir(data_dir, run_id)
    return {
        "run_output_dir": str(run_output_dir),
        "pipeline_summary_path": str(summary_path),
        "features_path": str(get_features_output_path(data_dir, run_id)),
        "scored_pairs_path": str(get_scored_pairs_output_path(data_dir, run_id)),
        "matching_scoring_metadata_path": str(get_scoring_metadata_path(data_dir, run_id)),
        "resolved_entities_path": str(get_resolved_entities_output_path(data_dir, run_id)),
        "clusters_path": str(get_clusters_output_path(data_dir, run_id)),
        "resolution_diagnostics_path": str(get_resolution_diagnostics_path(data_dir, run_id)),
        "resolution_components_path": str(get_resolution_components_path(data_dir, run_id)),
        "hitl_app_path": str((Path(__file__).resolve().parent / "hitl" / "streamlit_app.py")),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write the per-run pipeline summary JSON with atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
