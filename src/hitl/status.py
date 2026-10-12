"""Run status and diagnostics helpers for the HITL sidebar.

Provides safe loading of resolution diagnostics and scoring model versions,
and compact summaries for sidebar display.  All loaders are lenient — missing or corrupt files
produce None rather than exceptions, so the app never crashes on metadata gaps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.matching.writer import get_scoring_metadata_path
from src.resolution.writer import get_resolution_diagnostics_path

logger = logging.getLogger(__name__)

# Same name as src.matching.reranker.MODEL_METADATA_FILENAME. Not imported from
# there, because that module pulls LightGBM and scikit-learn into the UI.
INSTALLED_MODEL_METADATA_FILENAME = "reranker_model_metadata.json"


def _load_json_safe(path: Path) -> dict[str, Any] | None:
    """Parse one JSON object file, returning None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def load_diagnostics_safe(data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Load resolution diagnostics for one run, returning None on failure.

    Args:
        data_dir: Root data directory containing per-run outputs.
        run_id: Pipeline run identifier.

    Returns:
        Parsed diagnostics dict, or None if missing or unreadable.
    """
    return _load_json_safe(get_resolution_diagnostics_path(data_dir, run_id))


def _model_version(metadata: dict[str, Any] | None) -> str | None:
    """Pull a non-empty model_version string out of a metadata payload."""
    version = (metadata or {}).get("model_version")
    return str(version) if version else None


def load_run_model_version(data_dir: Path, run_id: str) -> str | None:
    """Return the reranker model version that scored one run, or None if unknown."""
    return _model_version(_load_json_safe(get_scoring_metadata_path(data_dir, run_id)))


def load_installed_model_version(data_dir: Path) -> str | None:
    """Return the version of the reranker model new runs in data_dir would use.

    Matching loads its model from the data dir by default, so this is the model
    a pipeline rerun would score with.
    """
    return _model_version(
        _load_json_safe(Path(data_dir) / INSTALLED_MODEL_METADATA_FILENAME)
    )


def model_version_warning(
    run_version: str | None,
    installed_version: str | None,
) -> str | None:
    """Explain a mismatch between the run's scoring model and the installed one.

    Returns None when the versions match or either one is unknown.
    """
    if run_version is None or installed_version is None:
        return None
    if run_version == installed_version:
        return None
    return (
        f"This run was scored with `{run_version}`, but the model installed in "
        f"the data dir is `{installed_version}`. Pair scores, clusters and "
        "routing reflect that model. Rerun the pipeline to evaluate the "
        "installed one."
    )


def diagnostics_sidebar_summary(diagnostics: dict[str, Any]) -> dict[str, str]:
    """Extract a compact key-value summary suitable for sidebar display.

    Args:
        diagnostics: Parsed resolution_diagnostics.json payload.

    Returns:
        Ordered dict with human-readable label→value pairs.
    """
    cluster_count = diagnostics.get("cluster_count", "?")
    singleton_rate = diagnostics.get("cluster_singleton_rate")
    singleton_str = f"{singleton_rate:.1%}" if singleton_rate is not None else "?"
    profile = diagnostics.get("selected_routing_profile", "?")

    return {
        "Clusters": str(cluster_count),
        "Singleton rate": singleton_str,
        "Routing profile": str(profile),
    }
