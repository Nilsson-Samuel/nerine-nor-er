"""Run status and diagnostics helpers for the HITL sidebar.

Provides safe loading of resolution diagnostics and compact summaries
for sidebar display.  All loaders are lenient — missing or corrupt files
produce None rather than exceptions, so the app never crashes on metadata gaps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.resolution.writer import get_resolution_diagnostics_path

logger = logging.getLogger(__name__)


def load_diagnostics_safe(data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Load resolution diagnostics for one run, returning None on failure.

    Args:
        data_dir: Root data directory containing per-run outputs.
        run_id: Pipeline run identifier.

    Returns:
        Parsed diagnostics dict, or None if missing or unreadable.
    """
    path = get_resolution_diagnostics_path(data_dir, run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load diagnostics for %s: %s", run_id, exc)
        return None


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
