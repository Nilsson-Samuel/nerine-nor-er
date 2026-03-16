"""Phase-1 resolution orchestration for retained components and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.resolution.clustering import (
    build_phase1_components,
    load_entity_ids,
    load_scored_pairs,
    summarize_components,
)


RESOLUTION_COMPONENTS_FILENAME = "resolution_components.json"
RESOLUTION_DIAGNOSTICS_FILENAME = "resolution_diagnostics.json"


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write one JSON artifact with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_resolution(data_dir: Path | str, run_id: str) -> dict[str, Any]:
    """Run phase-1 resolution and persist component summaries plus diagnostics."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    entity_ids = load_entity_ids(data_dir, run_id)
    scored_pairs = load_scored_pairs(data_dir, run_id)
    components, diagnostics = build_phase1_components(run_id, scored_pairs, entity_ids)
    component_payload = {
        "run_id": run_id,
        "component_count": len(components),
        "components": summarize_components(components),
    }

    _write_json(component_payload, data_dir / RESOLUTION_COMPONENTS_FILENAME)
    _write_json(diagnostics, data_dir / RESOLUTION_DIAGNOSTICS_FILENAME)
    return diagnostics
