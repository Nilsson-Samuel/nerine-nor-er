"""Blocking output writer — candidate_pairs.parquet and handoff_manifest.json.

Transforms unioned candidate records into a contract-valid parquet file and
generates a handoff manifest for the Developer A → B boundary.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.shared.paths import get_blocking_run_output_dir
from src.shared.schemas import CANDIDATE_PAIRS_SCHEMA, validate_contract_rules

logger = logging.getLogger(__name__)


def write_candidate_pairs(
    candidates: list[dict],
    run_id: str,
    data_dir: Path,
    con: duckdb.DuckDBPyConnection,
) -> int:
    """Write candidate_pairs.parquet from unioned candidate records.

    Args:
        candidates: List of candidate dicts from candidates.union_candidates().
        run_id: Run identifier to stamp on every row.
        data_dir: Pipeline data directory (per-run output dir is computed internally).
        con: DuckDB connection for registering the output table.

    Returns:
        Number of candidate pairs written.
    """
    out_dir = get_blocking_run_output_dir(data_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in candidates:
        rows.append({
            "run_id": run_id,
            "entity_id_a": c["entity_id_a"],
            "entity_id_b": c["entity_id_b"],
            "blocking_methods": c["blocking_methods"],
            "blocking_source": c["blocking_source"],
            "blocking_method_count": c["blocking_method_count"],
        })

    table = pa.table(
        {
            "run_id": pa.array([r["run_id"] for r in rows], type=pa.string()),
            "entity_id_a": pa.array(
                [r["entity_id_a"] for r in rows], type=pa.string(),
            ),
            "entity_id_b": pa.array(
                [r["entity_id_b"] for r in rows], type=pa.string(),
            ),
            "blocking_methods": pa.array(
                [r["blocking_methods"] for r in rows],
                type=pa.list_(pa.string()),
            ),
            "blocking_source": pa.array(
                [r["blocking_source"] for r in rows], type=pa.string(),
            ),
            "blocking_method_count": pa.array(
                [r["blocking_method_count"] for r in rows], type=pa.int8(),
            ),
        },
        schema=CANDIDATE_PAIRS_SCHEMA,
    )

    # Validate contract before writing
    errors = validate_contract_rules(table, "candidate_pairs")
    if errors:
        for e in errors:
            logger.error("candidate_pairs contract violation: %s", e)
        raise ValueError(
            f"candidate_pairs.parquet failed contract validation "
            f"with {len(errors)} error(s)"
        )

    out_path = out_dir / "candidate_pairs.parquet"
    pq.write_table(table, out_path)

    con.execute(
        f"CREATE OR REPLACE TABLE candidate_pairs "
        f"AS SELECT * FROM '{out_path}'"
    )

    logger.info("Wrote %d candidate pairs to %s", len(rows), out_path)
    return len(rows)


def write_handoff_manifest(
    run_id: str,
    entity_count: int,
    candidate_count: int,
    entity_types_present: list[str],
    data_dir: Path,
    embedding_dim: int = 768,
    k: int = 100,
) -> Path:
    """Generate handoff_manifest.json for the A→B boundary.

    Args:
        run_id: Run identifier.
        entity_count: Number of rows in entities.parquet (post-dedup mentions).
        candidate_count: Number of rows in candidate_pairs.parquet.
        entity_types_present: Sorted list of entity types found in this run.
        data_dir: Pipeline data directory (per-run output dir is computed internally).
        embedding_dim: Embedding dimensionality (default 768).
        k: FAISS top-k parameter used during blocking.

    Returns:
        Path to the written manifest file.
    """
    out_dir = get_blocking_run_output_dir(data_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mention_count": entity_count,
        "candidate_count": candidate_count,
        "embedding_dim": embedding_dim,
        "entity_types_present": sorted(entity_types_present),
        "k": k,
    }

    out_path = out_dir / "handoff_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Wrote handoff manifest to %s", out_path)
    return out_path
