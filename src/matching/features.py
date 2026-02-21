"""DuckDB join loader for the matching stage.

Resolves entity names from candidate pairs using a double JOIN against
entities.parquet. Feature computation is handled separately.
"""

from pathlib import Path

import duckdb
import polars as pl


# Double-join query: resolves name_a/name_b from entities for each candidate pair.
# Ordered so downstream feature builders always see a stable row sequence.
_LOAD_PAIRS_QUERY = """
SELECT
    c.run_id,
    c.entity_id_a,
    c.entity_id_b,
    e1.normalized AS name_a,
    e2.normalized AS name_b
FROM candidate_pairs c
JOIN entities e1 ON c.run_id = e1.run_id AND c.entity_id_a = e1.entity_id
JOIN entities e2 ON c.run_id = e2.run_id AND c.entity_id_b = e2.entity_id
WHERE c.run_id = ?
ORDER BY c.entity_id_a, c.entity_id_b
"""


def load_pairs_with_names(
    db_path_or_con: "duckdb.DuckDBPyConnection | Path | str",
    run_id: str,
) -> pl.DataFrame:
    """Load candidate pairs joined with entity names for a given run_id.

    Accepts either an existing DuckDB connection (with 'entities' and
    'candidate_pairs' already registered) or a data directory path (parquet
    files are registered as temporary views).

    Returns exactly five columns: run_id, entity_id_a, entity_id_b,
    name_a, name_b — ordered by entity_id_a, entity_id_b for stable output.

    Args:
        db_path_or_con: DuckDB connection with tables pre-registered, or a
            directory path containing entities.parquet and candidate_pairs.parquet.
        run_id: Pipeline run identifier to filter on.

    Returns:
        Polars DataFrame with columns
        [run_id, entity_id_a, entity_id_b, name_a, name_b].
    """
    if isinstance(db_path_or_con, duckdb.DuckDBPyConnection):
        table = db_path_or_con.execute(_LOAD_PAIRS_QUERY, [run_id]).fetch_arrow_table()
        return pl.from_arrow(table)

    # Path-based: use a context-managed connection so it is closed on exit.
    # read_parquet() takes the path as a Python argument - no SQL string interpolation.
    data_dir = Path(db_path_or_con)
    with duckdb.connect() as con:
        con.register("entities", con.read_parquet(str(data_dir / "entities.parquet")))
        con.register(
            "candidate_pairs",
            con.read_parquet(str(data_dir / "candidate_pairs.parquet")),
        )
        table = con.execute(_LOAD_PAIRS_QUERY, [run_id]).fetch_arrow_table()
        return pl.from_arrow(table)
