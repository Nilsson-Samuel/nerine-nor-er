"""Validation gate tests for the matching stage DuckDB loader and writer.

Covers:
- Loader returns exactly the five key columns.
- Row count matches mock fixture candidate count (2 rows).
- entity_id_a < entity_id_b ordering holds for all rows (no self-pairs, no reversals).
- write_string_features creates a readable parquet file at the expected path.
- Loader also works when handed an existing DuckDB connection with pre-registered views.
"""

from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from src.shared.fixtures import DEFAULT_RUN_ID, write_mock_handoff
from src.matching.features import load_pairs_with_names
from src.matching.writer import write_string_features


# The mock fixture always produces exactly 2 candidate pairs: one PER, one ORG.
_EXPECTED_PAIR_COUNT = 2
_EXPECTED_COLUMNS = {"run_id", "entity_id_a", "entity_id_b", "name_a", "name_b"}


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Write mock handoff files to a temp directory and return the path."""
    write_mock_handoff(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

def test_loader_returns_five_key_columns(handoff_dir: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_loader_no_extra_columns(handoff_dir: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    assert len(df.columns) == len(_EXPECTED_COLUMNS)


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

def test_loader_row_count_matches_fixture(handoff_dir: Path) -> None:
    # Default mock fixture produces exactly 2 candidate pairs (1 PER + 1 ORG).
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    assert len(df) == _EXPECTED_PAIR_COUNT


# ---------------------------------------------------------------------------
# Pair ordering invariants
# ---------------------------------------------------------------------------

def test_loader_entity_id_a_less_than_b(handoff_dir: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    assert (df["entity_id_a"] < df["entity_id_b"]).all(), (
        "entity_id_a must be strictly less than entity_id_b for every row"
    )


def test_loader_no_self_pairs(handoff_dir: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    assert (df["entity_id_a"] != df["entity_id_b"]).all()


def test_loader_rows_ordered_by_entity_id_a_then_b(handoff_dir: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    sorted_df = df.sort(["entity_id_a", "entity_id_b"])
    assert df.equals(sorted_df)


# ---------------------------------------------------------------------------
# run_id filter
# ---------------------------------------------------------------------------

def test_loader_run_id_column_matches_filter(handoff_dir: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    assert (df["run_id"] == DEFAULT_RUN_ID).all()


def test_loader_unknown_run_id_returns_empty(handoff_dir: Path) -> None:
    df = load_pairs_with_names(handoff_dir, "nonexistent_run")
    assert len(df) == 0
    assert set(df.columns) == _EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Pre-registered DuckDB connection
# ---------------------------------------------------------------------------

def test_loader_accepts_existing_connection(handoff_dir: Path) -> None:
    # Pre-register views and pass the connection directly.
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW entities AS SELECT * FROM read_parquet('{handoff_dir / 'entities.parquet'}')"
    )
    con.execute(
        f"CREATE VIEW candidate_pairs AS SELECT * FROM read_parquet"
        f"('{handoff_dir / 'candidate_pairs.parquet'}')"
    )
    df = load_pairs_with_names(con, DEFAULT_RUN_ID)
    assert len(df) == _EXPECTED_PAIR_COUNT
    assert set(df.columns) == _EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# writer: write_string_features
# ---------------------------------------------------------------------------

def test_write_string_features_creates_file(handoff_dir: Path, tmp_path: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    out_dir = tmp_path / "features_out"
    write_string_features(df, out_dir)
    assert (out_dir / "string_features.parquet").exists()


def test_write_string_features_readable_parquet(handoff_dir: Path, tmp_path: Path) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    out_dir = tmp_path / "features_out"
    write_string_features(df, out_dir)

    table = pq.read_table(out_dir / "string_features.parquet")
    assert table.num_rows == _EXPECTED_PAIR_COUNT
    assert set(table.schema.names) == _EXPECTED_COLUMNS


def test_write_string_features_creates_out_dir_if_absent(
    handoff_dir: Path, tmp_path: Path
) -> None:
    df = load_pairs_with_names(handoff_dir, DEFAULT_RUN_ID)
    nested = tmp_path / "a" / "b" / "c"
    write_string_features(df, nested)
    assert (nested / "string_features.parquet").exists()
