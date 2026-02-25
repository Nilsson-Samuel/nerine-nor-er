"""Parquet writers for intermediate matching artifacts."""

from pathlib import Path

import polars as pl


def write_string_features(df: pl.DataFrame, out_dir: Path) -> None:
    """Write string/token feature columns to string_features.parquet.

    Args:
        df: DataFrame containing feature columns (at minimum the five key
            columns from load_pairs_with_names, plus any computed features).
        out_dir: Directory to write string_features.parquet into.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "string_features.parquet")


def write_features(df: pl.DataFrame, out_dir: Path) -> None:
    """Write the integrated feature table to features.parquet.

    Args:
        df: Feature table containing pair keys and all computed feature columns.
        out_dir: Directory to write features.parquet into.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "features.parquet")
