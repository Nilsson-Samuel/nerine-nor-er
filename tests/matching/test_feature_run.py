"""Integration tests for matching feature-stage orchestration."""

from pathlib import Path

import polars as pl
import pytest

from src.matching.run import run_features
from src.shared.fixtures import DEFAULT_RUN_ID, write_mock_handoff


EXPECTED_FEATURE_COLUMNS = [
    "jaro_winkler_similarity",
    "levenshtein_ratio_similarity",
    "token_jaccard_similarity",
    "token_containment_ratio",
    "char_trigram_jaccard_similarity",
    "abbreviation_match_flag",
    "double_metaphone_overlap_flag",
    "cosine_sim_entity",
    "cosine_sim_context",
    "norwegian_id_match",
    "first_name_match",
    "last_name_match",
    "shared_doc_count",
    "blocking_method_count",
]
EXPECTED_OUTPUT_COLUMNS = ["run_id", "entity_id_a", "entity_id_b", *EXPECTED_FEATURE_COLUMNS]


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Write mock handoff artifacts into a temporary directory."""
    write_mock_handoff(tmp_path)
    return tmp_path


def test_run_features_writes_features_parquet(handoff_dir: Path) -> None:
    run_features(handoff_dir, DEFAULT_RUN_ID)
    assert (handoff_dir / "features.parquet").exists()


def test_run_features_row_count_matches_candidate_pairs(handoff_dir: Path) -> None:
    run_features(handoff_dir, DEFAULT_RUN_ID)
    features = pl.read_parquet(handoff_dir / "features.parquet")
    candidates = pl.read_parquet(handoff_dir / "candidate_pairs.parquet").filter(
        pl.col("run_id") == DEFAULT_RUN_ID
    )
    assert features.height == candidates.height


def test_run_features_output_columns_match_contract(handoff_dir: Path) -> None:
    run_features(handoff_dir, DEFAULT_RUN_ID)
    features = pl.read_parquet(handoff_dir / "features.parquet")

    assert features.columns == EXPECTED_OUTPUT_COLUMNS
    assert len(EXPECTED_FEATURE_COLUMNS) == 14
    for column in EXPECTED_FEATURE_COLUMNS:
        assert features[column].null_count() == 0
