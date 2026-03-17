"""Determinism and diagnostics integration tests for feature generation."""

import logging
import re
from pathlib import Path

import pytest

from src.matching.run import FEATURE_COLUMNS, run_features
from src.matching.writer import get_features_output_path
from src.shared.fixtures import DEFAULT_RUN_ID, write_mock_handoff


@pytest.fixture()
def handoff_dir(tmp_path: Path) -> Path:
    """Write mock handoff artifacts into a temporary directory."""
    write_mock_handoff(tmp_path)
    return tmp_path


def test_run_features_writes_byte_identical_output_on_repeated_runs(handoff_dir: Path) -> None:
    output_path = get_features_output_path(handoff_dir, DEFAULT_RUN_ID)

    run_features(handoff_dir, DEFAULT_RUN_ID)
    first_bytes = output_path.read_bytes()

    run_features(handoff_dir, DEFAULT_RUN_ID)
    second_bytes = output_path.read_bytes()

    assert first_bytes == second_bytes


def test_run_features_logs_diagnostics_for_all_feature_columns(
    handoff_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.matching.run")
    run_features(handoff_dir, DEFAULT_RUN_ID)

    pattern = re.compile(r"feature_diagnostics feature=([a-z0-9_]+)\b")
    logged_columns = {
        match.group(1)
        for record in caplog.records
        if (match := pattern.search(record.getMessage()))
    }

    assert logged_columns == set(FEATURE_COLUMNS)
