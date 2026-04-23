"""Tests for small pure helpers used by the HITL UI."""

from __future__ import annotations

from src.hitl.ui_utils import (
    LABEL_DELIMITER,
    build_option_label,
    escape_dot_label,
    format_metric_value,
    parse_option_id,
)


def test_build_option_label_uses_shared_delimiter() -> None:
    assert build_option_label("e1", "DNB ASA") == f"e1{LABEL_DELIMITER}DNB ASA"


def test_parse_option_id_uses_single_split() -> None:
    label = build_option_label("e1", f'Name{LABEL_DELIMITER}With delimiter')
    assert parse_option_id(label) == "e1"


def test_parse_option_id_returns_none_for_empty_value() -> None:
    assert parse_option_id(None) is None


def test_format_metric_value_handles_none() -> None:
    assert format_metric_value(None) == "-"


def test_format_metric_value_formats_numeric_values() -> None:
    assert format_metric_value(0.876) == "0.88"


def test_escape_dot_label_escapes_backslashes_and_quotes() -> None:
    assert escape_dot_label('A \\"quoted" value') == 'A \\\\\\"quoted\\" value'
