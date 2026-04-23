"""Small pure helpers for HITL UI labels, metrics, and DOT-safe text."""

from __future__ import annotations

from typing import Any


LABEL_DELIMITER = "  -  "


def build_option_label(item_id: str, label: str) -> str:
    """Build a stable selectbox label with the item ID first."""
    return f"{item_id}{LABEL_DELIMITER}{label}"


def parse_option_id(selected_label: str | None) -> str | None:
    """Extract the leading ID from a selectbox label."""
    if not selected_label:
        return None
    return selected_label.split(LABEL_DELIMITER, 1)[0]


def format_metric_value(value: Any) -> str:
    """Format nullable numeric metrics without crashing the inspector."""
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def escape_dot_label(value: str) -> str:
    """Escape Graphviz label text for quoted DOT strings."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
