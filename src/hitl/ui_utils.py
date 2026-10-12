"""Small pure helpers for HITL UI labels, metrics, DOT-safe text and context HTML."""

from __future__ import annotations

import html
from typing import Any

from src.hitl.context import MentionWindow


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


def format_mention_html(window: MentionWindow) -> str:
    """Render a mention window as escaped HTML with the mention highlighted."""
    before = ("…" if window.clipped_start else "") + html.escape(window.before)
    after = html.escape(window.after) + ("…" if window.clipped_end else "")
    mention = html.escape(window.mention)
    return (
        '<div style="white-space: pre-wrap; font-size: 0.9rem; line-height: 1.5; '
        'padding: 0.5rem 0.75rem; border-left: 3px solid #f0b429; margin-bottom: 0.5rem;">'
        f'{before}<mark style="background: #ffe08a; color: #000; padding: 0 2px;">'
        f"<b>{mention}</b></mark>{after}</div>"
    )
