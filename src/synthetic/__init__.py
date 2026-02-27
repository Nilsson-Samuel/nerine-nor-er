"""Synthetic dataset builders and helpers for pipeline development."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["build_matching_dataset", "validate_identity_groups_payload"]


def __getattr__(name: str) -> Any:
    """Lazy-load public helpers to avoid import-time side effects for `python -m`."""
    if name in __all__:
        module = import_module("src.synthetic.build_matching_dataset")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
