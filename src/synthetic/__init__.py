"""Synthetic dataset builders and helpers for pipeline development."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "build_matching_dataset": "src.synthetic.build_matching_dataset",
    "load_labeled_feature_matrix": "src.synthetic.build_matching_dataset",
    "validate_identity_groups_payload": "src.synthetic.build_matching_dataset",
    "validate_synthetic_data": "src.synthetic.validate",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazy-load public helpers to avoid import-time side effects for `python -m`."""
    if name in _EXPORT_MODULES:
        module = import_module(_EXPORT_MODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
