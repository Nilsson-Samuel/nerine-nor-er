"""Evaluation entrypoints for post-pipeline metrics and regression checks."""


def run_evaluation(*args, **kwargs):
    """Import the evaluation runner lazily for safe CLI module execution."""
    from src.evaluation.run import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)


__all__ = ["run_evaluation"]
