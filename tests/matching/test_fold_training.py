"""Tests for multi-run fold training helpers and compact fold summaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from scripts.run_case_fold_eval import _parse_fold_configs
from src.matching.fold_training import (
    FoldTrainingSource,
    build_macro_average_row,
    build_fold_summary_row,
    load_multi_run_labeled_feature_matrix,
    train_and_save_fold_model,
    write_aggregate_fold_reports_markdown,
    write_fold_metrics_csv,
    write_fold_summary_markdown,
)
from src.matching.run import run_features
from src.shared.paths import get_evaluation_labels_path
from src.synthetic.build_matching_dataset import build_matching_dataset, load_labeled_feature_matrix


def _variant_triplet(base: str, alias: str, alt: str, context_prefix: str, index: int) -> list[dict]:
    """Build three text/context variants for one synthetic identity group."""
    return [
        {
            "text": base,
            "context": f"{context_prefix} {base} appeared in report {index}.",
        },
        {
            "text": alias,
            "context": f"{context_prefix} alias {alias} appeared in report {index}.",
        },
        {
            "text": alt,
            "context": f"{context_prefix} alternate {alt} appeared in report {index}.",
        },
    ]


def _payload(run_id: str, suffix: str) -> dict:
    """Return a compact synthetic payload with enough label diversity per run."""
    groups: list[dict] = []
    person_groups = [
        (f"per_alice_{suffix}", f"Alice Hansen {suffix}", f"A. Hansen {suffix}", f"Alice H {suffix}"),
        (f"per_bob_{suffix}", f"Bjarne Olsen {suffix}", f"B. Olsen {suffix}", f"Bjarne O {suffix}"),
    ]
    for index, (group_id, base, alias, alt) in enumerate(person_groups, start=1):
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "PER",
                "doc_ids": [f"doc_{group_id}_a", f"doc_{group_id}_b"],
                "variants": _variant_triplet(base, alias, alt, "Interview", index),
            }
        )

    org_groups = [
        (f"org_dnb_{suffix}", f"Den Norske Bank {suffix}", f"DNB {suffix}", f"DNB ASA {suffix}"),
        (f"org_taxi_{suffix}", f"Oslo Taxi {suffix}", f"Oslo Taxi AS {suffix}", f"Taxi Oslo {suffix}"),
    ]
    for index, (group_id, base, alias, alt) in enumerate(org_groups, start=1):
        groups.append(
            {
                "group_id": group_id,
                "entity_type": "ORG",
                "doc_ids": [f"doc_{group_id}_a", f"doc_{group_id}_b"],
                "variants": _variant_triplet(base, alias, alt, "Company", index),
            }
        )

    return {
        "run_id": run_id,
        "groups": groups,
        "hard_negatives": [
            {"group_id_a": f"per_alice_{suffix}", "group_id_b": f"per_bob_{suffix}"},
            {"group_id_a": f"org_dnb_{suffix}", "group_id_b": f"org_taxi_{suffix}"},
        ],
    }


@pytest.fixture()
def fold_training_sources(tmp_path: Path) -> list[FoldTrainingSource]:
    """Build two isolated labeled runs for fold-training tests."""
    sources: list[FoldTrainingSource] = []
    for index, case_name in enumerate(["case_alpha", "case_beta"], start=1):
        run_id = f"fold_train_run_{index}"
        data_dir = tmp_path / case_name
        payload = _payload(run_id, str(index))
        identity_groups_path = tmp_path / f"{case_name}.json"
        identity_groups_path.write_text(json.dumps(payload), encoding="utf-8")
        build_matching_dataset(identity_groups_path, data_dir, max_pairs=2500, seed=7 + index)
        run_features(data_dir, run_id)
        labels_path = get_evaluation_labels_path(data_dir, run_id)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        pl.read_parquet(data_dir / "labels.parquet").write_parquet(labels_path)
        sources.append(
            FoldTrainingSource(
                case_name=case_name,
                data_dir=data_dir,
                run_id=run_id,
            )
        )
    return sources


def test_multi_run_loader_preserves_source_order_and_label_alignment(
    fold_training_sources: list[FoldTrainingSource],
) -> None:
    selected_sources = [fold_training_sources[1], fold_training_sources[0]]

    training_matrix = load_multi_run_labeled_feature_matrix(selected_sources)
    first_x, first_y = load_labeled_feature_matrix(
        selected_sources[0].data_dir,
        selected_sources[0].run_id,
    )
    second_x, second_y = load_labeled_feature_matrix(
        selected_sources[1].data_dir,
        selected_sources[1].run_id,
    )

    assert training_matrix.metadata["run_summaries"][0]["case_name"] == selected_sources[0].case_name
    assert training_matrix.metadata["run_summaries"][1]["case_name"] == selected_sources[1].case_name
    assert training_matrix.X_train.height == first_x.height + second_x.height
    assert training_matrix.y_train.len() == first_y.len() + second_y.len()
    assert training_matrix.X_train[: first_x.height].to_dict(as_series=False) == first_x.to_dict(
        as_series=False
    )
    assert training_matrix.X_train[first_x.height :].to_dict(as_series=False) == second_x.to_dict(
        as_series=False
    )
    assert training_matrix.y_train[: first_y.len()].to_list() == first_y.to_list()
    assert training_matrix.y_train[first_y.len() :].to_list() == second_y.to_list()


def test_train_and_save_fold_model_writes_fold_training_metadata(
    fold_training_sources: list[FoldTrainingSource],
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "fold_model"

    result = train_and_save_fold_model(
        fold_training_sources,
        model_dir,
        model_version="lightgbm_case_fold__demo",
    )

    metadata = json.loads((model_dir / "reranker_model_metadata.json").read_text(encoding="utf-8"))

    assert metadata["model_version"] == "lightgbm_case_fold__demo"
    assert metadata["training_param_source"] == "fold_training"
    assert metadata["fold_training"]["source_count"] == 2
    assert [row["case_name"] for row in metadata["fold_training"]["run_summaries"]] == [
        "case_alpha",
        "case_beta",
    ]
    assert result["training_metadata"]["labeled_row_count"] == metadata["fold_training"][
        "labeled_row_count"
    ]


def test_fold_summary_row_and_csv_capture_key_metrics(tmp_path: Path) -> None:
    report = {
        "metric_scope": {
            "evaluation_entity_count": 14,
            "evaluation_candidate_pair_count": 28,
        },
        "metrics": {
            "pairwise_precision": 0.7,
            "pairwise_recall": 0.54,
            "pairwise_f1": 0.61,
            "ari": 0.33,
            "nmi": 0.44,
            "bcubed_precision": 0.74,
            "bcubed_recall": 0.7,
            "bcubed_f1": 0.72,
            "bcubed_f0_5": 0.73,
        },
        "stage_metrics": {
            "blocking": {"gold_positive_pair_recall": 0.9},
            "matching": {
                "precision": 0.8,
                "recall": 0.5,
                "f1": 0.62,
            },
        },
    }
    training_metadata = {
        "labeled_row_count": 42,
        "positive_rate": 0.19,
    }

    row = build_fold_summary_row(
        fold_name="fold_demo",
        held_out_case="case_beta",
        train_cases=["case_alpha", "case_gamma"],
        test_run_id="abc123",
        training_metadata=training_metadata,
        evaluation_report=report,
    )
    csv_path = tmp_path / "fold_metrics.csv"
    write_fold_metrics_csv(csv_path, [row])
    csv_rows = pl.read_csv(csv_path).to_dicts()

    assert row["train_cases"] == ["case_alpha", "case_gamma"]
    assert row["pairwise_precision"] == 0.7
    assert row["pairwise_recall"] == 0.54
    assert row["pairwise_f1"] == 0.61
    assert row["bcubed_precision"] == 0.74
    assert row["bcubed_recall"] == 0.7
    assert row["bcubed_f0_5"] == 0.73
    assert row["blocking_positive_pair_recall"] == 0.9
    assert csv_rows == [
        {
            "fold_name": "fold_demo",
            "held_out_case": "case_beta",
            "train_cases": "case_alpha|case_gamma",
            "train_case_count": 2,
            "test_run_id": "abc123",
            "train_labeled_row_count": 42,
            "train_positive_rate": 0.19,
            "evaluation_entity_count": 14,
            "evaluation_candidate_pair_count": 28,
            "pairwise_precision": 0.7,
            "pairwise_recall": 0.54,
            "pairwise_f1": 0.61,
            "ari": 0.33,
            "nmi": 0.44,
            "bcubed_precision": 0.74,
            "bcubed_recall": 0.7,
            "bcubed_f1": 0.72,
            "bcubed_f0_5": 0.73,
            "blocking_positive_pair_recall": 0.9,
            "matching_pairwise_precision": 0.8,
            "matching_pairwise_recall": 0.5,
            "matching_pairwise_f1": 0.62,
        }
    ]


def test_fold_markdown_writers_include_full_metric_surface(tmp_path: Path) -> None:
    row = {
        "fold_name": "fold_demo",
        "held_out_case": "case_beta",
        "train_cases": ["case_alpha", "case_gamma"],
        "train_case_count": 2,
        "test_run_id": "abc123",
        "train_labeled_row_count": 42,
        "train_positive_rate": 0.19,
        "evaluation_entity_count": 14,
        "evaluation_candidate_pair_count": 28,
        "pairwise_precision": 0.7,
        "pairwise_recall": 0.54,
        "pairwise_f1": 0.61,
        "ari": 0.33,
        "nmi": 0.44,
        "bcubed_precision": 0.74,
        "bcubed_recall": 0.7,
        "bcubed_f1": 0.72,
        "bcubed_f0_5": 0.73,
        "blocking_positive_pair_recall": 0.9,
        "matching_pairwise_precision": 0.8,
        "matching_pairwise_recall": 0.5,
        "matching_pairwise_f1": 0.62,
    }
    payload = {
        "fold_name": "fold_demo",
        "held_out_case": "case_beta",
        "train_cases": ["case_alpha", "case_gamma"],
        "training": {"training_metadata": {"source_count": 2}},
        "held_out_run": {
            "evaluation_report_path": "/tmp/fold_demo/evaluation_report.json",
            "evaluation_markdown_report_path": "/tmp/fold_demo/evaluation_report.md",
        },
        "summary_row": row,
    }

    fold_markdown_path = tmp_path / "fold_summary.md"
    aggregate_markdown_path = tmp_path / "fold_reports.md"
    write_fold_summary_markdown(fold_markdown_path, payload)
    write_aggregate_fold_reports_markdown(aggregate_markdown_path, [row])

    fold_markdown = fold_markdown_path.read_text(encoding="utf-8")
    aggregate_markdown = aggregate_markdown_path.read_text(encoding="utf-8")
    macro_row = build_macro_average_row([row])

    assert "## Final Clustering Metrics" in fold_markdown
    assert "Pairwise precision" in fold_markdown
    assert "B-cubed F0.5" in fold_markdown
    assert "/tmp/fold_demo/evaluation_report.md" in fold_markdown
    assert "macro_avg" in aggregate_markdown
    assert f"{macro_row['pairwise_f1']:.3f}" in aggregate_markdown
    assert "B-cubed F0.5" in aggregate_markdown


def test_fold_config_rejects_duplicate_train_cases() -> None:
    payload = {
        "folds": [
            {
                "name": "fold_demo",
                "held_out_case": "case_gamma",
                "train_cases": ["case_alpha", "case_alpha"],
            }
        ]
    }

    with pytest.raises(ValueError, match="duplicate train cases"):
        _parse_fold_configs(payload)


def test_case_fold_runner_help_bootstraps_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, "scripts/run_case_fold_eval.py", "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run pragmatic held-out case-fold evaluation" in result.stdout
