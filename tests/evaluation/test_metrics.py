"""Tests for pure evaluation metric helpers."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    bcubed_metrics,
    clustering_metrics,
    mention_metrics,
    pairwise_metrics,
    positive_pairs_from_memberships,
)


def test_positive_pairs_from_memberships_builds_canonical_pairs() -> None:
    memberships = {
        "e3": "g2",
        "e1": "g1",
        "e2": "g1",
    }

    assert positive_pairs_from_memberships(memberships) == {("e1", "e2")}


def test_pairwise_metrics_compute_precision_recall_and_f1() -> None:
    metrics = pairwise_metrics(
        predicted_positive_pairs={("e1", "e2"), ("e1", "e3")},
        gold_positive_pairs={("e1", "e2"), ("e2", "e3")},
    )

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["true_positive_count"] == 1


def test_clustering_metrics_match_toy_merge_split_case() -> None:
    gold = {
        "e1": "g_a",
        "e2": "g_b",
        "e3": "g_b",
        "e4": "g_a",
    }
    predicted = {
        "e1": "c_1",
        "e2": "c_1",
        "e3": "c_2",
        "e4": "c_1",
    }

    metrics = clustering_metrics(gold, predicted)

    assert metrics["pairwise_precision"] == pytest.approx(1 / 3)
    assert metrics["pairwise_recall"] == pytest.approx(0.5)
    assert metrics["pairwise_f1"] == pytest.approx(0.4)
    assert 0.0 <= metrics["ari"] <= 1.0
    assert 0.0 <= metrics["nmi"] <= 1.0
    assert metrics["bcubed_precision"] == pytest.approx(2 / 3)
    assert metrics["bcubed_recall"] == pytest.approx(0.75)
    assert metrics["bcubed_f1"] == pytest.approx(12 / 17)
    assert metrics["bcubed_f0_5"] == pytest.approx(15 / 22)


def test_bcubed_metrics_include_precision_weighted_f_beta() -> None:
    metrics = bcubed_metrics(
        gold_membership_by_entity={
            "e1": "g_a",
            "e2": "g_a",
            "e3": "g_b",
            "e4": "g_b",
        },
        predicted_membership_by_entity={
            "e1": "c_a",
            "e2": "c_a",
            "e3": "c_b",
            "e4": "c_c",
        },
    )

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(6 / 7)
    assert metrics["f0_5"] == pytest.approx(15 / 16)
    assert metrics["f0_5"] > metrics["f1"]


def test_bcubed_metrics_require_matching_entity_sets() -> None:
    with pytest.raises(ValueError, match="gold/predicted entity sets differ"):
        bcubed_metrics({"e1": "g1"}, {"e1": "c1", "e2": "c2"})


def test_mention_metrics_score_exact_span_matches() -> None:
    metrics = mention_metrics(
        predicted_mentions={
            ("doc1", "PER", 0, 5),
            ("doc1", "PER", 10, 15),
        },
        gold_mentions={
            ("doc1", "PER", 0, 5),
            ("doc2", "ORG", 1, 4),
        },
    )

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
