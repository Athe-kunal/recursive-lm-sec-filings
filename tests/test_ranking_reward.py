"""Tests for ranking source metrics."""

import pytest

from rlm_sec.envs.rewards import compute_ranking_score


def test_compute_ranking_score_returns_precision_recall_f1():
    solution = "<source>10-K, Earnings</source>"
    ground_truth = {"relevant": ["10-K", "10-Q", "Earnings"]}

    score = compute_ranking_score(solution, ground_truth)

    assert score.format == 1.0
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(2 / 3)
    assert score.f1 == pytest.approx(0.8)
    assert score.correctness == pytest.approx(0.8)


def test_compute_ranking_score_without_source_tag_has_zero_format():
    solution = "No source tag"
    ground_truth = {"relevant": ["10-K"]}

    score = compute_ranking_score(solution, ground_truth)

    assert score.correctness == 0.0
    assert score.format == 0.0
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0
