"""Tests for ranking source metrics.

Covers:
  - compute_ranking_score   (pure function)
  - reward_ranking_precision (async metric, weight=0)
  - reward_ranking_recall    (async metric, weight=0)
"""

import asyncio

import pytest
import verifiers as vf
from typing import cast
from rlm_sec.envs.finance_env import reward_ranking_precision, reward_ranking_recall
from rlm_sec.envs.rewards import compute_ranking_score

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RANKING_INFO = {
    "task_type": "ranking",
    "ground_truth": {"relevant": ["10-K", "10-Q", "Earnings"]},
}


def _completion_with_answer(answer: str) -> vf.Messages:
    """Build assistant content with one <answer> block.

    Ranking (and QA) scoring use the *last* <answer>...</answer> span in the
    flattened completion. A trailing placeholder tag would override the real
    sources and drive precision/recall to zero.
    """
    return cast(
        vf.Messages,
        [{"role": "assistant", "content": f"<answer>{answer}</answer>"}],
    )


# ---------------------------------------------------------------------------
# compute_ranking_score — pure function
# ---------------------------------------------------------------------------


def test_compute_ranking_score_returns_precision_recall_f1():
    solution = "<answer>10-K, Earnings</answer>"
    ground_truth = {"relevant": ["10-K", "10-Q", "Earnings"]}

    score = compute_ranking_score(solution, ground_truth)

    assert score.format == 1.0
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(2 / 3)
    assert score.f1 == pytest.approx(0.8)
    assert score.correctness == pytest.approx(0.8)


def test_compute_ranking_score_without_answer_tag_has_zero_format():
    solution = "No source tag"
    ground_truth = {"relevant": ["10-K"]}

    score = compute_ranking_score(solution, ground_truth)

    assert score.correctness == 0.0
    assert score.format == 0.0
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_compute_ranking_score_perfect_match():
    solution = "<answer>10-K, DEF14A</answer>"
    ground_truth = {"relevant": ["10-K", "DEF14A"]}

    score = compute_ranking_score(solution, ground_truth)

    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(1.0)
    assert score.correctness == pytest.approx(1.0)


def test_compute_ranking_score_no_overlap():
    solution = "<answer>8-K</answer>"
    ground_truth = {"relevant": ["10-K", "DEF14A"]}

    score = compute_ranking_score(solution, ground_truth)

    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_compute_ranking_score_case_insensitive():
    solution = "<answer>10-k, earnings</answer>"
    ground_truth = {"relevant": ["10-K", "Earnings"]}

    score = compute_ranking_score(solution, ground_truth)

    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)


def test_compute_ranking_score_empty_relevant_returns_zero():
    solution = "<answer>10-K</answer>"
    ground_truth = {"relevant": []}

    score = compute_ranking_score(solution, ground_truth)

    assert score.correctness == 0.0
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


# ---------------------------------------------------------------------------
# reward_ranking_precision / reward_ranking_recall — async metric functions
# ---------------------------------------------------------------------------


class TestRankingMetricFunctions:
    def test_precision_partial_match(self):
        """2 predicted, both correct out of 3 relevant → precision = 1.0."""
        completion = _completion_with_answer("10-K, Earnings")
        score = asyncio.run(reward_ranking_precision(completion, _RANKING_INFO))
        assert score == pytest.approx(1.0)

    def test_recall_partial_match(self):
        """2 of 3 relevant sources found → recall = 2/3."""
        completion = _completion_with_answer("10-K, Earnings")
        score = asyncio.run(reward_ranking_recall(completion, _RANKING_INFO))
        assert score == pytest.approx(2 / 3)

    def test_precision_perfect_match(self):
        completion = _completion_with_answer("10-K, 10-Q, Earnings")
        score = asyncio.run(reward_ranking_precision(completion, _RANKING_INFO))
        assert score == pytest.approx(1.0)

    def test_recall_perfect_match(self):
        completion = _completion_with_answer("10-K, 10-Q, Earnings")
        score = asyncio.run(reward_ranking_recall(completion, _RANKING_INFO))
        assert score == pytest.approx(1.0)

    def test_precision_with_false_positives(self):
        """Predicted 4 sources, 3 correct → precision = 3/4."""
        completion = _completion_with_answer("10-K, 10-Q, Earnings, 8-K")
        score = asyncio.run(reward_ranking_precision(completion, _RANKING_INFO))
        assert score == pytest.approx(3 / 4)

    def test_precision_zero_when_no_answer_tag(self):
        completion = cast(
            vf.Messages, [{"role": "assistant", "content": "No source tags here."}]
        )
        score = asyncio.run(reward_ranking_precision(completion, _RANKING_INFO))
        assert score == 0.0

    def test_recall_zero_when_no_answer_tag(self):
        completion = cast(
            vf.Messages, [{"role": "assistant", "content": "No source tags here."}]
        )
        score = asyncio.run(reward_ranking_recall(completion, _RANKING_INFO))
        assert score == 0.0

    def test_none_info_returns_zero(self):
        completion = _completion_with_answer("10-K")
        assert asyncio.run(reward_ranking_precision(completion, None)) == 0.0
        assert asyncio.run(reward_ranking_recall(completion, None)) == 0.0
