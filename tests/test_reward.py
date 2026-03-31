"""Tests for QA format reward scoring and terminal correctness.

All tests call the pure scoring helpers and the async reward functions directly.
No HTTP server or env.step() is needed.

Valid action-format score breakdown for a single search action:
  1.0 (tool name recognised) + 1/3 (ticker) + 1/3 (year) + 1/3 (filing) = 2.0
"""

import asyncio
from typing import cast

import pytest
import verifiers as vf

from rlm_sec.envs.finance_env import reward_correctness, reward_format
from rlm_sec.envs.rewards import (
    compute_qa_company_to_ticker_score,
    compute_qa_format_score,
    compute_qa_ticker_match_score,
    compute_qa_year_match_score,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BASE_INFO = {
    "task_type": "qa",
    "ground_truth": {"target": "89.5B"},
}

_QA_INFO_WITH_GROUND_TRUTH = {
    "task_type": "qa",
    "ground_truth": {
        "target": "89.5B",
        "ticker": "AAPL",
        "ticker_or_company_name": "Apple Inc.",
        "year": "2023",
    },
}

_VALID_SEC_SEARCH = "<search>SECFilingTool(revenue, AAPL, 2023, 10-K)</search>"
_VALID_ET_SEARCH = "<search>EarningsTranscriptTool(guidance, AAPL, 2023, Q3)</search>"


def _make_completion(*assistant_contents: str) -> vf.Messages:
    """Interleave assistant messages with placeholder user messages.

    The final assistant message is never followed by a user turn so that
    completion[-1] always refers to the last assistant output (the answer).
    """
    messages = []
    for i, content in enumerate(assistant_contents):
        messages.append({"role": "assistant", "content": content})
        if i < len(assistant_contents) - 1:
            messages.append(
                {
                    "role": "user",
                    "content": "<information>Doc 1: Revenue was $89.5B.</information>",
                }
            )
    return cast(vf.Messages, messages)


# ---------------------------------------------------------------------------
# compute_qa_format_score — pure function, synchronous
# ---------------------------------------------------------------------------


class TestComputeQAFormatScore:
    def test_valid_sec_filing_returns_base_score(self):
        score = compute_qa_format_score(_VALID_SEC_SEARCH, {})
        assert score == 1.0

    def test_valid_earnings_transcript_returns_base_score(self):
        score = compute_qa_format_score(_VALID_ET_SEARCH, {})
        assert score == 1.0

    def test_wrong_arg_count_returns_zero(self):
        score = compute_qa_format_score(
            "<search>SECFilingTool(AAPL, 2023)</search>", {}
        )
        assert score == 0.0

    def test_unknown_tool_returns_zero(self):
        score = compute_qa_format_score(
            "<search>UnknownTool(revenue, AAPL, 2023, 10-K)</search>", {}
        )
        assert score == 0.0

    def test_no_search_tag_returns_zero(self):
        score = compute_qa_format_score("I need to look up revenue.", {})
        assert score == 0.0

    def test_company_name_to_ticker_counts_as_valid(self):
        score = compute_qa_format_score(
            "<search>CompanyNameToTickerTool(Apple Inc.)</search>", {}
        )
        assert score == 1.0


# ---------------------------------------------------------------------------
# Task 2 individual format components
# ---------------------------------------------------------------------------


class TestQAFormatComponents:
    _GT = {
        "ticker": "AAPL",
        "ticker_or_company_name": "Apple Inc.",
        "year": "2023",
    }

    # company-to-ticker
    def test_company_to_ticker_used(self):
        solution = "<search>CompanyNameToTickerTool(Apple Inc.)</search>"
        assert compute_qa_company_to_ticker_score(solution, self._GT) == 0.1

    def test_company_to_ticker_not_used(self):
        assert compute_qa_company_to_ticker_score(_VALID_SEC_SEARCH, self._GT) == 0.0

    def test_company_to_ticker_wrong_company_returns_zero(self):
        solution = "<search>CompanyNameToTickerTool(Microsoft)</search>"
        assert compute_qa_company_to_ticker_score(solution, self._GT) == 0.0

    # ticker match
    def test_correct_ticker_matched(self):
        assert compute_qa_ticker_match_score(_VALID_SEC_SEARCH, self._GT) == 0.1

    def test_wrong_ticker_returns_zero(self):
        solution = "<search>SECFilingTool(revenue, MSFT, 2023, 10-K)</search>"
        assert compute_qa_ticker_match_score(solution, self._GT) == 0.0

    def test_ticker_case_insensitive(self):
        solution = "<search>SECFilingTool(revenue, aapl, 2023, 10-K)</search>"
        assert compute_qa_ticker_match_score(solution, self._GT) == 0.1

    # year match
    def test_correct_year_matched(self):
        assert compute_qa_year_match_score(_VALID_SEC_SEARCH, self._GT) == 0.1

    def test_wrong_year_returns_zero(self):
        solution = "<search>SECFilingTool(revenue, AAPL, 2022, 10-K)</search>"
        assert compute_qa_year_match_score(solution, self._GT) == 0.0

    def test_company_to_ticker_ignores_data_source(self):
        gt = {
            "data_source": "some-other-dataset",
            "ticker": "AAPL",
            "ticker_or_company_name": "Apple Inc.",
            "year": "2023",
        }
        solution = "<search>CompanyNameToTickerTool(Apple Inc.)</search>"
        assert compute_qa_company_to_ticker_score(solution, gt) == 0.1

    def test_scores_ignore_data_source_for_ground_truth_matches(self):
        gt = {
            "data_source": "PatronusAI/financebench",
            "ticker": "AAPL",
            "ticker_or_company_name": "Apple Inc.",
            "year": "2023",
        }
        solution = (
            "<search>CompanyNameToTickerTool(Apple Inc.)</search>"
            "<search>SECFilingTool(revenue, AAPL, 2023, 10-K)</search>"
        )
        assert compute_qa_company_to_ticker_score(solution, gt) == 0.1
        assert compute_qa_ticker_match_score(solution, gt) == 0.1
        assert compute_qa_year_match_score(solution, gt) == 0.1

    def test_all_components_present_returns_one_point_three(self):
        solution = (
            "<search>CompanyNameToTickerTool(Apple Inc.)</search>"
            "<search>SECFilingTool(revenue, AAPL, 2023, 10-K)</search>"
        )
        assert compute_qa_format_score(solution, self._GT) == pytest.approx(1.3)

    def test_only_base_component_present_returns_one(self):
        solution = "<search>SECFilingTool(revenue, MSFT, 2022, 10-K)</search>"
        assert compute_qa_format_score(solution, self._GT) == pytest.approx(1.0)

    def test_company_to_ticker_only_needs_tool_call(self):
        gt = {
            "ticker": "AAPL",
            "ticker_or_company_name": "Apple Inc.",
            "year": "2023",
        }
        solution = "<search>CompanyNameToTickerTool(Apple Inc.)</search>"
        assert compute_qa_company_to_ticker_score(solution, gt) == 0.1

    def test_company_to_ticker_without_company_ground_truth_returns_zero(self):
        gt = {
            "ticker": "AAPL",
            "year": "2023",
        }
        solution = "<search>CompanyNameToTickerTool(Apple Inc.)</search>"
        assert compute_qa_company_to_ticker_score(solution, gt) == 0.0


# ---------------------------------------------------------------------------
# reward_correctness — async, tested via asyncio.run
# ---------------------------------------------------------------------------


class TestRewardCorrectness:
    def test_exact_match_returns_one(self):
        completion = _make_completion(_VALID_SEC_SEARCH, "<answer>89.5B</answer>")
        score = asyncio.run(reward_correctness(completion, _BASE_INFO))
        assert score == 1.0

    def test_wrong_answer_returns_zero(self):
        completion = _make_completion(_VALID_SEC_SEARCH, "<answer>50B</answer>")
        score = asyncio.run(reward_correctness(completion, _BASE_INFO))
        assert score == 0.0

    def test_missing_answer_tag_returns_zero(self):
        completion = cast(
            vf.Messages,
            [{"role": "assistant", "content": "Final thoughts, no answer tag."}],
        )
        score = asyncio.run(reward_correctness(completion, _BASE_INFO))
        assert score == 0.0

    def test_none_info_returns_zero(self):
        completion = _make_completion("<answer>89.5B</answer>")
        score = asyncio.run(reward_correctness(completion, None))
        assert score == 0.0


# ---------------------------------------------------------------------------
# reward_format — async, tested via asyncio.run
# ---------------------------------------------------------------------------


class TestRewardFormat:
    def test_terminal_step_base_format(self):
        """Single valid search + answer → base format score of 1.0."""
        completion = _make_completion(_VALID_SEC_SEARCH, "<answer>89.5B</answer>")
        score = asyncio.run(reward_format(completion, _BASE_INFO))
        assert score == pytest.approx(0.5)

    def test_terminal_step_all_components(self):
        """QA terminal format is normalized by assistant turns."""
        completion = cast(
            vf.Messages,
            [
                {
                    "role": "assistant",
                    "content": "<search>CompanyNameToTickerTool(Apple Inc.)</search>",
                },
                {"role": "user", "content": "<information>AAPL</information>"},
                {"role": "assistant", "content": _VALID_SEC_SEARCH},
                {
                    "role": "user",
                    "content": "<information>Doc 1: Revenue was $89.5B.</information>",
                },
                {"role": "assistant", "content": "<answer>89.5B</answer>"},
            ],
        )
        score = asyncio.run(reward_format(completion, _QA_INFO_WITH_GROUND_TRUTH))
        assert score == pytest.approx(1.3 / 3)

    def test_no_answer_tag_falls_back_to_intermediate_format(self):
        """Rollout ended at max_turns without an answer — last action has zero format."""
        completion = cast(
            vf.Messages,
            [{"role": "assistant", "content": "Final thoughts, no answer tag."}],
        )
        score = asyncio.run(reward_format(completion, _BASE_INFO))
        assert score == 0.0

    def test_wrong_answer_still_awards_base_format(self):
        """<answer> tag is present but content is wrong → format credit, no correctness."""
        completion = _make_completion(_VALID_SEC_SEARCH, "<answer>50B</answer>")
        score = asyncio.run(reward_format(completion, _BASE_INFO))
        assert score == pytest.approx(0.5)
