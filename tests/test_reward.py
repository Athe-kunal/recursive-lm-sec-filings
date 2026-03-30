"""Tests for format reward and terminal correctness scoring.

Retrieval (HTTP) is mocked via patch.object on FinanceSearchEnv._execute_tool
so no real server is needed.

Valid action-format score is:
1.0 (tool) + 1/3 (ticker) + 1/3 (year) + 1/3 (filing) = 2.0.
"""

from unittest.mock import patch

import pytest

from rlm_sec.envs.finance_env import FinanceSearchEnv, SearchEnvConfig

FAKE_RETRIEVAL = "\n<information>\nDoc 1: Revenue was $89.5B.\n</information>\n"
GROUND_TRUTH = {"target": "89.5B"}
EXPECTED_VALID_FORMAT_REWARD = pytest.approx(2.0)


def make_env(max_turns: int = 3) -> FinanceSearchEnv:
    cfg = SearchEnvConfig(topk=3, timeout=30, log_requests=False)
    env = FinanceSearchEnv(cfg, extras={"max_turns": max_turns})
    env.ground_truth = GROUND_TRUTH
    return env


# ── Format reward (intermediate steps) ────────────────────────────────────────


class TestFormatReward:
    def test_valid_sec_filing(self):
        env = make_env()
        with patch.object(FinanceSearchEnv, "_execute_tool", return_value=FAKE_RETRIEVAL):
            out = env.step("<search>SECFilingTool(revenue, AAPL, 2023, 10-K)</search>")

        assert out["done"] is False
        assert out["reward"].format == EXPECTED_VALID_FORMAT_REWARD
        assert "<information>" in out["observations"][0]["content"]

    def test_valid_earnings_transcript(self):
        env = make_env()
        with patch.object(FinanceSearchEnv, "_execute_tool", return_value=FAKE_RETRIEVAL):
            out = env.step(
                "<search>EarningsTranscriptTool(guidance, AAPL, 2023, Q3)</search>"
            )

        assert out["done"] is False
        assert out["reward"].format == EXPECTED_VALID_FORMAT_REWARD

    def test_wrong_arg_count_returns_zero(self):
        env = make_env()
        out = env.step("<search>SECFilingTool(AAPL, 2023)</search>")

        assert out["done"] is False
        assert out["reward"].format == 0.0
        assert "Invalid" in out["observations"][0]["content"]

    def test_unknown_tool_returns_zero(self):
        env = make_env()
        out = env.step("<search>UnknownTool(revenue, AAPL, 2023, 10-K)</search>")

        assert out["done"] is False
        assert out["reward"].format == 0.0

    def test_no_search_tag_returns_zero(self):
        env = make_env()
        out = env.step("I need to look up revenue.")

        assert out["done"] is False
        assert out["reward"].format == 0.0


# ── Correctness score (terminal steps) ────────────────────────────────────────


class TestCorrectnessScore:
    def _search_then_answer(self, env: FinanceSearchEnv, answer: str):
        with patch.object(FinanceSearchEnv, "_execute_tool", return_value=FAKE_RETRIEVAL):
            env.step("<search>SECFilingTool(revenue, AAPL, 2023, 10-K)</search>")
        return env.step(f"<answer>{answer}</answer>")

    def test_correct_answer(self):
        env = make_env()
        out = self._search_then_answer(env, "89.5B")

        assert out["done"] is True
        assert out["reward"].correctness == 1.0

    def test_wrong_answer_zero_correctness_but_format_credit(self):
        env = make_env()
        out = self._search_then_answer(env, "50B")

        assert out["done"] is True
        assert out["reward"].correctness == 0.0
        assert out["reward"].format == 1.0  # <answer> tag present → format credit

    def test_no_answer_tag_at_max_turns(self):
        env = make_env(max_turns=1)
        out = env.step("Final thoughts with no answer tag.")

        assert out["done"] is True
        assert out["reward"].correctness == 0.0
        assert out["reward"].format == 0.0  # no <answer> tag → no format credit


class TestTask2FormatScoring:
    def test_financebench_format_includes_lookup_ticker_year(self):
        env = make_env()
        env.ground_truth = {
            "target": "89.5B",
            "data_source": "PatronusAI/financebench",
            "ticker": "AAPL",
            "year": "2023",
        }
        with patch.object(FinanceSearchEnv, "_execute_tool", return_value=FAKE_RETRIEVAL):
            env.step("<search>CompanyNameToTickerTool(Apple Inc.)</search>")
            env.step("<search>SECFilingTool(revenue, AAPL, 2023, 10-K)</search>")
        out = env.step("<answer>89.5B</answer>")

        assert out["done"] is True
        assert out["reward"].correctness == 1.0
        assert out["reward"].format == 4.0
