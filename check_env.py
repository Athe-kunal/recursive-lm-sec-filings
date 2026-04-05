"""Barebones smoke test for FinanceSearchEnv with real OpenAI multi-turn.

Tests both qa and ranking task types end-to-end:
  - QA:      OpenAI issues <search> calls, env returns <information>, model answers.
  - Ranking: OpenAI reads the prompt and outputs <answer>relevant_types</answer>.

Usage:
    python check_env.py --task both --model gpt-4o-mini
    python check_env.py --task qa
    python check_env.py --task ranking
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, cast

from datasets import Dataset
from loguru import logger
from openai import OpenAI

import verifiers as vf
from rlm_sec.envs.finance_env import FinanceSearchEnv, create_finance_env
from rlm_sec.envs.rewards import (
    QAScoreResult,
    RankingScoreResult,
    compute_qa_format_score,
    compute_ranking_score,
    compute_score,
)
from rlm_sec.envs.tools import FINANCE_MAX_QA_TURNS, FINANCE_MAX_RANKING_TURNS
from rlm_sec.trainer.hf_dataloader import build_qa_prompt, build_ranking_prompt


# ---------------------------------------------------------------------------
# Dataset builders — one row each, schema matches QAExample + ground_truth
# ---------------------------------------------------------------------------


def build_qa_dataset(
    question: str,
    answer: str,
    ticker: str,
    year: str,
    filing_type: str,
) -> Dataset:
    """Build a single-row QA dataset compatible with FinanceSearchEnv."""
    row = {
        "prompt": build_qa_prompt(question),
        "answer": answer,
        "context": "",
        "year": year,
        "ticker_or_company_name": ticker,
        "filing_type": filing_type,
        "data_source": "smoke_test",
        "task_type": "qa",
        "env_class": "null",
        "relevant": [],
        "not_relevant": [],
        "ticker": ticker,
        "ground_truth": {
            "target": answer,
            "ticker": ticker,
            "ticker_or_company_name": ticker,
            "year": year,
            "data_source": "smoke_test",
        },
    }
    return Dataset.from_list([row])


def build_ranking_dataset(
    question: str,
    relevant: list[str],
    not_relevant: list[str],
) -> Dataset:
    """Build a single-row ranking dataset compatible with FinanceSearchEnv."""
    row = {
        "prompt": build_ranking_prompt(question),
        "answer": ", ".join(relevant),
        "context": "",
        "year": "",
        "ticker_or_company_name": "",
        "filing_type": "",
        "data_source": "smoke_test",
        "task_type": "ranking",
        "env_class": "null",
        "relevant": relevant,
        "not_relevant": not_relevant,
        "ticker": "",
        "ground_truth": {
            "relevant": relevant,
            "not_relevant": not_relevant,
        },
    }
    return Dataset.from_list([row])


# ---------------------------------------------------------------------------
# OpenAI + env multi-turn driver
# ---------------------------------------------------------------------------


def call_openai(client: OpenAI, model: str, messages: list[dict[str, Any]]) -> str:
    """Send messages to OpenAI and return the assistant reply."""
    response = client.chat.completions.create(
        model=model,
        messages=cast(list, messages),
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    logger.info(f"{content=}")
    return content


async def run_multiturn(
    env: FinanceSearchEnv,
    prompt: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    max_turns: int,
) -> list[dict[str, Any]]:
    """Drive a multi-turn conversation between OpenAI and the finance env.

    The loop:
      1. Ask OpenAI to produce an assistant turn given current messages.
      2. Pass the assistant turn to env.env_response.
      3. If env returns empty list, the episode is done (<answer> was produced).
      4. If it is the last turn and env still wants to return <information>,
         stop anyway — no next assistant step will consume it.
      5. Otherwise append the env's <information> reply and continue.
    """
    messages: list[dict[str, Any]] = list(prompt)
    state: vf.State = cast(vf.State, {})

    for turn in range(max_turns):
        is_last_turn = turn == max_turns - 1
        logger.info(f"Turn {turn + 1}/{max_turns}")

        assistant_content = call_openai(client, model, messages)
        messages.append({"role": "assistant", "content": assistant_content})

        env_replies = await env.env_response(
            messages=cast(vf.Messages, messages), state=state
        )
        logger.info(f"{env_replies=}")

        if not env_replies:
            logger.info(
                "env returned empty reply — episode complete (<answer> produced)."
            )
            break

        if is_last_turn:
            logger.info(
                "max turns reached — stopping without appending final <information>."
            )
            break

        messages.extend(cast(list, env_replies))

    return messages


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAEpisodeRewards:
    """Collected rewards for a QA episode."""

    per_turn_format: list[float]
    terminal: QAScoreResult


@dataclass(frozen=True)
class RankingEpisodeRewards:
    """Collected rewards for a ranking episode."""

    terminal: RankingScoreResult


def _flatten_assistant_messages(messages: list[dict[str, Any]]) -> str:
    """Concatenate all assistant message contents into a single string."""
    return "".join(
        str(msg.get("content", ""))
        for msg in messages
        if msg.get("role") == "assistant"
    )


def compute_qa_rewards(
    messages: list[dict[str, Any]], ground_truth: dict
) -> QAEpisodeRewards:
    """Compute per-turn intermediate format rewards and terminal QA reward."""
    per_turn_format: list[float] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        fmt = compute_qa_format_score(content, ground_truth)
        per_turn_format.append(fmt)

    full_completion = _flatten_assistant_messages(messages)
    terminal = compute_score(full_completion, ground_truth)
    return QAEpisodeRewards(per_turn_format=per_turn_format, terminal=terminal)


def compute_ranking_rewards(
    messages: list[dict[str, Any]], ground_truth: dict
) -> RankingEpisodeRewards:
    """Compute terminal ranking reward (single-turn, no intermediate rewards)."""
    full_completion = _flatten_assistant_messages(messages)
    terminal = compute_ranking_score(full_completion, ground_truth)
    return RankingEpisodeRewards(terminal=terminal)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_conversation(task_type: str, messages: list[dict[str, Any]]) -> None:
    """Pretty-print the full conversation for inspection."""
    print(f"\n{'=' * 60}")
    print(f"  {task_type.upper()} Smoke Test — Full Conversation")
    print(f"{'=' * 60}")
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        print(f"\n[{role}]\n{content}")
    print(f"\n{'=' * 60}\n")


def print_qa_rewards(rewards: QAEpisodeRewards) -> None:
    """Print per-turn and terminal rewards for a QA episode."""
    print("\n--- QA Rewards ---")
    for i, fmt in enumerate(rewards.per_turn_format, start=1):
        print(f"  Turn {i} intermediate format reward: {fmt:.3f}")
    print(f"  Terminal correctness : {rewards.terminal.correctness:.3f}")
    print(f"  Terminal format      : {rewards.terminal.format:.3f}")
    print()


def print_ranking_rewards(rewards: RankingEpisodeRewards) -> None:
    """Print terminal rewards for a ranking episode."""
    t = rewards.terminal
    print("\n--- Ranking Rewards ---")
    print(f"  Terminal correctness : {t.correctness:.3f}")
    print(f"  Terminal format      : {t.format:.3f}")
    print(f"  Precision            : {t.precision:.3f}")
    print(f"  Recall               : {t.recall:.3f}")
    print(f"  F1                   : {t.f1:.3f}")
    print()


# ---------------------------------------------------------------------------
# Per-task smoke runners
# ---------------------------------------------------------------------------


async def smoke_qa(client: OpenAI, model: str) -> None:
    """End-to-end QA smoke: OpenAI searches SEC filings then answers."""
    question = (
        "In 2023, for Hasbro, Inc., what percentage of the company's "
        "full-year revenues were earned in the second half of 2023?"
    )
    answer = (
        "In 2023, approximately 56% of the company's full-year revenues "
        "were earned in the second half of the year."
    )
    dataset = build_qa_dataset(
        question=question,
        answer=answer,
        ticker="HAS",
        year="2023",
        filing_type="10-K",
    )
    ground_truth: dict = dataset[0]["ground_truth"]
    logger.info(f"QA dataset built: {question=}")
    logger.info(f"{FINANCE_MAX_QA_TURNS=}")

    env = create_finance_env(dataset=dataset)
    prompt: list[dict[str, Any]] = cast(list, dataset[0]["prompt"])
    messages = await run_multiturn(
        env=env,
        prompt=prompt,
        client=client,
        model=model,
        max_turns=FINANCE_MAX_QA_TURNS,
    )
    print_conversation("qa", messages)
    rewards = compute_qa_rewards(messages=messages, ground_truth=ground_truth)
    print_qa_rewards(rewards)


async def smoke_ranking(client: OpenAI, model: str) -> None:
    """End-to-end ranking smoke: OpenAI classifies relevant document types."""
    question = (
        "What were Hasbro's key risk factors and full-year revenue "
        "breakdown in fiscal year 2023?"
    )
    relevant = ["10-K"]
    not_relevant = ["10-Q", "8-K", "Earnings", "DEF14A"]

    dataset = build_ranking_dataset(
        question=question,
        relevant=relevant,
        not_relevant=not_relevant,
    )
    ground_truth: dict = dataset[0]["ground_truth"]
    logger.info(f"Ranking dataset built: {question=}")
    logger.info(f"{FINANCE_MAX_RANKING_TURNS=}")

    env = create_finance_env(dataset=dataset)
    prompt: list[dict[str, Any]] = cast(list, dataset[0]["prompt"])
    messages = await run_multiturn(
        env=env,
        prompt=prompt,
        client=client,
        model=model,
        max_turns=FINANCE_MAX_RANKING_TURNS,
    )
    print_conversation("ranking", messages)
    rewards = compute_ranking_rewards(messages=messages, ground_truth=ground_truth)
    print_ranking_rewards(rewards)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main_async(model: str, task: str) -> None:
    """Run the selected smoke test(s)."""
    logger.info(f"{model=} {task=}")
    client = OpenAI()

    if task in ("qa", "both"):
        await smoke_qa(client=client, model=model)

    if task in ("ranking", "both"):
        await smoke_ranking(client=client, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test FinanceSearchEnv with OpenAI multi-turn."
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--task",
        choices=["qa", "ranking", "both"],
        default="both",
        help="Which task type to test.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(model=args.model, task=args.task))


if __name__ == "__main__":
    main()
