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
from typing import cast

from datasets import Dataset
from loguru import logger
from openai import OpenAI

import verifiers as vf
from rlm_sec.envs.finance_env import FinanceSearchEnv, create_finance_env
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

def call_openai(client: OpenAI, model: str, messages: vf.Messages) -> str:
    """Send messages to OpenAI and return the assistant reply."""
    response = client.chat.completions.create(
        model=model,
        messages=cast(list, messages),
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    logger.info(f"{content[:300]=}")
    return content


async def run_multiturn(
    env: FinanceSearchEnv,
    prompt: vf.Messages,
    client: OpenAI,
    model: str,
    max_turns: int,
) -> vf.Messages:
    """Drive a multi-turn conversation between OpenAI and the finance env.

    The loop:
      1. Ask OpenAI to produce an assistant turn given current messages.
      2. Pass the assistant turn to env.env_response.
      3. If env returns empty list, the episode is done (<answer> was produced).
      4. Otherwise append the env's <information> reply and continue.
    """
    messages: vf.Messages = list(prompt)
    state: vf.State = cast(vf.State, {})

    for turn in range(max_turns):
        logger.info(f"Turn {turn + 1}/{max_turns}")
        assistant_content = call_openai(client, model, messages)
        messages.append({"role": "assistant", "content": assistant_content})

        env_replies = await env.env_response(messages=messages, state=state)
        logger.info(f"{env_replies=}")

        if not env_replies:
            logger.info("env returned empty reply — episode complete.")
            break

        messages.extend(cast(list, env_replies))

    return messages


def print_conversation(task_type: str, messages: vf.Messages) -> None:
    """Pretty-print the full conversation for inspection."""
    print(f"\n{'=' * 60}")
    print(f"  {task_type.upper()} Smoke Test — Full Conversation")
    print(f"{'=' * 60}")
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        print(f"\n[{role}]\n{content}")
    print(f"\n{'=' * 60}\n")


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
    logger.info(f"QA dataset built: {question=}")
    logger.info(f"{FINANCE_MAX_QA_TURNS=}")

    env = create_finance_env(dataset=dataset)
    prompt: vf.Messages = cast(vf.Messages, dataset[0]["prompt"])
    messages = await run_multiturn(
        env=env,
        prompt=prompt,
        client=client,
        model=model,
        max_turns=FINANCE_MAX_QA_TURNS,
    )
    print_conversation("qa", messages)


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
    logger.info(f"Ranking dataset built: {question=}")
    logger.info(f"{FINANCE_MAX_RANKING_TURNS=}")

    env = create_finance_env(dataset=dataset)
    prompt: vf.Messages = cast(vf.Messages, dataset[0]["prompt"])
    messages = await run_multiturn(
        env=env,
        prompt=prompt,
        client=client,
        model=model,
        max_turns=FINANCE_MAX_RANKING_TURNS,
    )
    print_conversation("ranking", messages)


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
