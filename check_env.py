from __future__ import annotations

import argparse
import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol

from datasets import Dataset
from loguru import logger

import verifiers as vf
from rlm_sec.envs.finance_env import (
    FinanceSearchEnv,
    SearchEnvConfig,
    reward_correctness,
    reward_format,
    reward_qa_company_to_ticker,
    reward_qa_ticker_match,
    reward_qa_year_match,
    reward_ranking_precision,
    reward_ranking_recall,
)
from rlm_sec.envs.tools import FINANCE_MAX_QA_TURNS, FINANCE_MAX_RANKING_TURNS


@dataclass(frozen=True)
class SmokeTestConfig:
    """Runtime configuration for finance environment smoke tests."""

    question: str
    data: str
    model: str
    use_real_openai: bool
    search_mode: str
    ticker: str
    year: str
    filing_type: str
    retrieval_query: str


class ChatClient(Protocol):
    """Protocol for a client that answers a question using context."""

    def answer_question(self, question: str, context_data: str) -> str:
        """Answer a question with the provided context."""


class DummyOpenAIClient:
    """Offline fallback that mimics OpenAI responses for local tests."""

    def answer_question(self, question: str, context_data: str) -> str:
        logger.info(f"{question=}")
        logger.info(f"{context_data=}")
        return (
            "[DUMMY OPENAI CLIENT]\n"
            f"Question: {question}\n"
            "Answer: FinanceSearchEnv returned retrieval context successfully.\n"
            f"Context used: {context_data[:500]}"
        )


class OpenAIResponsesClient:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, model: str):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def answer_question(self, question: str, context_data: str) -> str:
        logger.info(f"{question=}")
        logger.info(f"{context_data=}")
        prompt = (
            "You are validating a finance retrieval environment. "
            "Use only the provided context and answer concisely.\n\n"
            f"Question: {question}\n"
            f"Context:\n{context_data}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


class DummyFinanceSearchTools:
    """Simple deterministic tools used when retrieval dependencies are unavailable."""

    async def sec_filing(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        logger.info(f"{query=}")
        logger.info(f"{ticker=}")
        logger.info(f"{year=}")
        logger.info(f"{filing_type=}")
        return (
            f"Dummy SEC filing result for {ticker} {year} {filing_type}. "
            f"Matched query: {query}."
        )

    async def earnings_transcript(
        self,
        query: str,
        ticker: str,
        year: str,
        quarter: str,
    ) -> str:
        logger.info(f"{query=}")
        logger.info(f"{ticker=}")
        logger.info(f"{year=}")
        logger.info(f"{quarter=}")
        return (
            f"Dummy earnings transcript result for {ticker} {year} {quarter}. "
            f"Matched query: {query}."
        )

    async def company_name_to_ticker(self, name: str) -> str:
        logger.info(f"{name=}")
        lookup = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
        }
        return lookup.get(name.strip().lower(), "UNKNOWN")


class RetrievalFinanceSearchTools:
    """Retrieval-backed tools that use Chroma hybrid search."""

    def __init__(self, top_k: int = 3):
        self._top_k = top_k
        self._vector_store = self._build_vector_store()

    def _build_vector_store(self) -> Any:
        from finance_data.dataloader.vector_store import ChromaVectorStore

        return ChromaVectorStore()

    async def sec_filing(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        logger.info(f"{query=}")
        logger.info(f"{ticker=}")
        logger.info(f"{year=}")
        logger.info(f"{filing_type=}")
        return self._search_and_format(
            query=query,
            ticker=ticker,
            year=year,
            filing_type=filing_type,
        )

    async def earnings_transcript(
        self,
        query: str,
        ticker: str,
        year: str,
        quarter: str,
    ) -> str:
        logger.info(f"{query=}")
        logger.info(f"{ticker=}")
        logger.info(f"{year=}")
        logger.info(f"{quarter=}")
        return self._search_and_format(
            query=query,
            ticker=ticker,
            year=year,
            filing_type=quarter,
        )

    async def company_name_to_ticker(self, name: str) -> str:
        from finance_data.filings.utils import company_to_ticker

        logger.info(f"{name=}")
        resolved = company_to_ticker(name)
        return (resolved or "UNKNOWN").upper()

    def _search_and_format(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        available_filings = self._vector_store.list_filings(ticker, year)
        logger.info(f"{available_filings=}")

        if not self._is_embedded(
            available_filings=available_filings, filing_type=filing_type
        ):
            return self._build_not_embedded_message(
                ticker=ticker,
                year=year,
                requested=filing_type,
                available_filings=available_filings,
            )

        hits = self._vector_store.hybrid_search(
            ticker=ticker,
            year=year,
            filing_type=filing_type,
            query=query,
            top_k=self._top_k,
        )
        logger.info(f"{hits=}")
        return self._hits_to_text(hits=hits)

    def _is_embedded(
        self,
        available_filings: list[dict[str, Any]],
        filing_type: str,
    ) -> bool:
        return any(item.get("filing_type") == filing_type for item in available_filings)

    def _build_not_embedded_message(
        self,
        ticker: str,
        year: str,
        requested: str,
        available_filings: list[dict[str, Any]],
    ) -> str:
        logger.info(f"{ticker=}")
        logger.info(f"{year=}")
        logger.info(f"{requested=}")
        logger.info(f"{available_filings=}")
        lines = [
            "Requested filing is not embedded for this ticker/year.",
            f"ticker={ticker}",
            f"year={year}",
            f"requested={requested}",
        ]
        if not available_filings:
            lines.append("available_filings=[]")
            return "\n".join(lines)

        lines.append("available_filings:")
        for filing in available_filings:
            lines.append(f"- {filing}")
        return "\n".join(lines)

    def _hits_to_text(self, hits: list[tuple[Any, float]]) -> str:
        if not hits:
            return "No retrieval hits found."

        lines: list[str] = []
        for index, (chunk, score) in enumerate(hits, start=1):
            chunk_dict = dataclasses.asdict(chunk)
            text = str(chunk_dict.get("text", "")).strip()
            metadata = {k: v for k, v in chunk_dict.items() if k != "text"}
            lines.append(
                f"Hit {index} | score={score:.4f} | metadata={metadata}\n{text}"
            )
        return "\n\n".join(lines)


def build_smoke_dataset(ticker: str, year: str) -> Dataset:
    """Create a minimal dataset row required by FinanceSearchEnv."""
    row = {
        "question": "What is the company ticker?",
        "task_type": "qa",
        "data_source": "sec_filings",
        "year": year,
        "ticker": ticker,
        "ticker_or_company_name": ticker,
        "ground_truth": {
            "target": ticker,
            "ticker": ticker,
            "year": year,
            "data_source": "sec_filings",
        },
    }
    return Dataset.from_list([row])


def build_env(tools: Any, ticker: str, year: str) -> FinanceSearchEnv:
    """Instantiate FinanceSearchEnv with provided tools and shared rubric."""
    dataset = build_smoke_dataset(ticker=ticker, year=year)
    rubric = vf.Rubric(funcs=[reward_correctness, reward_format], weights=[1.0, 1.0])
    rubric.add_metric(reward_ranking_precision)
    rubric.add_metric(reward_ranking_recall)
    rubric.add_metric(reward_qa_company_to_ticker)
    rubric.add_metric(reward_qa_ticker_match)
    rubric.add_metric(reward_qa_year_match)

    config = SearchEnvConfig(
        max_qa_turns=FINANCE_MAX_QA_TURNS,
        max_ranking_turns=FINANCE_MAX_RANKING_TURNS,
    )
    return FinanceSearchEnv(
        tools=tools,
        dataset=dataset,
        eval_dataset=dataset,
        rubric=rubric,
        max_qa_turns=config.max_qa_turns,
        max_ranking_turns=config.max_ranking_turns,
    )


def build_search_tools(search_mode: str) -> Any:
    """Construct search tools from the selected mode."""
    if search_mode == "dummy":
        return DummyFinanceSearchTools()

    try:
        return RetrievalFinanceSearchTools(top_k=3)
    except Exception as error:  # pylint: disable=broad-except
        logger.warning(f"{error=}")
        logger.warning("Falling back to dummy tools because retrieval tools failed.")
        return DummyFinanceSearchTools()


def build_chat_client(config: SmokeTestConfig) -> ChatClient:
    """Create either a real OpenAI client or an offline dummy client."""
    if not config.use_real_openai:
        return DummyOpenAIClient()

    try:
        return OpenAIResponsesClient(model=config.model)
    except ImportError:
        logger.warning("openai package is not installed. Falling back to dummy client.")
        return DummyOpenAIClient()
    except Exception as error:  # pylint: disable=broad-except
        logger.warning(f"{error=}")
        logger.warning(
            "Failed to initialize OpenAI client. Falling back to dummy client."
        )
        return DummyOpenAIClient()


async def execute_qa_two_turn_smoke(
    env: FinanceSearchEnv, config: SmokeTestConfig
) -> str:
    """Run QA-style search then answer (two assistant steps), like a short rollout.

    Turn 1: ``<search>`` → environment returns ``<information>``.
    Turn 2: same transcript plus ``<answer>`` → env terminates (empty response,
    ``final_env_response`` set). This matches ``max_qa_turns`` > 1; ranking uses
    ``max_ranking_turns=1`` and would not use this search-then-answer pattern.
    """
    search_action = (
        f"<search>SECFilingTool({config.retrieval_query}, {config.ticker}, "
        f"{config.year}, {config.filing_type})</search>"
    )
    answer_action = f"<answer>{config.ticker}</answer>"
    logger.info(f"{search_action=}")
    logger.info(f"{answer_action=}")

    state: vf.State = {}
    turn1_messages: vf.Messages = [{"role": "assistant", "content": search_action}]
    turn1_response = await env.env_response(messages=turn1_messages, state=state)
    logger.info(f"{turn1_response=}")

    turn1_text = ""
    if turn1_response:
        turn1_text = str(turn1_response[0].get("content", "") or "")

    turn2_messages: vf.Messages = [
        {"role": "assistant", "content": search_action},
        {"role": "user", "content": turn1_text or "\n<information>(empty)</information>\n"},
        {"role": "assistant", "content": answer_action},
    ]
    turn2_response = await env.env_response(messages=turn2_messages, state=state)
    logger.info(f"{turn2_response=}")
    logger.info(f"{state.get('final_env_response')=}")

    parts = [
        f"--- Turn 1 (search) ---\n{turn1_text or '(no <information> payload)'}",
        (
            "--- Turn 2 (answer) ---\n"
            f"env returned {len(turn2_response)} message(s); "
            f"final_env_response set: {state.get('final_env_response') is not None}"
        ),
    ]
    return "\n\n".join(parts)


def parse_args() -> SmokeTestConfig:
    """Parse command line args for smoke test execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for FinanceSearchEnv using retrieval hybrid search or dummy tools."
        )
    )
    parser.add_argument("--question", default="Summarize the retrieved context.")
    parser.add_argument(
        "--data",
        default="Explain revenue trend from the retrieved filing chunks.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--use-real-openai", action="store_true")
    parser.add_argument(
        "--search-mode",
        choices=["retrieval", "dummy"],
        default="retrieval",
        help="Use retrieval hybrid search by default; dummy mode is offline-only.",
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--filing-type", default="10-K")
    parser.add_argument("--retrieval-query", default="revenue growth")

    args = parser.parse_args()
    return SmokeTestConfig(
        question=args.question,
        data=args.data,
        model=args.model,
        use_real_openai=args.use_real_openai,
        search_mode=args.search_mode,
        ticker=args.ticker,
        year=args.year,
        filing_type=args.filing_type,
        retrieval_query=args.retrieval_query,
    )


async def run_smoke_test(config: SmokeTestConfig) -> str:
    """Run end-to-end smoke test and return model answer."""
    logger.info(f"{config=}")
    logger.info(
        f"{FINANCE_MAX_QA_TURNS=} {FINANCE_MAX_RANKING_TURNS=} "
        "(QA smoke uses two steps: search then answer.)"
    )
    tools = build_search_tools(search_mode=config.search_mode)
    env = build_env(tools=tools, ticker=config.ticker, year=config.year)

    env_data = await execute_qa_two_turn_smoke(env=env, config=config)
    logger.info(f"{env_data=}")

    context_data = f"User data: {config.data}\nEnvironment data: {env_data}"
    client = build_chat_client(config=config)
    answer = client.answer_question(question=config.question, context_data=context_data)
    logger.info(f"{answer=}")
    return answer


def main() -> None:
    """Entrypoint for script execution."""
    config = parse_args()
    answer = asyncio.run(run_smoke_test(config=config))
    print("\n=== Smoke Test Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
