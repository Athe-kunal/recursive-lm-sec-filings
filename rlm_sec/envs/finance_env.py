"""Prime-RL Verifiers environment for SEC filings and earnings search tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from rlm_sec.envs.rewards import compute_ranking_score, compute_score, extract_solution
from rlm_sec.envs.tools import call_search_api
from settings import (
    EARNINGS_TRANSCRIPT_TOOL_ENDPOINT,
    SEC_FILING_TOOL_ENDPOINT,
    env_settings,
)


@dataclass(frozen=True)
class SearchToolConfig:
    """Runtime configuration for SEC filings and transcript search tools."""

    sec_filing_tool_url: str = f"{env_settings.server_url}{SEC_FILING_TOOL_ENDPOINT}"
    earnings_transcript_tool_url: str = (
        f"{env_settings.server_url}{EARNINGS_TRANSCRIPT_TOOL_ENDPOINT}"
    )
    topk: int = 3
    timeout: int = 30
    log_requests: bool = False


class FinanceSearchTools:
    """Implements tool call handlers consumed by Verifiers ToolEnv."""

    def __init__(self, config: SearchToolConfig) -> None:
        self._config = config

    async def sec_filing_tool(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        """Search SEC filing chunks for a given ticker/year/filing type.

        Args:
            query: Natural language search query.
            ticker: Public company ticker symbol.
            year: Filing year.
            filing_type: SEC filing type (for example, 10-K).

        Returns:
            A formatted block of retrieved text chunks.
        """
        return self._search(
            service_url=self._config.sec_filing_tool_url,
            query=query,
            ticker=ticker,
            year=year,
            filing_type=filing_type,
        )

    async def earnings_transcript_tool(
        self,
        query: str,
        ticker: str,
        year: str,
        quarter: str,
    ) -> str:
        """Search earnings transcript chunks for a given ticker/year/quarter."""
        return self._search(
            service_url=self._config.earnings_transcript_tool_url,
            query=query,
            ticker=ticker,
            year=year,
            filing_type=quarter,
        )

    async def company_name_to_ticker_tool(self, name: str) -> str:
        """Resolve a company name into a ticker symbol.

        This tool currently returns the input if it already looks like a ticker-like
        token. For richer mappings, retain the existing backend service integration.
        """
        return name.strip().upper()

    def _search(
        self,
        service_url: str,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        response, error_msg = call_search_api(
            retrieval_service_url=service_url,
            query=query,
            ticker=ticker,
            year=year,
            filing_type=filing_type,
            topk=self._config.topk,
            timeout=self._config.timeout,
            log_requests=self._config.log_requests,
        )
        if error_msg:
            return f"Search error: {error_msg}"
        if not isinstance(response, list) or not response:
            return "No search results found."
        return _format_chunks(response)


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    """Render vector search chunks into a compact model-facing string."""
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        lines.append(f"Doc {idx}: {text}")
    return "\n".join(lines)


def _load_parquet(paths: list[str]) -> Dataset:
    """Load one or more parquet files and concatenate into a single dataset."""
    datasets_list: list[Dataset] = []
    for path in paths:
        dataset = load_dataset("parquet", data_files=path, split="train")
        datasets_list.append(dataset)
    if len(datasets_list) == 1:
        return datasets_list[0]
    return concatenate_datasets(datasets_list)


async def finance_reward(
    completion: list[dict[str, str]],
    task_type: str,
    answer: str,
    relevant: list[str],
) -> float:
    """Compute task-aware correctness reward for QA and ranking tasks."""
    response = completion[-1]["content"]
    if task_type == "qa":
        score, _ = compute_score(response, {"target": answer})
        return score
    if task_type == "ranking":
        score, _ = compute_ranking_score(response, {"relevant": relevant})
        return score
    return 0.0


async def answer_tag_format_reward(completion: list[dict[str, str]]) -> float:
    """Reward the model for returning an explicit <answer>...</answer> block."""
    response = completion[-1]["content"]
    return 1.0 if extract_solution(response) is not None else 0.0


def load_environment(
    train_paths: tuple[str, ...] = ("data/train.parquet",),
    eval_paths: tuple[str, ...] = ("data/validation.parquet",),
    max_turns: int = 4,
    topk: int = 3,
    timeout: int = 30,
    log_requests: bool = False,
):
    """Create a Prime-RL/Verifiers ToolEnv for finance search training.

    Args:
        train_paths: One or more parquet files for training rollouts.
        eval_paths: One or more parquet files for evaluation rollouts.
        max_turns: Maximum number of tool/assistant turns per rollout.
        topk: Number of retrieval chunks per tool call.
        timeout: HTTP timeout in seconds per retrieval call.
        log_requests: Whether to log retrieval requests.

    Returns:
        A configured `verifiers.ToolEnv` instance.
    """
    import verifiers as vf

    dataset = _load_parquet(list(train_paths))
    eval_dataset = _load_parquet(list(eval_paths))

    config = SearchToolConfig(topk=topk, timeout=timeout, log_requests=log_requests)
    tools = FinanceSearchTools(config)

    rubric = vf.Rubric(funcs=[finance_reward, answer_tag_format_reward])

    return vf.ToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        tools=[
            tools.sec_filing_tool,
            tools.earnings_transcript_tool,
            tools.company_name_to_ticker_tool,
        ],
        rubric=rubric,
        max_turns=max_turns,
    )
