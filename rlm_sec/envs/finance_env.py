from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import verifiers as vf
from datasets import Dataset
from omegaconf import DictConfig

from rlm_sec.envs.rewards import (
    DataTaskType,
    TaskType,
    compute_ranking_score,
    compute_score,
    reward_action_format,
)
from rlm_sec.envs.tools import FinanceSearchTools
from settings import (
    EARNINGS_TRANSCRIPT_TOOL_ENDPOINT,
    SEC_FILING_TOOL_ENDPOINT,
    env_settings,
)


ToolName = Literal[
    "SECFilingTool",
    "EarningsTranscriptTool",
    "CompanyNameToTickerTool",
]

SEARCH_TOOL_ROUTING: dict[ToolName, tuple[str, TaskType | None]] = {
    "SECFilingTool": ("sec_filing", "sec_filings"),
    "EarningsTranscriptTool": ("earnings_transcript", "earning_transcripts"),
    "CompanyNameToTickerTool": ("company_name_to_ticker", None),
}


@dataclass(frozen=True)
class SearchEnvConfig:
    """Configuration for finance retrieval tools."""

    log_requests: bool = False
    sec_filing_tool_url: str = f"{env_settings.server_url}{SEC_FILING_TOOL_ENDPOINT}"
    earnings_transcript_tool_url: str = (
        f"{env_settings.server_url}{EARNINGS_TRANSCRIPT_TOOL_ENDPOINT}"
    )
    topk: int = 3
    timeout: int = 30
    max_turns: int = 4


@dataclass(frozen=True)
class ParsedSearch:
    """Structured representation of a parsed <search> action."""

    action: str
    tool_name: ToolName | None = None
    query: str | None = None
    ticker: str | None = None
    year: str | None = None
    filing_type_or_quarter: str | None = None
    lookup_arg: str | None = None


@dataclass(frozen=True)
class RewardBreakdown:
    """Terminal and intermediate rewards exposed through rollout state."""

    correctness: float
    format: float


def _parse_tool_call(inner_action: str) -> tuple[str, str] | None:
    """Extract the tool name and raw args from `ToolName(arg1, ...)`."""
    call_match = re.match(r"(\w+)\((.+)\)$", inner_action, re.DOTALL)
    if call_match is None:
        return None
    return call_match.group(1).strip(), call_match.group(2)


def _parse_search_action(action: str) -> ParsedSearch:
    """Parse a `<search>...</search>` action into a typed object."""
    search_match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
    if search_match is None:
        return ParsedSearch(action=action)

    inner = search_match.group(1).strip()
    parsed_call = _parse_tool_call(inner)
    if parsed_call is None:
        return ParsedSearch(action=action)

    tool_name_str, raw_args = parsed_call
    if tool_name_str not in SEARCH_TOOL_ROUTING:
        return ParsedSearch(action=action)

    tool_name = tool_name_str  # type: ignore[assignment]
    if tool_name == "CompanyNameToTickerTool":
        return ParsedSearch(
            action=action, tool_name=tool_name, lookup_arg=raw_args.strip()
        )

    parts = [part.strip() for part in raw_args.rsplit(",", 3)]
    if len(parts) != 4:
        return ParsedSearch(action=action)

    query, ticker, year, filing_type_or_quarter = parts
    return ParsedSearch(
        action=action,
        tool_name=tool_name,
        query=query,
        ticker=ticker,
        year=year,
        filing_type_or_quarter=filing_type_or_quarter,
    )


def _intermediate_format_reward(parsed: ParsedSearch) -> float:
    """Compute action-format reward for a search action."""
    if parsed.tool_name is None:
        return 0.0

    route = SEARCH_TOOL_ROUTING[parsed.tool_name]
    task_type = route[1]
    if task_type is None:
        return 1.0

    return reward_action_format(
        tool_group_name=_tool_group_name(parsed.tool_name),
        ticker=parsed.ticker or "",
        year=parsed.year or "",
        filing_type=parsed.filing_type_or_quarter or "",
        task_type=task_type,
    )


def _tool_group_name(tool_name: ToolName) -> str:
    """Translate public tool names to historical tool group names."""
    match tool_name:
        case "SECFilingTool":
            return "SECFilingToolGroup"
        case "EarningsTranscriptTool":
            return "EarningsTranscriptToolGroup"
        case _:
            return "CompanyNameToTickerToolGroup"


def _compute_terminal_reward(
    completion_text: str, info: dict[str, Any]
) -> RewardBreakdown:
    """Compute terminal reward by task type (`qa` or `ranking`)."""
    task_type: DataTaskType = info.get("task_type", "qa")
    ground_truth = info.get("ground_truth", {})

    if task_type == "qa":
        correctness, fmt = compute_score(completion_text, ground_truth)
        return RewardBreakdown(correctness=correctness, format=fmt)

    if task_type == "ranking":
        correctness, fmt = compute_ranking_score(completion_text, ground_truth)
        return RewardBreakdown(correctness=correctness, format=fmt)

    return RewardBreakdown(correctness=0.0, format=0.0)


def _completion_to_text(completion: vf.Messages) -> str:
    """Flatten completion messages into a single string."""
    return "".join(message.get("content", "") for message in completion)


def _normalize_info(info: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional rollout info for reward functions."""
    if info is None:
        return {"task_type": "qa", "ground_truth": {}}
    return info


class FinanceSearchEnv(vf.MultiTurnEnv):
    """Custom Verifiers environment that preserves legacy `<search>` protocol."""

    def __init__(self, tools: FinanceSearchTools, **kwargs: Any):
        super().__init__(**kwargs)
        self._tools = tools

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        """Execute search tools from the most recent assistant action."""
        action = messages[-1].get("content", "") if messages else ""
        parsed = _parse_search_action(action)

        if "<answer>" in action and "</answer>" in action:
            state["final_env_response"] = []
            return []

        if parsed.tool_name is None:
            response = "\n<information>Invalid <search> format.</information>\n"
            return [{"role": "user", "content": response}]

        tool_output = await self._execute_parsed_action(parsed)
        return [
            {
                "role": "user",
                "content": f"\n<information>\n{tool_output}</information>\n",
            }
        ]

    async def _execute_parsed_action(self, parsed: ParsedSearch) -> str:
        """Route parsed action to the corresponding finance tool."""
        if parsed.tool_name == "SECFilingTool":
            return await self._tools.sec_filing(
                query=parsed.query or "",
                ticker=parsed.ticker or "",
                year=parsed.year or "",
                filing_type=parsed.filing_type_or_quarter or "",
            )

        if parsed.tool_name == "EarningsTranscriptTool":
            return await self._tools.earnings_transcript(
                query=parsed.query or "",
                ticker=parsed.ticker or "",
                year=parsed.year or "",
                quarter=parsed.filing_type_or_quarter or "",
            )

        return await self._tools.company_name_to_ticker(name=parsed.lookup_arg or "")


async def reward_correctness(
    completion: vf.Messages, info: dict[str, Any] | None
) -> float:
    """Reward function for terminal correctness."""
    normalized_info = _normalize_info(info)
    completion_text = _completion_to_text(completion)
    reward = _compute_terminal_reward(completion_text, normalized_info)
    return reward.correctness


async def reward_format(completion: vf.Messages, info: dict[str, Any] | None) -> float:
    """Reward function for format compliance.

    For intermediate search calls this captures action-format quality. For final
    answers, this returns the final-format reward from existing scorers.
    """
    normalized_info = _normalize_info(info)
    completion_text = _completion_to_text(completion)

    latest_action = completion[-1].get("content", "") if completion else ""
    if "<answer>" in latest_action and "</answer>" in latest_action:
        reward = _compute_terminal_reward(completion_text, normalized_info)
        return reward.format

    return _intermediate_format_reward(_parse_search_action(latest_action))


def create_finance_env(
    dataset: Dataset,
    eval_dataset: Dataset | None = None,
    env_config: SearchEnvConfig | DictConfig | None = None,
) -> FinanceSearchEnv:
    """Create a Prime RL compatible Verifiers environment."""
    if env_config is None:
        config = SearchEnvConfig()
    elif isinstance(env_config, SearchEnvConfig):
        config = env_config
    else:
        config = SearchEnvConfig(**dict(env_config))
    tools = FinanceSearchTools(
        sec_filing_tool_url=config.sec_filing_tool_url,
        earnings_transcript_tool_url=config.earnings_transcript_tool_url,
        topk=config.topk,
        timeout=config.timeout,
        log_requests=config.log_requests,
    )
    rubric = vf.Rubric(funcs=[reward_correctness, reward_format], weights=[1.0, 1.0])

    return FinanceSearchEnv(
        tools=tools,
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        max_turns=config.max_turns,
    )
