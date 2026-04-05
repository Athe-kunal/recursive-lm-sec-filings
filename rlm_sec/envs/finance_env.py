from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, cast

import verifiers as vf
from datasets import Dataset
from omegaconf import DictConfig

from rlm_sec.envs.rewards import (
    DataTaskType,
    TaskType,
    compute_qa_company_to_ticker_score,
    compute_qa_ticker_match_score,
    compute_qa_year_match_score,
    compute_ranking_score,
    compute_score,
    reward_action_format,
)
from rlm_sec.envs.tools import (
    FINANCE_MAX_QA_TURNS,
    FINANCE_MAX_RANKING_TURNS,
    FinanceSearchTools,
)
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
    max_qa_turns: int = FINANCE_MAX_QA_TURNS
    max_ranking_turns: int = FINANCE_MAX_RANKING_TURNS


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
class QARewardBreakdown:
    """Terminal reward payload for QA tasks."""

    correctness: float
    format: float


@dataclass(frozen=True)
class RankingRewardBreakdown:
    """Terminal reward payload for ranking tasks."""

    correctness: float
    format: float
    precision: float
    recall: float
    f1: float


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


def _build_ground_truth(normalized_info: dict[str, Any]) -> dict[str, Any]:
    """Build the ground_truth dict used by all scoring functions.

    Top-level info fields (data_source, year, ticker, ticker_or_company_name) are
    promoted into ground_truth so scoring helpers always find them in one place.
    """
    ground_truth = dict(normalized_info.get("ground_truth", {}))
    for key in ("data_source", "year", "ticker", "ticker_or_company_name"):
        if key in normalized_info and key not in ground_truth:
            ground_truth[key] = normalized_info[key]
    return ground_truth


def _compute_terminal_reward(
    completion_text: str, info: dict[str, Any]
) -> QARewardBreakdown | RankingRewardBreakdown:
    """Compute terminal reward by task type (`qa` or `ranking`)."""
    task_type: DataTaskType = info.get("task_type", "qa")
    ground_truth = _build_ground_truth(info)

    if task_type == "qa":
        qa_reward = compute_score(completion_text, ground_truth)
        return QARewardBreakdown(
            correctness=qa_reward.correctness,
            format=qa_reward.format,
        )

    if task_type == "ranking":
        ranking_reward = compute_ranking_score(completion_text, ground_truth)
        return RankingRewardBreakdown(
            correctness=ranking_reward.correctness,
            format=ranking_reward.format,
            precision=ranking_reward.precision,
            recall=ranking_reward.recall,
            f1=ranking_reward.f1,
        )

    return QARewardBreakdown(correctness=0.0, format=0.0)


def _completion_to_text(completion: vf.Messages) -> str:
    """Flatten completion messages into a single string."""
    return "".join(message.get("content", "") for message in completion)


def _assistant_turn_count(completion: vf.Messages) -> int:
    """Return the number of assistant turns in a rollout completion."""
    assistant_turns = sum(
        1
        for message in completion
        if isinstance(message, dict) and message.get("role") == "assistant"
    )
    return max(assistant_turns, 1)


def _normalize_info(info: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional rollout info for reward functions."""
    if info is None:
        return {"task_type": "qa", "ground_truth": {}}
    return info


def _max_turns_for_task_type(task_type: str, max_qa_turns: int, max_ranking_turns: int) -> int:
    """Return trajectory cap for this episode (assistant steps)."""
    if task_type == "ranking":
        return max_ranking_turns
    return max_qa_turns


def _rollout_task_type(state: vf.State) -> DataTaskType:
    """Read DataTaskType from rollout info (defaults to qa)."""
    info = state.get("info")
    if isinstance(info, dict):
        raw = info.get("task_type", "qa")
        if raw in ("qa", "ranking"):
            return cast(DataTaskType, raw)
    return "qa"


class FinanceSearchEnv(vf.MultiTurnEnv):
    """Custom Verifiers environment that preserves legacy `<search>` protocol."""

    def __init__(self, tools: FinanceSearchTools, **kwargs: Any):
        max_qa_turns = int(kwargs.pop("max_qa_turns", FINANCE_MAX_QA_TURNS))
        max_ranking_turns = int(kwargs.pop("max_ranking_turns", FINANCE_MAX_RANKING_TURNS))
        kwargs.setdefault("max_turns", max(max_qa_turns, max_ranking_turns))
        super().__init__(**kwargs)
        self._tools = tools
        self.max_qa_turns = max_qa_turns
        self.max_ranking_turns = max_ranking_turns

    @vf.stop
    async def max_turns_reached(self, state: vf.State) -> bool:
        """Stop when this episode's assistant-step budget is exhausted."""
        limit = _max_turns_for_task_type(
            _rollout_task_type(state), self.max_qa_turns, self.max_ranking_turns
        )
        return len(state["trajectory"]) >= limit and limit > 0

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

    When the final assistant message contains <answer> tags the terminal format
    score is returned (0-1.3 for QA with ground-truth matches, 0-1 for ranking).
    Otherwise the intermediate action-format score for the last search call is
    returned as a fallback (rollout ended at max_qa_turns or max_ranking_turns without an answer).
    """
    normalized_info = _normalize_info(info)
    completion_text = _completion_to_text(completion)

    latest_action = completion[-1].get("content", "") if completion else ""
    if "<answer>" in latest_action and "</answer>" in latest_action:
        reward = _compute_terminal_reward(completion_text, normalized_info)
        if normalized_info.get("task_type") == "qa":
            return reward.format / _assistant_turn_count(completion)
        return reward.format

    return _intermediate_format_reward(_parse_search_action(latest_action))


# ---------------------------------------------------------------------------
# Metric-only reward functions (weight=0 in the rubric — observability only).
# These never contribute to the training signal but appear in rollout logs.
# ---------------------------------------------------------------------------


async def reward_ranking_precision(
    completion: vf.Messages, info: dict[str, Any] | None
) -> float:
    """Metric: source-prediction precision for ranking episodes."""
    normalized_info = _normalize_info(info)
    if normalized_info.get("task_type") != "ranking":
        return 0.0
    completion_text = _completion_to_text(completion)
    ground_truth = _build_ground_truth(normalized_info)
    result = compute_ranking_score(completion_text, ground_truth)
    return result.precision


async def reward_ranking_recall(
    completion: vf.Messages, info: dict[str, Any] | None
) -> float:
    """Metric: source-prediction recall for ranking episodes."""
    normalized_info = _normalize_info(info)
    if normalized_info.get("task_type") != "ranking":
        return 0.0
    completion_text = _completion_to_text(completion)
    ground_truth = _build_ground_truth(normalized_info)
    result = compute_ranking_score(completion_text, ground_truth)
    return result.recall


async def reward_qa_company_to_ticker(
    completion: vf.Messages, info: dict[str, Any] | None
) -> float:
    """Metric: 1.0 when CompanyNameToTickerTool was used correctly."""
    normalized_info = _normalize_info(info)
    completion_text = _completion_to_text(completion)
    ground_truth = _build_ground_truth(normalized_info)
    return compute_qa_company_to_ticker_score(completion_text, ground_truth)


async def reward_qa_ticker_match(
    completion: vf.Messages, info: dict[str, Any] | None
) -> float:
    """Metric: 1.0 when the correct ticker appeared in a search call."""
    normalized_info = _normalize_info(info)
    completion_text = _completion_to_text(completion)
    ground_truth = _build_ground_truth(normalized_info)
    return compute_qa_ticker_match_score(completion_text, ground_truth)


async def reward_qa_year_match(
    completion: vf.Messages, info: dict[str, Any] | None
) -> float:
    """Metric: 1.0 when the correct year appeared in a search call."""
    normalized_info = _normalize_info(info)
    completion_text = _completion_to_text(completion)
    ground_truth = _build_ground_truth(normalized_info)
    return compute_qa_year_match_score(completion_text, ground_truth)


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
    rubric.add_metric(reward_ranking_precision)
    rubric.add_metric(reward_ranking_recall)
    rubric.add_metric(reward_qa_company_to_ticker)
    rubric.add_metric(reward_qa_ticker_match)
    rubric.add_metric(reward_qa_year_match)

    return FinanceSearchEnv(
        tools=tools,
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        max_qa_turns=config.max_qa_turns,
        max_ranking_turns=config.max_ranking_turns,
    )
