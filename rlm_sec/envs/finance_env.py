from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from typing import Any
from rlm_sec.envs.rewards import compute_score
import re
from typing import Dict, Optional, Union, NamedTuple
from dataclasses import dataclass
from omegaconf import DictConfig

from rlm_sec.envs.tools import (
    EarningsTranscriptToolGroup,
    SECFilingToolGroup,
)
from rlm_sec.envs.rewards import reward_action_format, TaskType
from settings import (
    EARNINGS_TRANSCRIPT_TOOL_ENDPOINT,
    SEC_FILING_TOOL_ENDPOINT,
    env_settings,
)


class RewardType(NamedTuple):
    correctness: float
    format: float


SEARCH_TOOL_ROUTING = {
    "SECFilingTool": (
        "SECFilingToolGroup",
        "sec_filing_to_markdown_embed_and_search",
    ),
    "EarningsTranscriptTool": (
        "EarningsTranscriptToolGroup",
        "earnings_transcript_to_embed_and_search",
    ),
}


@dataclass
class ParsedSearch:
    action: str
    query: Optional[str] = None
    ticker: Optional[str] = None
    year: Optional[str] = None
    filing_type_or_quarter: Optional[str] = None
    tool_group_name: Optional[str] = None
    tool_name: Optional[str] = None
    task_type: Optional[TaskType] = None

    @property
    def reward(self) -> float:
        if self.task_type is None:
            return 0.0
        return reward_action_format(
            tool_group_name=self.tool_group_name or "",
            ticker=self.ticker or "",
            year=self.year or "",
            filing_type=self.filing_type_or_quarter or "",
            task_type=self.task_type,
        )


@dataclass
class SearchEnvConfig:
    log_requests: bool = False
    sec_filing_tool_url: str = f"{env_settings.server_url}{SEC_FILING_TOOL_ENDPOINT}"
    earnings_transcript_tool_url: str = (
        f"{env_settings.server_url}{EARNINGS_TRANSCRIPT_TOOL_ENDPOINT}"
    )
    topk: int = 3
    timeout: int = 30


class FinanceSearchEnv(BaseTextEnv):
    def __init__(
        self,
        env_config: Union[SearchEnvConfig, DictConfig],
        extras: Dict[str, Any] = {},
    ):
        super().__init__()
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Register the base search tool plus the finance ingestion/search tools.
        # self.search_tool_group = SearchToolGroup(
        #     search_url=env_config.search_url,
        #     topk=env_config.topk,
        #     timeout=env_config.timeout,
        #     log_requests=env_config.log_requests,
        # )
        self.sec_filing_tool_group = SECFilingToolGroup(
            tool_url=env_config.sec_filing_tool_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.earnings_transcript_tool_group = EarningsTranscriptToolGroup(
            tool_url=env_config.earnings_transcript_tool_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.init_tool_groups(
            [
                # self.search_tool_group,
                self.sec_filing_tool_group,
                self.earnings_transcript_tool_group,
            ]
        )

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _get_tool_group_by_name(self, tool_group_name: str) -> Optional[Any]:
        """Return the registered tool group for the given name."""
        for group in self.tool_groups:
            if group.name == tool_group_name:
                return group
        return None

    def _parse_action(self, action: str) -> ParsedSearch:
        """Parse ``<search>ToolName(query, ticker, year, filing_type_or_quarter)</search>``.

        Always returns a ParsedSearch. On any parse failure, all fields except
        ``action`` are left as None and ``reward`` will be 0.

        Supported tool names: SECFilingTool, EarningsTranscriptTool.
        """
        search_match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        if not search_match:
            return ParsedSearch(action=action)

        inner = search_match.group(1).strip()
        call_match = re.match(r"(\w+)\((.+)\)$", inner, re.DOTALL)
        if not call_match:
            return ParsedSearch(action=action)

        tool_name_str = call_match.group(1).strip()
        if tool_name_str not in SEARCH_TOOL_ROUTING:
            return ParsedSearch(action=action)

        parts = [p.strip() for p in call_match.group(2).split(",")]
        if len(parts) != 4:
            return ParsedSearch(action=action)

        query, ticker, year, filing_type_or_quarter = parts
        tool_group_name, tool_name = SEARCH_TOOL_ROUTING[tool_name_str]
        task_type = (
            "sec_filings"
            if tool_group_name == "SECFilingToolGroup"
            else "earning_transcripts"
        )
        return ParsedSearch(
            action=action,
            query=query,
            ticker=ticker,
            year=year,
            filing_type_or_quarter=filing_type_or_quarter,
            tool_group_name=tool_group_name,
            tool_name=tool_name,
            task_type=task_type,
        )

    def _get_reward(self, action: str, done: bool) -> RewardType:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            correctness, format = compute_score(chat_history_str, self.ground_truth)
            return RewardType(
                correctness=correctness,
                format=format,
            )
        else:
            intermediate_reward = self._parse_action(action).reward
            return RewardType(
                correctness=0.0,
                format=intermediate_reward,
            )

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _execute_tool(
        self, tool_group_name: str, tool_name: str, tool_input: Any
    ) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)
        return "\n<information>\n" + tool_output + "</information>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=done, metadata={}
            )

        tool_group_name = None
        tool_name = None
        tool_input = None
        try:
            if "<search>" not in action or "</search>" not in action:
                observation = "\n<information></information>\n"
            else:
                parsed = self._parse_action(action)
                tool_input = parsed
                if parsed.tool_group_name is None:
                    observation = (
                        "\n<information>Invalid <search> format. Expected: "
                        "SECFilingTool(ticker, year, filing_type) or "
                        "EarningsTranscriptTool(ticker, year, quarter).</information>\n"
                        f"But got: {action}"
                    )
                else:
                    assert parsed.tool_group_name is not None
                    assert parsed.tool_name is not None
                    tool_group_name = parsed.tool_group_name
                    tool_name = parsed.tool_name
                    # The underlying tool signature is (query, ticker, year, filing_type_or_quarter).
                    # query is left empty; the server performs a broad retrieval over the filing.
                    tool_args = [
                        "",
                        parsed.ticker,
                        parsed.year,
                        parsed.filing_type_or_quarter,
                    ]
                    observation = self._execute_tool(
                        tool_group_name, tool_name, tool_args
                    )
        except Exception as e:
            error = str(e)
            observation = None

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        tool_metadata = {}
        if tool_group_name is not None:
            group = self._get_tool_group_by_name(tool_group_name)
            if group is not None and hasattr(group, "get_last_metadata"):
                tool_metadata = group.get_last_metadata()

        info = {
            "tool_group": tool_group_name,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_metadata": tool_metadata,
        }

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )
