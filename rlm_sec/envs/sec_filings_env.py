from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from typing import Any
from rlm_sec.envs.rewards import compute_score
from rlm_sec.envs.tools import SearchToolGroup
import re
from typing import Dict, Optional, List, Union, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig

from settings import env_settings


@dataclass
class SearchEnvConfig:
    log_requests: bool = False
    search_url: str = f"{env_settings.server_url}/vector_store/search"
    topk: int = 3
    timeout: int = 30


class SECSearchEnv(BaseTextEnv):
    def __init__(
        self,
        env_config: Union[SearchEnvConfig, DictConfig],
        extras: Dict[str, Any] = {},
    ):
        super().__init__()
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Initialize the tools
        # name is hardcoded to "SearchToolGroup", with tool name "search"
        self.tool_group = SearchToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> Optional[Tuple[str, str, str, str]]:
        """Parse ``<search>query, ticker, year, filing_type</search>`` from the action.

        The query field may contain commas; the last three comma-separated segments
        are interpreted as ticker, year, and filing_type.
        """
        match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        if not match:
            return None
        inner = match.group(1).strip()
        parts = inner.rsplit(",", 3)
        if len(parts) != 4:
            return None
        query, ticker, year, filing_type = (p.strip() for p in parts)
        return (query, ticker, year, filing_type)

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score(chat_history_str, self.ground_truth)
        else:
            # No reward for intermediate steps for Search tasks
            return 0

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

        tool_input = None
        try:
            if "<search>" not in action or "</search>" not in action:
                observation = "\n<information></information>\n"
            else:
                parsed = self._parse_action(action)
                tool_input = parsed
                if parsed is None:
                    observation = (
                        "\n<information>Invalid <search> format. Expected: "
                        "query, ticker, year, filing_type (comma-separated; "
                        "the query may contain commas).</information>\n"
                    )
                else:
                    observation = self._execute_tool(
                        "SearchToolGroup", "search", list(parsed)
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

        info = {
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": tool_input,
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
