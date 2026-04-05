"""Verifiers entrypoint for Prime RL environment loading."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset

from rlm_sec.envs.finance_env import SearchEnvConfig, create_finance_env


DEFAULT_TRAIN_DATA = "data/train.parquet"
DEFAULT_EVAL_DATA = "data/validation.parquet"


def _load_parquet_dataset(path: str) -> Dataset:
    """Load a parquet file as a Hugging Face Dataset."""
    dataset_path = Path(path)
    return Dataset.from_parquet(str(dataset_path))


def _build_search_env_config() -> SearchEnvConfig:
    """Read runtime configuration from environment variables."""
    return SearchEnvConfig(
        log_requests=os.getenv("RLM_SEC_LOG_REQUESTS", "false").lower() == "true",
        topk=int(os.getenv("RLM_SEC_TOPK", "3")),
        timeout=int(os.getenv("RLM_SEC_TIMEOUT", "30")),
        max_qa_turns=int(os.getenv("RLM_SEC_MAX_QA_TURNS", "4")),
        max_ranking_turns=int(os.getenv("RLM_SEC_MAX_RANKING_TURNS", "1")),
    )


def load_environment():
    """Prime RL / Verifiers standard loader entrypoint."""
    train_path = os.getenv("RLM_SEC_TRAIN_DATA", DEFAULT_TRAIN_DATA)
    eval_path = os.getenv("RLM_SEC_EVAL_DATA", DEFAULT_EVAL_DATA)
    train_dataset = _load_parquet_dataset(train_path)
    eval_dataset = _load_parquet_dataset(eval_path)
    return create_finance_env(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        env_config=_build_search_env_config(),
    )
