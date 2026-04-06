"""Utilities for async SFT rollout generation, caching, and JSONL persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
from loguru import logger
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

from datasets import Dataset
from openai import AsyncOpenAI
import verifiers as vf

from rlm_sec.envs.finance_env import FinanceSearchEnv, create_finance_env
from rlm_sec.envs.rewards import DataTaskType

_QA_TASK = "qa"
_RANKING_TASK = "ranking"
_JSONL_ENCODING = "utf-8"
_ROLLOUT_CONCURRENCY = 8

_COMMON_METADATA_FIELDS = (
    "context",
    "year",
    "ticker_or_company_name",
    "filing_type",
    "data_source",
    "task_type",
)
_QA_METADATA_FIELDS = ("answer",)
_RANKING_METADATA_FIELDS = ("relevant", "not_relevant")


class RolloutRecord(NamedTuple):
    """Represents one generated rollout row persisted to JSONL."""

    rollout_key: str
    metadata: dict[str, Any]
    prompt_messages: list[dict[str, str]]
    choices: list["RolloutChoice"]


class RolloutChoice(NamedTuple):
    """Represents one sampled rollout: the full conversation including prompt."""

    conversation: list[dict[str, str]]


class RolloutSummary(NamedTuple):
    """Summary statistics for a rollout generation pass."""

    output_path: str
    generated_count: int
    reused_count: int
    total_count: int


class PendingRollout(NamedTuple):
    """Represents one dataset row that still needs a generated rollout."""

    row_index: int
    rollout_key: str
    metadata: dict[str, Any]
    prompt_messages: list[dict[str, str]]


def _cast_prompt_messages(prompt: Any) -> list[dict[str, str]]:
    """Casts prompt to expected role/content messages format."""
    return cast(list[dict[str, str]], prompt)


def _build_metadata(example: dict[str, Any]) -> dict[str, Any]:
    """Builds rollout metadata from common and task-specific SFT fields."""
    metadata: dict[str, Any] = {}
    for field in _COMMON_METADATA_FIELDS:
        metadata[field] = example.get(field)

    task_type = str(example.get("task_type", _QA_TASK)).strip()
    task_specific_fields = _QA_METADATA_FIELDS
    if task_type == _RANKING_TASK:
        task_specific_fields = _RANKING_METADATA_FIELDS

    for field in task_specific_fields:
        metadata[field] = example.get(field)
    return metadata


def _compute_rollout_key(
    metadata: dict[str, Any],
    prompt_messages: list[dict[str, str]],
    model: str,
) -> str:
    """Computes a stable unique key used for JSONL caching."""
    payload = {
        "metadata": metadata,
        "model": model,
        "prompt": prompt_messages,
    }
    serialized_payload = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized_payload.encode(_JSONL_ENCODING)).hexdigest()


def _split_prompt_and_completion(
    conversation: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Splits a full conversation into shared prompt and sampled completion."""
    for message_index, message in enumerate(conversation):
        if message.get("role") == "assistant":
            return conversation[:message_index], conversation[message_index:]
    return conversation, []


def _load_cached_rollouts(jsonl_path: str, model: str) -> dict[str, RolloutRecord]:
    """Loads existing rollout records keyed by rollout_key from JSONL.

    Handles three historical formats:
    - New format: ``choices[].conversation`` holds the full conversation.
    - Old format: ``choices[].completion`` holds only assistant turns; merged with
      the top-level ``prompt`` to reconstruct the full conversation.
    - Legacy format: individual lines with ``conversation`` and ``candidate_index``;
      grouped by recomputed key.
    """
    path = Path(jsonl_path)
    if not path.exists():
        logger.info(f"cache file missing, starting fresh. {jsonl_path=}")
        return {}

    cached: dict[str, RolloutRecord] = {}
    legacy_choices_by_key: dict[str, list[tuple[int, RolloutChoice]]] = {}
    legacy_metadata_by_key: dict[str, dict[str, Any]] = {}
    legacy_prompt_by_key: dict[str, list[dict[str, str]]] = {}

    with path.open("r", encoding=_JSONL_ENCODING) as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            data = json.loads(stripped_line)

            if "choices" in data:
                rollout_key = str(data.get("rollout_key", ""))
                if not rollout_key:
                    logger.warning(
                        "ignoring cached line with missing rollout_key. "
                        f"{jsonl_path=} {line_number=}"
                    )
                    continue

                prompt = _cast_prompt_messages(data.get("prompt", []))
                serialized_choices = cast(list[dict[str, Any]], data.get("choices", []))
                choices: list[RolloutChoice] = []
                for choice_data in serialized_choices:
                    if "conversation" in choice_data:
                        conversation = _cast_prompt_messages(
                            choice_data["conversation"]
                        )
                    else:
                        completion = _cast_prompt_messages(
                            choice_data.get("completion", [])
                        )
                        conversation = prompt + completion
                    choices.append(RolloutChoice(conversation=conversation))

                cached[rollout_key] = RolloutRecord(
                    rollout_key=rollout_key,
                    metadata=cast(dict[str, Any], data.get("metadata", {})),
                    prompt_messages=prompt,
                    choices=choices,
                )
                continue

            rollout_key = str(data.get("rollout_key", ""))
            if not rollout_key:
                logger.warning(
                    "ignoring cached line with missing rollout_key. "
                    f"{jsonl_path=} {line_number=}"
                )
                continue

            metadata = cast(dict[str, Any], data.get("metadata", {}))
            conversation = _cast_prompt_messages(data.get("conversation", []))
            prompt_messages, _ = _split_prompt_and_completion(conversation)
            normalized_rollout_key = _compute_rollout_key(
                metadata=metadata,
                prompt_messages=prompt_messages,
                model=model,
            )
            candidate_index = int(data.get("candidate_index", 0))
            legacy_metadata_by_key[normalized_rollout_key] = metadata
            legacy_prompt_by_key[normalized_rollout_key] = prompt_messages
            legacy_choices_by_key.setdefault(normalized_rollout_key, []).append(
                (candidate_index, RolloutChoice(conversation=conversation))
            )

    for rollout_key, indexed_choices in legacy_choices_by_key.items():
        if rollout_key in cached:
            continue

        sorted_choices = [
            choice for _, choice in sorted(indexed_choices, key=lambda item: item[0])
        ]
        cached[rollout_key] = RolloutRecord(
            rollout_key=rollout_key,
            metadata=legacy_metadata_by_key[rollout_key],
            prompt_messages=legacy_prompt_by_key[rollout_key],
            choices=sorted_choices,
        )

    logger.info(f"loaded cached rollouts. {jsonl_path=} {len(cached)=}")
    return cached


def _write_rollout_records_sync(jsonl_path: str, records: list[RolloutRecord]) -> None:
    """Synchronously rewrites rollout JSONL with one record per dataset row."""
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding=_JSONL_ENCODING) as file_obj:
        for record in records:
            serialized_record = {
                "rollout_key": record.rollout_key,
                "metadata": record.metadata,
                "prompt": record.prompt_messages,
                "choices": [
                    {"conversation": choice.conversation} for choice in record.choices
                ],
            }
            file_obj.write(json.dumps(serialized_record, ensure_ascii=False) + "\n")


async def _write_rollout_records_async(
    jsonl_path: str, records: list[RolloutRecord]
) -> None:
    """Asynchronously rewrites rollout JSONL with merged rollout records."""
    await asyncio.to_thread(_write_rollout_records_sync, jsonl_path, records)


def _build_openai_client() -> AsyncOpenAI:
    """Creates an AsyncOpenAI client from OPENAI_API_KEY and OPENAI_BASE_URL."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set for rollout generation.")

    base_url_raw = os.getenv("OPENAI_BASE_URL", "").strip()
    base_url = base_url_raw or None

    logger.info(f"building OpenAI client. {base_url=}")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _extract_response_contents(response: Any) -> list[str]:
    """Extracts non-null assistant contents from chat completion choices."""
    contents: list[str] = []
    for choice in response.choices:
        content = choice.message.content or ""
        contents.append(content)
    return contents


async def _generate_assistant_responses(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    semaphore: asyncio.Semaphore,
    n: int,
) -> list[str]:
    """Generates assistant turns for the provided conversation history."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            temperature=temperature,
            n=n,
        )
    contents = _extract_response_contents(response)
    logger.info(f"assistant responses generated. {model=} {n=} {len(contents)=}")
    return contents


@dataclass
class _ContinuationResponseFn:
    """Callable that generates the next assistant turn via the OpenAI API.

    Wraps `_generate_assistant_responses` with fixed client/model/temperature/
    semaphore parameters so it can be passed to `FinanceSearchEnv.run_multiturn`
    as an `AsyncResponseFn` without using nested function definitions.
    """

    client: AsyncOpenAI
    model: str
    temperature: float
    semaphore: asyncio.Semaphore

    async def __call__(self, messages: list[dict[str, str]]) -> str:
        responses = await _generate_assistant_responses(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            semaphore=self.semaphore,
            n=1,
        )
        return responses[0] if responses else ""


def _build_rollout_record(
    pending_rollout: PendingRollout,
    choices: list[RolloutChoice],
) -> RolloutRecord:
    """Builds rollout record with shared prompt and sampled choices."""
    return RolloutRecord(
        rollout_key=pending_rollout.rollout_key,
        metadata=pending_rollout.metadata,
        prompt_messages=pending_rollout.prompt_messages,
        choices=choices,
    )


def _build_pending_rollouts(
    dataset: Dataset,
    cached_rollouts: dict[str, RolloutRecord],
    model: str,
) -> tuple[list[PendingRollout], int]:
    """Builds pending rollout jobs and counts how many were cache hits."""
    pending_rollouts: list[PendingRollout] = []
    reused_count = 0

    for index, example in enumerate(dataset):
        example_dict = cast(dict[str, Any], example)
        prompt_messages = _cast_prompt_messages(example_dict.get("prompt", []))
        metadata = _build_metadata(example_dict)
        rollout_key = _compute_rollout_key(
            metadata=metadata,
            prompt_messages=prompt_messages,
            model=model,
        )
        if rollout_key in cached_rollouts:
            logger.info(f"reusing cached rollout. {index=} {rollout_key=}")
            reused_count += 1
            continue

        pending_rollouts.append(
            PendingRollout(
                row_index=index,
                rollout_key=rollout_key,
                metadata=metadata,
                prompt_messages=prompt_messages,
            )
        )

    logger.info(f"prepared pending rollouts. {len(pending_rollouts)=} {reused_count=}")
    return pending_rollouts, reused_count


def _is_qa_rollout(metadata: dict[str, Any]) -> bool:
    """Returns True when the pending rollout is a QA example."""
    task_type = str(metadata.get("task_type", _QA_TASK)).strip()
    return task_type == _QA_TASK


async def _generate_rollout_record(
    env: FinanceSearchEnv,
    client: AsyncOpenAI,
    model: str,
    pending_rollout: PendingRollout,
    rollout_temperature: float,
    continuation_temperature: float,
    semaphore: asyncio.Semaphore,
    n: int,
) -> tuple[RolloutRecord, int]:
    """Generates n rollout choices for a single dataset row.

    Uses ``rollout_temperature`` for the initial diverse candidate sampling and
    ``continuation_temperature`` for subsequent reasoning turns in QA multi-turn.
    """
    initial_assistant_responses = await _generate_assistant_responses(
        client=client,
        model=model,
        messages=pending_rollout.prompt_messages,
        temperature=rollout_temperature,
        semaphore=semaphore,
        n=n,
    )
    logger.info(
        f"initial responses prepared. {pending_rollout.row_index=} "
        f"{n=} {len(initial_assistant_responses)=}"
    )

    response_fn = _ContinuationResponseFn(
        client=client,
        model=model,
        temperature=continuation_temperature,
        semaphore=semaphore,
    )
    raw_task_type = str(pending_rollout.metadata.get("task_type", _QA_TASK))
    task_type = cast(DataTaskType, raw_task_type)

    choices: list[RolloutChoice] = []
    for choice_index, initial_assistant_content in enumerate(
        initial_assistant_responses
    ):
        if _is_qa_rollout(pending_rollout.metadata):
            conversation = await env.run_multiturn(
                prompt_messages=pending_rollout.prompt_messages,
                initial_assistant_content=initial_assistant_content,
                response_fn=response_fn,
                task_type=task_type,
            )
        else:
            conversation = list(pending_rollout.prompt_messages) + [
                {"role": "assistant", "content": initial_assistant_content}
            ]

        choices.append(RolloutChoice(conversation=conversation))
        logger.info(
            "generated rollout candidate. "
            f"{pending_rollout.row_index=} {choice_index=} {pending_rollout.rollout_key=}"
        )

    record = _build_rollout_record(pending_rollout=pending_rollout, choices=choices)
    return record, len(choices)


async def generate_and_cache_rollouts_async(
    dataset: Dataset,
    output_jsonl_path: str,
    model: str,
    rollout_temperature: float = 1.0,
    continuation_temperature: float = 0.7,
    n: int = 1,
) -> RolloutSummary:
    """Generates rollouts for each row and persists cached JSONL records.

    ``rollout_temperature`` controls diversity across the ``n`` initial candidates
    sampled per prompt. ``continuation_temperature`` controls reasoning steps in
    subsequent QA multi-turn turns. Uses ``OPENAI_API_KEY`` (required) and
    ``OPENAI_BASE_URL`` (optional) for the HTTP client.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1. {n=}")

    client = _build_openai_client()
    env = create_finance_env(dataset=dataset)
    cached_rollouts = _load_cached_rollouts(output_jsonl_path, model=model)
    pending_rollouts, reused_count = _build_pending_rollouts(
        dataset=dataset,
        cached_rollouts=cached_rollouts,
        model=model,
    )

    semaphore = asyncio.Semaphore(_ROLLOUT_CONCURRENCY)
    tasks = [
        _generate_rollout_record(
            env=env,
            client=client,
            model=model,
            pending_rollout=pending_rollout,
            rollout_temperature=rollout_temperature,
            continuation_temperature=continuation_temperature,
            semaphore=semaphore,
            n=n,
        )
        for pending_rollout in pending_rollouts
    ]
    generated_results = await asyncio.gather(*tasks)
    generated_records: list[RolloutRecord] = []
    generated_count = 0
    for record, new_choice_count in generated_results:
        cached_rollouts[record.rollout_key] = record
        generated_records.append(record)
        generated_count += new_choice_count

    if generated_records:
        await _write_rollout_records_async(
            jsonl_path=output_jsonl_path,
            records=list(cached_rollouts.values()),
        )

    total_count = len(dataset)
    logger.info(
        f"rollout generation complete. {output_jsonl_path=} {generated_count=} "
        f"{reused_count=} {total_count=} {_ROLLOUT_CONCURRENCY=} {n=} "
        f"{rollout_temperature=} {continuation_temperature=}"
    )
    return RolloutSummary(
        output_path=output_jsonl_path,
        generated_count=generated_count,
        reused_count=reused_count,
        total_count=total_count,
    )


def _take_first_task_row(dataset: Dataset, task_type: str) -> dict[str, Any]:
    """Selects the first row matching task_type from a dataset."""
    for row in dataset:
        row_dict = cast(dict[str, Any], row)
        if row_dict.get("task_type") == task_type:
            return row_dict
    raise ValueError(f"No row found for task type. {task_type=}")


def build_smoke_dataset(dataset: Dataset) -> Dataset:
    """Builds a 2-row smoke dataset with one QA and one ranking sample."""
    qa_row = _take_first_task_row(dataset=dataset, task_type=_QA_TASK)
    ranking_row = _take_first_task_row(dataset=dataset, task_type=_RANKING_TASK)
    smoke_dataset = Dataset.from_list([qa_row, ranking_row])
    logger.info(f"smoke dataset prepared. {len(smoke_dataset)=}")
    return smoke_dataset
