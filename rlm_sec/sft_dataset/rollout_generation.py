"""Utilities for async SFT rollout generation, caching, and JSONL persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple, cast

from datasets import Dataset
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_QA_TASK = "qa"
_RANKING_TASK = "ranking"
_JSONL_ENCODING = "utf-8"
_ROLLOUT_CONCURRENCY = 8

_METADATA_FIELDS = (
    "answer",
    "context",
    "year",
    "ticker_or_company_name",
    "filing_type",
    "data_source",
    "task_type",
)


class RolloutRecord(NamedTuple):
    """Represents one generated rollout row persisted to JSONL."""

    rollout_key: str
    metadata: dict[str, Any]
    conversation: list[dict[str, str]]


class RolloutSummary(NamedTuple):
    """Summary statistics for a rollout generation pass."""

    output_path: str
    generated_count: int
    reused_count: int
    total_count: int


class PendingRollout(NamedTuple):
    """Represents one dataset row that still needs a generated rollout."""

    index: int
    rollout_key: str
    metadata: dict[str, Any]
    prompt_messages: list[dict[str, str]]


def _cast_prompt_messages(prompt: Any) -> list[dict[str, str]]:
    """Casts prompt to expected role/content messages format."""
    return cast(list[dict[str, str]], prompt)


def _build_metadata(example: dict[str, Any]) -> dict[str, Any]:
    """Builds rollout metadata from the canonical SFT metadata fields."""
    metadata: dict[str, Any] = {}
    for field in _METADATA_FIELDS:
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


def _load_cached_rollouts(jsonl_path: str) -> dict[str, dict[str, Any]]:
    """Loads existing rollout records keyed by rollout_key from JSONL."""
    path = Path(jsonl_path)
    if not path.exists():
        logger.info(f"cache file missing, starting fresh. {jsonl_path=}")
        return {}

    cached: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding=_JSONL_ENCODING) as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            data = json.loads(stripped_line)
            rollout_key = data.get("rollout_key", "")
            if not rollout_key:
                logger.warning(
                    "ignoring cached line with missing rollout_key. "
                    f"{jsonl_path=} {line_number=}"
                )
                continue
            cached[str(rollout_key)] = data

    logger.info(f"loaded cached rollouts. {jsonl_path=} {len(cached)=}")
    return cached


def _append_rollout_records_sync(jsonl_path: str, records: list[RolloutRecord]) -> None:
    """Synchronously appends multiple rollout records to JSONL."""
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding=_JSONL_ENCODING) as file_obj:
        for record in records:
            serialized_record = {
                "rollout_key": record.rollout_key,
                "metadata": record.metadata,
                "conversation": record.conversation,
            }
            file_obj.write(json.dumps(serialized_record, ensure_ascii=False) + "\n")


async def _append_rollout_records_async(
    jsonl_path: str, records: list[RolloutRecord]
) -> None:
    """Asynchronously appends multiple rollout records to JSONL."""
    await asyncio.to_thread(_append_rollout_records_sync, jsonl_path, records)


def _build_openai_client() -> AsyncOpenAI:
    """Creates an AsyncOpenAI client from OPENAI_API_KEY and OPENAI_BASE_URL."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set for rollout generation.")

    base_url_raw = os.getenv("OPENAI_BASE_URL", "").strip()
    base_url = base_url_raw or None

    logger.info(f"building OpenAI client. {base_url=}")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def _generate_assistant_response(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> str:
    """Generates one assistant turn for the provided conversation history."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    content = response.choices[0].message.content or ""
    logger.info(f"assistant response generated. {model=} {len(content)=}")
    return content


def _build_rollout_record(
    pending_rollout: PendingRollout,
    assistant_content: str,
) -> RolloutRecord:
    """Builds rollout record with full conversation and metadata."""
    conversation = list(pending_rollout.prompt_messages)
    conversation.append({"role": "assistant", "content": assistant_content})
    return RolloutRecord(
        rollout_key=pending_rollout.rollout_key,
        metadata=pending_rollout.metadata,
        conversation=conversation,
    )


def _build_pending_rollouts(
    dataset: Dataset,
    cached_rollouts: dict[str, dict[str, Any]],
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
            reused_count += 1
            logger.info(f"reusing cached rollout. {index=} {rollout_key=}")
            continue

        pending_rollouts.append(
            PendingRollout(
                index=index,
                rollout_key=rollout_key,
                metadata=metadata,
                prompt_messages=prompt_messages,
            )
        )

    logger.info(f"prepared pending rollouts. {len(pending_rollouts)=} {reused_count=}")
    return pending_rollouts, reused_count


async def _generate_rollout_record(
    client: AsyncOpenAI,
    model: str,
    pending_rollout: PendingRollout,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> RolloutRecord:
    """Generates one rollout and returns a serializable record."""
    assistant_content = await _generate_assistant_response(
        client=client,
        model=model,
        messages=pending_rollout.prompt_messages,
        temperature=temperature,
        semaphore=semaphore,
    )
    record = _build_rollout_record(
        pending_rollout=pending_rollout,
        assistant_content=assistant_content,
    )
    logger.info(f"generated rollout. {pending_rollout.index=} {record.rollout_key=}")
    return record


async def generate_and_cache_rollouts_async(
    dataset: Dataset,
    output_jsonl_path: str,
    model: str,
    temperature: float = 0.0,
) -> RolloutSummary:
    """Generates rollouts for each row and persists cached JSONL records.

    Uses ``OPENAI_API_KEY`` (required) and ``OPENAI_BASE_URL`` (optional; default
    OpenAI endpoint when unset or empty) for the HTTP client.
    """
    client = _build_openai_client()
    cached_rollouts = _load_cached_rollouts(output_jsonl_path)
    pending_rollouts, reused_count = _build_pending_rollouts(
        dataset=dataset,
        cached_rollouts=cached_rollouts,
        model=model,
    )

    semaphore = asyncio.Semaphore(_ROLLOUT_CONCURRENCY)
    tasks = [
        _generate_rollout_record(
            client=client,
            model=model,
            pending_rollout=pending_rollout,
            temperature=temperature,
            semaphore=semaphore,
        )
        for pending_rollout in pending_rollouts
    ]
    generated_records = await asyncio.gather(*tasks)

    if generated_records:
        await _append_rollout_records_async(
            jsonl_path=output_jsonl_path,
            records=generated_records,
        )

    generated_count = len(generated_records)
    total_count = len(dataset)
    logger.info(
        f"rollout generation complete. {output_jsonl_path=} {generated_count=} "
        f"{reused_count=} {total_count=} {_ROLLOUT_CONCURRENCY=}"
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
