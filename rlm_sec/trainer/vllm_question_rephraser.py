"""Minimal vLLM client to rephrase filing questions with explicit time scope."""

from __future__ import annotations

import asyncio
import json
from loguru import logger
import os
import random
from pathlib import Path
from typing import Final, NamedTuple

import fire
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field

_LOG = logger
_DEFAULT_BASE_URL: Final[str] = "http://localhost:8000/v1"
_DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
_ENV_OPENAI_API_KEY: Final[str] = "OPENAI_API_KEY"
_ENV_OPENAI_BASE_URL: Final[str] = "OPENAI_BASE_URL"
_ENV_OPENAI_MODEL: Final[str] = "OPENAI_MODEL"
_QUESTION_PREFIX: Final[str] = "Question:"


class QuestionRephraseStructuredResponse(BaseModel):
    """JSON schema for guided / structured chat completion (OpenAI-compatible servers)."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description=(
            "Step-by-step reasoning: whether the question already names the ticker, "
            "uses the required explicit time scope text, and avoids vague time wording."
        )
    )
    reformatted_question: str = Field(
        description=(
            "Single-sentence question for SEC filing retrieval. If the original already "
            "meets all requirements, return it verbatim; otherwise rewrite."
        )
    )


class RephraseQuestionResult(NamedTuple):
    """Outcome of a single rephrase call (reasoning plus final question text)."""

    reasoning: str
    reformatted_question: str


def _resolve_openai_base_url(base_url: str | None) -> str:
    if base_url is not None and base_url.strip():
        return base_url.strip()
    env_url = os.getenv(_ENV_OPENAI_BASE_URL, "").strip()
    return env_url or _DEFAULT_BASE_URL


def _resolve_openai_api_key(api_key: str | None) -> str:
    if api_key is not None and api_key.strip():
        return api_key.strip()
    env_key = os.getenv(_ENV_OPENAI_API_KEY, "").strip()
    return env_key or "EMPTY"


def _resolve_chat_model(model: str | None) -> str:
    if model is not None and model.strip():
        return model.strip()
    env_model = os.getenv(_ENV_OPENAI_MODEL, "").strip()
    if env_model:
        return env_model
    raise ValueError(
        "Chat model is required: pass model=... or set OPENAI_MODEL in the environment."
    )


def _normalize_filing_type(filing_type: str) -> str:
    return filing_type.strip().upper()


def _time_scope_text(year: str, filing_type: str) -> str:
    normalized = _normalize_filing_type(filing_type)
    quarterly_map = {
        "10-Q1": f"Q1 {year}",
        "10-Q2": f"Q2 {year}",
        "10-Q3": f"Q3 {year}",
        "10-Q4": f"Q4 {year}",
        "Q1": f"Q1 {year}",
        "Q2": f"Q2 {year}",
        "Q3": f"Q3 {year}",
        "Q4": f"Q4 {year}",
    }
    if normalized in quarterly_map:
        return quarterly_map[normalized]
    return f"fiscal year {year}"


def _build_system_prompt() -> str:
    return (
        "You rewrite questions for SEC filing retrieval. "
        "First work through your reasoning, then produce the final question. "
        "Your reply must follow the structured response format only (no extra prose). "
        "If the user's question already includes the ticker, the required explicit time scope, "
        "and avoids vague time words (current quarter, this year, recent, latest, now), "
        "copy the original question verbatim into the reformatted_question field. "
        "Otherwise produce a single rewritten sentence that includes the ticker and required time scope."
    )


def _build_user_prompt(question: str, ticker: str, year: str, filing_type: str) -> str:
    time_scope = _time_scope_text(year=year, filing_type=filing_type)
    return (
        f"Original question: {question}\n"
        f"Ticker: {ticker}\n"
        f"Year: {year}\n"
        f"Filing type: {filing_type}\n"
        f"Required time scope text: {time_scope}\n"
        "If the original question is already acceptable (ticker present, required time scope "
        "expressed explicitly, no vague time wording), return it unchanged in reformatted_question.\n"
        "Otherwise rewrite it to one sentence that explicitly contains the ticker and the "
        "required time scope text."
    )


def _build_rephrase_messages(
    question: str,
    ticker: str,
    year: str,
    filing_type: str,
) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": _build_system_prompt()},
        {
            "role": "user",
            "content": _build_user_prompt(
                question=question,
                ticker=ticker,
                year=year,
                filing_type=filing_type,
            ),
        },
    ]


async def rephrase_question_with_vllm_async(
    client: AsyncOpenAI,
    question: str,
    ticker: str,
    year: str,
    filing_type: str,
    resolved_model: str,
    semaphore: asyncio.Semaphore,
    timeout_seconds: int,
) -> RephraseQuestionResult:
    _LOG.info(f"{question=}")
    _LOG.info(f"{ticker=}")
    _LOG.info(f"{year=}")
    _LOG.info(f"{filing_type=}")
    _LOG.info(f"{resolved_model=}")

    messages = _build_rephrase_messages(
        question=question,
        ticker=ticker,
        year=year,
        filing_type=filing_type,
    )
    _LOG.info(f"{messages=}")

    async with semaphore:
        response = await client.chat.completions.parse(
            model=resolved_model,
            temperature=0.0,
            messages=messages,
            response_format=QuestionRephraseStructuredResponse,
            timeout=timeout_seconds,
        )
    _LOG.info(f"{response.model_dump()=}")
    message = response.choices[0].message
    parsed = message.parsed
    if parsed is None:
        raise ValueError(
            "Structured rephrase failed: missing parsed payload. "
            f"{message.refusal=} {message.content=}"
        )
    _LOG.info(f"{parsed.reasoning=}")
    _LOG.info(f"{parsed.reformatted_question=}")
    return RephraseQuestionResult(
        reasoning=parsed.reasoning.strip(),
        reformatted_question=parsed.reformatted_question.strip(),
    )


def _extract_question_from_prompt(prompt: list[dict[str, str]]) -> str:
    for message in reversed(prompt):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if _QUESTION_PREFIX in content:
            return content.split(_QUESTION_PREFIX, maxsplit=1)[1].strip()
        return content.strip()
    return ""


def _replace_question_in_prompt(
    prompt: list[dict[str, str]], question: str
) -> list[dict[str, str]]:
    updated_prompt: list[dict[str, str]] = []
    replaced = False
    for message in prompt:
        current_message = dict(message)
        if not replaced and current_message.get("role") == "user":
            content = current_message.get("content", "")
            if _QUESTION_PREFIX in content:
                prefix = (
                    content.split(_QUESTION_PREFIX, maxsplit=1)[0]
                    + _QUESTION_PREFIX
                    + " "
                )
                current_message["content"] = prefix + question
            else:
                current_message["content"] = question
            replaced = True
        updated_prompt.append(current_message)
    return updated_prompt


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _rephrase_jsonl_row_async(
    client: AsyncOpenAI,
    row: dict,
    row_index: int,
    *,
    resolved_model: str,
    semaphore: asyncio.Semaphore,
    timeout_seconds: int,
) -> tuple[int, dict]:
    """Run extract → async vLLM rephrase (semaphore-limited) → prompt replace."""
    _LOG.info(f"{row_index=}")
    prompt = row.get("prompt", [])
    question = _extract_question_from_prompt(prompt=prompt)
    ticker = str(row.get("ticker_or_company_name", "")).strip()
    year = str(row.get("year", "")).strip()
    filing_type = str(row.get("filing_type", "")).strip()
    _LOG.info(f"{question=}")
    _LOG.info(f"{ticker=}")
    _LOG.info(f"{year=}")
    _LOG.info(f"{filing_type=}")

    rephrase_outcome = await rephrase_question_with_vllm_async(
        client=client,
        question=question,
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        resolved_model=resolved_model,
        semaphore=semaphore,
        timeout_seconds=timeout_seconds,
    )
    updated_row = dict(row)
    updated_row["rephrased_question"] = rephrase_outcome.reformatted_question
    updated_row["rephrased_question_reasoning"] = rephrase_outcome.reasoning
    updated_row["prompt"] = _replace_question_in_prompt(
        prompt=prompt,
        question=rephrase_outcome.reformatted_question,
    )
    return row_index, updated_row


async def _rephrase_jsonl_rows_async(
    client: AsyncOpenAI,
    rows: list[dict],
    *,
    resolved_model: str,
    max_concurrent_requests: int,
    timeout_seconds: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    _LOG.info(f"{max_concurrent_requests=}")
    tasks = [
        _rephrase_jsonl_row_async(
            client=client,
            row=row,
            row_index=index,
            resolved_model=resolved_model,
            semaphore=semaphore,
            timeout_seconds=timeout_seconds,
        )
        for index, row in enumerate(rows)
    ]
    indexed_rows = await asyncio.gather(*tasks)
    indexed_rows.sort(key=lambda item: item[0])
    return [item[1] for item in indexed_rows]


def rephrase_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    max_concurrent_requests: int = 512,
    smoke_test: bool = False,
    smoke_test_seed: int | None = None,
) -> str:
    """Rephrase QA rows from a JSONL file using concurrent async OpenAI calls.

    When ``model``, ``base_url``, or ``api_key`` are omitted, reads ``OPENAI_MODEL``
    (required unless ``model`` is passed), ``OPENAI_BASE_URL``, and ``OPENAI_API_KEY``.

    Set ``smoke_test=True`` to process one randomly sampled input row (same pipeline);
    use ``smoke_test_seed`` for a deterministic draw. Output is always written to
    ``output_jsonl`` (one line when ``smoke_test`` is true).

    Up to ``max_concurrent_requests`` in-flight HTTP calls are allowed at once.
    """
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    resolved_model = _resolve_chat_model(model=model)
    resolved_base_url = _resolve_openai_base_url(base_url=base_url)
    resolved_api_key = _resolve_openai_api_key(api_key=api_key)
    _LOG.info(f"{input_jsonl=}")
    _LOG.info(f"{output_jsonl=}")
    _LOG.info(f"{resolved_model=}")
    _LOG.info(f"{resolved_base_url=}")
    _LOG.info(f"{smoke_test=}")
    _LOG.info(f"{smoke_test_seed=}")

    rows = _read_jsonl(path=input_path)
    if not rows:
        if smoke_test:
            raise ValueError(f"Input JSONL is empty (cannot smoke test): {input_path}")
        _LOG.info(f"empty input jsonl; {input_path=}")
        _write_jsonl(path=output_path, rows=[])
        return str(output_path)
    _LOG.info(f"{len(rows)=}")

    if smoke_test:
        rng = (
            random.Random(smoke_test_seed)
            if smoke_test_seed is not None
            else random.Random()
        )
        pick_index = rng.randrange(len(rows))
        rows = [rows[pick_index]]
        _LOG.info(f"{pick_index=}")

    client = AsyncOpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url.rstrip("/"),
        timeout=timeout_seconds,
    )
    rewritten_rows = asyncio.run(
        _rephrase_jsonl_rows_async(
            client=client,
            rows=rows,
            resolved_model=resolved_model,
            max_concurrent_requests=max(1, max_concurrent_requests),
            timeout_seconds=timeout_seconds,
        )
    )

    _write_jsonl(path=output_path, rows=rewritten_rows)
    return str(output_path)


def main() -> None:
    fire.Fire(rephrase_jsonl)


if __name__ == "__main__":
    main()
