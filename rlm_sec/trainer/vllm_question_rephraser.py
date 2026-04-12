"""Minimal vLLM client to rephrase filing questions with explicit time scope."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import fire
import requests

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_DEFAULT_BASE_URL: Final[str] = "http://localhost:8000/v1"
_DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
_QUESTION_PREFIX: Final[str] = "Question:"


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
        "Rewrite the user question for SEC filing retrieval. "
        "Always include ticker and explicit time scope. "
        "Never use vague time words like current quarter, this year, recent, latest, or now. "
        "Return one sentence only."
    )


def _build_user_prompt(question: str, ticker: str, year: str, filing_type: str) -> str:
    time_scope = _time_scope_text(year=year, filing_type=filing_type)
    return (
        f"Original question: {question}\n"
        f"Ticker: {ticker}\n"
        f"Year: {year}\n"
        f"Filing type: {filing_type}\n"
        f"Required time scope text: {time_scope}\n"
        "Rewrite the question so it explicitly contains ticker and required time scope text."
    )


def rephrase_question_with_vllm(
    question: str,
    ticker: str,
    year: str,
    filing_type: str,
    model: str,
    base_url: str = _DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    _LOG.info(f"{question=}")
    _LOG.info(f"{ticker=}")
    _LOG.info(f"{year=}")
    _LOG.info(f"{filing_type=}")
    _LOG.info(f"{model=}")
    _LOG.info(f"{base_url=}")

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
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
        ],
    }
    _LOG.info(f"{payload=}")

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    _LOG.info(f"{data=}")
    return data["choices"][0]["message"]["content"].strip()


def _extract_question_from_prompt(prompt: list[dict[str, str]]) -> str:
    for message in reversed(prompt):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if _QUESTION_PREFIX in content:
            return content.split(_QUESTION_PREFIX, maxsplit=1)[1].strip()
        return content.strip()
    return ""


def _replace_question_in_prompt(prompt: list[dict[str, str]], question: str) -> list[dict[str, str]]:
    updated_prompt: list[dict[str, str]] = []
    replaced = False
    for message in prompt:
        current_message = dict(message)
        if not replaced and current_message.get("role") == "user":
            content = current_message.get("content", "")
            if _QUESTION_PREFIX in content:
                prefix = content.split(_QUESTION_PREFIX, maxsplit=1)[0] + _QUESTION_PREFIX + " "
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


def rephrase_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    model: str,
    base_url: str = _DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Rephrase all QAExample questions from a JSONL file."""
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    _LOG.info(f"{input_jsonl=}")
    _LOG.info(f"{output_jsonl=}")
    _LOG.info(f"{model=}")

    rows = _read_jsonl(path=input_path)
    _LOG.info(f"{len(rows)=}")

    rewritten_rows: list[dict] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt", [])
        question = _extract_question_from_prompt(prompt=prompt)
        ticker = str(row.get("ticker_or_company_name", "")).strip()
        year = str(row.get("year", "")).strip()
        filing_type = str(row.get("filing_type", "")).strip()
        _LOG.info(f"{index=}")
        _LOG.info(f"{question=}")
        _LOG.info(f"{ticker=}")
        _LOG.info(f"{year=}")
        _LOG.info(f"{filing_type=}")

        rewritten_question = rephrase_question_with_vllm(
            question=question,
            ticker=ticker,
            year=year,
            filing_type=filing_type,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        updated_row = dict(row)
        updated_row["rephrased_question"] = rewritten_question
        updated_row["prompt"] = _replace_question_in_prompt(
            prompt=prompt,
            question=rewritten_question,
        )
        rewritten_rows.append(updated_row)

    _write_jsonl(path=output_path, rows=rewritten_rows)
    return str(output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    fire.Fire(rephrase_jsonl)


if __name__ == "__main__":
    main()
