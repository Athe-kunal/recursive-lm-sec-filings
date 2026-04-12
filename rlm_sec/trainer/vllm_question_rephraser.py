"""Minimal vLLM client to rephrase filing questions with explicit time scope."""

from __future__ import annotations

import logging
from typing import Final

import fire
import requests

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_DEFAULT_BASE_URL: Final[str] = "http://localhost:8000/v1"
_DEFAULT_TIMEOUT_SECONDS: Final[int] = 30


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


def run(
    question: str,
    ticker: str,
    year: str,
    filing_type: str,
    model: str,
    base_url: str = _DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Rephrase a filing question using a vLLM OpenAI-compatible endpoint."""
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
    return rewritten_question


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    fire.Fire(run)


if __name__ == "__main__":
    main()
