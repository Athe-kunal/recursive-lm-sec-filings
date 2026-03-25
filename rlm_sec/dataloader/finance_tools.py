"""Finance data tools for SEC filings and earnings transcripts workflows.

This module exposes SkyRL-compatible tools that:
1. Download SEC filings, run olmOCR to markdown, embed into vector store,
   and optionally search with a query.
2. Download earnings-call transcripts, persist markdown, embed into vector
   store, and optionally search with a query.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from skyrl_gym.tools.core import ToolGroup, tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _validate_non_empty(value: str, field_name: str) -> str:
    """Validate and normalize a required string argument.

    Args:
        value: Raw input string.
        field_name: Parameter name for error messages.

    Returns:
        A stripped non-empty string.

    Raises:
        ValueError: If the value is empty after stripping.
    """
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _validate_year(year: str) -> str:
    """Validate year input as a 4-digit numeric string."""
    normalized = _validate_non_empty(year, "year")
    if len(normalized) != 4 or not normalized.isdigit():
        raise ValueError("year must be a 4-digit string, for example '2025'")
    return normalized


def _run_async(coro: Any) -> Any:
    """Execute an async coroutine from synchronous code.

    This utility handles both cases:
    - No running event loop: uses ``asyncio.run``.
    - Running event loop present: executes in a dedicated loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _format_search_results(results: list[tuple[Any, float]]) -> str:
    """Format vector search results as readable plain text."""
    if not results:
        return "No search results found."

    formatted: list[str] = []
    for idx, (chunk, score) in enumerate(results, start=1):
        text = getattr(chunk, "text", "").strip()
        formatted.append(f"Doc {idx} (score={score:.4f}): {text}")
    return "\n\n".join(formatted)


def _to_quarter_label(raw_value: str) -> str:
    """Normalize quarter input to `Q1`..`Q4` label format."""
    normalized = _validate_non_empty(raw_value, "filing_type").upper().replace(" ", "")
    if normalized in {"Q1", "Q2", "Q3", "Q4"}:
        return normalized
    raise ValueError("filing_type for transcripts must be one of Q1, Q2, Q3, Q4")


class FinanceToolGroup(ToolGroup):
    """Tool group that orchestrates download/OCR/embed/search finance workflows."""

    def __init__(self, top_k: int = 5):
        """Initialize the tool group.

        Args:
            top_k: Maximum number of chunks returned by semantic search.
        """
        self.top_k = top_k
        self._vector_store = self._init_vector_store()
        super().__init__(name="FinanceToolGroup")

    def _init_vector_store(self) -> Any:
        """Create the shared vector store client.

        Returns:
            Initialized vector store instance.

        Raises:
            ImportError: If finance-data package imports are unavailable.
        """
        try:
            from finance_data.dataloader.vector_store import ChromaVectorStore
        except ImportError as exc:
            raise ImportError(
                "Missing finance_data package imports. Install the finance-data-llm "
                "dependencies that provide ChromaVectorStore."
            ) from exc

        return ChromaVectorStore()

    @tool
    def sec_filing_to_markdown_embed_and_search(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        """Fetch one SEC filing, OCR+embed it, then run semantic search.

        Args:
            query: Natural-language query used for semantic retrieval.
            ticker: Equity ticker (for example, ``AAPL``).
            year: Filing year (``YYYY``).
            filing_type: Filing selector (for example, ``10-K``, ``10-Q1``).

        Returns:
            Search results as formatted plain text.
        """
        query_value = _validate_non_empty(query, "query")
        ticker_value = _validate_non_empty(ticker, "ticker").upper()
        year_value = _validate_year(year)
        filing_type_value = _validate_non_empty(filing_type, "filing_type").upper()

        from finance_data.filings.sec_data import sec_main_to_markdown_and_embed

        payload = _run_async(
            sec_main_to_markdown_and_embed(
                ticker=ticker_value,
                year=year_value,
                filing_type=filing_type_value,
                force=False,
            )
        )

        sec_result = payload["sec_result"]
        indexed_filing_type = getattr(sec_result, "form_name", filing_type_value)
        results = self._vector_store.search(
            ticker=ticker_value,
            year=year_value,
            filing_type=indexed_filing_type,
            query=query_value,
            top_k=self.top_k,
        )
        return _format_search_results(results)

    @tool
    def earnings_transcript_to_embed_and_search(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        """Fetch one earnings transcript, embed it, then run semantic search.

        Args:
            query: Natural-language query used for semantic retrieval.
            ticker: Equity ticker symbol.
            year: Transcript year (``YYYY``).
            filing_type: Quarter label (``Q1`` .. ``Q4``).

        Returns:
            Search results as formatted plain text.
        """
        query_value = _validate_non_empty(query, "query")
        ticker_value = _validate_non_empty(ticker, "ticker").upper()
        year_value = _validate_year(year)
        quarter_value = _to_quarter_label(filing_type)

        from finance_data.earnings_transcripts.transcripts import (
            get_transcript_for_quarter_async,
            save_transcript_markdown,
        )

        transcript = _run_async(
            get_transcript_for_quarter_async(
                ticker=ticker_value,
                year=int(year_value),
                quarter=quarter_value,
            )
        )
        if transcript is None:
            raise FileNotFoundError(
                f"No transcript available for {ticker_value} {year_value} {quarter_value}."
            )

        markdown_path = Path(save_transcript_markdown(transcript))
        self._vector_store.from_earnings_transcript_markdown(
            ticker=ticker_value,
            year=year_value,
            transcript_paths=[markdown_path],
            force=False,
        )
        results = self._vector_store.search(
            ticker=ticker_value,
            year=year_value,
            filing_type=quarter_value,
            query=query_value,
            top_k=self.top_k,
        )
        return _format_search_results(results)
