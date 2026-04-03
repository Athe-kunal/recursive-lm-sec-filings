import dataclasses
import logging
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel

from finance_data.dataloader.text_splitter import Chunk
from finance_data.dataloader.vector_store import ChromaVectorStore
from finance_data.filings import models as filings_models
from settings import EARNINGS_TRANSCRIPT_TOOL_ENDPOINT, SEC_FILING_TOOL_ENDPOINT

log = logging.getLogger(__name__)

app = FastAPI()


class SecFilingRequest(BaseModel):
    query: str
    ticker: str
    year: str
    filing_type: filings_models.SecFilingType
    top_k: int = 3


class EarningsTranscriptRequest(BaseModel):
    query: str
    ticker: str
    year: str
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] | str  # quarter label e.g. "Q1"
    top_k: int = 3


def _is_filing_embedded(
    vector_store: ChromaVectorStore,
    ticker: str,
    year: str,
    filing_type: str,
) -> bool:
    existing = vector_store.list_filings(ticker, year)
    return any(f["filing_type"] == filing_type for f in existing)


def _hits_to_chunk_dicts(hits: list[tuple[Chunk, float]]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(chunk) for chunk, _ in hits]


def _tool_error_response(
    *,
    error: str,
    message: str,
    ticker: str,
    year: str,
    requested: str,
    vector_store: ChromaVectorStore,
) -> dict[str, Any]:
    available = vector_store.list_filings(ticker, year)
    log.info(
        f"tool error response: {error=} {message=} {ticker=} {year=} "
        f"{requested=} {available=}"
    )
    return {
        "error": error,
        "message": message,
        "ticker": ticker,
        "year": year,
        "requested": requested,
        "available_filings": available,
    }


@app.post(SEC_FILING_TOOL_ENDPOINT)
async def sec_filings_to_embed_and_search(
    request: SecFilingRequest,
) -> list[dict[str, Any]] | dict[str, Any]:
    log.info(
        f"{request.ticker=} {request.year=} {request.filing_type=} {request.query=}"
    )
    vector_store = ChromaVectorStore()

    if not _is_filing_embedded(
        vector_store, request.ticker, request.year, request.filing_type
    ):
        return _tool_error_response(
            error="not_embedded",
            message=(
                "The requested SEC filing is not embedded for this ticker and year."
            ),
            ticker=request.ticker,
            year=request.year,
            requested=str(request.filing_type),
            vector_store=vector_store,
        )

    try:
        hits = vector_store.hybrid_search(
            ticker=request.ticker,
            year=request.year,
            filing_type=request.filing_type,
            query=request.query,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        return _tool_error_response(
            error="search_failed",
            message=str(exc),
            ticker=request.ticker,
            year=request.year,
            requested=str(request.filing_type),
            vector_store=vector_store,
        )

    return _hits_to_chunk_dicts(hits)


@app.post(EARNINGS_TRANSCRIPT_TOOL_ENDPOINT)
async def earnings_transcript_to_embed_and_search(
    request: EarningsTranscriptRequest,
) -> list[dict[str, Any]] | dict[str, Any]:
    log.info(f"{request.ticker=} {request.year=} {request.quarter=} {request.query=}")
    vector_store = ChromaVectorStore()

    if not _is_filing_embedded(
        vector_store, request.ticker, request.year, request.quarter
    ):
        return _tool_error_response(
            error="not_embedded",
            message=(
                "The requested earnings transcript is not embedded for this ticker "
                "and year."
            ),
            ticker=request.ticker,
            year=request.year,
            requested=str(request.quarter),
            vector_store=vector_store,
        )

    try:
        hits = vector_store.hybrid_search(
            ticker=request.ticker,
            year=request.year,
            filing_type=request.quarter,
            query=request.query,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        return _tool_error_response(
            error="search_failed",
            message=str(exc),
            ticker=request.ticker,
            year=request.year,
            requested=str(request.quarter),
            vector_store=vector_store,
        )

    return _hits_to_chunk_dicts(hits)
