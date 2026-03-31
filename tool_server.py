import dataclasses
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from finance_data.dataloader.text_splitter import Chunk
from finance_data.dataloader.vector_store import ChromaVectorStore
from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
    save_transcript_markdown,
)
from finance_data.dataloader.pipeline import sec_main_to_markdown_and_embed
from settings import EARNINGS_TRANSCRIPT_TOOL_ENDPOINT, SEC_FILING_TOOL_ENDPOINT

log = logging.getLogger(__name__)

app = FastAPI()


class SecFilingRequest(BaseModel):
    query: str
    ticker: str
    year: str
    filing_type: str
    top_k: int = 3


class EarningsTranscriptRequest(BaseModel):
    query: str
    ticker: str
    year: str
    filing_type: str  # quarter label e.g. "Q1"
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


@app.post(SEC_FILING_TOOL_ENDPOINT)
async def sec_filings_to_embed_and_search(
    request: SecFilingRequest,
) -> list[dict[str, Any]]:
    log.info(
        f"{request.ticker=} {request.year=} {request.filing_type=} {request.query=}"
    )
    vector_store = ChromaVectorStore()

    if not _is_filing_embedded(
        vector_store, request.ticker, request.year, request.filing_type
    ):
        log.info(
            f"Embedding SEC filing: {request.ticker=} {request.year=} {request.filing_type=}"
        )
        await sec_main_to_markdown_and_embed(
            ticker=request.ticker,
            year=request.year,
            filing_type=request.filing_type,
        )

    try:
        hits = vector_store.search(
            ticker=request.ticker,
            year=request.year,
            filing_type=request.filing_type,
            query=request.query,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return _hits_to_chunk_dicts(hits)


@app.post(EARNINGS_TRANSCRIPT_TOOL_ENDPOINT)
async def earnings_transcript_to_embed_and_search(
    request: EarningsTranscriptRequest,
) -> list[dict[str, Any]]:
    log.info(
        f"{request.ticker=} {request.year=} {request.filing_type=} {request.query=}"
    )
    vector_store = ChromaVectorStore()

    if not _is_filing_embedded(
        vector_store, request.ticker, request.year, request.filing_type
    ):
        log.info(
            f"Fetching transcript: {request.ticker=} {request.year=} {request.filing_type=}"
        )
        transcript = await get_transcript_for_quarter_async(
            ticker=request.ticker,
            year=int(request.year),
            quarter=request.filing_type,
        )
        if transcript is None:
            raise HTTPException(
                status_code=404,
                detail=f"Transcript not found: {request.ticker=} {request.year=} {request.filing_type=}",
            )

        transcript_path = save_transcript_markdown(transcript)
        vector_store.from_earnings_transcript_markdown(
            ticker=request.ticker,
            year=request.year,
            transcript_paths=[transcript_path],
        )

    try:
        hits = vector_store.search(
            ticker=request.ticker,
            year=request.year,
            filing_type=request.filing_type,
            query=request.query,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return _hits_to_chunk_dicts(hits)
