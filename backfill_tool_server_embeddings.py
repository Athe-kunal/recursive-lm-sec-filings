"""Backfill SEC filing and transcript embeddings directly.

This script reads the local parquet splits from ``data/`` with Hugging Face
Datasets, derives the unique ticker/year targets, checks ChromaDB for coverage
gaps, and then embeds only the missing SEC filings and earnings transcripts
without routing through the FastAPI tool server.

Any individual failure is logged and skipped so the batch can continue.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from collections import Counter
from typing import NamedTuple

from loguru import logger as log

from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from finance_data.dataloader.pipeline import sec_main_to_markdown_and_embed
from finance_data.dataloader.vector_store import ChromaVectorStore
from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
    save_transcript_markdown,
)
from finance_data.filings.utils import company_to_ticker


SEC_FILING_TYPES: tuple[str, ...] = ("10-K", "10-Q1", "10-Q2", "10-Q3", "8-K", "DEF 14A")
EARNINGS_QUARTERS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

_QUARTER_RE = re.compile(r"^Q[1-4]$", re.IGNORECASE)


class TickerYear(NamedTuple):
    ticker: str
    year: str


class SecFilingJob(NamedTuple):
    ticker: str
    year: str
    filing_type: str


class TranscriptJob(NamedTuple):
    ticker: str
    year: str
    quarter: str


class JobLists(NamedTuple):
    sec_filing_jobs: list[SecFilingJob]
    transcript_jobs: list[TranscriptJob]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SEC filings and earnings transcripts directly."
    )
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--validation-path", default="data/validation.parquet")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def configure_logging() -> None:
    log.remove()
    log.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def load_local_dataset(train_path: str, validation_path: str) -> Dataset:
    data_files = [train_path, validation_path]
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    log.info(f"{data_files=}")
    log.info(f"{len(dataset)=}")
    return dataset


def resolve_ticker(company_name: str) -> str | None:
    resolved = company_to_ticker(company_name)
    ticker = (resolved or "").strip().upper() or None
    log.info(f"{company_name=} {ticker=}")
    return ticker


def collect_ticker_year_pairs(dataset: Dataset) -> set[TickerYear]:
    """Unique (ticker, year) pairs from the dataset; company names resolved via company_to_ticker."""
    resolved_cache: dict[str, str | None] = {}
    ticker_years: set[TickerYear] = set()
    unresolved: set[str] = set()

    for row in dataset:
        if str(row.get("task_type", "")).strip() != "qa":
            continue

        raw_name = str(row.get("ticker_or_company_name", "")).strip()
        raw_year = str(row.get("year", "")).strip()
        if not raw_name or not raw_year:
            continue

        if raw_name not in resolved_cache:
            resolved_cache[raw_name] = resolve_ticker(raw_name)

        ticker = resolved_cache[raw_name]
        if not ticker:
            unresolved.add(raw_name)
            continue

        ticker_years.add(TickerYear(ticker=ticker, year=raw_year))

    if unresolved:
        log.warning(f"{unresolved=} (could not resolve ticker, skipping)")
    log.info(f"{len(ticker_years)=} {len(unresolved)=}")
    return ticker_years


def find_present_filing_types(
    vector_store: ChromaVectorStore,
    ticker: str,
    year: str,
) -> set[str]:
    existing = vector_store.list_filings(ticker, year)
    return {str(f.get("filing_type", "")).strip() for f in existing if f.get("filing_type")}


def find_ticker_coverage_gaps(
    vector_store: ChromaVectorStore,
    ticker_year: TickerYear,
) -> tuple[set[str], set[str]]:
    """Returns (missing SEC filing types, missing earnings quarters) for a ticker/year."""
    present = find_present_filing_types(vector_store, ticker_year.ticker, ticker_year.year)

    missing_sec = set(SEC_FILING_TYPES) - present

    present_quarters = {t.upper() for t in present if _QUARTER_RE.fullmatch(t.upper())}
    missing_quarters = set(EARNINGS_QUARTERS) - present_quarters

    return missing_sec, missing_quarters


def build_job_lists(ticker_year_targets: set[TickerYear]) -> JobLists:
    """Check ChromaDB coverage and build job lists only for missing filings/transcripts."""
    vector_store = ChromaVectorStore()
    sec_filing_jobs: list[SecFilingJob] = []
    transcript_jobs: list[TranscriptJob] = []

    for ticker_year in sorted(ticker_year_targets):
        missing_sec, missing_quarters = find_ticker_coverage_gaps(vector_store, ticker_year)

        for filing_type in sorted(missing_sec):
            sec_filing_jobs.append(
                SecFilingJob(ticker=ticker_year.ticker, year=ticker_year.year, filing_type=filing_type)
            )

        for quarter in sorted(missing_quarters):
            transcript_jobs.append(
                TranscriptJob(ticker=ticker_year.ticker, year=ticker_year.year, quarter=quarter)
            )

        log.info(
            f"{ticker_year.ticker=} {ticker_year.year=} "
            f"missing_sec={sorted(missing_sec)} missing_quarters={sorted(missing_quarters)}"
        )

    log.info(f"{len(sec_filing_jobs)=} {len(transcript_jobs)=}")
    return JobLists(sec_filing_jobs=sec_filing_jobs, transcript_jobs=transcript_jobs)


async def embed_sec_filing(
    semaphore: asyncio.Semaphore,
    job: SecFilingJob,
) -> str:
    async with semaphore:
        try:
            await sec_main_to_markdown_and_embed(
                ticker=job.ticker,
                year=job.year,
                filing_type=job.filing_type,
            )
            log.info(f"Embedded SEC filing: {job.ticker=} {job.year=} {job.filing_type=}")
            return "embedded"
        except Exception as exc:  # noqa: BLE001
            log.error(
                f"SEC filing not embedded (missing or unavailable): "
                f"{job.ticker=} {job.year=} {job.filing_type=} {exc=}"
            )
            return "error"


async def embed_earnings_transcript(
    semaphore: asyncio.Semaphore,
    job: TranscriptJob,
) -> str:
    async with semaphore:
        try:
            transcript = await get_transcript_for_quarter_async(
                ticker=job.ticker,
                year=int(job.year),
                quarter=job.quarter,
            )
            if transcript is None:
                log.error(
                    f"Earnings transcript not present: "
                    f"{job.ticker=} {job.year=} {job.quarter=}"
                )
                return "missing_transcript"

            transcript_path = save_transcript_markdown(transcript)
            vector_store = ChromaVectorStore()
            vector_store.from_earnings_transcript_markdown(
                ticker=job.ticker,
                year=job.year,
                transcript_paths=[transcript_path],
            )
            log.info(
                f"Embedded transcript: {job.ticker=} {job.year=} {job.quarter=} {transcript_path=}"
            )
            return "embedded"
        except Exception as exc:  # noqa: BLE001
            log.error(
                f"Earnings transcript not embedded (missing or failed): "
                f"{job.ticker=} {job.year=} {job.quarter=} {exc=}"
            )
            return "error"


async def backfill_sec_filings(
    sec_filing_jobs: list[SecFilingJob],
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [embed_sec_filing(semaphore=semaphore, job=job) for job in sec_filing_jobs]
    log.info(f"Submitting SEC filing embedding tasks: {len(tasks)=}")
    outcomes = await asyncio.gather(*tasks)
    log.info(f"SEC filing task outcomes: {dict(Counter(outcomes))}")


async def backfill_earnings_transcripts(
    transcript_jobs: list[TranscriptJob],
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [embed_earnings_transcript(semaphore=semaphore, job=job) for job in transcript_jobs]
    log.info(f"Submitting earnings transcript embedding tasks: {len(tasks)=}")
    outcomes = await asyncio.gather(*tasks)
    log.info(f"Earnings transcript task outcomes: {dict(Counter(outcomes))}")


async def async_main(args: argparse.Namespace) -> None:
    dataset = load_local_dataset(
        train_path=args.train_path,
        validation_path=args.validation_path,
    )
    ticker_year_targets = collect_ticker_year_pairs(dataset)
    job_lists = build_job_lists(ticker_year_targets)

    log.info(f"{args.concurrency=} {len(job_lists.sec_filing_jobs)=} {len(job_lists.transcript_jobs)=}")
    started = time.perf_counter()
    await asyncio.gather(
        backfill_sec_filings(
            sec_filing_jobs=job_lists.sec_filing_jobs,
            concurrency=args.concurrency,
        ),
        backfill_earnings_transcripts(
            transcript_jobs=job_lists.transcript_jobs,
            concurrency=args.concurrency,
        ),
    )
    elapsed_s = time.perf_counter() - started
    log.info(f"Backfill finished in {elapsed_s:.1f}s (process exiting normally).")


def main() -> None:
    configure_logging()
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
