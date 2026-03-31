"""Backfill SEC filing and transcript embeddings directly.

This script reads the local parquet splits from ``data/`` with Hugging Face
Datasets, derives the unique ticker/year targets for the requested data
sources, and then embeds the matching SEC filings and earnings transcripts
without routing through the FastAPI tool server.

Any individual failure is logged and skipped so the batch can continue.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import NamedTuple

import yfinance as yf  # type: ignore[import-untyped]
from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from finance_data.dataloader.pipeline import sec_main_to_markdown_and_embed
from finance_data.dataloader.vector_store import ChromaVectorStore
from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
    save_transcript_markdown,
)


FINANCIAL_QA_SOURCE = "virattt/financial-qa-10K"
FINANCEBENCH_SOURCE = "PatronusAI/financebench"
SUPPORTED_SOURCES = {FINANCIAL_QA_SOURCE, FINANCEBENCH_SOURCE}
SEC_FILING_TYPES = ("10-K", "10-Q1", "10-Q2", "10-Q3", "8-K", "DEF 14A")
EARNINGS_QUARTERS = ("Q1", "Q2", "Q3", "Q4")

log = logging.getLogger(__name__)


class CompanyYear(NamedTuple):
    company_name: str
    year: str


class TickerYear(NamedTuple):
    ticker: str
    year: str
    data_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SEC filings and earnings transcripts directly."
    )
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--validation-path", default="data/validation.parquet")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_local_dataset(train_path: str, validation_path: str) -> Dataset:
    data_files = [train_path, validation_path]
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    log.info(f"{data_files=}")
    log.info(f"{len(dataset)=}")
    return dataset


def collect_financial_qa_years(dataset: Dataset) -> set[str]:
    years = {
        str(row["year"]).strip()
        for row in dataset
        if row["data_source"] == FINANCIAL_QA_SOURCE and str(row["year"]).strip()
    }
    log.info(f"{years=}")
    return years


def collect_financebench_company_years(dataset: Dataset) -> set[CompanyYear]:
    company_years = {
        CompanyYear(
            company_name=str(row["ticker_or_company_name"]).strip(),
            year=str(row["year"]).strip(),
        )
        for row in dataset
        if row["data_source"] == FINANCEBENCH_SOURCE
        and str(row["ticker_or_company_name"]).strip()
        and str(row["year"]).strip()
    }
    log.info(f"{len(company_years)=}")
    return company_years


def extract_year_from_filing(filing: str) -> str:
    return filing.split("_", 1)[0]


def load_financial_qa_ticker_years(target_years: set[str]) -> set[TickerYear]:
    if not target_years:
        return set()

    raw_dataset = load_dataset(FINANCIAL_QA_SOURCE, split="train")
    ticker_years = {
        TickerYear(
            ticker=str(row["ticker"]).strip().upper(),
            year=extract_year_from_filing(str(row["filing"])),
            data_source=FINANCIAL_QA_SOURCE,
        )
        for row in raw_dataset
        if extract_year_from_filing(str(row["filing"])) in target_years
    }
    log.info(f"{len(ticker_years)=}")
    return ticker_years


def normalize_company_name(company_name: str) -> str:
    return " ".join(company_name.replace("&", "and").split()).lower()


def choose_equity_symbol(quotes: list[dict]) -> str | None:
    for quote in quotes:
        if quote.get("quoteType") != "EQUITY":
            continue
        symbol = str(quote.get("symbol", "")).strip().upper()
        if not symbol or "." in symbol:
            continue
        return symbol
    return None


def resolve_ticker_from_company_name(company_name: str) -> str | None:
    search = yf.Search(query=company_name, max_results=10)
    ticker = choose_equity_symbol(getattr(search, "quotes", []))
    log.info(f"{company_name=} {ticker=}")
    return ticker


def resolve_financebench_ticker_years(
    company_years: set[CompanyYear],
) -> tuple[set[TickerYear], set[str]]:
    resolved_tickers: dict[str, str | None] = {}
    unresolved_companies: set[str] = set()
    ticker_years: set[TickerYear] = set()

    for company_year in sorted(company_years):
        company_name = company_year.company_name
        if company_name not in resolved_tickers:
            resolved_tickers[company_name] = resolve_ticker_from_company_name(
                company_name
            )

        ticker = resolved_tickers[company_name]
        if not ticker:
            unresolved_companies.add(company_name)
            continue

        ticker_years.add(
            TickerYear(
                ticker=ticker,
                year=company_year.year,
                data_source=FINANCEBENCH_SOURCE,
            )
        )

    log.info(f"{len(ticker_years)=} {len(unresolved_companies)=}")
    return ticker_years, unresolved_companies


def build_ticker_year_targets(dataset: Dataset) -> set[TickerYear]:
    source_values = {str(row["data_source"]).strip() for row in dataset}
    present_sources = SUPPORTED_SOURCES.intersection(source_values)
    log.info(f"{present_sources=}")

    financial_qa_years = collect_financial_qa_years(dataset)
    financebench_company_years = collect_financebench_company_years(dataset)

    financial_qa_targets = load_financial_qa_ticker_years(financial_qa_years)
    financebench_targets, unresolved_companies = resolve_financebench_ticker_years(
        financebench_company_years
    )

    if unresolved_companies:
        log.warning(f"{unresolved_companies=}")

    ticker_year_targets = financial_qa_targets | financebench_targets
    log.info(f"{len(ticker_year_targets)=}")
    return ticker_year_targets


def is_filing_embedded(
    vector_store: ChromaVectorStore,
    ticker: str,
    year: str,
    filing_type: str,
) -> bool:
    existing = vector_store.list_filings(ticker, year)
    return any(filing["filing_type"] == filing_type for filing in existing)


async def embed_sec_filing(
    semaphore: asyncio.Semaphore,
    ticker_year: TickerYear,
    filing_type: str,
) -> None:
    async with semaphore:
        vector_store = ChromaVectorStore()
        if is_filing_embedded(
            vector_store=vector_store,
            ticker=ticker_year.ticker,
            year=ticker_year.year,
            filing_type=filing_type,
        ):
            log.info(
                f"Skipping already embedded SEC filing: {ticker_year.ticker=} "
                f"{ticker_year.year=} {filing_type=}"
            )
            return

        try:
            await sec_main_to_markdown_and_embed(
                ticker=ticker_year.ticker,
                year=ticker_year.year,
                filing_type=filing_type,
            )
            log.info(
                f"Embedded SEC filing: {ticker_year.ticker=} "
                f"{ticker_year.year=} {filing_type=}"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                f"Skipping failed SEC filing: {ticker_year.ticker=} "
                f"{ticker_year.year=} {filing_type=} {exc=}"
            )


async def embed_earnings_transcript(
    semaphore: asyncio.Semaphore,
    ticker_year: TickerYear,
    quarter: str,
) -> None:
    async with semaphore:
        vector_store = ChromaVectorStore()
        if is_filing_embedded(
            vector_store=vector_store,
            ticker=ticker_year.ticker,
            year=ticker_year.year,
            filing_type=quarter,
        ):
            log.info(
                f"Skipping already embedded transcript: {ticker_year.ticker=} "
                f"{ticker_year.year=} {quarter=}"
            )
            return

        try:
            transcript = await get_transcript_for_quarter_async(
                ticker=ticker_year.ticker,
                year=int(ticker_year.year),
                quarter=quarter,
            )
            if transcript is None:
                log.warning(
                    f"Skipping missing transcript: {ticker_year.ticker=} "
                    f"{ticker_year.year=} {quarter=}"
                )
                return

            transcript_path = save_transcript_markdown(transcript)
            vector_store.from_earnings_transcript_markdown(
                ticker=ticker_year.ticker,
                year=ticker_year.year,
                transcript_paths=[transcript_path],
            )
            log.info(
                f"Embedded transcript: {ticker_year.ticker=} "
                f"{ticker_year.year=} {quarter=} {transcript_path=}"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                f"Skipping failed transcript: {ticker_year.ticker=} "
                f"{ticker_year.year=} {quarter=} {exc=}"
            )


async def backfill_sec_filings(
    ticker_year_targets: set[TickerYear],
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        embed_sec_filing(
            semaphore=semaphore,
            ticker_year=ticker_year,
            filing_type=filing_type,
        )
        for ticker_year in sorted(ticker_year_targets)
        for filing_type in SEC_FILING_TYPES
    ]
    log.info(f"Submitting SEC filing embedding tasks: {len(tasks)=}")
    await asyncio.gather(*tasks)


async def backfill_earnings_transcripts(
    ticker_year_targets: set[TickerYear],
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        embed_earnings_transcript(
            semaphore=semaphore,
            ticker_year=ticker_year,
            quarter=quarter,
        )
        for ticker_year in sorted(ticker_year_targets)
        for quarter in EARNINGS_QUARTERS
    ]
    log.info(f"Submitting earnings transcript embedding tasks: {len(tasks)=}")
    await asyncio.gather(*tasks)


async def async_main(args: argparse.Namespace) -> None:
    dataset = load_local_dataset(
        train_path=args.train_path,
        validation_path=args.validation_path,
    )
    ticker_year_targets = build_ticker_year_targets(dataset)

    log.info(f"{args.concurrency=} {len(ticker_year_targets)=}")
    await asyncio.gather(
        backfill_sec_filings(
            ticker_year_targets=ticker_year_targets,
            concurrency=args.concurrency,
        ),
        backfill_earnings_transcripts(
            ticker_year_targets=ticker_year_targets,
            concurrency=args.concurrency,
        ),
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
