from __future__ import annotations

import re
import sys
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from datasets import load_dataset
from loguru import logger
from finance_data.filings.utils import company_to_ticker
from tqdm import tqdm


# Must match backfill_tool_server_embeddings.SEC_FILING_TYPES / EARNINGS_QUARTERS
SEC_FILING_TYPES: tuple[str, ...] = (
    "10-K",
    "10-Q1",
    "10-Q2",
    "10-Q3",
    "8-K",
    "DEF 14A",
)
EARNINGS_QUARTERS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

_QUARTER_RE = re.compile(r"^Q[1-4]$", re.IGNORECASE)

train_dataset = load_dataset("parquet", data_files="data/train.parquet")["train"]
val_dataset = load_dataset("parquet", data_files="data/validation.parquet")["train"]


def configure_logging() -> None:
    """Stderr sink with ANSI colors for highlighted lines (e.g. coverage gaps)."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        colorize=True,
    )


def collect_ticker_year_pairs() -> list[tuple[str, object]]:
    """Pairs of (normalized ticker, raw year value as stored in the dataset row)."""
    all_tickers_year: list[tuple[str, object]] = []

    for dataset in (train_dataset, val_dataset):
        for row in dataset:
            if row["task_type"] != "qa":
                continue

            raw_name = row["ticker_or_company_name"]
            resolved = company_to_ticker(raw_name) if raw_name else None
            ticker = (resolved or "").strip().upper()
            year_raw: object = row["year"]
            all_tickers_year.append((ticker, year_raw))

    return all_tickers_year


def year_metadata_candidates(raw_year: object) -> list[str | int | float]:
    """Values to try in Chroma filters; ingest may use int, str, or float years."""
    candidates: list[str | int | float] = []

    if isinstance(raw_year, bool):
        return [raw_year]

    if isinstance(raw_year, int):
        candidates.extend([raw_year, str(raw_year)])
        return list(dict.fromkeys(candidates))

    if isinstance(raw_year, float):
        if raw_year.is_integer():
            i = int(raw_year)
            candidates.extend([i, str(i)])
        else:
            candidates.append(raw_year)
        return list(dict.fromkeys(candidates))

    text = str(raw_year).strip()
    candidates.append(text)
    try:
        as_float = float(text)
        if as_float.is_integer():
            i = int(as_float)
            candidates.extend([i, str(i)])
    except ValueError:
        pass

    return list(dict.fromkeys(candidates))


def build_ticker_year_where(
    ticker: str, year_values: list[str | int | float]
) -> dict[str, object]:
    """Chroma metadata equality is strict; OR across plausible year representations."""
    if not year_values:
        raise ValueError(f"year_values must be non-empty for {ticker=}")

    if len(year_values) == 1:
        return {"$and": [{"ticker": ticker}, {"year": year_values[0]}]}

    return {
        "$or": [
            {"$and": [{"ticker": ticker}, {"year": value}]} for value in year_values
        ]
    }


def get_ticker_year_metadatas(
    collection: Collection,
    ticker: str,
    raw_year: object,
) -> list[Any]:
    """All chunk metadatas for this ticker/year (one row per embedded chunk)."""
    year_values = year_metadata_candidates(raw_year)
    where_filter = build_ticker_year_where(ticker, year_values)
    results = collection.get(
        where=where_filter,
        limit=None,
        include=["metadatas"],
    )
    metadatas = results.get("metadatas")
    if not isinstance(metadatas, list):
        return []
    return metadatas


def distinct_filing_types(metadatas: list[Any]) -> set[str]:
    """Unique filing_type values from chunk metadata (e.g. 10-K, Q1)."""
    found: set[str] = set()
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        filing_type = str(meta.get("filing_type", "")).strip()
        if filing_type:
            found.add(filing_type)
    return found


def transcript_quarters_present(filing_types: set[str]) -> set[str]:
    """Normalize transcript labels to Q1..Q4 (matches vector_store.resolve_transcript_quarters)."""
    quarters: set[str] = set()
    for ft in filing_types:
        u = ft.upper()
        if _QUARTER_RE.fullmatch(u):
            quarters.add(u)
    return quarters


def coverage_gaps(filing_types: set[str]) -> tuple[set[str], set[str]]:
    """Returns (missing SEC forms, missing earnings quarters)."""
    present_sec = filing_types.intersection(SEC_FILING_TYPES)
    missing_sec = set(SEC_FILING_TYPES) - present_sec

    present_q = transcript_quarters_present(filing_types)
    missing_transcripts = set(EARNINGS_QUARTERS) - present_q

    return missing_sec, missing_transcripts


def process_one(
    collection: Collection, pair: tuple[str, object]
) -> tuple[str, str] | None:
    ticker, raw_year = pair
    year = str(raw_year)
    if not ticker:
        line = f"{ticker=} {year=} no_ticker"
        logger.opt(colors=True).info(f"<red>{line}</red>")
        return (ticker, year)

    metadatas = get_ticker_year_metadatas(collection, ticker, raw_year)
    n_chunks = len(metadatas)
    filing_types = distinct_filing_types(metadatas)
    missing_sec, missing_transcripts = coverage_gaps(filing_types)

    line = (
        f"{ticker=} {year=} {n_chunks=} "
        f"missing_sec={sorted(missing_sec)} missing_transcripts={sorted(missing_transcripts)}"
    )
    has_gap = n_chunks == 0 or bool(missing_sec) or bool(missing_transcripts)
    if has_gap:
        logger.opt(colors=True).info(f"<red>{line}</red>")
    else:
        logger.info(line)

    if has_gap:
        return (ticker, year)
    return None


def main() -> None:
    configure_logging()

    chroma_path = "chroma_db"
    collection_name = "sec_filings"
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)

    all_tickers_year = collect_ticker_year_pairs()
    logger.info(f"{len(all_tickers_year)=}")
    missing_pairs: list[tuple[str, str]] = []

    for pair in tqdm(
        all_tickers_year,
        desc="Checking ChromaDB",
        file=sys.stdout,
    ):
        result = process_one(collection, pair)
        if result is not None:
            missing_pairs.append(result)

    for ticker, year in missing_pairs:
        print(f"{ticker}\t{year}")


if __name__ == "__main__":
    main()
