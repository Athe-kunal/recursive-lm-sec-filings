import asyncio
import re
from dataclasses import asdict, dataclass
from tqdm import tqdm
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from src.sec_dataloader import company_to_ticker, load_sec_filings
from src.sec_data_utils.sec_data import SecResults
from loguru import logger
from pathlib import Path


@dataclass(slots=True)
class QAExample:
    question: str
    answer: str
    context: str
    year: str
    ticker_or_company_name: str
    filing_type: str


QA_FEATURES = Features(
    {
        "question": Value("string"),
        "answer": Value("string"),
        "context": Value("string"),
        "year": Value("string"),
        "ticker_or_company_name": Value("string"),
        "filing_type": Value("string"),
    }
)


def load_financial_qa() -> Dataset:
    ds = load_dataset("virattt/financial-qa-10K", split="train")

    def transform(row: dict) -> dict:
        year, filing_type = row["filing"].split("_", 1)
        example = QAExample(
            question=row["question"],
            answer=row["answer"],
            context=row["context"],
            year=year,
            ticker_or_company_name=row["ticker"],
            filing_type=filing_type,
        )
        return asdict(example)

    return ds.map(transform, remove_columns=ds.column_names, features=QA_FEATURES)


def load_financebench() -> Dataset:
    ds = load_dataset("PatronusAI/financebench", split="train")

    def transform(row: dict) -> dict:
        context = " ".join(
            e["evidence_text"] for e in row["evidence"] if "evidence_text" in e
        )
        example = QAExample(
            question=row["question"],
            answer=row["answer"],
            context=context,
            year=str(row["doc_period"]),
            ticker_or_company_name=row["company"],
            filing_type=row["doc_type"],
        )
        return asdict(example)

    return ds.map(transform, remove_columns=ds.column_names, features=QA_FEATURES)


def load_combined_qa() -> Dataset:
    return concatenate_datasets([load_financial_qa(), load_financebench()])


def is_ticker(symbol: str) -> bool:
    """
    Detect if a string already looks like a stock ticker.
    """
    return bool(re.fullmatch(r"[A-Z]{1,6}", symbol))


async def dispatch_sec_filings(
    ds: Dataset,
) -> list[list[tuple[SecResults, Path]]]:
    """
    Resolve unique (ticker, year) pairs and dispatch SEC filing requests concurrently.
    """
    unique_pairs: set[tuple[str, str]] = set(
        zip(ds["ticker_or_company_name"], ds["year"])
    )

    ticker_cache: dict[str, str | None] = {}
    resolved_pairs: set[tuple[str, str]] = set()

    for company_or_ticker, year in tqdm(
        sorted(unique_pairs),
        desc="Resolving tickers",
    ):
        ticker: str | None

        if is_ticker(company_or_ticker):
            ticker = company_or_ticker
        else:
            if company_or_ticker not in ticker_cache:
                ticker_cache[company_or_ticker] = company_to_ticker(company_or_ticker)

            ticker = ticker_cache[company_or_ticker]

        if ticker is not None:
            resolved_pairs.add((ticker, year))

    resolved_pair_list: list[tuple[str, str]] = sorted(resolved_pairs)[:1]

    logger.info(
        f"dispatch_sec_filings: dispatching {len(resolved_pair_list)} SEC filing requests"
    )

    tasks: list[asyncio.Future[list[tuple[SecResults, Path]]]] = [
        asyncio.create_task(load_sec_filings(ticker, year))
        for ticker, year in resolved_pair_list
    ]

    return await asyncio.gather(*tasks)


async def main() -> None:
    ds = load_combined_qa()
    results = await dispatch_sec_filings(ds)
    logger.info(f"Loaded {len(results)} SEC filing batches")


if __name__ == "__main__":
    asyncio.run(main())
