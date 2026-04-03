import functools
import logging
import os
import re
import random
from dataclasses import asdict, dataclass

from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)
from finance_data.filings.utils import company_to_ticker
import yfinance as yf

logger = logging.getLogger(__name__)


def get_company_name(ticker: str) -> str | None:
    yf_ticker = yf.Ticker(ticker)
    return yf_ticker.info.get("longName")


LOAD_FROM_CACHE_FILE: bool = os.getenv("LOAD_FROM_CACHE_FILE", "").lower() in {
    "1",
    "true",
    "yes",
}
FILTER_UNSUPPORTED_TICKERS: bool = os.getenv(
    "FILTER_UNSUPPORTED_TICKERS", ""
).lower() in {"1", "true", "yes"}
CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "sec_filings"
QUESTION_REPHRASER_RANDOM = random.Random(2026)
MIN_SUPPORTED_YEAR = 2020


DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."

DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. In your reasoning, decide which document "
    "source is most likely to contain the answer (e.g. annual report, quarterly filing, "
    "earnings call, proxy statement, or current report) before calling any tool.\n\n"
    "You have three tools available:\n\n"
    "Tool 1 — Resolve company name to ticker (use this if you only have a company name):\n"
    '  Schema: {"company_name": string}\n'
    "  <search>CompanyNameToTickerTool(company_name)</search>\n"
    "  Example: <search>CompanyNameToTickerTool(Apple Inc.)</search>\n\n"
    "Tool 2 — SEC Filings (annual, quarterly, current, and proxy reports):\n"
    '  Schema: {"query": string, "ticker": string, "year": string, "filing_type": string}\n'
    "  filing_type is one of: 10-K (annual), 10-Q1, 10-Q2, 10-Q3 (quarterly),\n"
    "  8-K (current report / material events), DEF 14A (proxy statement).\n"
    "  <search>SECFilingTool(query, ticker, year, filing_type)</search>\n"
    "  Example: <search>SECFilingTool(cash flow from operations, AAPL, 2023, 10-K)</search>\n\n"
    "Tool 3 — Earnings Call Transcripts:\n"
    '  Schema: {"query": string, "ticker": string, "year": string, "quarter": string}\n'
    "  quarter is one of: Q1, Q2, Q3, Q4.\n"
    "  <search>EarningsTranscriptTool(query, ticker, year, quarter)</search>\n"
    "  Example: <search>EarningsTranscriptTool(cash flow from operations, MSFT, 2023, Q2)</search>\n\n"
    "The search engine will return results between <information> and </information>. "
    "You can search as many times as needed. Once you have sufficient information, "
    "provide the final answer inside <answer> and </answer> without additional explanation. "
    "For example, <answer> The revenue increased by 16%. </answer>. \n\nQuestion: "
)

RANKING_USER_CONTENT_PREFIX = (
    "Given the question below, identify which document types are most relevant to answer it. "
    "You must conduct reasoning inside <think> and </think> tags before outputting the answer.\n\n"
    "Available document types:\n"
    "  DEF14A  — proxy statement (executive pay, board nominees, shareholder votes)\n"
    "  10-K    — annual report (full-year financials, risk factors, business overview)\n"
    "  10-Q    — quarterly report (interim financials, quarter-over-quarter trends)\n"
    "  8-K     — current report (material events: earnings releases, M&A, leadership changes)\n"
    "  Earnings — earnings call transcript (management commentary, analyst Q&A)\n\n"
    "Output the relevant document types inside <sources> and </sources> as a "
    "comma-separated list. Use only these values: DEF14A, 10-K, 10-Q, 8-K, Earnings.\n"
    "For example, <sources> 10-K, Earnings </sources>.\n\nQuestion: "
)


def build_qa_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"content": DEFAULT_SYSTEM_CONTENT, "role": "system"},
        {"content": DEFAULT_USER_CONTENT_PREFIX + question, "role": "user"},
    ]


def build_ranking_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"content": DEFAULT_SYSTEM_CONTENT, "role": "system"},
        {"content": RANKING_USER_CONTENT_PREFIX + question, "role": "user"},
    ]


@dataclass(slots=True)
class QAExample:
    prompt: list[dict[str, str]]
    answer: str
    context: str
    year: str
    ticker_or_company_name: str
    filing_type: str
    data_source: str
    task_type: str


QA_FEATURES = Features(
    {
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "answer": Value("string"),
        "context": Value("string"),
        "year": Value("string"),
        "ticker_or_company_name": Value("string"),
        "filing_type": Value("string"),
        "data_source": Value("string"),
        "task_type": Value("string"),
    }
)


def extract_year_from_filing(filing: str) -> str:
    return filing.split("_", 1)[0]


def extract_filing_type(filing: str) -> str:
    return filing.split("_", 1)[1]


def extract_year_from_doc_period(doc_period: str | int) -> str:
    match = re.search(r"\b(\d{4})\b", str(doc_period))
    if match:
        return match.group(1)
    return str(doc_period)


def is_year_supported(year: str) -> bool:
    if not year.isdigit():
        return False
    return int(year) >= MIN_SUPPORTED_YEAR


def normalize_ticker_value(raw_value: str | None) -> str:
    if raw_value is None:
        return ""
    return raw_value.strip().upper()


def resolve_to_ticker(raw_value: str | None) -> str:
    if raw_value is None:
        return ""
    resolved_ticker = company_to_ticker(raw_value)
    normalized_ticker = (resolved_ticker or "").strip().upper()
    if normalized_ticker:
        return normalized_ticker
    return normalize_ticker_value(raw_value)


def normalize_year(raw_year: str | int | float | None) -> str:
    if raw_year is None:
        return ""
    if isinstance(raw_year, float):
        if raw_year.is_integer():
            return str(int(raw_year))
        return str(raw_year).strip()
    return str(raw_year).strip()


def build_ticker_year_pair(ticker: str, year: str) -> tuple[str, str]:
    return ticker, year


def parse_ticker_year_from_metadata(metadata: dict) -> tuple[str, str] | None:
    ticker = normalize_ticker_value(str(metadata.get("ticker", "")))
    year = normalize_year(metadata.get("year"))
    if not ticker or not year:
        return None
    return build_ticker_year_pair(ticker, year)


@functools.lru_cache(maxsize=1)
def load_available_ticker_year_pairs() -> set[tuple[str, str]]:
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION)
    results = collection.get(include=["metadatas"], limit=None)
    metadatas = results.get("metadatas", [])
    available_pairs: set[tuple[str, str]] = set()
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue
        pair = parse_ticker_year_from_metadata(metadata)
        if pair is None:
            continue
        available_pairs.add(pair)
    logger.info("%s", f"{len(available_pairs)=}")
    return available_pairs


def has_training_data_for_ticker_year(ticker_or_company_name: str, year: str) -> bool:
    available_pairs = load_available_ticker_year_pairs()
    normalized_ticker = resolve_to_ticker(ticker_or_company_name)
    normalized_year = normalize_year(year)
    pair = build_ticker_year_pair(normalized_ticker, normalized_year)
    has_pair = pair in available_pairs
    if not has_pair:
        logger.info("%s", f"{pair=}")
    return has_pair


def is_covered_qa_example_row(row: dict) -> bool:
    return has_training_data_for_ticker_year(
        ticker_or_company_name=row["ticker_or_company_name"],
        year=row["year"],
    )


def maybe_filter_unsupported_ticker_rows(dataset: Dataset) -> Dataset:
    if not FILTER_UNSUPPORTED_TICKERS:
        logger.info("%s", f"{FILTER_UNSUPPORTED_TICKERS=}")
        return dataset
    logger.info("%s", f"{FILTER_UNSUPPORTED_TICKERS=}")
    return dataset.filter(
        is_covered_qa_example_row,
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
    )


def question_mentions_year(question: str, year: str) -> bool:
    return re.search(rf"\b{re.escape(year)}\b", question) is not None


def normalize_for_match(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def question_mentions_company_name(question: str, company_name: str) -> bool:
    normalized_question = normalize_for_match(question)
    normalized_company_name = normalize_for_match(company_name)
    return normalized_company_name in normalized_question


def _lowercase_question(question: str) -> str:
    return question[:1].lower() + question[1:]


@functools.lru_cache(maxsize=1024)
def resolve_company_name(ticker: str) -> str:
    company_name = get_company_name(ticker)
    return company_name if company_name else ticker


def rephrase_financebench_question(question: str, year: str) -> str:
    if question_mentions_year(question, year):
        return question
    return f"For the year {year}, {_lowercase_question(question)}"


def rephrase_financial_qa_question(question: str, company_name: str, year: str) -> str:
    has_company_name = question_mentions_company_name(question, company_name)
    has_year = question_mentions_year(question, year)
    if has_company_name and has_year:
        return question
    lowercased = _lowercase_question(question)
    templates = [
        "For {company_name} in {year}, {question}",
        "In {year}, for {company_name}, {question}",
        "Considering {company_name}'s {year} filing, {question}",
    ]
    template = QUESTION_REPHRASER_RANDOM.choice(templates)
    return template.format(company_name=company_name, year=year, question=lowercased)


def transform_financial_qa_row(row: dict) -> dict:
    year = extract_year_from_filing(row["filing"])
    filing_type = extract_filing_type(row["filing"])
    company_name = resolve_company_name(row["ticker"])
    question = rephrase_financial_qa_question(row["question"], company_name, year)
    example = QAExample(
        prompt=build_qa_prompt(question),
        answer=row["answer"],
        context=row["context"],
        year=year,
        ticker_or_company_name=company_name,
        filing_type=filing_type,
        data_source="virattt/financial-qa-10K",
        task_type="qa",
    )
    return asdict(example)


def is_supported_financial_qa_row(row: dict) -> bool:
    year = extract_year_from_filing(row["filing"])
    return is_year_supported(year)


def load_financial_qa() -> Dataset:
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    filtered_ds = ds.filter(
        is_supported_financial_qa_row, load_from_cache_file=LOAD_FROM_CACHE_FILE
    )
    return filtered_ds.map(
        transform_financial_qa_row,
        remove_columns=filtered_ds.column_names,
        features=QA_FEATURES,
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
    )


def build_financebench_context(evidence: list[dict[str, str]]) -> str:
    return " ".join(e["evidence_text"] for e in evidence if "evidence_text" in e)


def transform_financebench_row(row: dict) -> dict:
    year = extract_year_from_doc_period(row["doc_period"])
    question = rephrase_financebench_question(row["question"], year)
    context = build_financebench_context(row["evidence"])
    example = QAExample(
        prompt=build_qa_prompt(question),
        answer=row["answer"],
        context=context,
        year=year,
        ticker_or_company_name=row["company"],
        filing_type=row["doc_type"],
        data_source="PatronusAI/financebench",
        task_type="qa",
    )
    return asdict(example)


def is_supported_financebench_row(row: dict) -> bool:
    year = extract_year_from_doc_period(row["doc_period"])
    return is_year_supported(year)


def load_financebench() -> Dataset:
    ds = load_dataset("PatronusAI/financebench", split="train")
    filtered_ds = ds.filter(
        is_supported_financebench_row, load_from_cache_file=LOAD_FROM_CACHE_FILE
    )
    return filtered_ds.map(
        transform_financebench_row,
        remove_columns=filtered_ds.column_names,
        features=QA_FEATURES,
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
    )


def load_combined_qa() -> Dataset:
    combined_dataset = concatenate_datasets([load_financial_qa(), load_financebench()])
    return maybe_filter_unsupported_ticker_rows(combined_dataset)


INDEX_TO_FILING_TYPE: dict[str, str] = {
    "0": "DEF14A",
    "1": "10-K",
    "2": "10-Q",
    "3": "8-K",
    "4": "Earnings",
}

QUESTION_PATTERN = re.compile(r"Question:\s*(.+?)(?:\n\nDocument Types)", re.DOTALL)


@dataclass(slots=True)
class DocumentRankingExample:
    prompt: list[dict[str, str]]
    relevant: list[str]
    not_relevant: list[str]
    data_source: str
    task_type: str


DOCUMENT_RANKING_FEATURES = Features(
    {
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "relevant": Sequence(Value("string")),
        "not_relevant": Sequence(Value("string")),
        "data_source": Value("string"),
        "task_type": Value("string"),
    }
)


def extract_question(content: str) -> str:
    match = QUESTION_PATTERN.search(content)
    return match.group(1).strip() if match else ""


def parse_qrel(qrel: dict[str, int]) -> tuple[list[str], list[str]]:
    relevant = sorted(
        (idx for idx, score in qrel.items() if score > 0),
        key=lambda idx: qrel[idx],
        reverse=True,
    )
    not_relevant = [idx for idx, score in qrel.items() if score == 0]
    return (
        [INDEX_TO_FILING_TYPE[idx] for idx in relevant],
        [INDEX_TO_FILING_TYPE[idx] for idx in not_relevant],
    )


def load_finance_agent_bench(file_path: str) -> Dataset:
    ds = load_dataset("json", data_files=file_path, split="train")

    return ds.map(
        transform_finance_agent_bench_row,
        remove_columns=ds.column_names,
        features=DOCUMENT_RANKING_FEATURES,
    )


def transform_finance_agent_bench_row(row: dict) -> dict:
    content = row["messages"][0]["content"]
    question = extract_question(content)
    relevant, not_relevant = parse_qrel(row["qrel"])
    example = DocumentRankingExample(
        prompt=build_ranking_prompt(question),
        relevant=relevant,
        not_relevant=not_relevant,
        data_source="financeAgentBench",
        task_type="ranking",
    )
    return asdict(example)
