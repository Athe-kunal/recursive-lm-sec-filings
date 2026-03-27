from dataclasses import asdict, dataclass
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset

DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you lack some knowledge, "
    "you can call one of the two search tools below.\n\n"
    "Tool 1 — SEC Filings (annual and quarterly reports):\n"
    "  <search>SECFilingTool(ticker, year, filing_type)</search>\n"
    "  filing_type is one of: 10-K (annual), 10-Q1, 10-Q2, 10-Q3 (quarterly).\n"
    "  Example: <search>SECFilingTool(AAPL, 2023, 10-K)</search>\n\n"
    "Tool 2 — Earnings Call Transcripts:\n"
    "  <search>EarningsTranscriptTool(ticker, year, quarter)</search>\n"
    "  quarter is one of: Q1, Q2, Q3, Q4.\n"
    "  Example: <search>EarningsTranscriptTool(MSFT, 2023, Q2)</search>\n\n"
    "The search engine will return results between <information> and </information>. "
    "You can search as many times as needed. Once you have sufficient information, "
    "provide the final answer inside <answer> and </answer> without additional explanation. "
    "For example, <answer> The revenue increased by 16%. </answer>. Question: "
)


def build_qa_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"content": DEFAULT_SYSTEM_CONTENT, "role": "system"},
        {"content": DEFAULT_USER_CONTENT_PREFIX + question, "role": "user"},
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


QA_FEATURES = Features(
    {
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "answer": Value("string"),
        "context": Value("string"),
        "year": Value("string"),
        "ticker_or_company_name": Value("string"),
        "filing_type": Value("string"),
        "data_source": Value("string"),
    }
)


def load_financial_qa() -> Dataset:
    ds = load_dataset("virattt/financial-qa-10K", split="train")

    def transform(row: dict) -> dict:
        year, filing_type = row["filing"].split("_", 1)
        example = QAExample(
            prompt=build_qa_prompt(row["question"]),
            answer=row["answer"],
            context=row["context"],
            year=year,
            ticker_or_company_name=row["ticker"],
            filing_type=filing_type,
            data_source="virattt/financial-qa-10K",
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
            prompt=build_qa_prompt(row["question"]),
            answer=row["answer"],
            context=context,
            year=str(row["doc_period"]),
            ticker_or_company_name=row["company"],
            filing_type=row["doc_type"],
            data_source="PatronusAI/financebench",
        )
        return asdict(example)

    return ds.map(transform, remove_columns=ds.column_names, features=QA_FEATURES)


def load_combined_qa() -> Dataset:
    return concatenate_datasets([load_financial_qa(), load_financebench()])
