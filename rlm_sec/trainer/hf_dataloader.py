from dataclasses import asdict, dataclass
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset


@dataclass(slots=True)
class QAExample:
    question: str
    answer: str
    context: str
    year: str
    ticker_or_company_name: str
    filing_type: str
    data_source: str


QA_FEATURES = Features(
    {
        "question": Value("string"),
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
            question=row["question"],
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
            question=row["question"],
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
