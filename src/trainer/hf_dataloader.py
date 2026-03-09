from datasets import Dataset, concatenate_datasets, load_dataset


def load_financial_qa() -> Dataset:
    ds = load_dataset("virattt/financial-qa-10K", split="train")

    def transform(row):
        year, filing_type = row["filing"].split("_", 1)
        return {
            "question": row["question"],
            "answer": row["answer"],
            "context": row["context"],
            "year": year,
            "filing_type": filing_type,
        }

    return ds.map(transform, remove_columns=ds.column_names)


def load_financebench() -> Dataset:
    ds = load_dataset("PatronusAI/financebench", split="train")

    def transform(row):
        context = " ".join(
            e["evidence_text"] for e in row["evidence"] if "evidence_text" in e
        )
        return {
            "question": row["question"],
            "answer": row["answer"],
            "context": context,
            "year": str(row["doc_period"]),
            "filing_type": row["doc_type"],
        }

    return ds.map(transform, remove_columns=ds.column_names)


def load_combined_qa() -> Dataset:
    return concatenate_datasets([load_financial_qa(), load_financebench()])


if __name__ == "__main__":
    ds = load_combined_qa()
    print(ds[0])
