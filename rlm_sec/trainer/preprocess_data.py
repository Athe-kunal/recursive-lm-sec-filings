import argparse
import os

from datasets import Dataset, concatenate_datasets
from loguru import logger

from rlm_sec.trainer import hf_dataloader


def make_qa_map_fn():
    """Project a QA example into the unified schema.

    Ranking-only fields (relevant, not_relevant) are filled with empty lists
    so the schema is compatible for concatenation with ranking examples.
    """
    def process_fn(example: dict, idx: int) -> dict:
        return {
            "data_source": example["data_source"],
            "prompt": example["prompt"],
            "env_class": "null",
            "task_type": example["task_type"],
            "answer": example["answer"],
            "context": example["context"],
            "year": example["year"],
            "ticker_or_company_name": example["ticker_or_company_name"],
            "filing_type": example["filing_type"],
            "relevant": [],
            "not_relevant": [],
        }

    return process_fn


def make_ranking_map_fn():
    """Project a ranking example into the unified schema.

    QA-only fields (answer, context, year, ticker_or_company_name, filing_type)
    are filled with empty strings so the schema is compatible for concatenation
    with QA examples.
    """
    def process_fn(example: dict, idx: int) -> dict:
        return {
            "data_source": example["data_source"],
            "prompt": example["prompt"],
            "env_class": "null",
            "task_type": example["task_type"],
            "answer": "",
            "context": "",
            "year": "",
            "ticker_or_company_name": "",
            "filing_type": "",
            "relevant": example["relevant"],
            "not_relevant": example["not_relevant"],
        }

    return process_fn


def build_combined_dataset(finance_agent_bench_file: str) -> Dataset:
    qa_ds = hf_dataloader.load_combined_qa().map(
        function=make_qa_map_fn(), with_indices=True
    )
    ranking_ds = hf_dataloader.load_finance_agent_bench(finance_agent_bench_file).map(
        function=make_ranking_map_fn(), with_indices=True
    )
    combined = concatenate_datasets([qa_ds, ranking_ds])
    logger.info(f"Combined dataset: qa={len(qa_ds)}, ranking={len(ranking_ds)}, total={len(combined)}")
    return combined


def save_split(
    dataset: Dataset,
    train_path: str,
    val_path: str,
    test_size: float,
    seed: int,
):
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    logger.info(f"{train_path}: train={len(split['train'])}, validation={len(split['test'])}")
    split["train"].to_parquet(train_path)
    split["test"].to_parquet(val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument(
        "--finance_agent_bench_file",
        type=str,
        default="document_ranking_kaggle_dev.jsonl",
    )

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    combined = build_combined_dataset(args.finance_agent_bench_file)
    save_split(
        dataset=combined,
        train_path=os.path.join(args.output_dir, "train.parquet"),
        val_path=os.path.join(args.output_dir, "validation.parquet"),
        test_size=args.test_size,
        seed=args.seed,
    )
