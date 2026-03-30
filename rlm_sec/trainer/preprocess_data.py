import argparse
import os

from loguru import logger
from datasets import Dataset
from typing import Callable
from rlm_sec.trainer import hf_dataloader


def make_qa_map_fn():
    def process_fn(example: dict, idx: int) -> dict:
        return {
            "data_source": example["data_source"],
            "prompt": example["prompt"],
            "env_class": "null",
            "answer": example["answer"],
            "context": example["context"],
            "year": example["year"],
            "ticker_or_company_name": example["ticker_or_company_name"],
            "filing_type": example["filing_type"],
        }

    return process_fn


def make_ranking_map_fn():
    def process_fn(example: dict, idx: int) -> dict:
        return {
            "data_source": example["data_source"],
            "prompt": example["prompt"],
            "relevant": example["relevant"],
            "not_relevant": example["not_relevant"],
            "env_class": "null",
        }

    return process_fn


def save_split(
    dataset: Dataset,
    map_fn: Callable,
    train_path: str,
    val_path: str,
    test_size: float,
    seed: int,
):
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    train_ds = split["train"].map(function=map_fn, with_indices=True)
    val_ds = split["test"].map(function=map_fn, with_indices=True)
    logger.info(f"{train_path}: train={len(train_ds)}, validation={len(val_ds)}")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)


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

    # save_split(
    #     dataset=hf_dataloader.load_combined_qa(),
    #     map_fn=make_qa_map_fn(),
    #     train_path=os.path.join(args.output_dir, "qa_train.parquet"),
    #     val_path=os.path.join(args.output_dir, "qa_validation.parquet"),
    #     test_size=args.test_size,
    #     seed=args.seed,
    # )

    save_split(
        dataset=hf_dataloader.load_finance_agent_bench(args.finance_agent_bench_file),
        map_fn=make_ranking_map_fn(),
        train_path=os.path.join(
            args.output_dir, "finance_agent_bench", "train.parquet"
        ),
        val_path=os.path.join(
            args.output_dir, "finance_agent_bench", "validation.parquet"
        ),
        test_size=args.test_size,
        seed=args.seed,
    )
