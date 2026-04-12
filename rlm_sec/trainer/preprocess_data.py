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
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--finance_agent_bench_file",
        type=str,
        default="document_ranking_kaggle_dev.jsonl",
    )
    parser.add_argument(
        "--rephrased_qa_jsonl",
        type=str,
        default="data/rephrased_out.jsonl",
        help="JSONL of rephrased QA rows (cast to QAExample) for merge mode.",
    )
    parser.add_argument(
        "--merge_rephrased_with_parquet_splits",
        action="store_true",
        help=(
            "If set, load rephrased JSONL + a fraction of existing train.parquet + full "
            "validation.parquet from output_dir, concatenate, then train_test_split."
        ),
    )
    parser.add_argument(
        "--train_subsample_fraction",
        type=float,
        default=0.5,
        help="Fraction of existing train.parquet rows to include when merging (default half).",
    )

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")

    if args.merge_rephrased_with_parquet_splits:
        merged = hf_dataloader.build_merged_dataset_from_rephrased_and_parquet(
            rephrased_jsonl=args.rephrased_qa_jsonl,
            train_parquet=train_path,
            validation_parquet=val_path,
            train_subsample_fraction=args.train_subsample_fraction,
            seed=args.seed,
        )
        logger.info(
            f"Merged rephrased + train fraction + val: {len(merged)=} "
            f"(rephrased jsonl={args.rephrased_qa_jsonl})"
        )
        save_split(
            dataset=merged,
            train_path=train_path,
            val_path=val_path,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        combined = build_combined_dataset(args.finance_agent_bench_file)
        save_split(
            dataset=combined,
            train_path=train_path,
            val_path=val_path,
            test_size=args.test_size,
            seed=args.seed,
        )
