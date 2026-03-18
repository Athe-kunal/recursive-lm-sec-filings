import argparse
import os

from loguru import logger

from rlm_sec.trainer import hf_dataloader


def make_map_fn():
    def process_fn(example: dict, idx: int) -> dict:
        return {
            "data_source": example["data_source"],
            "prompt": [
                {"role": "user", "content": example["question"]},
            ],
            "env_class": "null",
            "answer": example["answer"],
            "context": example["context"],
            "year": example["year"],
            "ticker_or_company_name": example["ticker_or_company_name"],
            "filing_type": example["filing_type"],
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    combined_qa = hf_dataloader.load_combined_qa()
    split_data = combined_qa.train_test_split(test_size=args.test_size, seed=args.seed)
    train_dataset, val_dataset = split_data["train"], split_data["test"]
    logger.info(f"Split: train={len(train_dataset)}, validation={len(val_dataset)}")

    train_dataset = train_dataset.map(
        function=make_map_fn(),
        with_indices=True,
    )
    val_dataset = val_dataset.map(
        function=make_map_fn(),
        with_indices=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
