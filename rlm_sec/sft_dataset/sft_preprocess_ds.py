"""Load/subsample SFT parquet data and optionally generate cached rollouts."""

from __future__ import annotations

import argparse
import asyncio
from loguru import logger

from datasets import Dataset, concatenate_datasets, load_dataset

from rlm_sec.sft_dataset.rollout_generation import (
    RolloutSummary,
    build_smoke_dataset,
    generate_and_cache_rollouts_async,
)

_DEFAULT_TRAIN_PATH = "data/train.parquet"
_DEFAULT_TRAIN_QA_N = 1396
_DEFAULT_TRAIN_RANKING_N = 1002
_QA_TASK = "qa"
_RANKING_TASK = "ranking"


def _row_is_qa(example: dict) -> bool:
    return example["task_type"] == _QA_TASK


def _row_is_ranking(example: dict) -> bool:
    return example["task_type"] == _RANKING_TASK


def load_sft_dataset(
    train_path: str = _DEFAULT_TRAIN_PATH,
    seed: int = 42,
    train_qa_n: int = _DEFAULT_TRAIN_QA_N,
    train_ranking_n: int = _DEFAULT_TRAIN_RANKING_N,
) -> Dataset:
    """Load parquet train data and build a subsampled train dataset.

    The full train parquet is filtered by ``task_type``, shuffled with ``seed``,
    and the first ``train_qa_n`` / ``train_ranking_n`` rows are taken per task.

    Args:
        train_path: Path to the training parquet file.
        seed: RNG seed for reproducible subsampling and shuffles.
        train_qa_n: Number of QA rows to keep from the train split.
        train_ranking_n: Number of ranking rows to keep from the train split.

    Returns:
        The train SFT dataset with sampled QA and ranking rows.
    """
    logger.info(f"{train_path=}, {seed=}, {train_qa_n=}, {train_ranking_n=}")

    full_train = load_dataset("parquet", data_files=train_path)["train"]

    qa_train = full_train.filter(_row_is_qa)
    ranking_train = full_train.filter(_row_is_ranking)

    logger.info(f"{len(qa_train)=}, {len(ranking_train)=}")

    if len(qa_train) < train_qa_n:
        raise ValueError(
            f"Not enough QA rows in train split: need {train_qa_n}, have {len(qa_train)}"
        )
    if len(ranking_train) < train_ranking_n:
        raise ValueError(
            "Not enough ranking rows in train split: need "
            f"{train_ranking_n}, have {len(ranking_train)}"
        )

    qa_sample = qa_train.shuffle(seed=seed).select(range(train_qa_n))
    ranking_sample = ranking_train.shuffle(seed=seed + 1).select(range(train_ranking_n))

    train_sft_dataset = concatenate_datasets([qa_sample, ranking_sample]).shuffle(
        seed=seed + 2
    )

    logger.info(f"{len(train_sft_dataset)=} (expected {train_qa_n + train_ranking_n})")

    return train_sft_dataset


async def build_train_dataset_with_rollouts_async(
    train_path: str,
    rollout_output_jsonl_path: str,
    model: str,
    seed: int = 42,
    train_qa_n: int = _DEFAULT_TRAIN_QA_N,
    train_ranking_n: int = _DEFAULT_TRAIN_RANKING_N,
    rollout_temperature: float = 1.0,
    continuation_temperature: float = 0.7,
    n: int = 1,
    smoke_test: bool = False,
) -> RolloutSummary:
    """Builds sampled data and writes async model rollouts to cached JSONL."""
    train_dataset = load_sft_dataset(
        train_path=train_path,
        seed=seed,
        train_qa_n=train_qa_n,
        train_ranking_n=train_ranking_n,
    )
    dataset_for_rollout = (
        build_smoke_dataset(train_dataset) if smoke_test else train_dataset
    )
    logger.info(
        f"starting rollout generation. {rollout_output_jsonl_path=} {smoke_test=} "
        f"{len(dataset_for_rollout)=}"
    )
    return await generate_and_cache_rollouts_async(
        dataset=dataset_for_rollout,
        output_jsonl_path=rollout_output_jsonl_path,
        model=model,
        rollout_temperature=rollout_temperature,
        continuation_temperature=continuation_temperature,
        n=n,
    )


def parse_args() -> argparse.Namespace:
    """Parses CLI args for dataset loading and rollout generation."""
    parser = argparse.ArgumentParser(
        description="Build SFT train dataset and optionally generate rollout JSONL."
    )
    parser.add_argument("--train_path", default=_DEFAULT_TRAIN_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_qa_n", type=int, default=_DEFAULT_TRAIN_QA_N)
    parser.add_argument("--train_ranking_n", type=int, default=_DEFAULT_TRAIN_RANKING_N)
    parser.add_argument("--rollout_output_jsonl_path", default="sft_data_n4.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--rollout_temperature", type=float, default=1.0)
    parser.add_argument("--continuation_temperature", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    """Runs CLI entrypoint for loading data and optional rollout generation."""
    args = parse_args()
    logger.info(f"{args=}")

    if not args.rollout_output_jsonl_path:
        dataset = load_sft_dataset(
            train_path=args.train_path,
            seed=args.seed,
            train_qa_n=args.train_qa_n,
            train_ranking_n=args.train_ranking_n,
        )
        logger.info(f"dataset loaded without rollout generation. {len(dataset)=}")
        return

    summary = asyncio.run(
        build_train_dataset_with_rollouts_async(
            train_path=args.train_path,
            rollout_output_jsonl_path=args.rollout_output_jsonl_path,
            model=args.model,
            seed=args.seed,
            train_qa_n=args.train_qa_n,
            train_ranking_n=args.train_ranking_n,
            rollout_temperature=args.rollout_temperature,
            continuation_temperature=args.continuation_temperature,
            n=args.n,
            smoke_test=args.smoke_test,
        )
    )
    logger.info(f"rollout generation finished. {summary=}")


if __name__ == "__main__":
    main()
