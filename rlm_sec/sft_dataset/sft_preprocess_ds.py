"""Load/subsample SFT parquet data and optionally generate cached rollouts."""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import Counter, defaultdict

from loguru import logger

from datasets import Dataset, concatenate_datasets, load_dataset

from rlm_sec.sft_dataset.rollout_generation import (
    RolloutSummary,
    build_smoke_dataset,
    generate_and_cache_rollouts_async,
)

_DEFAULT_TRAIN_PATH = "data/train.parquet"
_DEFAULT_VALIDATION_PATH = "data/validation.parquet"
_DEFAULT_TRAIN_QA_N = 1396
_DEFAULT_TRAIN_RANKING_N = 1002
_QA_TASK = "qa"
_RANKING_TASK = "ranking"


def _row_is_qa(example: dict) -> bool:
    return example["task_type"] == _QA_TASK


def _row_is_ranking(example: dict) -> bool:
    return example["task_type"] == _RANKING_TASK


def _normalize_qa_filing_type_row(example: dict) -> dict:
    """Maps legacy ``10K`` labels to ``10-K`` for QA rows (tool schema alignment)."""
    filing_type = str(example.get("filing_type", "")).strip()
    if filing_type.upper() == "10K":
        return {**example, "filing_type": "10-K"}
    return example


def _indices_by_filing_type(qa_dataset: Dataset) -> dict[str, list[int]]:
    """Groups QA row indices by ``filing_type`` after any normalization."""
    filing_types = qa_dataset["filing_type"]
    indices_by_type: dict[str, list[int]] = defaultdict(list)
    for row_index, filing_type in enumerate(filing_types):
        key = str(filing_type).strip()
        indices_by_type[key].append(row_index)
    return dict(indices_by_type)


def _stratified_qa_row_indices(
    indices_by_type: dict[str, list[int]],
    train_qa_n: int,
    seed: int,
) -> list[int]:
    """Picks ``train_qa_n`` QA indices with near-equal counts per ``filing_type``.

    Every distinct ``filing_type`` present in ``indices_by_type`` receives at least
    one selected row when ``train_qa_n`` is at least the number of types. Shortfalls
    from sparse types are filled from remaining rows in deterministic round-robin
    order until ``train_qa_n`` rows are chosen or the pool is exhausted.
    """
    rng = random.Random(seed)
    types_sorted = sorted(indices_by_type.keys())
    num_types = len(types_sorted)
    if num_types == 0:
        raise ValueError("No QA rows with filing_type after normalization.")
    if train_qa_n < num_types:
        raise ValueError(
            f"{train_qa_n=} must be >= {num_types=} (distinct filing_type values) "
            "so each filing_type can appear at least once in the QA sample."
        )

    base_quota = train_qa_n // num_types
    remainder = train_qa_n % num_types
    shuffled_types = types_sorted[:]
    rng.shuffle(shuffled_types)
    quota_by_type = {t: base_quota for t in types_sorted}
    for filing_type in shuffled_types[:remainder]:
        quota_by_type[filing_type] += 1

    selected: list[int] = []
    leftovers_by_type: dict[str, list[int]] = {}
    for filing_type in types_sorted:
        pool = indices_by_type[filing_type][:]
        rng.shuffle(pool)
        take_n = min(quota_by_type[filing_type], len(pool))
        selected.extend(pool[:take_n])
        leftovers_by_type[filing_type] = pool[take_n:]

    shortfall = train_qa_n - len(selected)
    cycle_order = shuffled_types[:]
    while shortfall > 0:
        progressed = False
        for filing_type in cycle_order:
            if shortfall <= 0:
                break
            bucket = leftovers_by_type[filing_type]
            if not bucket:
                continue
            selected.append(bucket.pop())
            shortfall -= 1
            progressed = True
        if not progressed:
            break

    if len(selected) != train_qa_n:
        raise ValueError(
            f"Stratified QA sampling failed: need {train_qa_n=} rows, "
            f"selected {len(selected)=}. Check QA pool size and quotas."
        )
    rng.shuffle(selected)
    return selected


def load_sft_dataset(
    train_path: str = _DEFAULT_TRAIN_PATH,
    seed: int = 46,
    train_qa_n: int = _DEFAULT_TRAIN_QA_N,
    train_ranking_n: int = _DEFAULT_TRAIN_RANKING_N,
) -> Dataset:
    """Load parquet train data and build a subsampled train dataset.

    QA rows: ``filing_type`` ``10K`` is normalized to ``10-K``, then ``train_qa_n``
    rows are chosen with stratified sampling so every ``filing_type`` in the QA
    pool is represented and counts per type are as equal as the row counts allow.

    Ranking rows: filtered, shuffled with ``seed + 1``, and the first
    ``train_ranking_n`` rows are taken.

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

    qa_train = full_train.filter(_row_is_qa).map(_normalize_qa_filing_type_row)
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

    indices_by_type = _indices_by_filing_type(qa_train)
    pool_sizes_by_type = {k: len(v) for k, v in indices_by_type.items()}
    logger.info(f"{sorted(indices_by_type.keys())=} {pool_sizes_by_type=}")
    qa_indices = _stratified_qa_row_indices(
        indices_by_type=indices_by_type,
        train_qa_n=train_qa_n,
        seed=seed,
    )
    qa_sample = qa_train.select(qa_indices)
    filing_type_column = qa_train["filing_type"]
    per_type_selected = Counter(
        str(filing_type_column[i]).strip() for i in qa_indices
    )
    logger.info(f"{dict(per_type_selected)=}")

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
        build_smoke_dataset(train_dataset, seed=seed) if smoke_test else train_dataset
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
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--train_qa_n", type=int, default=_DEFAULT_TRAIN_QA_N)
    parser.add_argument("--train_ranking_n", type=int, default=_DEFAULT_TRAIN_RANKING_N)
    parser.add_argument("--rollout_output_jsonl_path", default="sft_data_full.jsonl")
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
