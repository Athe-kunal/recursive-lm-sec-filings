"""Load and subsample SFT parquet splits for supervised fine-tuning."""

import logging

from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

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
    """Load parquet splits and build subsampled train SFT data plus full validation.

    The full train parquet is filtered by ``task_type``, shuffled with ``seed``,
    and the first ``train_qa_n`` / ``train_ranking_n`` rows are taken per task.
    The validation split is returned in full (all QA and ranking rows).

    Args:
        train_path: Path to the training parquet file.
        seed: RNG seed for reproducible subsampling and shuffles.
        train_qa_n: Number of QA rows to keep from the train split.
        train_ranking_n: Number of ranking rows to keep from the train split.

    Returns:
        ``train_sft_dataset`` — HuggingFace ``Dataset``
        instances; the train set is QA and ranking subsets concatenated and shuffled.
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
