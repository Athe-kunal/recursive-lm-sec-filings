"""Streamlit UI to browse train/validation parquet rows for manual QA validation."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import streamlit as st
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "train.parquet"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "validation.parquet"

DATASET_CACHE_KEY = "dataset_validator_datasets"
# Logical row position (not used as a widget ``key`` — avoids Streamlit
# "cannot be modified after the widget ... is instantiated" on nav buttons).
ROW_INDEX_STATE_KEY_TEMPLATE = "dataset_validator_row_index_{split}"
FILTER_SIGNATURE_KEY_TEMPLATE = "dataset_validator_filter_sig_{split}"

# User templates end with "Question: " then the task question (see hf_dataloader.py).
QUESTION_SUFFIX_PATTERN = re.compile(r"Question:\s*(.*)", re.DOTALL)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


def load_split_parquet(path: Path) -> Dataset:
    if not path.is_file():
        logger.error(f"{path=} not found")
        raise FileNotFoundError(f"Parquet not found: {path}")
    dataset = load_dataset("parquet", data_files=str(path))["train"]
    logger.info(f"{path=} {dataset.num_rows=}")
    return dataset


def get_datasets(train_path: Path, val_path: Path) -> tuple[Dataset, Dataset]:
    train_ds = load_split_parquet(train_path)
    val_ds = load_split_parquet(val_path)
    return train_ds, val_ds


def prompt_messages_to_markdown(messages: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        blocks.append(f"**{role}**\n\n{content}")
    return "\n\n---\n\n".join(blocks)


def user_prompt_text(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    if messages:
        return str(messages[-1].get("content", ""))
    return ""


def text_after_question_marker(user_content: str) -> str | None:
    """Return text after the first ``Question:`` match, or None if absent."""
    match = QUESTION_SUFFIX_PATTERN.search(user_content)
    if not match:
        return None
    return match.group(1).strip()


def format_list_cell(values: object) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        if not values:
            return "_(empty)_"
        return "\n".join(f"- {item}" for item in values)
    return str(values)


def render_example_row(row: dict) -> None:
    st.subheader("Metadata")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Year", row.get("year") or "—")
    meta_cols[1].metric("Ticker / company", row.get("ticker_or_company_name") or "—")
    meta_cols[2].metric("Filing type", row.get("filing_type") or "—")
    meta_cols[3].metric("Task type", row.get("task_type") or "—")

    st.caption(
        f"data_source={row.get('data_source')} | env_class={row.get('env_class')}"
    )

    st.subheader("Question")
    user_text = user_prompt_text(row.get("prompt") or [])
    extracted = text_after_question_marker(user_text) if user_text else None
    if extracted is None:
        st.markdown("_No `Question:` marker found in the user message._")
    elif not extracted:
        st.markdown("_Matched `Question:` but the captured text is empty._")
    else:
        st.markdown(extracted)

    with st.expander("Full prompt (all messages)", expanded=False):
        st.markdown(prompt_messages_to_markdown(row.get("prompt") or []))

    st.subheader("Answer")
    st.markdown(row.get("answer") or "_Empty_")

    st.subheader("Document ranking labels")
    rank_cols = st.columns(2)
    with rank_cols[0]:
        st.markdown("**Relevant**")
        st.markdown(format_list_cell(row.get("relevant")))
    with rank_cols[1]:
        st.markdown("**Not relevant**")
        st.markdown(format_list_cell(row.get("not_relevant")))

    ctx = row.get("context") or ""
    if ctx:
        with st.expander("Context (training field)", expanded=False):
            st.text_area(
                "context",
                value=ctx,
                height=240,
                disabled=True,
                label_visibility="collapsed",
            )


def clamp_index(value: int, max_index: int) -> int:
    if max_index < 0:
        return 0
    if value < 0:
        return 0
    if value > max_index:
        return max_index
    return value


def indices_for_task_filter(dataset: Dataset, task_types: list[str]) -> list[int]:
    labels = dataset["task_type"]
    selected = set(task_types)
    return [i for i, label in enumerate(labels) if label in selected]


def sync_cursor_after_filter_change(
    split: str,
    filter_signature: str,
    max_cursor: int,
) -> None:
    index_key = ROW_INDEX_STATE_KEY_TEMPLATE.format(split=split)
    sig_key = FILTER_SIGNATURE_KEY_TEMPLATE.format(split=split)
    prior_sig = st.session_state.get(sig_key)
    if prior_sig != filter_signature:
        st.session_state[index_key] = 0
        st.session_state[sig_key] = filter_signature
    st.session_state[index_key] = clamp_index(
        int(st.session_state.get(index_key, 0)),
        max_cursor,
    )


def resolve_row_index(
    dataset: Dataset,
    cursor: int,
    filtered_indices: list[int] | None,
) -> int:
    if filtered_indices is None:
        return cursor
    return int(filtered_indices[cursor])


def main() -> None:
    _configure_logging()
    st.set_page_config(
        page_title="Parquet dataset validator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Train / validation parquet viewer")

    with st.sidebar:
        st.header("Data paths")
        train_path = Path(st.text_input("Train parquet", value=str(DEFAULT_TRAIN_PATH)))
        val_path = Path(
            st.text_input("Validation parquet", value=str(DEFAULT_VAL_PATH))
        )
        if st.button("Reload parquet files"):
            if DATASET_CACHE_KEY in st.session_state:
                del st.session_state[DATASET_CACHE_KEY]
            st.session_state.pop("_dataset_validator_paths", None)
            st.rerun()

        split = st.radio("Split", options=["train", "validation"], horizontal=True)

    cache_key = (
        str(train_path.resolve()),
        str(val_path.resolve()),
    )
    if (
        DATASET_CACHE_KEY not in st.session_state
        or st.session_state.get("_dataset_validator_paths") != cache_key
    ):
        train_ds, val_ds = get_datasets(train_path, val_path)
        st.session_state[DATASET_CACHE_KEY] = {"train": train_ds, "validation": val_ds}
        st.session_state["_dataset_validator_paths"] = cache_key

    datasets = st.session_state[DATASET_CACHE_KEY]
    active_ds: Dataset = datasets[split]
    num_rows = int(active_ds.num_rows)

    task_type_values = sorted(
        {str(x) for x in active_ds.unique("task_type") if x is not None}
    )
    filter_task = st.sidebar.multiselect(
        "Filter task_type (optional)",
        options=task_type_values,
        default=[],
    )

    filtered_indices: list[int] | None = None
    if filter_task:
        filtered_indices = indices_for_task_filter(active_ds, filter_task)
        if not filtered_indices:
            st.warning("No rows match the selected task_type filters.")
            st.stop()

    if filtered_indices is None:
        max_cursor = max(0, num_rows - 1)
        filter_signature = f"{split}:all"
    else:
        max_cursor = max(0, len(filtered_indices) - 1)
        filter_signature = f"{split}:{'|'.join(sorted(filter_task))}"

    sync_cursor_after_filter_change(split, filter_signature, max_cursor)

    index_key = ROW_INDEX_STATE_KEY_TEMPLATE.format(split=split)
    position_for_slider = int(st.session_state[index_key])

    st.sidebar.metric("Rows in split", num_rows)
    if filtered_indices is not None:
        st.sidebar.metric("Rows after filter", len(filtered_indices))

    slider_value = st.sidebar.slider(
        "Position in current view",
        min_value=0,
        max_value=max_cursor,
        value=position_for_slider,
        step=1,
    )
    st.session_state[index_key] = int(slider_value)
    position_after_slider = int(st.session_state[index_key])

    nav_cols = st.columns([1, 1, 1, 6])
    if nav_cols[0].button("⏮ First"):
        st.session_state[index_key] = 0
        st.rerun()
    if nav_cols[1].button("◀ Previous"):
        st.session_state[index_key] = clamp_index(
            position_after_slider - 1, max_cursor
        )
        st.rerun()
    if nav_cols[2].button("Next ▶"):
        st.session_state[index_key] = clamp_index(
            position_after_slider + 1, max_cursor
        )
        st.rerun()

    current_cursor = int(st.session_state[index_key])
    row_index = resolve_row_index(active_ds, current_cursor, filtered_indices)
    row = active_ds[row_index]

    if filtered_indices is None:
        st.caption(f"Dataset row index: {row_index} / {max(0, num_rows - 1)}")
    else:
        st.caption(
            f"Filtered position: {current_cursor} / {max_cursor} — "
            f"dataset row index: {row_index}"
        )

    raw_json = json.dumps(row, ensure_ascii=False, indent=2, default=str)
    with st.expander("Raw row JSON"):
        st.code(raw_json, language="json")

    render_example_row(row)


if __name__ == "__main__":
    main()
