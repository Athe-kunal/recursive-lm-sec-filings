"""Build OpenAI Batch API JSONL requests for numeric synthetic SEC QA generation.

This script builds OpenAI Batch API request JSONL that asks the model to produce
structured QA pairs from numeric paragraph and table contexts extracted from SEC
filings and earnings transcripts. It can also parse batch output JSONL and
convert it into a training-ready QA dataset JSONL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger

from mcp_synthetic_qa_server import (
    EARNINGS_ROOT,
    MAX_QUESTIONS_PER_FILING_TYPE,
    MIN_QUESTIONS_PER_FILING_TYPE,
    OVERWRITE_OUTPUT,
    RANDOM_SEED,
    SEC_ROOT,
    _FilingRecord,
    _collect_records,
    _extract_content_after_first_page,
    _extract_markdown_tables,
    _extract_paragraphs,
    _has_financial_scale_context,
    _is_toc_style_markdown_table,
    _normalize_whitespace,
)

_DEFAULT_MODEL = "gpt-4.1-mini"
_DEFAULT_BATCH_INPUT_PATH = "openai_batch_synthetic_qa_requests.jsonl"
_DEFAULT_BATCH_OUTPUT_PATH = "openai_batch_synthetic_qa_output.jsonl"
_DEFAULT_DATASET_OUTPUT_PATH = "synthetic_qa_dataset_from_batch.jsonl"
_DEFAULT_MAX_PARAGRAPH_CONTEXTS_PER_FILING = 2
_DEFAULT_MAX_TABLE_CONTEXTS_PER_FILING = 2

_NUMERIC_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z])(?:\$?\d[\d,]*(?:\.\d+)?%?)(?![A-Za-z])")

_SYNTHETIC_QA_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "synthetic_qa_pairs",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "qas": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["question", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["qas"],
        "additionalProperties": False,
    },
}


@dataclasses.dataclass(slots=True)
class SyntheticQAExample:
    question: str
    answer: str
    context: str
    context_type: str
    year: str
    ticker_or_company_name: str
    filing_type: str
    data_source: str
    task_type: str = "qa"


class _RequestPayload(NamedTuple):
    custom_id: str
    body: dict[str, Any]


class _ParsedCustomID(NamedTuple):
    ticker: str
    year: str
    filing_type: str
    context_type: str


class _ParsedBatchRow(NamedTuple):
    examples: list[SyntheticQAExample]
    custom_id: str
    qa_count: int


class _NumericContextSelection(NamedTuple):
    paragraph_contexts: list[str]
    table_contexts: list[str]


def _build_system_prompt() -> str:
    return (
        "You generate synthetic financial QA pairs from SEC filings and earnings "
        "transcripts. Use only facts in the provided context. Keep numbers exact. "
        "Do not invent values. Return only JSON matching the schema."
    )


def _build_user_prompt(record: _FilingRecord, context: str, context_type: str) -> str:
    prompt = (
        "Generate 1 to 3 QA pairs from this context.\n"
        f"ticker_or_company_name={record.ticker_or_company_name}\n"
        f"year={record.year}\n"
        f"filing_type={record.filing_type}\n"
        f"context_type={context_type}\n"
        "Each question should be answerable directly from the context and should "
        "focus on numeric facts.\n"
        "Context:\n"
        f"{context}"
    )
    return prompt


def _build_custom_id(record: _FilingRecord, context_type: str, sample_idx: int) -> str:
    return (
        f"ticker={record.ticker_or_company_name}|year={record.year}|"
        f"filing_type={record.filing_type}|context_type={context_type}|sample_idx={sample_idx}"
    )


def _build_request_payload(
    record: _FilingRecord,
    context: str,
    context_type: str,
    sample_idx: int,
    model: str,
) -> _RequestPayload:
    custom_id = _build_custom_id(
        record=record,
        context_type=context_type,
        sample_idx=sample_idx,
    )
    body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": _build_system_prompt()}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _build_user_prompt(
                            record=record,
                            context=context,
                            context_type=context_type,
                        ),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": _SYNTHETIC_QA_RESPONSE_SCHEMA["name"],
                "schema": _SYNTHETIC_QA_RESPONSE_SCHEMA["schema"],
                "strict": _SYNTHETIC_QA_RESPONSE_SCHEMA["strict"],
            }
        },
    }
    payload = _RequestPayload(custom_id=custom_id, body=body)
    return payload


def _request_row(payload: _RequestPayload) -> dict[str, Any]:
    row = {
        "custom_id": payload.custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": payload.body,
    }
    return row


def _serialize_jsonl_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False)


def _write_jsonl_lines(output_path: Path, lines: list[str], overwrite_output: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite_output else "a"
    with output_path.open(mode, encoding="utf-8") as file_obj:
        for line in lines:
            file_obj.write(line + "\n")


def _has_numeric_tokens(text: str) -> bool:
    return _NUMERIC_TOKEN_PATTERN.search(text) is not None


def _is_numeric_paragraph(paragraph: str) -> bool:
    has_numeric_tokens = _has_numeric_tokens(paragraph)
    has_scale_terms = _has_financial_scale_context(paragraph)
    return has_numeric_tokens or has_scale_terms


def _is_numeric_table(markdown_table: str) -> bool:
    has_numeric_tokens = _has_numeric_tokens(markdown_table)
    has_scale_terms = _has_financial_scale_context(markdown_table)
    return has_numeric_tokens or has_scale_terms


def _sample_contexts(
    contexts: list[str],
    max_contexts: int,
    randomizer: random.Random,
) -> list[str]:
    if not contexts:
        return []
    if len(contexts) <= max_contexts:
        return contexts
    sampled_contexts = randomizer.sample(contexts, k=max_contexts)
    return sampled_contexts


def _select_numeric_contexts(
    markdown_text: str,
    randomizer: random.Random,
    max_paragraph_contexts: int,
    max_table_contexts: int,
) -> _NumericContextSelection:
    content = _extract_content_after_first_page(markdown_text)

    paragraphs = _extract_paragraphs(content)
    numeric_paragraphs = [
        paragraph for paragraph in paragraphs if _is_numeric_paragraph(paragraph)
    ]

    tables = _extract_markdown_tables(content)
    non_toc_tables = [table for table in tables if not _is_toc_style_markdown_table(table)]
    numeric_tables = [table for table in non_toc_tables if _is_numeric_table(table)]

    sampled_paragraphs = _sample_contexts(
        contexts=numeric_paragraphs,
        max_contexts=max_paragraph_contexts,
        randomizer=randomizer,
    )
    sampled_tables = _sample_contexts(
        contexts=numeric_tables,
        max_contexts=max_table_contexts,
        randomizer=randomizer,
    )

    logger.info(
        f"{len(paragraphs)=} {len(numeric_paragraphs)=} "
        f"{len(tables)=} {len(non_toc_tables)=} {len(numeric_tables)=} "
        f"{len(sampled_paragraphs)=} {len(sampled_tables)=}"
    )

    selection = _NumericContextSelection(
        paragraph_contexts=sampled_paragraphs,
        table_contexts=sampled_tables,
    )
    return selection


def _build_request_rows_for_contexts(
    record: _FilingRecord,
    contexts: list[str],
    context_type: str,
    question_count: int,
    model: str,
) -> list[str]:
    serialized_rows: list[str] = []
    for context_index, context in enumerate(contexts):
        for sample_idx in range(question_count):
            payload = _build_request_payload(
                record=record,
                context=context,
                context_type=context_type,
                sample_idx=(context_index * question_count) + sample_idx,
                model=model,
            )
            row = _request_row(payload=payload)
            serialized_rows.append(_serialize_jsonl_row(row))
    return serialized_rows


def build_batch_requests(
    sec_root: Path,
    earnings_root: Path,
    output_jsonl_path: Path,
    model: str,
    overwrite_output: bool,
    max_paragraph_contexts_per_filing: int,
    max_table_contexts_per_filing: int,
) -> int:
    randomizer = random.Random(RANDOM_SEED)
    records = _collect_records(sec_root=sec_root, earnings_root=earnings_root)

    serialized_rows: list[str] = []
    for record in records:
        markdown_text = record.file_path.read_text(encoding="utf-8", errors="ignore")
        context_selection = _select_numeric_contexts(
            markdown_text=markdown_text,
            randomizer=randomizer,
            max_paragraph_contexts=max_paragraph_contexts_per_filing,
            max_table_contexts=max_table_contexts_per_filing,
        )
        question_count = randomizer.randint(
            MIN_QUESTIONS_PER_FILING_TYPE,
            MAX_QUESTIONS_PER_FILING_TYPE,
        )

        logger.info(
            f"{record.file_path=} {question_count=} "
            f"{len(context_selection.paragraph_contexts)=} "
            f"{len(context_selection.table_contexts)=}"
        )

        paragraph_rows = _build_request_rows_for_contexts(
            record=record,
            contexts=context_selection.paragraph_contexts,
            context_type="paragraph",
            question_count=question_count,
            model=model,
        )
        table_rows = _build_request_rows_for_contexts(
            record=record,
            contexts=context_selection.table_contexts,
            context_type="table",
            question_count=question_count,
            model=model,
        )
        serialized_rows.extend(paragraph_rows)
        serialized_rows.extend(table_rows)

    _write_jsonl_lines(
        output_path=output_jsonl_path,
        lines=serialized_rows,
        overwrite_output=overwrite_output,
    )
    request_count = len(serialized_rows)
    logger.info(f"{request_count=} {output_jsonl_path=}")
    return request_count


def _extract_response_text(row: dict[str, Any]) -> str:
    response = row.get("response", {})
    body = response.get("body", {})

    output_text = body.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output_items = body.get("output", [])
    for item in output_items:
        contents = item.get("content", [])
        for content in contents:
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value
    raise ValueError(f"Missing response text for row: {row.get('custom_id')}")


def _parse_custom_id(custom_id: str) -> _ParsedCustomID:
    values: dict[str, str] = {}
    for pair in custom_id.split("|"):
        if "=" not in pair:
            continue
        key, value = pair.split("=", maxsplit=1)
        values[key] = value

    parsed = _ParsedCustomID(
        ticker=values["ticker"],
        year=values["year"],
        filing_type=values["filing_type"],
        context_type=values["context_type"],
    )
    return parsed


def _parse_qas_from_response_text(text: str) -> list[dict[str, str]]:
    data = json.loads(text)
    qas = data["qas"]
    if not isinstance(qas, list):
        raise ValueError("The model response 'qas' field must be a list.")
    return qas


def _extract_context_from_request_body(request_body: dict[str, Any]) -> str:
    user_input = request_body.get("input", [])
    if len(user_input) <= 1:
        return ""

    content_items = user_input[1].get("content", [])
    if not content_items:
        return ""

    prompt_text = str(content_items[0].get("text", ""))
    marker = "Context:\n"
    marker_index = prompt_text.find(marker)
    if marker_index < 0:
        return prompt_text

    context = prompt_text[marker_index + len(marker) :]
    return context.strip()


def _build_examples_from_batch_row(row: dict[str, Any]) -> _ParsedBatchRow:
    custom_id = str(row["custom_id"])
    custom_id_fields = _parse_custom_id(custom_id=custom_id)
    response_text = _extract_response_text(row=row)
    qas = _parse_qas_from_response_text(text=response_text)

    request_body = row.get("request", {}).get("body", {})
    context = _extract_context_from_request_body(request_body=request_body)

    examples: list[SyntheticQAExample] = []
    for qa in qas:
        question = _normalize_whitespace(str(qa["question"]))
        answer = _normalize_whitespace(str(qa["answer"]))
        example = SyntheticQAExample(
            question=question,
            answer=answer,
            context=context,
            context_type=custom_id_fields.context_type,
            year=custom_id_fields.year,
            ticker_or_company_name=custom_id_fields.ticker,
            filing_type=custom_id_fields.filing_type,
            data_source="openai_batch_generated",
        )
        examples.append(example)

    parsed_batch_row = _ParsedBatchRow(
        examples=examples,
        custom_id=custom_id,
        qa_count=len(examples),
    )
    return parsed_batch_row


def build_dataset_from_batch_output(
    batch_output_jsonl_path: Path,
    output_dataset_jsonl_path: Path,
    overwrite_output: bool,
) -> int:
    dataset_rows: list[str] = []
    with batch_output_jsonl_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            parsed = _build_examples_from_batch_row(row=row)
            logger.info(f"{parsed.custom_id=} {parsed.qa_count=}")
            for example in parsed.examples:
                dataset_rows.append(
                    json.dumps(dataclasses.asdict(example), ensure_ascii=False)
                )

    _write_jsonl_lines(
        output_path=output_dataset_jsonl_path,
        lines=dataset_rows,
        overwrite_output=overwrite_output,
    )
    example_count = len(dataset_rows)
    logger.info(f"{example_count=} {output_dataset_jsonl_path=}")
    return example_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OpenAI Batch API JSONL requests for synthetic numeric SEC QA, "
            "or parse batch output into dataset JSONL."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_requests_parser = subparsers.add_parser(
        "build-requests",
        help="Create JSONL input for OpenAI Batch API.",
    )
    build_requests_parser.add_argument(
        "--sec-root",
        type=Path,
        default=Path(SEC_ROOT),
    )
    build_requests_parser.add_argument(
        "--earnings-root",
        type=Path,
        default=Path(EARNINGS_ROOT),
    )
    build_requests_parser.add_argument(
        "--output-jsonl-path",
        type=Path,
        default=Path(_DEFAULT_BATCH_INPUT_PATH),
    )
    build_requests_parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
    )
    build_requests_parser.add_argument(
        "--max-paragraph-contexts-per-filing",
        type=int,
        default=_DEFAULT_MAX_PARAGRAPH_CONTEXTS_PER_FILING,
    )
    build_requests_parser.add_argument(
        "--max-table-contexts-per-filing",
        type=int,
        default=_DEFAULT_MAX_TABLE_CONTEXTS_PER_FILING,
    )
    build_requests_parser.add_argument(
        "--overwrite-output",
        action="store_true",
        default=OVERWRITE_OUTPUT,
    )

    parse_output_parser = subparsers.add_parser(
        "build-dataset",
        help="Convert downloaded OpenAI Batch output JSONL into synthetic QA dataset JSONL.",
    )
    parse_output_parser.add_argument(
        "--batch-output-jsonl-path",
        type=Path,
        default=Path(_DEFAULT_BATCH_OUTPUT_PATH),
    )
    parse_output_parser.add_argument(
        "--output-dataset-jsonl-path",
        type=Path,
        default=Path(_DEFAULT_DATASET_OUTPUT_PATH),
    )
    parse_output_parser.add_argument(
        "--overwrite-output",
        action="store_true",
        default=OVERWRITE_OUTPUT,
    )

    args = parser.parse_args()
    return args


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"Expected positive integer for {name}, but got {value}.")


def main() -> None:
    args = _parse_args()
    logger.info(f"{args=}")

    if args.command == "build-requests":
        _validate_positive_int(
            name="max_paragraph_contexts_per_filing",
            value=args.max_paragraph_contexts_per_filing,
        )
        _validate_positive_int(
            name="max_table_contexts_per_filing",
            value=args.max_table_contexts_per_filing,
        )
        build_batch_requests(
            sec_root=args.sec_root,
            earnings_root=args.earnings_root,
            output_jsonl_path=args.output_jsonl_path,
            model=args.model,
            overwrite_output=args.overwrite_output,
            max_paragraph_contexts_per_filing=args.max_paragraph_contexts_per_filing,
            max_table_contexts_per_filing=args.max_table_contexts_per_filing,
        )
        return

    if args.command == "build-dataset":
        build_dataset_from_batch_output(
            batch_output_jsonl_path=args.batch_output_jsonl_path,
            output_dataset_jsonl_path=args.output_dataset_jsonl_path,
            overwrite_output=args.overwrite_output,
        )
        return

    raise ValueError(f"Unknown {args.command=}")


if __name__ == "__main__":
    main()
