"""Build, submit, and parse OpenAI Batch API jobs for SEC / earnings QA.

Each line is one POST ``/v1/chat/completions`` request with system and user
messages containing sampled paragraph and table contexts. The model returns
structured QA pairs (JSON schema + Pydantic validation). Batch output is
turned into trainer ``QAExample`` JSONL rows.

CLI entrypoint uses Python Fire: ``build-requests``, ``build-dataset``,
``submit-batch`` (see ``python openai_batch_synthetic_qa.py --help``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import re
from pathlib import Path
from typing import Any, NamedTuple
import dotenv
import fire
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
import requests
from tqdm.auto import tqdm

dotenv.load_dotenv(".env")

from rlm_sec.trainer.hf_dataloader import (
    QAExample,
    build_qa_prompt as _trainer_build_qa_prompt,
)

_ALLOWED_SEC_FILING_TYPES = {
    "10-K",
    "10-Q1",
    "10-Q2",
    "10-Q3",
    "10-Q4",
    "8-K",
    "DEF 14A",
}
_ALLOWED_TRANSCRIPT_FILING_TYPES = {"Q1", "Q2", "Q3", "Q4"}
_DEFAULT_MODEL = "gpt-4.1-mini"
_DEFAULT_BATCH_INPUT_PATH = "openai_batch_synthetic_qa_requests.jsonl"
_DEFAULT_BATCH_OUTPUT_PATH = "openai_batch_synthetic_qa_output.jsonl"
_DEFAULT_DATASET_OUTPUT_PATH = "synthetic_qa_dataset_from_batch.jsonl"
_DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
_DEFAULT_COMPLETION_WINDOW = "24h"
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
_DEFAULT_MAX_PARAGRAPH_CONTEXTS_PER_FILING = 2
_DEFAULT_MAX_TABLE_CONTEXTS_PER_FILING = 2
_SMOKE_TEST_REQUEST_LIMIT = 2
_FINANCIAL_SCALE_CONTEXT_PATTERN = re.compile(
    r"%|\b(?:millions?|billions?)\b", re.IGNORECASE
)
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:\$?\d[\d,]*(?:\.\d+)?%?)(?![A-Za-z])"
)
_PAGE_SPLIT_PATTERN = re.compile(
    r"\n\s*(?:---\s*)?page\s+\d+\s*(?:---\s*)?\n", re.IGNORECASE
)
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")
_TABLE_BLOCK_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_ROW_PATTERN = re.compile(r"<tr[\s\S]*?</tr>", re.IGNORECASE)
_CELL_PATTERN = re.compile(r"<(?:td|th)[^>]*>([\s\S]*?)</(?:td|th)>", re.IGNORECASE)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_FINANCIAL_SCALE_CONTEXT_PATTERN = re.compile(
    r"%|\b(?:millions?|billions?)\b", re.IGNORECASE
)
_TOC_ITEM_MARKER_PATTERN = re.compile(r"Item\s+\d+[A-Z]{0,2}\.", re.IGNORECASE)

SEC_ROOT = "localworkspace/markdown/sec_data"
EARNINGS_ROOT = "earnings_transcripts_data"
RANDOM_SEED = 42
OVERWRITE_OUTPUT = True

_CONTEXT_HEADER = "Contexts:"


class BatchSyntheticQAPair(BaseModel):
    """One question/answer pair from the model (matches batch JSON schema)."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str


class BatchSyntheticQAResponse(BaseModel):
    """Root object returned under structured output for each batch line."""

    model_config = ConfigDict(extra="forbid")

    qas: list[BatchSyntheticQAPair] = Field(min_length=1, max_length=3)


def _batch_response_json_schema() -> dict[str, Any]:
    """JSON Schema for OpenAI ``json_schema`` format (strict, no ``$ref``)."""

    return {
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
    }


class _RequestPayload(NamedTuple):
    custom_id: str
    body: dict[str, Any]


class _ParsedCustomID(NamedTuple):
    ticker: str
    year: str
    filing_type: str


class _ParsedBatchRow(NamedTuple):
    examples: list[QAExample]
    custom_id: str
    qa_count: int


class _UploadedFile(NamedTuple):
    file_id: str
    response_json: dict[str, Any]


class _CreatedBatch(NamedTuple):
    batch_id: str
    response_json: dict[str, Any]


class _FilingRecord(NamedTuple):
    """One batch row after grouping. SEC rows use a single path; earnings may merge
    multiple transcript files for the same ticker/year/quarter (unique ``custom_id``).
    """

    file_paths: tuple[Path, ...]
    ticker_or_company_name: str
    year: str
    filing_type: str
    data_source: str


def _build_system_prompt() -> str:
    return (
        "Your task is to generate synthetic financial question-answer pairs from SEC filings "
        "and earnings transcripts. Given the provided text, formulate 1 to 3 "
        "questions and answers that are directly supported by the context. Look "
        "at the provided metadata such as company identifier, reporting year, and "
        "document type, and use it as subtle context so the questions feel "
        "grounded in the document, but do not overemphasize or restate the hint "
        "unless it is natural. Questions can be numeric or briefly descriptive. "
        "Use only facts in the provided contexts. Keep numbers exact when used. "
        "Do not invent values. Return only JSON matching the schema."
    )


def _document_type_hint(filing_type: str) -> str:
    subtle_descriptions = {
        "10-K": "an annual filing",
        "10-Q1": "an early-quarter filing",
        "10-Q2": "a mid-year quarter filing",
        "10-Q3": "a late-year quarter filing",
        "10-Q4": "a year-end quarter filing",
        "8-K": "a current-event filing",
        "DEF 14A": "a proxy filing",
        "Q1": "a first-quarter earnings call transcript",
        "Q2": "a second-quarter earnings call transcript",
        "Q3": "a third-quarter earnings call transcript",
        "Q4": "a fourth-quarter earnings call transcript",
    }
    return subtle_descriptions.get(filing_type, "a company financial document")


def _build_user_prompt(
    record: _FilingRecord,
    paragraph_contexts: list[str],
    table_contexts: list[str],
) -> str:
    document_type_hint = _document_type_hint(filing_type=record.filing_type)
    lines: list[str] = [
        "Generate 1 to 3 question-answer pairs from the contexts below.",
        (
            "Task: given the text, formulate questions and answers that are "
            "directly supported by the provided context."
        ),
        (
            f"Company identifier: {record.ticker_or_company_name}. "
            f"Reporting year: {record.year}. "
            f"Document type: {record.filing_type} ({document_type_hint})."
        ),
        (
            "Each question must be answerable directly from the contexts and "
            "may focus on numeric facts, non-numeric facts, or short descriptive facts."
        ),
        "Keep the wording natural, concise, and specific.",
        "",
        _CONTEXT_HEADER,
        "",
    ]
    if paragraph_contexts:
        lines.append("## Paragraphs")
        for index, paragraph in enumerate(paragraph_contexts, start=1):
            lines.extend([f"### Paragraph {index}", paragraph, ""])
    if table_contexts:
        lines.append("## Tables")
        for index, table in enumerate(table_contexts, start=1):
            lines.extend([f"### Table {index}", table, ""])
    return "\n".join(lines).rstrip() + "\n"


def _build_custom_id(record: _FilingRecord) -> str:
    return (
        f"ticker={record.ticker_or_company_name}|year={record.year}|"
        f"filing_type={record.filing_type}"
    )


def _has_financial_scale_context(text: str) -> bool:
    return _FINANCIAL_SCALE_CONTEXT_PATTERN.search(text) is not None


def _is_toc_style_markdown_table(table: str) -> bool:
    item_markers = _TOC_ITEM_MARKER_PATTERN.findall(table)
    if len(item_markers) >= 3:
        return True
    lines = table.split("\n")
    if not lines:
        return False
    first_line_lower = lines[0].lower()
    if "page" in first_line_lower and "part " in first_line_lower:
        return True
    return False


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    no_tags = re.sub(_TAG_PATTERN, " ", text)
    return _normalize_whitespace(no_tags)


def _extract_content_after_first_page(markdown_text: str) -> str:
    pages = re.split(_PAGE_SPLIT_PATTERN, markdown_text)
    if len(pages) <= 1:
        return markdown_text
    return "\n".join(pages[1:])


def _extract_paragraphs(markdown_text: str) -> list[str]:
    blocks = _PARAGRAPH_SPLIT_PATTERN.split(markdown_text)
    paragraphs: list[str] = []
    for block in blocks:
        cleaned = _strip_html(block)
        if len(cleaned) < 120:
            continue
        if "table" in block.lower():
            continue
        paragraphs.append(cleaned)
    return paragraphs


def _to_markdown_row(cells: list[str]) -> str:
    escaped_cells = [cell.replace("|", "\\|") for cell in cells]
    return "| " + " | ".join(escaped_cells) + " |"


def _html_table_to_markdown(table_html: str) -> str | None:
    row_matches = re.findall(_ROW_PATTERN, table_html)
    rows: list[list[str]] = []
    for row_html in row_matches:
        cell_matches = re.findall(_CELL_PATTERN, row_html)
        normalized_cells = [_strip_html(cell) for cell in cell_matches]
        normalized_cells = [cell for cell in normalized_cells if cell]
        if not normalized_cells:
            continue
        rows.append(normalized_cells)

    if len(rows) < 2:
        return None

    header = rows[0]
    markdown_lines = [_to_markdown_row(header)]
    markdown_lines.append(_to_markdown_row(["---"] * len(header)))
    for row in rows[1:]:
        padded_row = row + [""] * max(0, len(header) - len(row))
        markdown_lines.append(_to_markdown_row(padded_row[: len(header)]))
    return "\n".join(markdown_lines)


def _extract_markdown_tables(markdown_text: str) -> list[str]:
    html_tables = re.findall(_TABLE_BLOCK_PATTERN, markdown_text)
    markdown_tables: list[str] = []
    for html_table in html_tables:
        markdown_table = _html_table_to_markdown(html_table)
        if markdown_table is None:
            continue
        markdown_tables.append(markdown_table)
    return markdown_tables


def _has_numeric_tokens(text: str) -> bool:
    return _NUMERIC_TOKEN_PATTERN.search(text) is not None


def _is_numeric_text(text: str) -> bool:
    return _has_numeric_tokens(text) or _has_financial_scale_context(text)


def _sample_contexts(
    contexts: list[str],
    max_contexts: int,
    randomizer: random.Random,
) -> list[str]:
    if not contexts:
        return []
    if len(contexts) <= max_contexts:
        return contexts
    return randomizer.sample(contexts, k=max_contexts)


def _extract_numeric_contexts(markdown_text: str) -> tuple[list[str], list[str]]:
    """All numeric paragraph/table contexts in one markdown document (no sampling)."""
    content = _extract_content_after_first_page(markdown_text)

    paragraphs = _extract_paragraphs(content)
    numeric_paragraphs = [p for p in paragraphs if _is_numeric_text(p)]

    tables = _extract_markdown_tables(content)
    non_toc_tables = [t for t in tables if not _is_toc_style_markdown_table(t)]
    numeric_tables = [t for t in non_toc_tables if _is_numeric_text(t)]

    return numeric_paragraphs, numeric_tables


def _gather_numeric_contexts_for_record(
    record: _FilingRecord,
    randomizer: random.Random,
    max_paragraph_contexts: int,
    max_table_contexts: int,
) -> tuple[list[str], list[str]]:
    """Pool numeric contexts from every file in ``record.file_paths``, then sample."""
    all_paragraphs: list[str] = []
    all_tables: list[str] = []
    for path in record.file_paths:
        markdown_text = path.read_text(encoding="utf-8", errors="ignore")
        paras, tabs = _extract_numeric_contexts(markdown_text=markdown_text)
        all_paragraphs.extend(paras)
        all_tables.extend(tabs)
    sampled_paragraphs = _sample_contexts(
        all_paragraphs, max_paragraph_contexts, randomizer
    )
    sampled_tables = _sample_contexts(all_tables, max_table_contexts, randomizer)
    return sampled_paragraphs, sampled_tables


def _parse_sec_record(file_path: Path) -> _FilingRecord | None:
    parent_name = file_path.parent.name
    if "-" not in parent_name:
        return None
    ticker, year = parent_name.rsplit("-", maxsplit=1)
    filing_type = file_path.stem
    if filing_type not in _ALLOWED_SEC_FILING_TYPES:
        return None
    return _FilingRecord(
        file_paths=(file_path,),
        ticker_or_company_name=ticker,
        year=year,
        filing_type=filing_type,
        data_source="generated_sec_data_markdown",
    )


def _parse_transcript_record(file_path: Path) -> _FilingRecord | None:
    if len(file_path.parts) < 3:
        return None
    ticker = file_path.parents[1].name
    year = file_path.parent.name
    quarter = file_path.stem.split("_")[0]
    if quarter not in _ALLOWED_TRANSCRIPT_FILING_TYPES:
        return None
    return _FilingRecord(
        file_paths=(file_path,),
        ticker_or_company_name=ticker,
        year=year,
        filing_type=quarter,
        data_source="generated_earnings_data_markdown",
    )


def _merge_earnings_transcript_records_by_quarter(
    records: list[_FilingRecord],
) -> list[_FilingRecord]:
    """Collapse multiple earnings transcript files per (ticker, year, quarter) so
    ``custom_id`` is unique and contexts are pooled from every file in the group.
    """
    sec_records: list[_FilingRecord] = []
    earnings_by_key: dict[tuple[str, str, str], list[Path]] = {}
    for record in records:
        if record.data_source != "generated_earnings_data_markdown":
            sec_records.append(record)
            continue
        key = (
            record.ticker_or_company_name,
            record.year,
            record.filing_type,
        )
        earnings_by_key.setdefault(key, []).extend(record.file_paths)

    merged_earnings: list[_FilingRecord] = []
    for key in sorted(earnings_by_key.keys(), key=lambda k: (k[0], k[1], k[2])):
        paths = earnings_by_key[key]
        unique_paths = tuple(sorted(set(paths), key=lambda p: str(p)))
        ticker, year, filing_type = key
        if len(unique_paths) > 1:
            logger.info(
                f"merged earnings transcripts for {ticker=} {year=} {filing_type=} "
                f"{len(unique_paths)=} {unique_paths=}"
            )
        merged_earnings.append(
            _FilingRecord(
                file_paths=unique_paths,
                ticker_or_company_name=ticker,
                year=year,
                filing_type=filing_type,
                data_source="generated_earnings_data_markdown",
            )
        )
    merged = sec_records + merged_earnings
    logger.info(
        f"{len(records)=} after_earnings_merge={len(merged)} "
        f"{len(sec_records)=} {len(merged_earnings)=}"
    )
    return merged


def _collect_records(sec_root: Path, earnings_root: Path) -> list[_FilingRecord]:
    records: list[_FilingRecord] = []
    for file_path in sorted(sec_root.glob("*/*.md")):
        record = _parse_sec_record(file_path)
        if record is not None:
            records.append(record)
    for file_path in sorted(earnings_root.glob("*/*/Q*.md")):
        record = _parse_transcript_record(file_path)
        if record is not None:
            records.append(record)
    logger.info(f"{sec_root=} {earnings_root=} {len(records)=}")
    records = _merge_earnings_transcript_records_by_quarter(records=records)
    return records


def _build_request_payload(
    record: _FilingRecord,
    paragraph_contexts: list[str],
    table_contexts: list[str],
    model: str,
) -> _RequestPayload:
    custom_id = _build_custom_id(record=record)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _build_system_prompt(),
            },
            {
                "role": "user",
                "content": _build_user_prompt(
                    record=record,
                    paragraph_contexts=paragraph_contexts,
                    table_contexts=table_contexts,
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_qa_pairs",
                "schema": _batch_response_json_schema(),
                "strict": True,
            },
        },
    }
    return _RequestPayload(custom_id=custom_id, body=body)


def _request_row(payload: _RequestPayload) -> dict[str, Any]:
    return {
        "custom_id": payload.custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": payload.body,
    }


def _read_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Expected OPENAI_API_KEY to be set in the environment.")
    return api_key


def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def _validate_input_jsonl_path(input_jsonl_path: Path) -> None:
    if not input_jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL file does not exist: {input_jsonl_path}")
    if not input_jsonl_path.is_file():
        raise ValueError(f"Expected a file path, but got: {input_jsonl_path}")
    if input_jsonl_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a .jsonl file, but got: {input_jsonl_path}")


def _parse_metadata_json(metadata_json: str) -> dict[str, str] | None:
    if not metadata_json.strip():
        return None

    parsed = json.loads(metadata_json)
    if not isinstance(parsed, dict):
        raise ValueError("Expected --metadata-json to decode to a JSON object.")

    normalized_metadata: dict[str, str] = {}
    for key, value in parsed.items():
        normalized_metadata[str(key)] = str(value)
    return normalized_metadata


def _upload_batch_input_file(
    input_jsonl_path: Path,
    api_base_url: str,
    headers: dict[str, str],
) -> _UploadedFile:
    upload_url = f"{api_base_url}/files"
    with input_jsonl_path.open("rb") as file_obj:
        response = requests.post(
            upload_url,
            headers=headers,
            data={"purpose": "batch"},
            files={"file": (input_jsonl_path.name, file_obj, "application/jsonl")},
            timeout=_DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
    response.raise_for_status()

    response_json = response.json()
    file_id = str(response_json["id"])
    logger.info(f"{input_jsonl_path=} {file_id=}")
    return _UploadedFile(file_id=file_id, response_json=response_json)


def _build_batch_create_payload(
    uploaded_file_id: str,
    endpoint: str,
    completion_window: str,
    metadata: dict[str, str] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_file_id": uploaded_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def _create_batch_job(
    uploaded_file_id: str,
    api_base_url: str,
    headers: dict[str, str],
    endpoint: str,
    completion_window: str,
    metadata: dict[str, str] | None,
) -> _CreatedBatch:
    batch_url = f"{api_base_url}/batches"
    payload = _build_batch_create_payload(
        uploaded_file_id=uploaded_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata,
    )
    response = requests.post(
        batch_url,
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=_DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    response_json = response.json()
    batch_id = str(response_json["id"])
    status = str(response_json.get("status", "unknown"))
    logger.info(f"{batch_id=} {status=} {endpoint=} {completion_window=}")
    return _CreatedBatch(batch_id=batch_id, response_json=response_json)


def _write_batch_response_json(
    output_json_path: Path | None,
    batch_response_json: dict[str, Any],
) -> None:
    if output_json_path is None:
        return

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(batch_response_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"{output_json_path=}")


def _write_jsonl_lines(
    output_path: Path, lines: list[str], overwrite_output: bool
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite_output else "a"
    with output_path.open(mode, encoding="utf-8") as file_obj:
        for line in lines:
            file_obj.write(line + "\n")


def _order_records_for_request_build(
    records: list[_FilingRecord],
    smoke_test: bool,
    randomizer: random.Random,
) -> list[_FilingRecord]:
    ordered_records = list(records)
    if smoke_test:
        randomizer.shuffle(ordered_records)
    return ordered_records


def _has_reached_smoke_test_limit(smoke_test: bool, request_count: int) -> bool:
    return smoke_test and request_count >= _SMOKE_TEST_REQUEST_LIMIT


def build_batch_requests(
    sec_root: Path,
    earnings_root: Path,
    output_jsonl_path: Path,
    model: str,
    overwrite_output: bool,
    max_paragraph_contexts_per_filing: int,
    max_table_contexts_per_filing: int,
    smoke_test: bool,
) -> int:
    randomizer = random.Random(RANDOM_SEED)
    records = _collect_records(sec_root=sec_root, earnings_root=earnings_root)
    ordered_records = _order_records_for_request_build(
        records=records,
        smoke_test=smoke_test,
        randomizer=randomizer,
    )

    serialized_rows: list[str] = []
    with tqdm(
        ordered_records,
        desc="Building batch requests",
        unit="filing",
    ) as progress:
        for record in progress:
            paragraph_contexts, table_contexts = _gather_numeric_contexts_for_record(
                record=record,
                randomizer=randomizer,
                max_paragraph_contexts=max_paragraph_contexts_per_filing,
                max_table_contexts=max_table_contexts_per_filing,
            )
            if not paragraph_contexts and not table_contexts:
                tqdm.write(
                    f"skip {record.file_paths=} "
                    f"(no numeric paragraph or table contexts)"
                )
                progress.set_postfix_str("skipped", refresh=True)
                continue

            payload = _build_request_payload(
                record=record,
                paragraph_contexts=paragraph_contexts,
                table_contexts=table_contexts,
                model=model,
            )
            serialized_rows.append(
                json.dumps(_request_row(payload), ensure_ascii=False)
            )
            file_label = (
                record.file_paths[0].name
                if len(record.file_paths) == 1
                else f"{len(record.file_paths)}_md_files"
            )
            progress.set_postfix(
                p=len(paragraph_contexts),
                t=len(table_contexts),
                file=file_label,
                refresh=True,
            )
            if _has_reached_smoke_test_limit(
                smoke_test=smoke_test,
                request_count=len(serialized_rows),
            ):
                break

    _write_jsonl_lines(
        output_path=output_jsonl_path,
        lines=serialized_rows,
        overwrite_output=overwrite_output,
    )
    request_count = len(serialized_rows)
    logger.info(f"{request_count=} {output_jsonl_path=}")
    return request_count


def submit_batch_job(
    input_jsonl_path: Path,
    endpoint: str = _DEFAULT_BATCH_ENDPOINT,
    completion_window: str = _DEFAULT_COMPLETION_WINDOW,
    api_base_url: str = _DEFAULT_API_BASE_URL,
    metadata_json: str = "",
    output_json_path: Path | None = None,
) -> _CreatedBatch:
    _validate_input_jsonl_path(input_jsonl_path=input_jsonl_path)

    normalized_api_base_url = api_base_url.rstrip("/")
    api_key = _read_api_key()
    headers = _build_headers(api_key=api_key)
    metadata = _parse_metadata_json(metadata_json=metadata_json)

    uploaded_file = _upload_batch_input_file(
        input_jsonl_path=input_jsonl_path,
        api_base_url=normalized_api_base_url,
        headers=headers,
    )
    created_batch = _create_batch_job(
        uploaded_file_id=uploaded_file.file_id,
        api_base_url=normalized_api_base_url,
        headers=headers,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata,
    )
    _write_batch_response_json(
        output_json_path=output_json_path,
        batch_response_json=created_batch.response_json,
    )
    return created_batch


def _extract_response_text(row: dict[str, Any]) -> str:
    response = row.get("response", {})
    body = response.get("body", {})

    choices = body.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str):
                    return text_value

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

    return _ParsedCustomID(
        ticker=values["ticker"],
        year=values["year"],
        filing_type=values["filing_type"],
    )


def _parse_qas_from_response_text(text: str) -> list[BatchSyntheticQAPair]:
    parsed = BatchSyntheticQAResponse.model_validate_json(text)
    return list(parsed.qas)


def _extract_context_from_request_body(request_body: dict[str, Any]) -> str:
    messages = request_body.get("messages", [])
    if len(messages) <= 1:
        return ""

    user_message = messages[1]
    prompt_content = user_message.get("content", "")
    if isinstance(prompt_content, list):
        if not prompt_content:
            return ""
        prompt_text = str(prompt_content[0].get("text", ""))
    else:
        prompt_text = str(prompt_content)

    if not prompt_text:
        return ""
    marker_index = prompt_text.find(_CONTEXT_HEADER)
    if marker_index < 0:
        return prompt_text.strip()

    after_header = prompt_text[marker_index + len(_CONTEXT_HEADER) :].lstrip("\n")
    return after_header.strip()


def _build_trainer_qa_example(
    qa: BatchSyntheticQAPair,
    context: str,
    meta: _ParsedCustomID,
) -> QAExample:
    question = _normalize_whitespace(qa.question)
    answer = _normalize_whitespace(qa.answer)
    return QAExample(
        prompt=_trainer_build_qa_prompt(question=question),
        answer=answer,
        context=context,
        year=meta.year,
        ticker_or_company_name=meta.ticker,
        filing_type=meta.filing_type,
        data_source="generated_openai_batch_synthetic_qa",
        task_type="qa",
    )


def _build_examples_from_batch_row(row: dict[str, Any]) -> _ParsedBatchRow:
    custom_id = str(row["custom_id"])
    meta = _parse_custom_id(custom_id=custom_id)
    response_text = _extract_response_text(row=row)
    qas = _parse_qas_from_response_text(text=response_text)

    request_body = row.get("request", {}).get("body", {})
    context = _extract_context_from_request_body(request_body=request_body)

    examples: list[QAExample] = []
    for qa in qas:
        examples.append(_build_trainer_qa_example(qa=qa, context=context, meta=meta))

    return _ParsedBatchRow(
        examples=examples,
        custom_id=custom_id,
        qa_count=len(examples),
    )


def build_qa_examples_from_batch_output(
    batch_output_jsonl_path: Path,
    output_dataset_jsonl_path: Path,
    overwrite_output: bool,
) -> list[QAExample]:
    dataset_rows: list[str] = []
    qa_examples: list[QAExample] = []
    with batch_output_jsonl_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            parsed = _build_examples_from_batch_row(row=row)
            logger.info(f"{parsed.custom_id=} {parsed.qa_count=}")
            for example in parsed.examples:
                qa_examples.append(example)
                dataset_rows.append(
                    json.dumps(dataclasses.asdict(example), ensure_ascii=False)
                )

    _write_jsonl_lines(
        output_path=output_dataset_jsonl_path,
        lines=dataset_rows,
        overwrite_output=overwrite_output,
    )
    logger.info(f"{len(qa_examples)=} {output_dataset_jsonl_path=}")
    return qa_examples


def build_dataset_from_batch_output(
    batch_output_jsonl_path: Path,
    output_dataset_jsonl_path: Path,
    overwrite_output: bool,
) -> int:
    qa_examples = build_qa_examples_from_batch_output(
        batch_output_jsonl_path=batch_output_jsonl_path,
        output_dataset_jsonl_path=output_dataset_jsonl_path,
        overwrite_output=overwrite_output,
    )
    return len(qa_examples)


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"Expected positive integer for {name}, but got {value}.")


def _cli_build_requests(
    sec_root: Path = Path(SEC_ROOT),
    earnings_root: Path = Path(EARNINGS_ROOT),
    output_jsonl_path: Path = Path(_DEFAULT_BATCH_INPUT_PATH),
    model: str = _DEFAULT_MODEL,
    overwrite_output: bool = OVERWRITE_OUTPUT,
    max_paragraph_contexts_per_filing: int = _DEFAULT_MAX_PARAGRAPH_CONTEXTS_PER_FILING,
    max_table_contexts_per_filing: int = _DEFAULT_MAX_TABLE_CONTEXTS_PER_FILING,
    smoke_test: bool = False,
) -> int:
    """Create JSONL input for OpenAI Batch API."""
    _validate_positive_int(
        name="max_paragraph_contexts_per_filing",
        value=max_paragraph_contexts_per_filing,
    )
    _validate_positive_int(
        name="max_table_contexts_per_filing",
        value=max_table_contexts_per_filing,
    )
    return build_batch_requests(
        sec_root=sec_root,
        earnings_root=earnings_root,
        output_jsonl_path=output_jsonl_path,
        model=model,
        overwrite_output=overwrite_output,
        max_paragraph_contexts_per_filing=max_paragraph_contexts_per_filing,
        max_table_contexts_per_filing=max_table_contexts_per_filing,
        smoke_test=smoke_test,
    )


def _cli_build_dataset(
    batch_output_jsonl_path: str,
    output_dataset_jsonl_path: Path = Path(_DEFAULT_DATASET_OUTPUT_PATH),
    overwrite_output: bool = OVERWRITE_OUTPUT,
) -> int:
    """Convert OpenAI Batch output JSONL into synthetic QA dataset JSONL."""
    return build_dataset_from_batch_output(
        batch_output_jsonl_path=Path(batch_output_jsonl_path),
        output_dataset_jsonl_path=output_dataset_jsonl_path,
        overwrite_output=overwrite_output,
    )


def _cli_submit_batch(
    input_jsonl_path: str,
    endpoint: str = _DEFAULT_BATCH_ENDPOINT,
    completion_window: str = _DEFAULT_COMPLETION_WINDOW,
    api_base_url: str = _DEFAULT_API_BASE_URL,
    metadata_json: str = "",
    output_json_path: Path | None = None,
) -> str:
    """Upload a JSONL file and create an OpenAI Batch API job; returns batch id."""
    created_batch = submit_batch_job(
        input_jsonl_path=Path(input_jsonl_path),
        endpoint=endpoint,
        completion_window=completion_window,
        api_base_url=api_base_url,
        metadata_json=metadata_json,
        output_json_path=output_json_path,
    )
    return created_batch.batch_id


def main() -> None:
    fire.Fire(
        {
            "build-requests": _cli_build_requests,
            "build-dataset": _cli_build_dataset,
            "submit-batch": _cli_submit_batch,
        },
        name="openai_batch_synthetic_qa",
    )


if __name__ == "__main__":
    main()
