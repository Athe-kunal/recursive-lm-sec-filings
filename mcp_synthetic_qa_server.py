"""MCP server for synthetic QA dataset building from local markdown filings."""

from __future__ import annotations

import dataclasses
import json
import random
import re
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger
from mcp.server.fastmcp import FastMCP

_ALLOWED_SEC_FILING_TYPES = {
    "10-K",
    "10-Q1",
    "10-Q2",
    "10-Q3",
    "8-K",
    "DEF 14A",
    "10-Q4",
}
_ALLOWED_TRANSCRIPT_FILING_TYPES = {"Q1", "Q2", "Q3", "Q4"}
_PAGE_SPLIT_PATTERN = re.compile(
    r"\n\s*(?:---\s*)?page\s+\d+\s*(?:---\s*)?\n", re.IGNORECASE
)
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")


@dataclasses.dataclass(slots=True)
class SyntheticQAExample:
    question: str
    answer: str
    context: str
    year: str
    ticker_or_company_name: str
    filing_type: str
    data_source: str
    task_type: str = "qa"


class _FilingRecord(NamedTuple):
    file_path: Path
    ticker_or_company_name: str
    year: str
    filing_type: str
    data_source: str


class _ContextSelection(NamedTuple):
    paragraph_context: str | None
    table_context: str | None


mcp = FastMCP("synthetic-qa-server")


def _normalize_whitespace(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized


def _extract_content_after_first_page(markdown_text: str) -> str:
    pages = re.split(_PAGE_SPLIT_PATTERN, markdown_text)
    if len(pages) <= 1:
        return markdown_text
    content = "\n".join(pages[1:])
    return content


def _extract_paragraphs(markdown_text: str) -> list[str]:
    raw_blocks = _PARAGRAPH_SPLIT_PATTERN.split(markdown_text)
    paragraphs: list[str] = []
    for block in raw_blocks:
        normalized = _normalize_whitespace(block)
        if len(normalized) < 120:
            continue
        if "|" in normalized:
            continue
        paragraphs.append(normalized)
    logger.info(f"{len(paragraphs)=}")
    return paragraphs


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    is_separator = bool(re.fullmatch(r"\|?\s*[:\-\| ]+\|?\s*", stripped))
    return is_separator


def _extract_markdown_tables(markdown_text: str) -> list[str]:
    lines = markdown_text.splitlines()
    candidate_tables: list[list[str]] = []
    current_table: list[str] = []

    for line in lines:
        stripped = line.strip()
        if "|" not in stripped:
            if current_table:
                candidate_tables.append(current_table)
                current_table = []
            continue
        current_table.append(stripped)

    if current_table:
        candidate_tables.append(current_table)

    markdown_tables: list[str] = []
    for table_lines in candidate_tables:
        if len(table_lines) < 2:
            continue
        if not any(_is_table_separator(line) for line in table_lines):
            continue
        markdown_table = "\n".join(table_lines)
        markdown_tables.append(markdown_table)

    logger.info(f"{len(markdown_tables)=}")
    return markdown_tables


def _parse_sec_record(file_path: Path) -> _FilingRecord | None:
    parent_name = file_path.parent.name
    if "-" not in parent_name:
        return None

    ticker, year = parent_name.rsplit("-", maxsplit=1)
    filing_type = file_path.stem
    if filing_type not in _ALLOWED_SEC_FILING_TYPES:
        return None

    record = _FilingRecord(
        file_path=file_path,
        ticker_or_company_name=ticker,
        year=year,
        filing_type=filing_type,
        data_source="sec_data_markdown",
    )
    return record


def _parse_transcript_record(file_path: Path) -> _FilingRecord | None:
    if len(file_path.parts) < 3:
        return None

    ticker = file_path.parents[1].name
    year = file_path.parent.name
    quarter = file_path.stem.split("_")[0]
    if quarter not in _ALLOWED_TRANSCRIPT_FILING_TYPES:
        return None

    record = _FilingRecord(
        file_path=file_path,
        ticker_or_company_name=ticker,
        year=year,
        filing_type=quarter,
        data_source="earnings_data_markdown",
    )
    return record


def _parse_record_from_file_path(file_path: Path) -> _FilingRecord | None:
    sec_record = _parse_sec_record(file_path)
    if sec_record is not None:
        return sec_record

    transcript_record = _parse_transcript_record(file_path)
    if transcript_record is not None:
        return transcript_record

    return None


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
    return records


def _resolve_earnings_root(preferred_root: Path) -> Path:
    if preferred_root.exists():
        return preferred_root
    fallback_root = Path("earnings_transcripts_data")
    return fallback_root


def _choose_random_contexts(markdown_text: str, random_seed: int | None) -> _ContextSelection:
    rng = random_seed if random_seed is not None else 0
    randomizer = random.Random(rng)

    content_after_page_one = _extract_content_after_first_page(markdown_text)
    paragraphs = _extract_paragraphs(content_after_page_one)
    markdown_tables = _extract_markdown_tables(content_after_page_one)

    paragraph_context = randomizer.choice(paragraphs) if paragraphs else None
    table_context = randomizer.choice(markdown_tables) if markdown_tables else None
    logger.info(f"{paragraph_context is not None=} {table_context is not None=}")

    selection = _ContextSelection(
        paragraph_context=paragraph_context,
        table_context=table_context,
    )
    return selection


def _build_example(
    record: _FilingRecord,
    question: str,
    answer: str,
    context: str,
    data_source: str,
) -> SyntheticQAExample:
    example = SyntheticQAExample(
        question=_normalize_whitespace(question),
        answer=_normalize_whitespace(answer),
        context=context,
        year=record.year,
        ticker_or_company_name=record.ticker_or_company_name,
        filing_type=record.filing_type,
        data_source=data_source,
    )
    return example


def _append_jsonl_record(output_path: Path, example: SyntheticQAExample) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_obj:
        serialized = json.dumps(dataclasses.asdict(example), ensure_ascii=False)
        file_obj.write(serialized + "\n")


@mcp.tool()
def list_filing_records(
    sec_root: str = "localworkspace/markdown/sec_data",
    earnings_root: str = "earnings_transcripts_data",
) -> list[dict[str, str]]:
    """List available filing markdown records with parsed metadata."""
    sec_root_path = Path(sec_root)
    earnings_root_path = _resolve_earnings_root(Path(earnings_root))

    records = _collect_records(sec_root=sec_root_path, earnings_root=earnings_root_path)
    serialized_records: list[dict[str, str]] = []
    for record in records:
        serialized_records.append(
            {
                "file_path": str(record.file_path),
                "ticker_or_company_name": record.ticker_or_company_name,
                "year": record.year,
                "filing_type": record.filing_type,
                "data_source": record.data_source,
            }
        )
    logger.info(f"{len(serialized_records)=}")
    return serialized_records


@mcp.tool()
def sample_contexts_for_filing(
    file_path: str,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Pick one random paragraph and one random markdown table after page 1."""
    path = Path(file_path)
    record = _parse_record_from_file_path(path)
    if record is None:
        raise ValueError(f"unsupported markdown file path: {file_path}")

    markdown_text = path.read_text(encoding="utf-8", errors="ignore")
    context_selection = _choose_random_contexts(
        markdown_text=markdown_text,
        random_seed=random_seed,
    )

    response = {
        "file_path": str(path),
        "ticker_or_company_name": record.ticker_or_company_name,
        "year": record.year,
        "filing_type": record.filing_type,
        "paragraph_context": context_selection.paragraph_context,
        "table_context": context_selection.table_context,
    }
    logger.info(f"{path=} {random_seed=}")
    return response


@mcp.tool()
def write_synthetic_qa_record(
    output_jsonl_path: str,
    file_path: str,
    context: str,
    question: str,
    answer: str,
    data_source: str = "mcp_client_generated",
) -> dict[str, str]:
    """Write one QA record using question/answer returned by an MCP client."""
    path = Path(file_path)
    record = _parse_record_from_file_path(path)
    if record is None:
        raise ValueError(f"unsupported markdown file path: {file_path}")

    example = _build_example(
        record=record,
        question=question,
        answer=answer,
        context=context,
        data_source=data_source,
    )

    output_path = Path(output_jsonl_path)
    _append_jsonl_record(output_path=output_path, example=example)

    result = {
        "output_jsonl_path": str(output_path),
        "ticker_or_company_name": example.ticker_or_company_name,
        "year": example.year,
        "filing_type": example.filing_type,
    }
    logger.info(f"{output_path=} {path=} {data_source=}")
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
