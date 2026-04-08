"""MCP server for synthetic QA dataset building from local markdown filings."""

from __future__ import annotations

import dataclasses
import json
import random
import re
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger
from mcp.server.fastmcp import Context, FastMCP

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
_PAGE_SPLIT_PATTERN = re.compile(
    r"\n\s*(?:---\s*)?page\s+\d+\s*(?:---\s*)?\n", re.IGNORECASE
)
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")
_TABLE_BLOCK_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_ROW_PATTERN = re.compile(r"<tr[\s\S]*?</tr>", re.IGNORECASE)
_CELL_PATTERN = re.compile(r"<(?:td|th)[^>]*>([\s\S]*?)</(?:td|th)>", re.IGNORECASE)
_TAG_PATTERN = re.compile(r"<[^>]+>")
# Percent or spelled-out large scale (e.g. "in millions", "$ billions").
_FINANCIAL_SCALE_CONTEXT_PATTERN = re.compile(
    r"%|\b(?:millions?|billions?)\b", re.IGNORECASE
)
# SEC index-of-contents style tables (Item 1., Part I | Page, ...).
_TOC_ITEM_MARKER_PATTERN = re.compile(r"Item\s+\d+[A-Z]{0,2}\.", re.IGNORECASE)


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
    contexts: list[str]
    paragraph_context: str | None
    table_context: str | None


mcp = FastMCP("synthetic-qa-server")


def _normalize_whitespace(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized


def _strip_html(text: str) -> str:
    no_tags = re.sub(_TAG_PATTERN, " ", text)
    clean = _normalize_whitespace(no_tags)
    return clean


def _extract_content_after_first_page(markdown_text: str) -> str:
    pages = re.split(_PAGE_SPLIT_PATTERN, markdown_text)
    if len(pages) <= 1:
        return markdown_text
    content = "\n".join(pages[1:])
    return content


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
    logger.info(f"{len(paragraphs)=}")
    return paragraphs


def _to_markdown_row(cells: list[str]) -> str:
    escaped_cells = [cell.replace("|", "\\|") for cell in cells]
    row = "| " + " | ".join(escaped_cells) + " |"
    return row


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
    markdown_table = "\n".join(markdown_lines)
    return markdown_table


def _extract_markdown_tables(markdown_text: str) -> list[str]:
    html_tables = re.findall(_TABLE_BLOCK_PATTERN, markdown_text)
    markdown_tables: list[str] = []
    for html_table in html_tables:
        markdown_table = _html_table_to_markdown(html_table)
        if markdown_table is None:
            continue
        markdown_tables.append(markdown_table)
    logger.info(f"{len(markdown_tables)=}")
    return markdown_tables


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


def _choose_numeric_preferred_paragraph(
    paragraphs: list[str],
    randomizer: random.Random,
) -> str | None:
    financial_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if _has_financial_scale_context(paragraph)
    ]
    logger.info(f"{len(financial_paragraphs)=}")
    if financial_paragraphs:
        return randomizer.choice(financial_paragraphs)
    if not paragraphs:
        return None
    return randomizer.choice(paragraphs)


def _choose_numeric_preferred_table(
    markdown_tables: list[str],
    randomizer: random.Random,
) -> str | None:
    non_toc_tables = [
        table
        for table in markdown_tables
        if not _is_toc_style_markdown_table(table)
    ]
    financial_tables = [
        table
        for table in non_toc_tables
        if _has_financial_scale_context(table)
    ]
    logger.info(
        f"{len(financial_tables)=} {len(non_toc_tables)=} {len(markdown_tables)=}"
    )
    if financial_tables:
        return randomizer.choice(financial_tables)
    if non_toc_tables:
        return randomizer.choice(non_toc_tables)
    return None


def _choose_contexts(
    markdown_text: str, randomizer: random.Random
) -> _ContextSelection:
    content = _extract_content_after_first_page(markdown_text)
    paragraphs = _extract_paragraphs(content)
    markdown_tables = _extract_markdown_tables(content)
    paragraph_context = _choose_numeric_preferred_paragraph(paragraphs, randomizer)
    table_context = _choose_numeric_preferred_table(markdown_tables, randomizer)

    contexts: list[str] = []
    if paragraph_context is not None:
        contexts.append(paragraph_context)
    if table_context is not None:
        contexts.append(table_context)
    logger.info(
        f"{len(contexts)=} {paragraph_context is not None=} {table_context is not None=}"
    )

    selection = _ContextSelection(
        contexts=contexts,
        paragraph_context=paragraph_context,
        table_context=table_context,
    )
    return selection


def _extract_text_from_sample_response(sample_response: Any) -> str:
    if isinstance(sample_response, str):
        return sample_response
    text = getattr(sample_response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(sample_response, "content", None)
    if isinstance(content, str):
        return content
    serialized = json.dumps(sample_response, default=str)
    return serialized


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        parsed = json.loads(stripped)
        return parsed
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match is None:
        raise ValueError(f"Could not parse JSON object from client response: {text}")
    parsed = json.loads(match.group(0))
    return parsed


async def _request_qa_from_mcp_client(
    ctx: Context,
    record: _FilingRecord,
    context: str,
) -> tuple[str, str]:
    prompt = (
        "Generate exactly one QA pair as strict JSON with keys question and answer. "
        "Use only the provided context.\n"
        f"ticker_or_company_name={record.ticker_or_company_name}\n"
        f"year={record.year}\n"
        f"filing_type={record.filing_type}\n"
        f"context:\n{context}"
    )

    sample_response = await ctx.sample(prompt)
    raw_text = _extract_text_from_sample_response(sample_response)
    parsed = _extract_json_object(raw_text)
    question = _normalize_whitespace(str(parsed["question"]))
    answer = _normalize_whitespace(str(parsed["answer"]))
    logger.info(f"{len(question)=} {len(answer)=}")
    return question, answer


def _build_example(
    record: _FilingRecord,
    question: str,
    answer: str,
    context: str,
) -> SyntheticQAExample:
    example = SyntheticQAExample(
        question=question,
        answer=answer,
        context=context,
        year=record.year,
        ticker_or_company_name=record.ticker_or_company_name,
        filing_type=record.filing_type,
        data_source="mcp_client_generated",
    )
    return example


def _prepare_output_file(output_path: Path, overwrite_output: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite_output:
        output_path.write_text("", encoding="utf-8")


def _append_example(output_path: Path, example: SyntheticQAExample) -> None:
    with output_path.open("a", encoding="utf-8") as file_obj:
        serialized = json.dumps(dataclasses.asdict(example), ensure_ascii=False)
        file_obj.write(serialized + "\n")


@mcp.tool()
async def generate_synthetic_qa_dataset(
    output_jsonl_path: str = "synthetic_qa_dataset.jsonl",
    sec_root: str = "localworkspace/markdown/sec_data",
    earnings_root: str = "earnings_transcripts_data",
    random_seed: int = 42,
    min_questions_per_filing_type: int = 2,
    max_questions_per_filing_type: int = 3,
    overwrite_output: bool = True,
    ctx: Context | None = None,
) -> dict[str, int | str]:
    """Iterate filings, ask MCP client for QA from sampled contexts, and write JSONL."""
    if ctx is None:
        raise ValueError("ctx is required for MCP sampling.")
    if min_questions_per_filing_type < 1:
        raise ValueError("min_questions_per_filing_type must be >= 1")
    if max_questions_per_filing_type < min_questions_per_filing_type:
        raise ValueError(
            "max_questions_per_filing_type must be >= min_questions_per_filing_type"
        )

    randomizer = random.Random(random_seed)
    sec_root_path = Path(sec_root)
    earnings_root_path = Path(earnings_root)
    output_path = Path(output_jsonl_path)

    records = _collect_records(sec_root=sec_root_path, earnings_root=earnings_root_path)
    _prepare_output_file(output_path=output_path, overwrite_output=overwrite_output)

    written_examples = 0
    for record in records:
        markdown_text = record.file_path.read_text(encoding="utf-8", errors="ignore")
        context_selection = _choose_contexts(
            markdown_text=markdown_text, randomizer=randomizer
        )
        if not context_selection.contexts:
            logger.warning(f"Skipping filing with no contexts. {record.file_path=}")
            continue

        question_count = randomizer.randint(
            min_questions_per_filing_type,
            max_questions_per_filing_type,
        )
        logger.info(f"{record.file_path=} {question_count=}")
        for _ in range(question_count):
            context = randomizer.choice(context_selection.contexts)
            question, answer = await _request_qa_from_mcp_client(
                ctx=ctx,
                record=record,
                context=context,
            )
            example = _build_example(
                record=record,
                question=question,
                answer=answer,
                context=context,
            )
            _append_example(output_path=output_path, example=example)
            written_examples += 1

    result = {
        "output_jsonl_path": str(output_path),
        "filings_processed": len(records),
        "examples_written": written_examples,
    }
    logger.info(f"{result=}")
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
