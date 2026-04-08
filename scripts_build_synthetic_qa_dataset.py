"""Build synthetic QA task data from SEC markdown and earnings transcripts."""

from __future__ import annotations

import dataclasses
import json
from loguru import logger
import random
import re
from pathlib import Path
from typing import Iterable


_ALLOWED_SEC_FILINGS = {"10-Q1", "10-Q2", "10-Q3", "8-K", "DEF 14A"}
_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_REGEX = re.compile(r"\s+")


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


@dataclasses.dataclass(slots=True)
class FileMetadata:
    path: Path
    ticker: str
    year: str
    filing_type: str


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_REGEX.sub(" ", text).strip()


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("**Date:"):
        return True
    if stripped.startswith("---"):
        return True
    return False


def read_clean_lines(file_path: Path) -> list[str]:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        if is_noise_line(line):
            continue
        normalized = normalize_whitespace(line)
        if not normalized:
            continue
        cleaned_lines.append(normalized)
    cleaned_lines_count = len(cleaned_lines)
    logger.debug(f"{file_path=} {cleaned_lines_count=}")
    return cleaned_lines


def build_paragraphs(cleaned_lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in cleaned_lines:
        if len(line) < 2:
            if current:
                paragraphs.append(normalize_whitespace(" ".join(current)))
                current = []
            continue
        current.append(line)
        if line.endswith(".") or line.endswith("?") or line.endswith("!"):
            paragraphs.append(normalize_whitespace(" ".join(current)))
            current = []
    if current:
        paragraphs.append(normalize_whitespace(" ".join(current)))
    return paragraphs


def should_keep_sentence(sentence: str) -> bool:
    if len(sentence) < 60 or len(sentence) > 420:
        return False
    lowered = sentence.lower()
    blocked_terms = (
        "table of contents",
        "conference call operator",
        "forward-looking",
        "click here",
        "signature",
        "item ",
    )
    if any(term in lowered for term in blocked_terms):
        return False
    if sentence.upper() == sentence:
        return False
    if not re.search(r"[a-zA-Z]", sentence):
        return False
    return True


def sentence_score(sentence: str) -> int:
    score = 0
    lowered = sentence.lower()
    if re.search(r"\d", sentence):
        score += 2
    if "%" in sentence:
        score += 2
    if "$" in sentence:
        score += 2
    keywords = (
        "revenue",
        "growth",
        "net income",
        "cash",
        "guidance",
        "margin",
        "members",
        "eps",
        "dividend",
        "debt",
    )
    for keyword in keywords:
        if keyword in lowered:
            score += 1
    return score


def split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_SPLIT_REGEX.split(paragraph)
    sentences = [normalize_whitespace(part) for part in parts]
    return [sentence for sentence in sentences if should_keep_sentence(sentence)]


def build_context_window(paragraph: str, answer_sentence: str) -> str:
    raw_sentences = [
        normalize_whitespace(part)
        for part in _SENTENCE_SPLIT_REGEX.split(paragraph)
        if normalize_whitespace(part)
    ]
    if not raw_sentences:
        return answer_sentence
    try:
        answer_index = raw_sentences.index(answer_sentence)
    except ValueError:
        return answer_sentence

    start_index = max(0, answer_index - 1)
    end_index = min(len(raw_sentences), answer_index + 2)
    return normalize_whitespace(" ".join(raw_sentences[start_index:end_index]))


def build_question(metadata: FileMetadata, sentence: str, index: int) -> str:
    period_phrase = period_descriptor(filing_type=metadata.filing_type)
    prompt_prefix = prompt_prefix_options(
        ticker=metadata.ticker,
        year=metadata.year,
        period_phrase=period_phrase,
        index=index,
    )
    sentence_lower = sentence.lower()
    variant = index % 3

    if "eps" in sentence_lower:
        options = (
            "how much did EPS grow, or what EPS figure was reported?",
            "what EPS figure was reported?",
            "what was the reported EPS?",
        )
        return prompt_prefix + options[variant]
    if "free cash flow" in sentence_lower and "cash" in sentence_lower:
        options = (
            "how much free cash flow and ending cash was reported?",
            "what were the free cash flow and ending cash amounts?",
            "what free cash flow and quarter-end cash figures were reported?",
        )
        return prompt_prefix + options[variant]
    if "free cash flow" in sentence_lower:
        options = (
            "how much free cash flow was generated?",
            "what free cash flow amount was reported?",
            "what did the company report for free cash flow?",
        )
        return prompt_prefix + options[variant]
    if "revenue" in sentence_lower or "sales" in sentence_lower:
        options = (
            "how much revenue or sales was reported?",
            "what revenue or sales figure was reported?",
            "what did the company report for revenue or sales?",
        )
        return prompt_prefix + options[variant]
    if "net income" in sentence_lower:
        options = (
            "how much net income was reported?",
            "what net income figure was reported?",
            "what did the company report for net income?",
        )
        return prompt_prefix + options[variant]
    if "operating margin" in sentence_lower or "margin" in sentence_lower:
        options = (
            "what operating margin or margin figure was reported?",
            "what margin percentage was reported?",
            "what did the company report for margin?",
        )
        return prompt_prefix + options[variant]
    if "guidance" in sentence_lower:
        options = (
            "what guidance was provided?",
            "what guidance figures were reported?",
            "what did management say about guidance?",
        )
        return prompt_prefix + options[variant]
    if "dividend" in sentence_lower:
        options = (
            "what dividend detail was reported?",
            "what dividend figure was reported?",
            "what did the company report about its dividend?",
        )
        return prompt_prefix + options[variant]
    if "debt" in sentence_lower:
        options = (
            "what debt figure or debt update was reported?",
            "what debt amount was reported?",
            "what did the company report about debt?",
        )
        return prompt_prefix + options[variant]
    if "member" in sentence_lower:
        options = (
            "how many members were reported?",
            "what membership figure was reported?",
            "what did the company report about membership?",
        )
        return prompt_prefix + options[variant]
    options = (
        "what was reported in the company update?",
        "what key figure was reported?",
        "what did the company state in its update?",
    )
    return prompt_prefix + options[variant]


def period_descriptor(filing_type: str) -> str:
    filing_to_period = {
        "10-Q1": "early",
        "Q1": "early",
        "10-Q2": "mid-year",
        "Q2": "mid-year",
        "10-Q3": "late",
        "Q3": "late",
        "Q4": "year-end",
    }
    return filing_to_period.get(filing_type, "")


def prompt_prefix_options(
    ticker: str,
    year: str,
    period_phrase: str,
    index: int,
) -> str:
    style = index % 3
    if period_phrase:
        if style == 0:
            return f"In {period_phrase} {year}, for {ticker}, "
        if style == 1:
            return f"For {ticker} in {period_phrase} {year}, "
        return f"For {ticker}, in {period_phrase} {year}, "
    if style == 0:
        return f"In {year}, for {ticker}, "
    if style == 1:
        return f"For {ticker} in {year}, "
    return f"For {ticker}, in {year}, "


def generate_examples_for_file(
    metadata: FileMetadata,
    examples_per_file: int,
) -> list[SyntheticQAExample]:
    cleaned_lines = read_clean_lines(metadata.path)
    paragraphs = build_paragraphs(cleaned_lines)
    candidates: list[tuple[int, str, str]] = []

    for paragraph in paragraphs:
        sentences = split_sentences(paragraph)
        for sentence in sentences:
            score = sentence_score(sentence)
            candidates.append((score, sentence, paragraph))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: examples_per_file * 2]

    examples: list[SyntheticQAExample] = []
    used_answers: set[str] = set()
    for _, sentence, paragraph in selected:
        if sentence in used_answers:
            continue
        if len(examples) >= examples_per_file:
            break
        question = build_question(
            metadata=metadata, sentence=sentence, index=len(examples)
        )
        context = build_context_window(paragraph=paragraph, answer_sentence=sentence)
        example = SyntheticQAExample(
            question=question,
            answer=sentence,
            context=context,
            year=metadata.year,
            ticker_or_company_name=metadata.ticker,
            filing_type=metadata.filing_type,
            data_source="generated",
        )
        examples.append(example)
        used_answers.add(sentence)

    examples_count = len(examples)
    logger.info(f"generated file examples. {metadata.path=} {examples_count=}")
    return examples


def collect_sec_files(base_dir: Path) -> list[FileMetadata]:
    records: list[FileMetadata] = []
    for company_year_dir in sorted(base_dir.glob("*")):
        if not company_year_dir.is_dir():
            continue
        ticker, year = split_ticker_year(directory_name=company_year_dir.name)
        for file_path in sorted(company_year_dir.glob("*.md")):
            filing_type = file_path.stem
            if filing_type not in _ALLOWED_SEC_FILINGS:
                continue
            records.append(
                FileMetadata(
                    path=file_path,
                    ticker=ticker,
                    year=year,
                    filing_type=filing_type,
                )
            )
    records_count = len(records)
    logger.info(f"collected sec files. {records_count=}")
    return records


def split_ticker_year(directory_name: str) -> tuple[str, str]:
    if "-" not in directory_name:
        return directory_name, ""
    ticker, year = directory_name.rsplit("-", maxsplit=1)
    return ticker, year


def collect_transcript_files(base_dir: Path) -> list[FileMetadata]:
    records: list[FileMetadata] = []
    for ticker_dir in sorted(base_dir.glob("*")):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for year_dir in sorted(ticker_dir.glob("*")):
            if not year_dir.is_dir():
                continue
            year = year_dir.name
            for file_path in sorted(year_dir.glob("Q*.md")):
                quarter = file_path.stem.split("_")[0]
                if quarter not in {"Q1", "Q2", "Q3", "Q4"}:
                    continue
                records.append(
                    FileMetadata(
                        path=file_path,
                        ticker=ticker,
                        year=year,
                        filing_type=quarter,
                    )
                )
    records_count = len(records)
    logger.info(f"collected transcript files. {records_count=}")
    return records


def sample_file_metadata(
    sec_records: list[FileMetadata],
    transcript_records: list[FileMetadata],
    target_examples: int,
    examples_per_file: int,
    seed: int,
) -> list[FileMetadata]:
    random.seed(seed)
    required_files = max(1, target_examples // examples_per_file)
    combined_records = sec_records + transcript_records
    if len(combined_records) <= required_files:
        return combined_records
    sampled = random.sample(combined_records, required_files)
    sampled_count = len(sampled)
    logger.info(f"sampled files. {required_files=} {sampled_count=}")
    return sampled


def write_jsonl(output_path: Path, examples: Iterable[SyntheticQAExample]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file_obj:
        for example in examples:
            file_obj.write(
                json.dumps(dataclasses.asdict(example), ensure_ascii=False) + "\n"
            )
            count += 1
    return count


def main() -> None:
    sec_base_dir = Path("localworkspace/markdown/sec_data")
    transcript_base_dir = Path("earnings_transcripts_data")
    output_path = Path("data/generated_synthetic_qa.jsonl")

    target_examples = 900
    examples_per_file = 3
    seed = 42

    sec_records = collect_sec_files(base_dir=sec_base_dir)
    transcript_records = collect_transcript_files(base_dir=transcript_base_dir)
    sampled_records = sample_file_metadata(
        sec_records=sec_records,
        transcript_records=transcript_records,
        target_examples=target_examples,
        examples_per_file=examples_per_file,
        seed=seed,
    )

    all_examples: list[SyntheticQAExample] = []
    for metadata in sampled_records:
        file_examples = generate_examples_for_file(
            metadata=metadata,
            examples_per_file=examples_per_file,
        )
        all_examples.extend(file_examples)

    if len(all_examples) > target_examples:
        all_examples = all_examples[:target_examples]

    written_count = write_jsonl(output_path=output_path, examples=all_examples)
    logger.info(f"finished generating examples. {output_path=} {written_count=}")


if __name__ == "__main__":
    main()
