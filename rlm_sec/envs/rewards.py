import re
import string
import datetime
from dataclasses import dataclass
from typing import Literal

_TICKER_SYMBOL_RE = re.compile(r"^(?:[A-Z]{1,5}|[A-Z]{1,4}\.[A-Z])$")
_VALID_FILING_TYPES = frozenset({"10-K", "10-Q1", "10-Q2", "10-Q3"})
_VALID_QUARTER_TYPES = frozenset({"Q1", "Q2", "Q3", "Q4"})
_VALID_RANKING_FILING_TYPES = frozenset({"DEF14A", "10-K", "10-Q", "8-K", "EARNINGS"})

# TaskType governs intermediate action-format rewards (which tool was called).
TaskType = Literal["sec_filings", "earning_transcripts"]

# DataTaskType governs the final episode reward (what the episode is training).
DataTaskType = Literal["qa", "ranking"]

_VALID_TOOL_GROUP_NAMES = frozenset(
    {"SECFilingToolGroup", "EarningsTranscriptToolGroup", "CompanyNameToTickerToolGroup"}
)

_SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_SOURCE_PATTERN = re.compile(r"<sources?>(.*?)</sources?>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class QAScoreResult:
    """Scoring result for QA tasks."""

    correctness: float
    format: float


@dataclass(frozen=True)
class RankingScoreResult:
    """Scoring result for ranking tasks."""

    correctness: float
    format: float
    precision: float
    recall: float
    f1: float


def _reward_ticker(ticker: str) -> float:
    """Return 1 if *ticker* looks like a US stock symbol, else 0."""
    normalized = ticker.strip()
    if not normalized:
        return 0.0
    if _TICKER_SYMBOL_RE.fullmatch(normalized):
        return 1 / 3
    return 0.0


def _reward_year(year: str) -> float:
    if len(year) != 4:
        return 0.0
    curr_year = datetime.datetime.now().year
    output_year = int(year)
    if output_year <= curr_year:
        return 1 / 3
    return 0.0


def _reward_filing_type(filing_type: str, task_type: TaskType) -> float:
    """Return 1 if *filing_type* is 10-K or 10-Q1..10-Q3, else 0.

    Quarter labels match stored stems (e.g. ``10-Q3`` is the third 10-Q).
    """
    normalized = filing_type.strip().upper()
    match task_type:
        case "sec_filings":
            valid_types = _VALID_FILING_TYPES
        case "earning_transcripts":
            valid_types = _VALID_QUARTER_TYPES
        case _:
            raise ValueError(f"Invalid task type: {task_type}")
    if not normalized:
        return 0.0
    if normalized in valid_types:
        return 1 / 3
    return 0.0


def _reward_tool_name(tool_group_name: str) -> float:
    """Return 1.0 if tool_group_name is a recognized tool group, else 0.0."""
    return 1.0 if tool_group_name in _VALID_TOOL_GROUP_NAMES else 0.0


def reward_action_format(
    tool_group_name: str,
    ticker: str,
    year: str,
    filing_type: str,
    task_type: TaskType,
) -> float:
    """Score pre-parsed action components across tool name, ticker, year, and filing type."""
    return (
        _reward_tool_name(tool_group_name)
        + _reward_ticker(ticker)
        + _reward_year(year)
        + _reward_filing_type(filing_type, task_type)
    )


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_sources(solution_str: str) -> list[str]:
    """Extract source labels from the last <source> or <sources> tag."""
    matches = list(_SOURCE_PATTERN.finditer(solution_str))
    if not matches:
        return []

    raw_sources = matches[-1].group(1)
    normalized_sources = []
    for raw_source in raw_sources.split(","):
        normalized = raw_source.strip()
        if normalized:
            normalized_sources.append(normalized)
    return normalized_sources


def compute_precision_recall_f1(
    predicted: list[str], relevant: list[str]
) -> tuple[float, float, float]:
    """Compute precision/recall/F1 over case-insensitive source sets."""
    predicted_set = {value.upper() for value in predicted}
    relevant_set = {value.upper() for value in relevant}

    if not predicted_set and not relevant_set:
        return 1.0, 1.0, 1.0
    if not relevant_set:
        return 0.0, 0.0, 0.0

    true_positives = len(predicted_set & relevant_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(relevant_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _find_search_calls(solution_str: str) -> list[tuple[str, list[str]]]:
    """Return parsed tool calls as (tool_name, args)."""
    parsed_calls: list[tuple[str, list[str]]] = []
    for match in _SEARCH_PATTERN.finditer(solution_str):
        inner = match.group(1).strip()
        call_match = re.match(r"(\w+)\((.+)\)$", inner, re.DOTALL)
        if not call_match:
            continue
        tool_name = call_match.group(1).strip()
        args = [part.strip() for part in call_match.group(2).rsplit(",", 3)]
        parsed_calls.append((tool_name, args))
    return parsed_calls


def _matches_dataset(data_source: str, accepted: set[str]) -> bool:
    normalized = data_source.strip().lower()
    return normalized in accepted


def _has_valid_qa_search_call(search_calls: list[tuple[str, list[str]]]) -> bool:
    """Return True when at least one recognized retrieval tool call is well-formed."""
    for tool_name, args in search_calls:
        if tool_name == "CompanyNameToTickerTool" and len(args) == 1:
            return True
        if tool_name in {"SECFilingTool", "EarningsTranscriptTool"} and len(args) == 4:
            return True
    return False


def compute_qa_format_score(solution_str: str, ground_truth: dict) -> float:
    """Compute QA format score from tool usage and dataset-specific format checks."""
    search_calls = _find_search_calls(solution_str)
    valid_tool_call = _has_valid_qa_search_call(search_calls)
    base_format_score = 1.0 if valid_tool_call else 0.0

    data_source = str(ground_truth.get("data_source", ""))
    task2_sources = {
        "virattt/financial-qa-10k",
        "viratt-finance-data",
        "patronusai/financebench",
        "finance bench",
    }
    if not _matches_dataset(data_source, task2_sources):
        return base_format_score

    used_company_to_ticker = any(
        tool_name == "CompanyNameToTickerTool" for tool_name, _ in search_calls
    )

    expected_ticker = str(
        ground_truth.get("ticker", ground_truth.get("ticker_or_company_name", ""))
    ).strip()
    expected_year = str(ground_truth.get("year", "")).strip()

    has_expected_ticker = False
    has_expected_year = False
    for tool_name, args in search_calls:
        if tool_name in {"SECFilingTool", "EarningsTranscriptTool"} and len(args) == 4:
            ticker = args[1].upper()
            year = args[2]
            if expected_ticker and ticker == expected_ticker.upper():
                has_expected_ticker = True
            if expected_year and year == expected_year:
                has_expected_year = True

    return base_format_score + (1.0 if used_company_to_ticker else 0.0) + (
        1.0 if has_expected_ticker else 0.0
    ) + (1.0 if has_expected_year else 0.0)


def compute_score(
    solution_str, ground_truth, method="strict", format_score=1.0, score=1.0
) -> QAScoreResult:
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)

    qa_format_score = compute_qa_format_score(solution_str, ground_truth)
    if answer is None:
        return QAScoreResult(correctness=0.0, format=0.0)
    if em_check(answer, ground_truth["target"]):
        return QAScoreResult(correctness=score, format=qa_format_score)
    return QAScoreResult(correctness=0.0, format=qa_format_score)


def extract_searched_filing_types(chat_history_str: str) -> set[str]:
    """Parse all <search> calls in the chat history and return the set of
    filing types / quarters used as the last argument.

    Works for both 4-arg tool calls (SECFilingTool / EarningsTranscriptTool)
    and ignores 1-arg calls (CompanyNameToTickerTool).
    """
    filing_types: set[str] = set()
    for match in _SEARCH_PATTERN.finditer(chat_history_str):
        inner = match.group(1).strip()
        call_match = re.match(r"\w+\((.+)\)$", inner, re.DOTALL)
        if not call_match:
            continue
        parts = [p.strip() for p in call_match.group(1).split(",")]
        if len(parts) == 4:
            filing_types.add(parts[3].upper())
    return filing_types


def _answer_covers_relevant(answer: str, relevant: list[str]) -> bool:
    """Return True if the answer string contains every relevant filing type."""
    answer_upper = answer.upper()
    return all(filing_type.upper() in answer_upper for filing_type in relevant)


def compute_ranking_score(
    solution_str: str,
    ground_truth: dict,
    format_score: float = 1.0,
    score: float = 1.0,
) -> RankingScoreResult:
    """Score ranking by predicted sources and return source metrics.

    Correctness is F1 between the predicted sources and relevant sources.
    Format reward is 1.0 when <source> (or <sources>) tags are present.
    """
    relevant: list[str] = ground_truth.get("relevant", [])
    predicted_sources = extract_sources(solution_str)
    has_source_tag = bool(predicted_sources)
    format_reward = format_score if has_source_tag else 0.0

    if not relevant:
        return RankingScoreResult(
            correctness=0.0, format=format_reward, precision=0.0, recall=0.0, f1=0.0
        )

    precision, recall, f1 = compute_precision_recall_f1(predicted_sources, relevant)
    correctness = score * f1
    return RankingScoreResult(
        correctness=correctness,
        format=format_reward,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def compute_score_subem(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
