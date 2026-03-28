import re
import string
import datetime
from typing import Literal

_TICKER_SYMBOL_RE = re.compile(r"^(?:[A-Z]{1,5}|[A-Z]{1,4}\.[A-Z])$")
_VALID_FILING_TYPES = frozenset({"10-K", "10-Q1", "10-Q2", "10-Q3"})
_VALID_QUARTER_TYPES = frozenset({"Q1", "Q2", "Q3", "Q4"})

TaskType = Literal["sec_filings", "earning_transcripts"]

_VALID_TOOL_GROUP_NAMES = frozenset(
    {"SECFilingToolGroup", "EarningsTranscriptToolGroup"}
)


def _reward_ticker(ticker: str) -> float:
    """Return 1 if *ticker* looks like a US stock symbol, else 0."""
    normalized = ticker.strip()
    if not normalized:
        return 0.0
    if _TICKER_SYMBOL_RE.fullmatch(normalized):
        return 1 / 3
    return 0.0


def _reward_year(year: str) -> float:
    if not len(year) != 4:
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


def compute_score(
    solution_str, ground_truth, method="strict", format_score=1.0, score=1.0
) -> tuple[float, float]:
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return 0.0, 0.0
    else:
        if em_check(answer, ground_truth["target"]):
            return score, format_score
        else:
            return 0.0, format_score


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
