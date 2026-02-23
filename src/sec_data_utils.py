import re
import concurrent.futures
import requests
import pdfkit
from typing import Final, Union, Optional
from pathlib import Path
import os
from loguru import logger
from ratelimit import limits, sleep_and_retry

SEC_ARCHIVE_URL: Final[str] = "https://www.sec.gov/Archives/edgar/data"
SEC_VIEWER_URL: Final[str] = "https://www.sec.gov/ix?doc=/Archives/edgar/data"
SEC_SEARCH_URL: Final[str] = "http://www.sec.gov/cgi-bin/browse-edgar"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"


def _drop_dashes(accession_number: Union[str, int]) -> str:
    """Converts the accession number to the no dash representation."""
    accession_number = str(accession_number).replace("-", "")
    return accession_number.zfill(18)


def _add_dashes(accession_number: Union[str, int]) -> str:
    """Adds the dashes back into the accession number"""
    accession_number = str(accession_number).replace("-", "").zfill(18)
    return f"{accession_number[:10]}-{accession_number[10:12]}-{accession_number[12:]}"


def archive_url(cik: Union[str, int], accession_number: Union[str, int]) -> str:
    """Builds the archive URL for the SEC accession number."""
    filename = f"{_add_dashes(accession_number)}.txt"
    accession_number = _drop_dashes(accession_number)
    return f"{SEC_ARCHIVE_URL}/{cik}/{accession_number}/{filename}"


def _get_session(
    company: Optional[str] = "Indiana-University-Bloomington",
    email: Optional[str] = "athecolab@gmail.com",
) -> requests.Session:
    """Creates a requests sessions with the appropriate headers set."""
    if company is None:
        company = os.environ.get("SEC_API_ORGANIZATION")
    if email is None:
        email = os.environ.get("SEC_API_EMAIL")
    assert company
    assert email
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{company} {email}",
            "Content-Type": "text/html",
        }
    )
    return session


def _search_url(cik: Union[str, int]) -> str:
    search_string = f"CIK={cik}&Find=Search&owner=exclude&action=getcompany"
    url = f"{SEC_SEARCH_URL}?{search_string}"
    return url


@sleep_and_retry
@limits(calls=2, period=1)
def get_cik_by_ticker(ticker: str) -> str:
    """Gets a CIK number from a stock ticker by running a search on the SEC website."""
    cik_re = re.compile(r".*CIK=(\d{10}).*")
    url = _search_url(ticker)
    company = "Indiana-University-Bloomington"
    email = "athecolab@gmail.com"
    headers = {
        "User-Agent": f"{company} {email}",
        "Content-Type": "text/html",
    }
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    results = cik_re.findall(response.text)
    return str(results[0])


def viewer_url(
    cik: Union[str, int],
    accession_number: Union[str, int],
    primary_document: str,
) -> str:
    """Builds the SEC inline XBRL viewer URL for the primary .htm document."""
    acc_no_dashes = _drop_dashes(accession_number)
    return f"{SEC_VIEWER_URL}/{cik}/{acc_no_dashes}/{primary_document}"


def save_filings_as_pdfs(
    filings: list[tuple[Union[str, int], Union[str, int], str, Union[str, Path]]],
    company: str,
    email: str,
    max_workers: int = 4,
) -> list[Path]:
    """Render each filing's primary .htm document to PDF via pdfkit/wkhtmltopdf.

    Args:
        filings: List of (cik, accession_number, primary_document, output_path) tuples.
        company: Company name for SEC User-Agent header.
        email: Contact e-mail for SEC User-Agent header.
        max_workers: Maximum number of concurrent wkhtmltopdf processes.

    Returns:
        List of Path objects pointing to the saved PDFs (same order as input).
    """
    user_agent = f"{company} {email}"
    pdf_options = {
        "quiet": "",
        "custom-header": [("User-Agent", user_agent)],
        "custom-header-propagation": "",
        "no-stop-slow-scripts": "",
        "javascript-delay": 2000,
        "page-size": "A4",
        "encoding": "UTF-8",
    }

    def _save_one(
        cik: Union[str, int],
        accession_number: Union[str, int],
        primary_document: str,
        output_path: Union[str, Path],
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = viewer_url(cik, accession_number, primary_document)
        logger.info(f"Rendering {url} → {output_path}")
        pdfkit.from_url(url, str(output_path), options=pdf_options)
        logger.info(f"Saved PDF: {output_path}")
        return output_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_save_one, cik, acc_num, primary_doc, out_path)
            for cik, acc_num, primary_doc, out_path in filings
        ]
        results = [f.result() for f in futures]

    return results
