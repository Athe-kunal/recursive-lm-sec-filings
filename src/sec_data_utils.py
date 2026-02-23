import re
import asyncio
import requests
from typing import Final, Union, Optional
from pathlib import Path
import os
from loguru import logger
from ratelimit import limits, sleep_and_retry
from playwright.async_api import async_playwright

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


async def save_filings_as_pdfs(
    filings: list[tuple[Union[str, int], Union[str, int], str, Union[str, Path]]],
    company: str,
    email: str,
    max_concurrent: int = 4,
) -> list[Path]:
    """Render each filing's primary .htm document to PDF via a headless browser.

    Args:
        filings: List of (cik, accession_number, primary_document, output_path) tuples.
        company: Company name for SEC User-Agent header.
        email: Contact e-mail for SEC User-Agent header.
        max_concurrent: Maximum number of simultaneous browser pages.

    Returns:
        List of Path objects pointing to the saved PDFs (same order as input).
    """
    sem = asyncio.Semaphore(max_concurrent)
    user_agent = f"{company} {email}"

    async def _save_one(
        cik: Union[str, int],
        accession_number: Union[str, int],
        primary_document: str,
        output_path: Union[str, Path],
        context,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = viewer_url(cik, accession_number, primary_document)
        async with sem:
            logger.info(f"Rendering {url} → {output_path}")
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=120_000)
                await page.pdf(
                    path=str(output_path),
                    format="A4",
                    print_background=True,
                )
            finally:
                await page.close()
        logger.info(f"Saved PDF: {output_path}")
        return output_path

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            user_agent=user_agent,
            extra_http_headers={"Content-Type": "text/html"},
        )
        try:
            results = await asyncio.gather(
                *[
                    _save_one(cik, acc_num, primary_doc, out_path, context)
                    for cik, acc_num, primary_doc, out_path in filings
                ]
            )
        finally:
            await context.close()
            await browser.close()

    return list(results)
