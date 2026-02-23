import re
import asyncio
import aiohttp
import requests
from typing import Final, Union, Optional
from pathlib import Path
import os
from loguru import logger
from ratelimit import limits, sleep_and_retry
from playwright.async_api import async_playwright

SEC_ARCHIVE_URL: Final[str] = "https://www.sec.gov/Archives/edgar/data"
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


async def _get_main_html_url(
    session: aiohttp.ClientSession,
    cik: Union[str, int],
    accession_number: Union[str, int],
    user_agent: str,
) -> str:
    """Fetches the filing index JSON and returns the URL of the primary HTML document."""
    acc_no_dashes = _drop_dashes(accession_number)
    acc_with_dashes = _add_dashes(accession_number)
    index_url = f"{SEC_ARCHIVE_URL}/{cik}/{acc_no_dashes}/{acc_with_dashes}-index.json"
    request_headers = {
        "User-Agent": user_agent,
        "Content-Type": "text/html",
    }
    async with session.get(index_url, headers=request_headers) as response:
        response.raise_for_status()
        index_data = await response.json(content_type=None)

    # Prefer the explicitly marked primary document
    primary_doc = index_data.get("primaryDocument", index_data.get("primary_doc", ""))
    if primary_doc and re.search(r"\.html?$", primary_doc, re.IGNORECASE):
        return f"{SEC_ARCHIVE_URL}/{cik}/{acc_no_dashes}/{primary_doc}"

    # Fall back to first document with an .htm/.html extension
    for doc in index_data.get("documents", []):
        doc_name = doc.get("document", doc.get("name", ""))
        if re.search(r"\.html?$", doc_name, re.IGNORECASE):
            return f"{SEC_ARCHIVE_URL}/{cik}/{acc_no_dashes}/{doc_name}"

    raise ValueError(
        f"No HTML document found in filing index for accession {accession_number}"
    )


async def save_filings_as_pdfs(
    filings: list[tuple[Union[str, int], Union[str, int], Union[str, Path]]],
    company: str,
    email: str,
    max_concurrent: int = 4,
) -> list[Path]:
    """Fetch each filing's primary HTML document and save it as a PDF.

    Args:
        filings: List of (cik, accession_number, output_path) tuples.
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
        output_path: Union[str, Path],
        browser,
        session: aiohttp.ClientSession,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with sem:
            html_url = await _get_main_html_url(session, cik, accession_number, user_agent)
            logger.info(f"Rendering {html_url} → {output_path}")
            page = await browser.new_page()
            try:
                await page.set_extra_http_headers({
                    "User-Agent": user_agent,
                    "Content-Type": "text/html",
                })
                await page.goto(html_url, wait_until="networkidle", timeout=120_000)
                await page.pdf(
                    path=str(output_path),
                    format="A4",
                    print_background=True,
                )
            finally:
                await page.close()
        logger.info(f"Saved PDF: {output_path}")
        return output_path

    async with aiohttp.ClientSession() as session:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                results = await asyncio.gather(
                    *[
                        _save_one(cik, acc_num, out_path, browser, session)
                        for cik, acc_num, out_path in filings
                    ]
                )
            finally:
                await browser.close()

    return list(results)
