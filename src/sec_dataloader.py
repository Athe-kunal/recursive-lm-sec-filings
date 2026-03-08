from pathlib import Path

import yfinance as yf
from loguru import logger

from src.ocr.run_ocr import DEFAULT_MODEL, DEFAULT_SERVER, DEFAULT_WORKSPACE
from src.sec_data_utils.sec_data import (
    SecResults,
    get_sec_results,
    save_sec_results_as_pdfs,
)


def company_to_ticker(name: str) -> str | None:
    """Resolve a company name to its stock ticker symbol via Yahoo Finance.

    Args:
        name: The company name to look up (e.g. ``"Apple Inc"``).

    Returns:
        The ticker symbol string (e.g. ``"AAPL"``), or ``None`` if no match
        is found.
    """
    results = yf.Search(name).quotes

    if not results:
        return None

    return results[0]["symbol"]


def fetch_sec_filings(
    ticker: str,
    year: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
    company: str = "Indiana University Bloomington",
    email: str = "astmohap@iu.edu",
    pdf_base_dir: str = "sec_data",
) -> list[Path]:
    """Fetch SEC filings for a given ticker and year, saving each as a PDF.

    Filing metadata is always retrieved from the SEC API so the full set of
    expected filings is known.  Individual PDFs that already exist on disk are
    skipped — only the missing ones are downloaded and rendered.

    Args:
        ticker: Stock ticker symbol (e.g. ``"NVDA"``).
        year: Four-digit fiscal year string (e.g. ``"2025"``).
        filing_types: SEC form types to retrieve (default ``["10-K", "10-Q"]``).
        include_amends: When ``True``, amended forms (``/A`` suffix) are also
            fetched alongside the originals.
        company: Organisation name sent in the SEC ``User-Agent`` header.
        email: Contact e-mail sent in the SEC ``User-Agent`` header.
        pdf_base_dir: Root directory under which PDFs are saved.  Each ticker/
            year combination is stored in a sub-directory
            ``{pdf_base_dir}/{ticker}-{year}/``.

    Returns:
        Ordered list of :class:`~pathlib.Path` objects pointing to all PDFs
        (both pre-existing and newly downloaded).

    Raises:
        RuntimeError: Re-raised from the SEC API layer when the submissions
            endpoint returns a non-200 response.
        ValueError: If no filings of the requested types are found for the
            given ticker and year.
    """
    sec_results: list[SecResults] = get_sec_results(
        ticker=ticker,
        year=year,
        filing_types=filing_types,
        include_amends=include_amends,
        company=company,
        email=email,
    )

    if not sec_results:
        raise ValueError(
            f"No filings found for {ticker=} {year=} with {filing_types=}."
        )

    output_dir = Path(pdf_base_dir) / f"{ticker}-{year}"
    existing_paths: list[Path] = []
    missing_results: list[SecResults] = []

    for sr in sec_results:
        pdf_path = output_dir / f"{sr.form_name}.pdf"
        if pdf_path.exists():
            logger.info(f"PDF already exists, skipping download: {pdf_path}")
            existing_paths.append(pdf_path)
        else:
            missing_results.append(sr)

    if missing_results:
        logger.info(
            f"Downloading {len(missing_results)} missing PDF(s) for "
            f"{ticker}-{year}…"
        )
        new_paths = save_sec_results_as_pdfs(
            sec_results=missing_results,
            ticker=ticker,
            year=year,
            company=company,
            email=email,
        )
    else:
        new_paths = []

    all_paths = existing_paths + new_paths
    logger.info(f"fetch_sec_filings: {len(all_paths)} PDF(s) ready in {output_dir}")
    return all_paths


def _derive_markdown_path(pdf_path: Path, workspace: str) -> Path:
    """Compute the expected markdown output path for *pdf_path*.

    The olmocr pipeline stores markdowns under::

        {workspace}/markdown/{relative_pdf_path_without_leading_slash}/{stem}.md

    For a local path such as ``sec_data/NVDA-2025/10-K.pdf`` and the default
    workspace the result is::

        ./localworkspace/markdown/sec_data/NVDA-2025/10-K.md

    Args:
        pdf_path: Path to the source PDF file.
        workspace: Root workspace directory used by the olmocr pipeline.

    Returns:
        Expected :class:`~pathlib.Path` for the corresponding markdown file.
    """
    relative = str(pdf_path).lstrip("/")
    parts = [p for p in relative.split("/") if p and p != ".."]
    safe_relative = "/".join(parts)
    md_filename = Path(safe_relative).stem + ".md"
    dir_path = str(Path(safe_relative).parent)
    return Path(workspace) / "markdown" / dir_path / md_filename


def run_ocr_on_filings(
    pdf_dir: str,
    workspace: str = DEFAULT_WORKSPACE,
    server: str = DEFAULT_SERVER,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    """Run OCR on all PDFs inside *pdf_dir* and return their markdown paths.

    After the pipeline returns, the expected markdown paths are derived from the
    PDF filenames using the same convention as the pipeline's
    ``get_markdown_path`` helper and validated to exist.

    Args:
        pdf_dir: Directory containing the ``*.pdf`` files to process.
        workspace: Root workspace directory where the olmocr pipeline writes
            its outputs (markdowns, results, done flags, …).
        server: Base URL of the running olmOCR vLLM server.
        model: Model name as registered on the vLLM server.

    Returns:
        List of :class:`~pathlib.Path` objects pointing to the markdown files,
        in the same order as the PDFs found on disk.

    Raises:
        FileNotFoundError: If *pdf_dir* does not exist or contains no PDF
            files (re-raised from :func:`src.ocr.run_ocr.run`).
        RuntimeError: If the pipeline completes but any expected markdown file
            is absent on disk.
    """
    from src.ocr.run_ocr import run as _run_ocr

    pdf_dir_path = Path(pdf_dir)
    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))

    _run_ocr(pdf_dir=str(pdf_dir_path), workspace=workspace, server=server, model=model)

    expected_md_paths = [_derive_markdown_path(p, workspace) for p in pdf_files]
    not_produced = [p for p in expected_md_paths if not p.exists()]
    if not_produced:
        raise RuntimeError(
            f"OCR completed but {len(not_produced)} markdown file(s) were not "
            f"produced: {not_produced}"
        )

    return expected_md_paths


def load_sec_filings(
    ticker: str,
    year: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
    company: str = "Indiana University Bloomington",
    email: str = "astmohap@iu.edu",
    pdf_base_dir: str = "sec_data",
    workspace: str = DEFAULT_WORKSPACE,
    server: str = DEFAULT_SERVER,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    """Fetch SEC filings and convert them to markdown via OCR.

    This is the primary entry point for the data-loading pipeline.  It
    orchestrates two steps:

    1. :func:`fetch_sec_filings` — downloads any PDFs that are not yet present
       on disk.
    2. :func:`run_ocr_on_filings` — runs the olmocr pipeline on the PDF
       directory and returns markdown paths, skipping files that have already
       been processed.

    Args:
        ticker: Stock ticker symbol (e.g. ``"NVDA"``).
        year: Four-digit fiscal year string (e.g. ``"2025"``).
        filing_types: SEC form types to retrieve (default ``["10-K", "10-Q"]``).
        include_amends: When ``True``, amended forms (``/A`` suffix) are also
            fetched alongside the originals.
        company: Organisation name sent in the SEC ``User-Agent`` header.
        email: Contact e-mail sent in the SEC ``User-Agent`` header.
        pdf_base_dir: Root directory under which PDFs are saved.
        workspace: Root workspace directory used by the olmocr pipeline.
        server: Base URL of the running olmOCR vLLM server.
        model: Model name as registered on the vLLM server.

    Returns:
        List of :class:`~pathlib.Path` objects pointing to the produced
        markdown files (one per filing).

    Raises:
        ValueError: If no filings are found for the requested ticker/year.
        RuntimeError: On SEC API failure or if OCR does not produce expected
            output files.
        FileNotFoundError: If the PDF directory is empty after the download
            step.
    """
    pdf_paths = fetch_sec_filings(
        ticker=ticker,
        year=year,
        filing_types=filing_types,
        include_amends=include_amends,
        company=company,
        email=email,
        pdf_base_dir=pdf_base_dir,
    )

    pdf_dir = str(Path(pdf_base_dir) / f"{ticker}-{year}")
    markdown_paths = run_ocr_on_filings(
        pdf_dir=pdf_dir,
        workspace=workspace,
        server=server,
        model=model,
    )

    logger.info(
        f"load_sec_filings: {len(markdown_paths)} markdown(s) ready for "
        f"{ticker}-{year}."
    )
    return markdown_paths


if __name__ == "__main__":
    paths = load_sec_filings(ticker="LVS", year="2023")
    for p in paths:
        print(p)
