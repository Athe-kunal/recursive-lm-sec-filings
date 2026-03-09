from __future__ import annotations

import os
from code import InteractiveConsole
from dataclasses import dataclass
from pathlib import Path

from src.ocr.run_ocr import DEFAULT_MODEL, DEFAULT_SERVER, DEFAULT_WORKSPACE
from src.sec_dataloader import load_sec_filings


@dataclass
class MarkdownReplEnvironment:
    ticker: str
    year: str
    filing_type: str
    markdown_path: Path
    markdown_text: str
    namespace: dict[str, object]
    console: InteractiveConsole


def markdown_to_repl_env(
    markdown_path: Path,
    ticker: str,
    year: str,
) -> MarkdownReplEnvironment:
    resolved_path = markdown_path.resolve()
    markdown_text = resolved_path.read_text(encoding="utf-8")
    filing_type = resolved_path.stem
    namespace: dict[str, object] = {
        "ticker": ticker,
        "year": year,
        "filing_type": filing_type,
        "markdown_path": resolved_path,
        "markdown_text": markdown_text,
    }
    console = InteractiveConsole(locals=namespace)

    return MarkdownReplEnvironment(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        markdown_path=resolved_path,
        markdown_text=markdown_text,
        namespace=namespace,
        console=console,
    )


def load_sec_filing_repl_envs(
    ticker: str,
    year: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
) -> list[MarkdownReplEnvironment]:
    company = os.getenv("SEC_COMPANY", "Indiana University Bloomington")
    email = os.getenv("SEC_EMAIL", "astmohap@iu.edu")
    pdf_base_dir = os.getenv("SEC_PDF_BASE_DIR", "sec_data")
    workspace = os.getenv("OLMOCR_WORKSPACE", DEFAULT_WORKSPACE)
    server = os.getenv("OLMOCR_SERVER", DEFAULT_SERVER)
    model = os.getenv("OLMOCR_MODEL", DEFAULT_MODEL)

    markdown_paths = load_sec_filings(
        ticker=ticker,
        year=year,
        filing_types=filing_types,
        include_amends=include_amends,
        company=company,
        email=email,
        pdf_base_dir=pdf_base_dir,
        workspace=workspace,
        server=server,
        model=model,
    )
    return [
        markdown_to_repl_env(markdown_path=path, ticker=ticker, year=year)
        for path in markdown_paths
    ]
