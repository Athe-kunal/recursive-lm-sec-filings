from typing import NamedTuple
import asyncio
import requests
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger
from src.sec_data_utils import *


class SecResults(NamedTuple):
    dashes_acc_num: str
    form_name: str
    filing_date: str
    report_date: str


def sec_main(
    ticker: str,
    year: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
    company: str = "Unstructured Technologies",
    email: str = "support@unstructured.io",
) -> tuple[list[SecResults], list[Path]]:
    cik = get_cik_by_ticker(ticker)
    logger.info(f"For {ticker=} found {cik=}")

    rgld_cik = int(cik.lstrip("0"))

    forms = []
    if include_amends:
        for ft in filing_types:
            forms.append(ft)
            forms.append(ft + "/A")
    else:
        forms = list(filing_types)

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        json_data = response.json()
    else:
        raise RuntimeError(
            f"Unable to fetch submissions. Status code: {response.status_code}"
        )

    filings = json_data["filings"]
    recent_filings = filings["recent"]
    sec_form_names: list[str] = []
    form_lists: list[SecResults] = []

    for acc_num, form_name, filing_date, report_date in zip(
        recent_filings["accessionNumber"],
        recent_filings["form"],
        recent_filings["filingDate"],
        recent_filings["reportDate"],
        strict=True,
    ):
        if form_name in forms and report_date.startswith(str(year)):
            display_name = form_name
            if form_name == "10-Q":
                datetime_obj = datetime.strptime(report_date, "%Y-%m-%d")
                quarter = pd.Timestamp(datetime_obj).quarter
                display_name = f"10-Q{quarter}"
                if display_name in sec_form_names:
                    display_name += "-1"
            no_dashes_acc_num = re.sub("-", "", acc_num)
            form_lists.append(
                SecResults(
                    dashes_acc_num=no_dashes_acc_num,
                    form_name=display_name,
                    filing_date=filing_date,
                    report_date=report_date,
                )
            )
            sec_form_names.append(display_name)

    output_dir = Path("sec_data") / f"{ticker}-{year}"
    filings_to_save = [
        (rgld_cik, sr.dashes_acc_num, output_dir / f"{sr.form_name}.pdf")
        for sr in form_lists
    ]

    pdf_paths = asyncio.run(
        save_filings_as_pdfs(
            filings=filings_to_save,
            company=company,
            email=email,
        )
    )

    logger.info(f"Saved {len(pdf_paths)} PDFs to {output_dir}")
    return form_lists, pdf_paths


if __name__ == "__main__":
    data = sec_main(ticker="AAPL", year="2025")
