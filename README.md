# recursive-lm-sec-filings

Pipeline for fetching SEC filings and converting them to markdown via OCR.

## Functionality

### `src/sec_dataloader.py`

| Function | Description |
|---|---|
| `company_to_ticker(name)` | Resolve a company name to its ticker symbol via Yahoo Finance. |
| `fetch_sec_filings(ticker, year, ...)` | Download SEC filings as PDFs into `sec_data/{ticker}-{year}/`. Skips PDFs that already exist. |
| `run_ocr_on_filings(pdf_dir, ...)` | Run olmOCR on a PDF directory and return markdown file paths. The pipeline skips already-processed files via done-flags. |
| `load_sec_filings(ticker, year, ...)` | End-to-end: fetch PDFs → run OCR → return markdown paths. |

### Key paths

- PDFs: `sec_data/{ticker}-{year}/{form_name}.pdf`
- Markdowns: `localworkspace/markdown/sec_data/{ticker}-{year}/{form_name}.md`

### Requirements

- Running olmOCR vLLM server (default: `http://localhost:8000/v1`). Override with the `OLMOCR_SERVER` env var.
