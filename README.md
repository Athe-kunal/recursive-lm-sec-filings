# recursive-lm-sec-filings

Training/data pipeline built on top of the published `finance-data-llm` package.

## What changed

This repository no longer keeps local SEC download/OCR implementations.
Instead, it imports those features directly from `finance_data_llm`:

- SEC filings metadata/download utilities
- OCR execution and markdown generation
- Server entrypoint

`settings.py` in this repo is still the local place for environment defaults used by
the training and vector-index workflows.

## Server

The server command is exposed through project scripts:

```toml
[project.scripts]
finance-data-llm-server = "finance_data.cli:main"
```

### Spin up the server

From this repository:

```bash
uv sync
uv run finance-data-llm-server
```

Or with pip:

```bash
pip install -e .
finance-data-llm-server
```

## Paths used by the pipeline

- SEC PDFs: `{sec_data_dir}/{ticker}-{year}/{form_name}.pdf`
- OCR markdowns: `{olmocr_workspace}/markdown/{sec_data_dir}/{ticker}-{year}/{form_name}.pdf.md`
