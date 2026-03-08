import asyncio
import glob
import os
import sys
from pathlib import Path

DEFAULT_SERVER = os.environ.get("OLMOCR_SERVER", "http://localhost:8000/v1")
DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
DEFAULT_WORKSPACE = "./localworkspace"


def run(
    pdf_dir: str,
    workspace: str = DEFAULT_WORKSPACE,
    server: str = DEFAULT_SERVER,
    model: str = DEFAULT_MODEL,
) -> None:
    pdfs = glob.glob(str(Path(pdf_dir) / "*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    sys.argv = [
        "olmocr_pipeline",
        workspace,
        "--server",
        server,
        "--model",
        model,
        "--markdown",
        "--pdfs",
        *pdfs,
    ]

    from src.ocr.olmocr_pipeline import main

    asyncio.run(main())


if __name__ == "__main__":
    run(pdf_dir="sec_data/CSCO-2025")
