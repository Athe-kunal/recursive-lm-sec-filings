from __future__ import annotations

import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"

from pathlib import Path
from typing import List

import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import LayoutDetection
from tqdm import tqdm

from src.ocr.bbox_postprocess import Box, postprocess, draw_boxes


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[np.ndarray]:
    """Convert each PDF page to a BGR numpy array (no files written)."""
    pages = convert_from_path(pdf_path, dpi=dpi)
    return [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) for page in pages]


def run_layout_detection(
    images: List[np.ndarray],
    model: LayoutDetection,
    batch_size: int = 128,
) -> List[List[Box]]:
    """Run layout detection on all pages in batches, with a tqdm progress bar."""
    results: List[List[Box]] = [[] for _ in images]
    batches = list(range(0, len(images), batch_size))
    for start in tqdm(batches, desc="Layout detection", unit="batch"):
        batch = images[start : start + batch_size]
        output = model.predict(batch, batch_size=batch_size, layout_nms=True)
        for offset, res in enumerate(output):
            results[start + offset] = res.json["res"]["boxes"]
    return results


def postprocess_boxes(raw_boxes: List[List[Box]]) -> List[List[Box]]:
    """Apply bbox post-processing to each page's raw boxes."""
    return [
        postprocess(boxes)
        for boxes in tqdm(raw_boxes, desc="Post-processing pages", unit="page")
    ]


def save_visualizations(
    images: List[np.ndarray],
    processed_boxes: List[List[Box]],
    pdf_stem: str,
    output_dir: str,
) -> None:
    """Draw and save one annotated image per page into output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, (image, boxes) in enumerate(
        tqdm(
            zip(images, processed_boxes),
            total=len(images),
            desc="Saving visualizations",
            unit="page",
        )
    ):
        save_path = out / f"{pdf_stem}_page_{i + 1}_merged.png"
        draw_boxes(image, boxes, save_path=str(save_path))


def process_pdf(
    pdf_path: str,
    output_dir: str = "./output",
    dpi: int = 150,
    batch_size: int = 4,
    device: str = "cpu",
    threshold: float = 0.3,
) -> None:
    """End-to-end pipeline: PDF → images → layout detection → post-process → visualize."""
    images = pdf_to_images(pdf_path, dpi=dpi)

    model = LayoutDetection(
        model_name="PP-DocLayout_plus-L",
        device=device,
        enable_hpi=True,
        threshold=threshold,
    )

    raw_boxes = run_layout_detection(images, model, batch_size=batch_size)
    processed_boxes = postprocess_boxes(raw_boxes)
    save_visualizations(images, processed_boxes, Path(pdf_path).stem, output_dir)


if __name__ == "__main__":
    process_pdf(pdf_path="sec_data/NVDA-2025/10-K.pdf", output_dir="output")
