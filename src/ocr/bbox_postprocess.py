from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

Box = Dict[str, Any]

# ---------------------------------------------------------------------------
# Label groups
# ---------------------------------------------------------------------------

TITLE_LABELS = {"doc_title", "paragraph_title", "figure_title"}
IGNORED_LABELS = {"footer"}
# Text boxes must stay independent – never merged with others
STANDALONE_LABELS = {"text", "table"}


# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------


def _coord(box: Box):
    """Return (x1, y1, x2, y2) as floats."""
    c = box["coordinate"]
    return float(c[0]), float(c[1]), float(c[2]), float(c[3])


def _union(box_a: Box, box_b: Box) -> Box:
    """Return a new box that is the bounding union of box_a and box_b."""
    ax1, ay1, ax2, ay2 = _coord(box_a)
    bx1, by1, bx2, by2 = _coord(box_b)
    merged_coord = [
        min(ax1, bx1),
        min(ay1, by1),
        max(ax2, bx2),
        max(ay2, by2),
    ]
    return {
        "cls_id": box_a["cls_id"],
        "label": box_a["label"],
        "score": max(box_a["score"], box_b["score"]),
        "coordinate": merged_coord,
    }


def _overlap(box_a: Box, box_b: Box) -> bool:
    """Return True if box_a and box_b have any overlapping area."""
    ax1, ay1, ax2, ay2 = _coord(box_a)
    bx1, by1, bx2, by2 = _coord(box_b)
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


# ---------------------------------------------------------------------------
# Rule 1 – filter footer
# ---------------------------------------------------------------------------


def filter_footer(boxes: List[Box]) -> List[Box]:
    """Drop all boxes whose label is in IGNORED_LABELS (e.g. 'footer')."""
    return [b for b in boxes if b["label"] not in IGNORED_LABELS]


# ---------------------------------------------------------------------------
# Rule 2 – merge overlapping boxes
# ---------------------------------------------------------------------------


def merge_overlapping(boxes: List[Box]) -> List[Box]:
    """
    Iteratively union any pair of boxes that overlap.
    Text boxes never initiate or participate in overlap merges (rule 4:
    text must not be merged with other texts or tables).
    Repeats until no more merges are possible.
    """
    changed = True
    while changed:
        changed = False
        merged: List[Box] = []
        used = [False] * len(boxes)

        for i, box_i in enumerate(boxes):
            if used[i]:
                continue
            if box_i["label"] in STANDALONE_LABELS:
                # Text boxes are never merged; carry them forward as-is.
                merged.append(box_i)
                used[i] = True
                continue

            current = box_i
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                box_j = boxes[j]
                if box_j["label"] in STANDALONE_LABELS:
                    continue  # text boxes never get absorbed either
                if _overlap(current, box_j):
                    current = _union(current, box_j)
                    used[j] = True
                    changed = True

            merged.append(current)
            used[i] = True

        boxes = merged

    return boxes


# ---------------------------------------------------------------------------
# Rule 3 – merge title-like boxes with the next content box below
# ---------------------------------------------------------------------------


def merge_titles_forward(boxes: List[Box]) -> List[Box]:
    """
    Greedily collect consecutive title-like boxes (doc_title, paragraph_title,
    figure_title) and merge the whole run into the first text or table that
    follows them.

    Walk top-to-bottom:
    - Accumulate titles into a pending group.
    - As soon as a text or table is encountered, union it with the entire
      pending group and emit the result; clear the group.
    - If a non-title / non-text / non-table box is encountered while titles
      are pending, flush the pending titles as-is first, then emit that box.
    - Any remaining pending titles at the end are emitted as-is.
    """
    boxes = sorted(boxes, key=lambda b: _coord(b)[1])

    result: List[Box] = []
    pending_titles: List[Box] = []

    for box in boxes:
        label = box["label"]

        if label in TITLE_LABELS:
            pending_titles.append(box)

        elif label in ("text", "table") and pending_titles:
            # Merge all accumulated titles into this text/table box.
            merged = box
            for t in pending_titles:
                merged = _union(merged, t)
            merged["label"] = label  # keep text/table as the final label
            result.append(merged)
            pending_titles = []

        else:
            # Non-title, non-text/table box (image, formula, etc.) — flush any
            # pending titles first, then emit this box unchanged.
            result.extend(pending_titles)
            pending_titles = []
            result.append(box)

    # Flush any titles that never found a text/table below them.
    result.extend(pending_titles)

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def postprocess(boxes: List[Box]) -> List[Box]:
    boxes = filter_footer(boxes)
    boxes = merge_overlapping(boxes)
    boxes = merge_titles_forward(boxes)
    return boxes


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "text": (220, 20, 60),  # crimson
    "table": (0, 128, 0),  # green
    "paragraph_title": (255, 165, 0),  # orange
    "doc_title": (30, 144, 255),  # dodger blue
    "figure_title": (148, 0, 211),  # purple
    "image": (0, 206, 209),  # dark turquoise
    "formula": (255, 215, 0),  # gold
    "header": (105, 105, 105),  # dim grey
}
DEFAULT_COLOR = (128, 128, 128)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes(image_path: str, boxes: List[Box], save_path: str) -> None:
    """Draw post-processed bounding boxes on the source image and save."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in _coord(box)]
        label = box["label"]
        score = box["score"]
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        tag = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(tag, FONT, 0.7, 2)
        tag_y = max(y1 - 5, th + 5)
        cv2.rectangle(
            img, (x1, tag_y - th - baseline), (x1 + tw, tag_y + baseline), color, -1
        )
        cv2.putText(img, tag, (x1, tag_y), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"Saved post-processed image → {save_path}")


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_boxes_from_json(json_path: str) -> List[Box]:
    with open(json_path) as f:
        data = json.load(f)
    return data["boxes"]
