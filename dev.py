import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"

from pathlib import Path
from paddleocr import LayoutDetection
from bbox_postprocess import postprocess, draw_boxes

model = LayoutDetection(
    model_name="PP-DocLayout_plus-L", device="cpu", enable_hpi=True, threshold=0.3
)

image_path = "sec_data/output_images/page_14.png"
output = model.predict(image_path, batch_size=1, layout_nms=True)

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")

    raw_boxes = res.json["res"]["boxes"]
    merged_boxes = postprocess(raw_boxes)

    stem = Path(image_path).stem
    draw_boxes(image_path, merged_boxes, save_path=f"./output/{stem}_merged.png")
