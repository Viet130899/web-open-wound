"""
Wound detection.

Uses the same YOLO11-seg (Ultralytics) checkpoint as Block 2, but reports the
bounding box of the highest-confidence detection (the "detect" view).
Pixel -> mm conversion is done via the green calibration tag (Block 1).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_BLOCK1_DIR = Path(__file__).resolve().parent.parent / "block1"
if str(_BLOCK1_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK1_DIR))

from tag_segmentation import calibrate_pixel_per_mm  # noqa: E402

TAG_DIAMETER_MM = 15.0
CONF_THRESHOLD = 0.25


class WoundDetector:
    def __init__(self, checkpoint_path):
        self.model = YOLO(str(checkpoint_path))

    def predict(self, img_bgr):
        pixel_per_mm, tag_circle = calibrate_pixel_per_mm(img_bgr, TAG_DIAMETER_MM)

        result = self.model.predict(img_bgr, conf=CONF_THRESHOLD, verbose=False)[0]

        overlay = img_bgr.copy()
        area_px = 0
        length_px = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            best_i = int(np.argmax(confs))
            x1, y1, x2, y2 = (int(v) for v in boxes_xyxy[best_i])
            bw, bh = x2 - x1, y2 - y1
            area_px = int(bw * bh)
            length_px = float(max(bw, bh))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, "wound", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if tag_circle is not None:
            cx, cy, r_px = tag_circle
            cv2.circle(overlay, (cx, cy), r_px, (0, 255, 0), 2)

        area_mm2 = round(area_px / (pixel_per_mm ** 2), 2) if pixel_per_mm else 0.0
        length_mm = round(length_px / pixel_per_mm, 2) if pixel_per_mm else 0.0

        return {
            "overlay_bgr": overlay,
            "area_px": area_px,
            "area_mm2": area_mm2,
            "length_mm": length_mm,
            "pixel_per_mm": pixel_per_mm,
        }
