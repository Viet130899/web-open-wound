"""
Block 2 — wound segmentation.

Wraps a YOLO11-seg (Ultralytics) checkpoint that segments the wound region.
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


class WoundPredictor:
    def __init__(self, seg_checkpoint, grade_checkpoint=None):
        self.model = YOLO(str(seg_checkpoint))
        # Model classes are the wound grades themselves (DTI, Unstageable,
        # stage-1..stage-4), matching _OW_GRADE_INFO's 0-5 order 1:1 — no
        # separate grade_checkpoint needed.

    def predict_with_overlay(self, img_bgr):
        pixel_per_mm, tag_circle = calibrate_pixel_per_mm(img_bgr, TAG_DIAMETER_MM)

        result = self.model.predict(img_bgr, conf=CONF_THRESHOLD, verbose=False)[0]

        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None and len(result.masks.data) > 0:
            for m in result.masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask[m_resized > 0.5] = 255

        grade = None
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            grade = int(classes[int(np.argmax(confs))])

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_px = int(np.count_nonzero(mask))

        length_px = 0.0
        if cnts:
            best = max(cnts, key=cv2.contourArea)
            (_, _), (rw, rh), _ = cv2.minAreaRect(best)
            length_px = max(rw, rh)

        area_mm2 = round(area_px / (pixel_per_mm ** 2), 2) if pixel_per_mm else 0.0
        length_mm = round(length_px / pixel_per_mm, 2) if pixel_per_mm else 0.0

        overlay = img_bgr.copy()
        colored = np.zeros_like(overlay)
        colored[mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)
        if tag_circle is not None:
            cx, cy, r_px = tag_circle
            cv2.circle(overlay, (cx, cy), r_px, (0, 255, 0), 2)

        result_dict = {
            "mask": mask,
            "overlay_bgr": overlay,
            "area_px": area_px,
            "area_mm2": area_mm2,
            "length_mm": length_mm,
            "pixel_per_mm": pixel_per_mm,
        }
        if grade is not None:
            result_dict["grade"] = grade
        return result_dict
