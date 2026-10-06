import cv2
import numpy as np


def segment_green_tag(image):
    """
    Segment green circular tag.

    Args:
        image (str | np.ndarray):
            - Nếu là str: đường dẫn ảnh.
            - Nếu là np.ndarray: ảnh BGR đã đọc bằng OpenCV.

    Returns:
        np.ndarray: Binary mask (uint8, 0 hoặc 255).
    """
    # Đọc ảnh nếu truyền vào là path
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image}")
    else:
        img_bgr = image

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([55, 70, 50])
    upper = np.array([95, 255, 230])

    mask = cv2.inRange(hsv, lower, upper)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    H, W = mask.shape

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.0005 * H * W:
            continue

        per = cv2.arcLength(c, True)
        if per == 0:
            continue

        circularity = 4 * np.pi * area / (per * per)
        if circularity < 0.8:
            continue

        if area > best_area:
            best = c
            best_area = area

    clean_mask = np.zeros_like(mask)
    if best is not None:
        cv2.drawContours(clean_mask, [best], -1, 255, cv2.FILLED)

    return clean_mask


def calibrate_pixel_per_mm(image, tag_diameter_mm=15.0):
    """
    Tính tỷ lệ pixel/mm dựa trên tag tròn xanh lá làm vật chuẩn.

    Args:
        image (str | np.ndarray): path hoặc ảnh BGR.
        tag_diameter_mm (float): đường kính thật của tag (mm).

    Returns:
        tuple[float | None, tuple | None]:
            - pixel_per_mm, hoặc None nếu không tìm thấy tag.
            - (center_x, center_y, radius_px) của tag, hoặc None.
    """
    mask = segment_green_tag(image)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    best = max(cnts, key=cv2.contourArea)
    (cx, cy), radius_px = cv2.minEnclosingCircle(best)
    if radius_px <= 0:
        return None, None

    diameter_px = 2 * radius_px
    pixel_per_mm = diameter_px / tag_diameter_mm
    return pixel_per_mm, (int(cx), int(cy), int(radius_px))


if __name__ == "__main__":
    import os
    import glob
    import cv2
    import matplotlib.pyplot as plt

    folder = "/Users/long/workspace/tag-segmentation/20260729_4aa414fc/images"

    # Đọc tất cả ảnh trong folder
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))

    image_paths = sorted(image_paths)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Segment
        mask = segment_green_tag(img)

        # Overlay mask lên ảnh
        overlay = img.copy()
        overlay[mask > 0] = (0, 0, 255)  # đỏ
        vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        # Tìm contour từ mask
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if cnts is not None and len(cnts) > 0:
            best = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(best)
            center = (int(x), int(y))
            radius = int(radius)

            # Vẽ hình tròn lên ảnh
            cv2.circle(vis, center, radius, (0, 255, 0), 2)  # xanh lá
            
        # ================= Visualization =================
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Overlay + Circle")
        axes[1].axis("off")

        plt.suptitle(os.path.basename(img_path))
        plt.tight_layout()
        plt.show()