from __future__ import annotations

import cv2
import numpy as np


def _frame_pixels(image: np.ndarray, frame_width: int) -> np.ndarray:
    top = image[:frame_width, :, :]
    bottom = image[-frame_width:, :, :]
    left = image[:, :frame_width, :]
    right = image[:, -frame_width:, :]
    return np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )


def remove_uniform_border(
    image: np.ndarray,
    *,
    max_crop_ratio: float = 0.12,
    min_content_ratio: float = 0.012,
    color_distance_threshold: float = 18.0,
    padding: int = 2,
) -> np.ndarray:
    """Trim page-colored margins after perspective correction.

    The crop is deliberately conservative: it only removes margins similar to
    the outer frame color and will not trim more than ``max_crop_ratio`` from
    any side.
    """
    height, width = image.shape[:2]
    if height < 24 or width < 24:
        return image

    frame_width = max(2, min(height, width) // 40)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_color = np.median(_frame_pixels(lab, frame_width), axis=0)
    color_distance = np.linalg.norm(lab - border_color, axis=2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    content_mask = (color_distance > color_distance_threshold) | (edges > 0)
    if not np.any(content_mask):
        return image

    row_ratio = np.mean(content_mask, axis=1)
    col_ratio = np.mean(content_mask, axis=0)
    rows = np.where(row_ratio >= min_content_ratio)[0]
    cols = np.where(col_ratio >= min_content_ratio)[0]
    if len(rows) == 0 or len(cols) == 0:
        return image

    top = max(0, int(rows[0]) - padding)
    bottom = min(height, int(rows[-1]) + padding + 1)
    left = max(0, int(cols[0]) - padding)
    right = min(width, int(cols[-1]) + padding + 1)

    max_y_crop = int(round(height * max_crop_ratio))
    max_x_crop = int(round(width * max_crop_ratio))
    if top > max_y_crop:
        top = 0
    if height - bottom > max_y_crop:
        bottom = height
    if left > max_x_crop:
        left = 0
    if width - right > max_x_crop:
        right = width

    if bottom - top < height * 0.5 or right - left < width * 0.5:
        return image
    if top == 0 and bottom == height and left == 0 and right == width:
        return image
    return image[top:bottom, left:right].copy()
