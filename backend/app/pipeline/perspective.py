from pathlib import Path

import cv2
import numpy as np

from app.pipeline.image_ops import normalized_box_to_pixels, save_jpeg
from app.pipeline.border_removal import remove_uniform_border


def _points_from_mask(mask: dict | None, width: int, height: int) -> np.ndarray | None:
    if not mask:
        return None
    polygon = mask.get("polygon")
    if not polygon:
        return None
    points = np.array(
        [[float(point["x"]) * width, float(point["y"]) * height] for point in polygon],
        dtype=np.float32,
    )
    if len(points) < 4:
        return None
    if len(points) == 4:
        return points

    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
    perimeter = cv2.arcLength(hull, True)
    for epsilon_ratio in (0.01, 0.02, 0.035, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if len(approx) == 4:
            return approx.astype(np.float32)

    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.float32)


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    summed = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(summed)]
    rect[2] = points[np.argmax(summed)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def crop_and_correct_photo(
    page_image: np.ndarray,
    bounding_box: dict,
    mask: dict | None = None,
    *,
    trim_border: bool = True,
) -> np.ndarray:
    height, width = page_image.shape[:2]
    src_points = _points_from_mask(mask, width, height)
    if src_points is None:
        x1, y1, x2, y2 = normalized_box_to_pixels(bounding_box, width, height)
        src_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    src_points = _order_points(src_points)
    top_width = np.linalg.norm(src_points[1] - src_points[0])
    bottom_width = np.linalg.norm(src_points[2] - src_points[3])
    left_height = np.linalg.norm(src_points[3] - src_points[0])
    right_height = np.linalg.norm(src_points[2] - src_points[1])
    target_width = max(1, int(round(max(top_width, bottom_width))))
    target_height = max(1, int(round(max(left_height, right_height))))
    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_points, destination)
    corrected = cv2.warpPerspective(page_image, matrix, (target_width, target_height))
    if trim_border:
        corrected = remove_uniform_border(corrected)
    return corrected


def save_corrected_photo(page_image: np.ndarray, bounding_box: dict, mask: dict | None, output_path: Path) -> None:
    corrected = crop_and_correct_photo(page_image, bounding_box, mask)
    save_jpeg(output_path, corrected)
