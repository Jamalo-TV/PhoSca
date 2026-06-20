import hashlib
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError(f"Could not write image: {path}")


def calculate_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalized_box_to_pixels(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(width - 1, int(round(float(box["x1"]) * width)))),
        max(0, min(height - 1, int(round(float(box["y1"]) * height)))),
        max(0, min(width, int(round(float(box["x2"]) * width)))),
        max(0, min(height, int(round(float(box["y2"]) * height)))),
    )


def pixel_box_to_normalized(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> dict[str, float]:
    return {
        "x1": max(0.0, min(1.0, x / image_width)),
        "y1": max(0.0, min(1.0, y / image_height)),
        "x2": max(0.0, min(1.0, (x + w) / image_width)),
        "y2": max(0.0, min(1.0, (y + h) / image_height)),
    }


def box_center(box: dict[str, float]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2.0, (float(box["y1"]) + float(box["y2"])) / 2.0)


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    x_left = max(float(a["x1"]), float(b["x1"]))
    y_top = max(float(a["y1"]), float(b["y1"]))
    x_right = min(float(a["x2"]), float(b["x2"]))
    y_bottom = min(float(a["y2"]), float(b["y2"]))
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (float(a["x2"]) - float(a["x1"])) * (float(a["y2"]) - float(a["y1"]))
    area_b = (float(b["x2"]) - float(b["x1"])) * (float(b["y2"]) - float(b["y1"]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

