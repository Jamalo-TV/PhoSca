from dataclasses import dataclass
from math import hypot
from pathlib import Path

import numpy as np

from app.config import Settings
from app.models import OCRTextType
from app.pipeline.image_ops import box_center, box_iou


@dataclass(frozen=True)
class OCRBlock:
    text: str
    bounding_box: dict[str, float]
    confidence: float
    engine: str


@dataclass(frozen=True)
class ClassifiedOCRBlock:
    text: str
    bounding_box: dict[str, float]
    confidence: float
    engine: str
    text_type: OCRTextType
    photo_id: str | None
    reason: str


def _normalize_paddle_box(points: list, width: int, height: int) -> dict[str, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return {
        "x1": max(0.0, min(1.0, min(xs) / width)),
        "y1": max(0.0, min(1.0, min(ys) / height)),
        "x2": max(0.0, min(1.0, max(xs) / width)),
        "y2": max(0.0, min(1.0, max(ys) / height)),
    }


def run_paddle_ocr(image: np.ndarray, settings: Settings) -> tuple[list[OCRBlock], dict]:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # noqa: BLE001 - dependency is optional until installed in runtime image.
        return [], {"ocr_engine": "paddleocr", "available": False, "error": str(exc)}

    height, width = image.shape[:2]
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, det_model_dir=str(settings.paddleocr_model_path))
    result = ocr.ocr(image[:, :, ::-1], cls=True)
    blocks: list[OCRBlock] = []
    for line in result or []:
        for entry in line or []:
            points, text_info = entry
            text, confidence = text_info
            blocks.append(
                OCRBlock(
                    text=str(text),
                    bounding_box=_normalize_paddle_box(points, width, height),
                    confidence=float(confidence),
                    engine="paddleocr",
                )
            )
    return blocks, {"ocr_engine": "paddleocr", "available": True, "block_count": len(blocks)}


def run_sidecar_ocr(image_path: Path) -> tuple[list[OCRBlock], dict]:
    sidecar_path = image_path.with_suffix(".ocr.json")
    if not sidecar_path.exists():
        return [], {"ocr_engine": "sidecar", "available": False}
    import json

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    blocks = [
        OCRBlock(
            text=str(item["text"]),
            bounding_box=item["bounding_box"],
            confidence=float(item.get("confidence", 1.0)),
            engine="sidecar",
        )
        for item in payload.get("blocks", [])
    ]
    return blocks, {"ocr_engine": "sidecar", "available": True, "block_count": len(blocks)}


def classify_ocr_blocks(blocks: list[OCRBlock], photos: list[dict]) -> list[ClassifiedOCRBlock]:
    classified: list[ClassifiedOCRBlock] = []
    for block in blocks:
        best_overlap_photo: dict | None = None
        best_overlap = 0.0
        for photo in photos:
            overlap = box_iou(block.bounding_box, photo["bounding_box"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_photo = photo

        if best_overlap_photo and best_overlap > 0.1:
            classified.append(
                ClassifiedOCRBlock(
                    text=block.text,
                    bounding_box=block.bounding_box,
                    confidence=block.confidence,
                    engine=block.engine,
                    text_type=OCRTextType.caption,
                    photo_id=str(best_overlap_photo["id"]),
                    reason=f"text overlaps photo by IoU {best_overlap:.2f}",
                )
            )
            continue

        center = box_center(block.bounding_box)
        if block.bounding_box["y2"] <= 0.15:
            classified.append(
                ClassifiedOCRBlock(
                    text=block.text,
                    bounding_box=block.bounding_box,
                    confidence=block.confidence,
                    engine=block.engine,
                    text_type=OCRTextType.directory_name,
                    photo_id=None,
                    reason="text is in top 15% of page without photo overlap",
                )
            )
            continue

        nearest_photo: dict | None = None
        nearest_distance = float("inf")
        for photo in photos:
            px, py = box_center(photo["bounding_box"])
            distance = hypot(center[0] - px, center[1] - py)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_photo = photo

        if nearest_photo is not None and nearest_distance < 0.35:
            classified.append(
                ClassifiedOCRBlock(
                    text=block.text,
                    bounding_box=block.bounding_box,
                    confidence=block.confidence,
                    engine=block.engine,
                    text_type=OCRTextType.caption,
                    photo_id=str(nearest_photo["id"]),
                    reason=f"text linked to nearest photo at normalized distance {nearest_distance:.2f}",
                )
            )
        else:
            classified.append(
                ClassifiedOCRBlock(
                    text=block.text,
                    bounding_box=block.bounding_box,
                    confidence=block.confidence,
                    engine=block.engine,
                    text_type=OCRTextType.unknown,
                    photo_id=None,
                    reason="text did not overlap a photo and was not near a photo or page header",
                )
            )
    return classified

