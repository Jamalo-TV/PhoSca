from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.image_ops import pixel_box_to_normalized
from app.pipeline.quality import evaluate_segmentation_geometry


@dataclass(frozen=True)
class SegmentationDetection:
    bounding_box: dict[str, float]
    mask: dict
    confidence: float
    aspect_ratio: float
    geometry_valid: bool
    review_reasons: list[str]


@dataclass(frozen=True)
class SegmentationResult:
    detections: list[SegmentationDetection]
    metadata: dict


def _polygon_from_contour(contour: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)
    if len(points) < 4:
        x, y, w, h = cv2.boundingRect(contour)
        points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return [{"x": float(x) / width, "y": float(y) / height} for x, y in points[:12]]


def detect_photos_classical(image: np.ndarray, *, min_area_ratio: float = 0.02) -> SegmentationResult:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[SegmentationDetection] = []
    image_area = width * height
    for contour in contours:
        area = cv2.contourArea(contour)
        if area / image_area < min_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20:
            continue
        quality = evaluate_segmentation_geometry(
            w,
            h,
            min_aspect_ratio=0.5,
            max_aspect_ratio=3.0,
        )
        detections.append(
            SegmentationDetection(
                bounding_box=pixel_box_to_normalized(x, y, w, h, width, height),
                mask={"polygon": _polygon_from_contour(contour, width, height)},
                confidence=0.75,
                aspect_ratio=quality.aspect_ratio,
                geometry_valid=quality.geometry_valid,
                review_reasons=quality.review_reasons,
            )
        )

    detections.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    return SegmentationResult(
        detections=detections,
        metadata={"model": "classical_contour", "inference_time_ms": 0.0, "fallback_used": True},
    )


def _parse_simple_box_outputs(outputs: list[np.ndarray], width: int, height: int, settings: Settings) -> list[SegmentationDetection]:
    detections: list[SegmentationDetection] = []
    for output in outputs:
        array = np.asarray(output)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or array.shape[-1] < 5:
            continue
        for row in array:
            confidence = float(row[4])
            if confidence < settings.yolo_confidence_threshold:
                continue
            x1, y1, x2, y2 = [float(value) for value in row[:4]]
            if max(x1, y1, x2, y2) > 1.5:
                x1, x2 = x1 / width, x2 / width
                y1, y2 = y1 / height, y2 / height
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            quality = evaluate_segmentation_geometry(
                w,
                h,
                min_aspect_ratio=settings.segmentation_min_aspect_ratio,
                max_aspect_ratio=settings.segmentation_max_aspect_ratio,
            )
            box = {
                "x1": max(0.0, min(1.0, x1)),
                "y1": max(0.0, min(1.0, y1)),
                "x2": max(0.0, min(1.0, x2)),
                "y2": max(0.0, min(1.0, y2)),
            }
            detections.append(
                SegmentationDetection(
                    bounding_box=box,
                    mask={"polygon": [
                        {"x": box["x1"], "y": box["y1"]},
                        {"x": box["x2"], "y": box["y1"]},
                        {"x": box["x2"], "y": box["y2"]},
                        {"x": box["x1"], "y": box["y2"]},
                    ]},
                    confidence=confidence,
                    aspect_ratio=quality.aspect_ratio,
                    geometry_valid=quality.geometry_valid,
                    review_reasons=quality.review_reasons,
                )
            )
    return detections


def segment_album_page(image: np.ndarray, settings: Settings) -> SegmentationResult:
    metadata: dict = {
        "model_path": str(settings.yolo_model_path),
        "confidence_threshold": settings.yolo_confidence_threshold,
    }
    if Path(settings.yolo_model_path).exists():
        try:
            import onnxruntime as ort

            started = perf_counter()
            session = ort.InferenceSession(str(settings.yolo_model_path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            target_height = int(input_shape[2] if isinstance(input_shape[2], int) else 640)
            target_width = int(input_shape[3] if isinstance(input_shape[3], int) else 640)
            resized = cv2.resize(image, (target_width, target_height))
            tensor = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            outputs = session.run(None, {input_name: tensor[np.newaxis, ...]})
            detections = _parse_simple_box_outputs(outputs, image.shape[1], image.shape[0], settings)
            metadata.update(
                {
                    "model": "yolov8_seg_onnx",
                    "onnxruntime_version": ort.__version__,
                    "inference_time_ms": round((perf_counter() - started) * 1000, 2),
                    "fallback_used": False,
                }
            )
            if detections:
                return SegmentationResult(detections=detections, metadata=metadata)
            metadata["onnx_parse_warning"] = "No supported detections were parsed from model outputs."
        except Exception as exc:  # noqa: BLE001 - model loading should fail into reviewable metadata.
            metadata["onnx_error"] = str(exc)

    if not settings.enable_classical_segmentation_fallback:
        return SegmentationResult(detections=[], metadata={**metadata, "fallback_used": False})

    fallback = detect_photos_classical(image)
    return SegmentationResult(detections=fallback.detections, metadata={**metadata, **fallback.metadata})
