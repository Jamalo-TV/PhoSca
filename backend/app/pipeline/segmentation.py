from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.album_windows import ALBUM_WINDOW_OCCLUDED, ALBUM_WINDOW_SOURCE, ALBUM_WINDOW_STOCK_UNCERTAIN, album_window_leaves
from app.pipeline.geometry import (
    normalized_polygon_overlap,
    normalized_vertices_to_pixels,
    order_quad_cyclic,
    pixel_vertices_to_normalized,
    quad_from_polygon,
)
from app.pipeline.image_ops import normalized_box_to_pixels, pixel_box_to_normalized
from app.pipeline.mounted_prints import mounted_print_evidence
from app.pipeline.page_deskew import album_rotation, deskewed_view
from app.pipeline.quality import evaluate_segmentation_geometry
from app.pipeline.segmentation_filter import paper_and_edge_rejection_reasons, paper_around_nested_print

_CLASSICAL_MIN_AREA_RATIO = 0.02


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


@dataclass(frozen=True)
class _ContourCandidate:
    box: tuple[int, int, int, int]
    contour: np.ndarray
    area_ratio: float
    source: str = "edge"


@dataclass(frozen=True)
class _AxisLine:
    position: float
    span_start: float
    span_end: float
    length: float


@dataclass(frozen=True)
class _CandidateAssessment:
    confidence: float
    edge_support: float
    border_contrast: float
    area_ratio: float
    rectangularity: float
    interior_texture: float
    edge_supported_sides: int
    outline_supported_sides: int
    exterior_background_support: float
    exterior_background_sides: int
    rejected: bool
    rejection_reasons: list[str]


@dataclass(frozen=True)
class _LetterboxTransform:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    scale: float
    resized_width: int
    resized_height: int
    pad_x: float
    pad_y: float


@lru_cache(maxsize=4)
def _cached_onnx_session(model_path: str, modified_ns: int):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available and hasattr(ort, "preload_dlls"):
        # On Windows the CUDA DLLs bundled with PyTorch are not necessarily on
        # PATH.  ONNX Runtime can preload them before the CUDA provider is
        # created; failures still fall back safely to the CPU provider.
        try:
            ort.preload_dlls()
        except Exception:  # pragma: no cover - environment-specific DLL setup
            pass
    providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
    return ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])


def _frame_pixels(image: np.ndarray, frame_width: int) -> np.ndarray:
    top = image[:frame_width, :, :]
    bottom = image[-frame_width:, :, :]
    left = image[:, :frame_width, :]
    right = image[:, -frame_width:, :]
    return np.concatenate(
        [
            top.reshape(-1, image.shape[2]),
            bottom.reshape(-1, image.shape[2]),
            left.reshape(-1, image.shape[2]),
            right.reshape(-1, image.shape[2]),
        ],
        axis=0,
    )


def _clamp_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = points.astype(np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, height - 1))
    return clipped


def _normalized_polygon(points: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
    normalized = pixel_vertices_to_normalized(points, width, height)
    return [
        {
            "x": float(x),
            "y": float(y),
        }
        for x, y in normalized
    ]


def _build_letterbox_transform(
    source_width: int, source_height: int, input_width: int, input_height: int
) -> _LetterboxTransform:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source image dimensions must be positive for YOLO letterboxing.")
    scale = min(input_width / source_width, input_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    pad_x = float((input_width - resized_width) // 2)
    pad_y = float((input_height - resized_height) // 2)
    return _LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        input_width=input_width,
        input_height=input_height,
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def _letterbox_image(image: np.ndarray, input_width: int, input_height: int) -> tuple[np.ndarray, _LetterboxTransform]:
    source_height, source_width = image.shape[:2]
    transform = _build_letterbox_transform(source_width, source_height, input_width, input_height)
    resized = cv2.resize(
        image,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_AREA if transform.scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((input_height, input_width, image.shape[2]), 114, dtype=image.dtype)
    left = int(transform.pad_x)
    top = int(transform.pad_y)
    canvas[top : top + transform.resized_height, left : left + transform.resized_width] = resized
    return canvas, transform


def _model_points_to_source(points: np.ndarray, transform: _LetterboxTransform) -> np.ndarray:
    mapped = points.astype(np.float32).copy()
    scale = max(transform.scale, 1e-6)
    mapped[:, 0] = (mapped[:, 0] - transform.pad_x) / scale
    mapped[:, 1] = (mapped[:, 1] - transform.pad_y) / scale
    return _clamp_points(mapped, transform.source_width, transform.source_height)


def _source_box_from_input_box(
    input_box: dict[str, float], transform: _LetterboxTransform, *, native: bool
) -> dict[str, float] | None:
    width = transform.resized_width if native else transform.scale * transform.source_width
    height = transform.resized_height if native else transform.scale * transform.source_height
    source_box = {
        "x1": (input_box["x1"] * transform.input_width - transform.pad_x) / width,
        "y1": (input_box["y1"] * transform.input_height - transform.pad_y) / height,
        "x2": (input_box["x2"] * transform.input_width - transform.pad_x) / width,
        "y2": (input_box["y2"] * transform.input_height - transform.pad_y) / height,
    }
    return _clip_normalized_box(source_box)


def _order_points(points: np.ndarray) -> np.ndarray:
    return order_quad_cyclic(points)


def _enhanced_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=18, sigmaY=18)
    normalized = cv2.addWeighted(enhanced, 1.45, background, -0.45, 24)
    return cv2.GaussianBlur(normalized, (3, 3), 0)


def _edge_support_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    lower = int(max(24, min(120, median * 0.55)))
    upper = int(max(lower + 30, min(220, median * 1.35)))
    edges = cv2.Canny(gray, lower, upper)
    fixed_edges = cv2.Canny(gray, 45, 140)
    edges = cv2.bitwise_or(edges, fixed_edges)
    return cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)


def _quadrilateral_from_contour(contour: np.ndarray) -> tuple[np.ndarray, str]:
    hull = cv2.convexHull(contour.reshape(-1, 1, 2))
    perimeter = cv2.arcLength(hull, True)
    hull_area = max(1.0, cv2.contourArea(hull))

    for epsilon_ratio in (0.012, 0.018, 0.025, 0.035, 0.05, 0.075):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        approx_area = cv2.contourArea(approx)
        if approx_area < hull_area * 0.78:
            continue
        return _order_points(approx.reshape(-1, 2).astype(np.float32)), "classical_border_quad"

    rect = cv2.minAreaRect(hull)
    return _order_points(cv2.boxPoints(rect).astype(np.float32)), "classical_min_area_rect"


def _quadrilateral_geometry_dimensions(
    points: np.ndarray, fallback_width: int, fallback_height: int
) -> tuple[float, float]:
    if len(points) != 4:
        return float(fallback_width), float(fallback_height)
    ordered = _order_points(points.astype(np.float32))
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    corrected_width = max(top_width, bottom_width)
    corrected_height = max(left_height, right_height)
    if corrected_width <= 0 or corrected_height <= 0:
        return float(fallback_width), float(fallback_height)
    return float(corrected_width), float(corrected_height)


def _line_from_points(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    a = float(p1[1] - p2[1])
    b = float(p2[0] - p1[0])
    c = float(p1[0] * p2[1] - p2[0] * p1[1])
    norm = float(np.hypot(a, b))
    if norm == 0:
        return 0.0, 0.0, 0.0
    return a / norm, b / norm, c / norm


def _line_distance(line: tuple[float, float, float], point: np.ndarray) -> float:
    return abs(line[0] * float(point[0]) + line[1] * float(point[1]) + line[2])


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    fit = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([float(fit[0]), float(fit[1])], dtype=np.float32)
    origin = np.array([float(fit[2]), float(fit[3])], dtype=np.float32)
    if float(np.linalg.norm(direction)) == 0:
        return None
    return direction / np.linalg.norm(direction), origin


def _intersect_parametric_lines(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    d1, p1 = first
    d2, p2 = second
    matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float32)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-5:
        return None
    t, _ = np.linalg.solve(matrix, p2 - p1)
    return p1 + d1 * t


def _segment_near_side(
    line: np.ndarray,
    side_start: np.ndarray,
    side_end: np.ndarray,
    *,
    distance_tolerance: float,
    angle_tolerance_cos: float,
) -> bool:
    start = line[:2].astype(np.float32)
    end = line[2:].astype(np.float32)
    segment = end - start
    side = side_end - side_start
    segment_length = float(np.linalg.norm(segment))
    side_length = float(np.linalg.norm(side))
    if segment_length < max(16.0, side_length * 0.18) or side_length <= 0:
        return False

    segment_direction = segment / segment_length
    side_direction = side / side_length
    if abs(float(np.dot(segment_direction, side_direction))) < angle_tolerance_cos:
        return False

    side_line = _line_from_points(side_start, side_end)
    midpoint = (start + end) / 2.0
    if _line_distance(side_line, midpoint) > distance_tolerance:
        return False

    projection_start = float(np.dot(start - side_start, side_direction))
    projection_end = float(np.dot(end - side_start, side_direction))
    overlap_start = max(min(projection_start, projection_end), -side_length * 0.12)
    overlap_end = min(max(projection_start, projection_end), side_length * 1.12)
    return overlap_end - overlap_start > side_length * 0.12


def _refine_quadrilateral_with_lines(
    image: np.ndarray,
    points: np.ndarray,
    line_cache: dict[tuple[int, int, int, int], np.ndarray | None],
) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    ordered = _order_points(points.astype(np.float32))
    x, y, w, h = cv2.boundingRect(ordered.astype(np.int32))
    pad = int(max(12, min(width, height) * 0.025))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return ordered, False

    roi = (x1, y1, x2, y2)
    if roi not in line_cache:
        roi_gray = _enhanced_gray(image[y1:y2, x1:x2])
        try:
            detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except Exception:
            detector = cv2.createLineSegmentDetector()
        if len(line_cache) >= 16:
            line_cache.pop(next(iter(line_cache)))
        line_cache[roi] = detector.detect(roi_gray)[0]
    detected = line_cache[roi]
    if detected is None or len(detected) < 4:
        return ordered, False

    lines = detected.reshape(-1, 4).astype(np.float32)
    lines[:, [0, 2]] += x1
    lines[:, [1, 3]] += y1
    # The L1 length is an upper bound. Leave near-threshold cases to the scalar check.
    line_length_upper_bounds = np.abs(lines[:, 2:] - lines[:, :2]).sum(axis=1, dtype=np.float64)

    fitted_sides: list[tuple[np.ndarray, np.ndarray] | None] = []
    for index in range(4):
        side_start = ordered[index]
        side_end = ordered[(index + 1) % 4]
        side_length = float(np.linalg.norm(side_end - side_start))
        distance_tolerance = max(8.0, side_length * 0.06, min(width, height) * 0.012)
        support_points: list[np.ndarray] = []
        minimum_length = max(16.0, side_length * 0.18) * (1.0 - 1e-6)
        for line in lines[line_length_upper_bounds >= minimum_length]:
            if _segment_near_side(
                line,
                side_start,
                side_end,
                distance_tolerance=distance_tolerance,
                angle_tolerance_cos=0.92,
            ):
                support_points.extend([line[:2], line[2:]])
        fitted_sides.append(_fit_line(np.array(support_points, dtype=np.float32)) if len(support_points) >= 4 else None)

    if any(side is None for side in fitted_sides):
        return ordered, False

    intersections: list[np.ndarray] = []
    for index in range(4):
        previous_side = fitted_sides[index - 1]
        current_side = fitted_sides[index]
        assert previous_side is not None and current_side is not None
        point = _intersect_parametric_lines(previous_side, current_side)
        if point is None:
            return ordered, False
        intersections.append(point)

    refined = _order_points(_clamp_points(np.array(intersections, dtype=np.float32), width, height))
    if not cv2.isContourConvex(refined.reshape(-1, 1, 2)):
        return ordered, False

    original_area = max(1.0, cv2.contourArea(ordered.reshape(-1, 1, 2)))
    refined_area = cv2.contourArea(refined.reshape(-1, 1, 2))
    if refined_area < original_area * 0.72 or refined_area > original_area * 1.28:
        return ordered, False
    return refined, True


def _mean_score(scores: list[float]) -> float:
    return float(np.mean(scores)) if scores else 0.0


def _edge_support_scores(edge_map: np.ndarray, points: np.ndarray) -> list[float]:
    height, width = edge_map.shape[:2]
    ordered = _order_points(points.astype(np.float32))
    side_scores: list[float] = []
    for index in range(4):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        length = float(np.linalg.norm(end - start))
        samples = max(12, min(120, int(length / 6)))
        hits = 0
        for t in np.linspace(0.06, 0.94, samples):
            point = start + (end - start) * float(t)
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < width and 0 <= y < height and edge_map[y, x] > 0:
                hits += 1
        side_scores.append(hits / samples)
    return side_scores


def _border_contrast_scores(lab: np.ndarray, points: np.ndarray) -> list[float]:
    height, width = lab.shape[:2]
    ordered = _order_points(points.astype(np.float32))
    centroid = np.mean(ordered, axis=0)
    offset = max(3.0, min(width, height) * 0.006)
    side_scores: list[float] = []
    for index in range(4):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 0:
            continue
        normal = np.array([-edge[1], edge[0]], dtype=np.float32) / length
        distances: list[float] = []
        for t in np.linspace(0.12, 0.88, max(10, min(80, int(length / 10)))):
            midpoint = start + edge * float(t)
            first = midpoint + normal * offset
            second = midpoint - normal * offset
            if np.linalg.norm(first - centroid) < np.linalg.norm(second - centroid):
                inside, outside = first, second
            else:
                inside, outside = second, first
            ix = int(round(float(np.clip(inside[0], 0, width - 1))))
            iy = int(round(float(np.clip(inside[1], 0, height - 1))))
            ox = int(round(float(np.clip(outside[0], 0, width - 1))))
            oy = int(round(float(np.clip(outside[1], 0, height - 1))))
            distances.append(float(np.linalg.norm(lab[iy, ix].astype(np.float32) - lab[oy, ox])))
        if distances:
            side_scores.append(min(1.0, float(np.median(distances)) / 30.0))
        else:
            side_scores.append(0.0)
    return side_scores


def _frame_background_lab_model(lab: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = lab.shape[:2]
    frame_width = max(6, min(width, height) // 24)
    frame = _frame_pixels(lab, frame_width).astype(np.float32)
    background = np.median(frame, axis=0)
    distances = np.linalg.norm(frame - background, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = median_distance + max(12.0, 2.6 * mad)
    return background.astype(np.float32), threshold


def _exterior_background_scores(
    lab: np.ndarray, points: np.ndarray, lab_background: tuple[np.ndarray, float]
) -> list[float]:
    height, width = lab.shape[:2]
    background, threshold = lab_background
    ordered = _order_points(points.astype(np.float32))
    centroid = np.mean(ordered, axis=0)
    min_dimension = min(width, height)
    offsets = (max(4.0, min_dimension * 0.008), max(8.0, min_dimension * 0.018))
    side_scores: list[float] = []

    for index in range(4):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 0:
            side_scores.append(0.0)
            continue

        normal = np.array([-edge[1], edge[0]], dtype=np.float32) / length
        middle = start + edge * 0.5
        first = middle + normal * offsets[0]
        second = middle - normal * offsets[0]
        outside_direction = normal if np.linalg.norm(first - centroid) > np.linalg.norm(second - centroid) else -normal
        hits = 0
        samples = 0
        for t in np.linspace(0.14, 0.86, max(10, min(70, int(length / 12)))):
            midpoint = start + edge * float(t)
            for offset in offsets:
                outside = midpoint + outside_direction * offset
                x = int(round(float(outside[0])))
                y = int(round(float(outside[1])))
                if not (0 <= x < width and 0 <= y < height):
                    continue
                samples += 1
                if float(np.linalg.norm(lab[y, x] - background)) <= threshold:
                    hits += 1
        side_scores.append(hits / samples if samples else 0.0)
    return side_scores


def _rectangularity_score(points: np.ndarray) -> float:
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(contour))
    rect = cv2.minAreaRect(contour)
    box_area = float(rect[1][0] * rect[1][1])
    if box_area <= 0:
        return 0.0
    return float(max(0.0, min(1.0, area / box_area)))


def _interior_texture_score(laplacian: np.ndarray, points: np.ndarray) -> float:
    height, width = laplacian.shape
    _, _, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
    inset = max(3, min(21, int(round(min(box_width, box_height) * 0.06))))
    polygon = _clamp_points(points, width, height).astype(np.int32)
    x, y, polygon_width, polygon_height = cv2.boundingRect(polygon)
    radius = inset // 2
    left, top = max(0, x - radius), max(0, y - radius)
    right, bottom = min(width, x + polygon_width + radius), min(height, y + polygon_height + radius)
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon - np.array([left, top], dtype=np.int32), 255)
    if cv2.countNonZero(mask) < max(64, int(width * height * 0.002)):
        return 0.0
    kernel = np.ones((inset | 1, inset | 1), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    if cv2.countNonZero(eroded) > 0:
        mask = eroded
    texture = float(np.std(laplacian[top:bottom, left:right][mask > 0]))
    return float(max(0.0, min(1.0, texture / 42.0)))


def _assess_candidate(
    lab: np.ndarray,
    edge_map: np.ndarray,
    lab_background: tuple[np.ndarray, float],
    laplacian: np.ndarray,
    points: np.ndarray,
    *,
    area_ratio: float,
    polygon_source: str,
    detector_source: str,
    refined_by_lines: bool,
) -> _CandidateAssessment:
    edge_scores = _edge_support_scores(edge_map, points)
    border_scores = _border_contrast_scores(lab, points)
    exterior_scores = _exterior_background_scores(lab, points, lab_background)
    edge_support = _mean_score(edge_scores)
    border_contrast = _mean_score(border_scores)
    exterior_background_support = _mean_score(exterior_scores)
    rectangularity = _rectangularity_score(points)
    interior_texture = _interior_texture_score(laplacian, points)
    edge_supported_sides = sum(score >= 0.10 for score in edge_scores)
    outline_supported_sides = sum(
        max(edge_score, border_score, exterior_score * 0.6) >= 0.16
        for edge_score, border_score, exterior_score in zip(edge_scores, border_scores, exterior_scores, strict=False)
    )
    exterior_background_sides = sum(score >= 0.55 for score in exterior_scores)
    confidence = _candidate_confidence(
        edge_support=edge_support,
        border_contrast=border_contrast,
        area_ratio=area_ratio,
        polygon_source=polygon_source,
        refined_by_lines=refined_by_lines,
    )

    rejection_reasons: list[str] = []
    if rectangularity < 0.62:
        rejection_reasons.append("candidate_not_rectangular_enough")
    if area_ratio > 0.62:
        rejection_reasons.append("candidate_too_large_for_single_photo")
    image_height, image_width = lab.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
    touches_frame = sum(
        [
            x <= 2,
            y <= 2,
            x + box_width >= image_width - 2,
            y + box_height >= image_height - 2,
        ]
    )
    if (
        area_ratio > 0.18
        and touches_frame >= 1
        and (box_width / image_width > 0.62 or box_height / image_height > 0.62)
    ):
        rejection_reasons.append("frame_touching_container_candidate")
    if edge_support < 0.08 and border_contrast < 0.16 and interior_texture < 0.32:
        rejection_reasons.append("weak_photo_boundary")
    if area_ratio < 0.16 and outline_supported_sides <= 2 and exterior_background_sides <= 1:
        rejection_reasons.append("incomplete_photo_outline")
    if (
        detector_source == "contrast_region"
        and edge_support < 0.12
        and border_contrast < 0.22
        and interior_texture < 0.16
    ):
        rejection_reasons.append("smooth_contrast_false_positive")
    contour_width, contour_height = _quadrilateral_geometry_dimensions(points, 0, 0)
    aspect_ratio = contour_width / contour_height if contour_height > 0 else 0.0
    if detector_source == "contrast_region" and interior_texture < 0.08 and (aspect_ratio < 0.55 or aspect_ratio > 2.1):
        rejection_reasons.append("smooth_extreme_aspect_false_positive")
    if polygon_source == "classical_min_area_rect" and edge_support < 0.16 and border_contrast < 0.24:
        rejection_reasons.append("unsupported_min_area_rectangle")
    if (
        detector_source in {"edge_contour", "line_pairs"}
        and area_ratio < 0.12
        and exterior_background_support < 0.18
        and exterior_background_sides == 0
    ):
        rejection_reasons.append("internal_edge_without_album_background")

    if rejection_reasons:
        confidence = round(max(0.25, confidence - 0.18), 3)

    return _CandidateAssessment(
        confidence=confidence,
        edge_support=edge_support,
        border_contrast=border_contrast,
        area_ratio=area_ratio,
        rectangularity=rectangularity,
        interior_texture=interior_texture,
        edge_supported_sides=edge_supported_sides,
        outline_supported_sides=outline_supported_sides,
        exterior_background_support=exterior_background_support,
        exterior_background_sides=exterior_background_sides,
        rejected=bool(rejection_reasons),
        rejection_reasons=rejection_reasons,
    )


def _candidate_confidence(
    *,
    edge_support: float,
    border_contrast: float,
    area_ratio: float,
    polygon_source: str,
    refined_by_lines: bool,
) -> float:
    size_score = min(1.0, max(0.0, area_ratio / 0.08))
    confidence = 0.42 + 0.28 * edge_support + 0.24 * border_contrast + 0.06 * size_score
    if refined_by_lines:
        confidence += 0.04
    if polygon_source == "classical_min_area_rect":
        confidence -= 0.06
    return round(float(max(0.35, min(0.98, confidence))), 3)


def _box_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[float, float]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0, 0.0
    intersection = (right - left) * (bottom - top)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0, intersection / min(area_a, area_b)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return box[2] * box[3]


def _mostly_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    _, overlap_min = _box_overlap_ratio(inner, outer)
    return overlap_min > 0.85


def _is_page_sized_box(box: tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    x, y, w, h = box
    width_ratio = w / image_width
    height_ratio = h / image_height
    touches = sum(
        [
            x <= 2,
            y <= 2,
            x + w >= image_width - 2,
            y + h >= image_height - 2,
        ]
    )
    return (width_ratio > 0.85 and height_ratio > 0.85) or touches >= 3


def _candidate_from_contour(
    contour: np.ndarray,
    *,
    width: int,
    height: int,
    image_area: int,
    min_area_ratio: float,
    source: str,
) -> _ContourCandidate | None:
    area = cv2.contourArea(contour)
    area_ratio = area / image_area
    if area_ratio < min_area_ratio or area_ratio > 0.75:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h else 0
    if aspect_ratio < 0.4 or aspect_ratio > 2.35:
        return None
    if w < 20 or h < 20 or _is_page_sized_box((x, y, w, h), width, height):
        return None
    return _ContourCandidate(box=(x, y, w, h), contour=contour, area_ratio=area_ratio, source=source)


def _overlap_1d(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _merge_axis_lines(lines: list[_AxisLine], tolerance: float) -> list[_AxisLine]:
    if not lines:
        return []
    merged: list[_AxisLine] = []
    current = sorted(lines, key=lambda line: line.position)
    group: list[_AxisLine] = [current[0]]
    for line in current[1:]:
        if abs(line.position - np.mean([item.position for item in group])) <= tolerance:
            group.append(line)
            continue
        merged.append(_merge_axis_line_group(group))
        group = [line]
    merged.append(_merge_axis_line_group(group))
    return merged


def _merge_axis_line_group(group: list[_AxisLine]) -> _AxisLine:
    total_length = sum(line.length for line in group)
    if total_length <= 0:
        position = float(np.mean([line.position for line in group]))
    else:
        position = float(sum(line.position * line.length for line in group) / total_length)
    return _AxisLine(
        position=position,
        span_start=float(min(line.span_start for line in group)),
        span_end=float(max(line.span_end for line in group)),
        length=float(sum(line.length for line in group)),
    )


def _find_line_pair_candidates(image: np.ndarray, min_area_ratio: float) -> list[_ContourCandidate]:
    height, width = image.shape[:2]
    image_area = width * height
    edge_map = _edge_support_map(image)
    min_dimension = min(width, height)
    min_line_length = max(60, int(min_dimension * 0.16))
    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=max(45, int(min_dimension * 0.045)),
        minLineLength=min_line_length,
        maxLineGap=max(12, int(min_dimension * 0.025)),
    )
    if lines is None:
        return []

    horizontal: list[_AxisLine] = []
    vertical: list[_AxisLine] = []
    for raw_line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in raw_line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_line_length:
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180
        if angle <= 24 or angle >= 156:
            horizontal.append(
                _AxisLine(position=(y1 + y2) / 2.0, span_start=min(x1, x2), span_end=max(x1, x2), length=length)
            )
        elif 66 <= angle <= 114:
            vertical.append(
                _AxisLine(position=(x1 + x2) / 2.0, span_start=min(y1, y2), span_end=max(y1, y2), length=length)
            )

    horizontal = _merge_axis_lines(horizontal, max(10.0, min_dimension * 0.018))
    vertical = _merge_axis_lines(vertical, max(10.0, min_dimension * 0.018))
    candidates: list[_ContourCandidate] = []
    for top_index, top in enumerate(horizontal):
        for bottom in horizontal[top_index + 1 :]:
            y1 = min(top.position, bottom.position)
            y2 = max(top.position, bottom.position)
            rect_height = y2 - y1
            if rect_height < min_dimension * 0.10:
                continue
            for left_index, left in enumerate(vertical):
                for right in vertical[left_index + 1 :]:
                    x1 = min(left.position, right.position)
                    x2 = max(left.position, right.position)
                    rect_width = x2 - x1
                    if rect_width < min_dimension * 0.10:
                        continue
                    area_ratio = (rect_width * rect_height) / image_area
                    if area_ratio < min_area_ratio or area_ratio > 0.35:
                        continue
                    aspect_ratio = rect_width / rect_height if rect_height else 0
                    if aspect_ratio < 0.4 or aspect_ratio > 2.35:
                        continue
                    horizontal_overlap = min(
                        _overlap_1d(top.span_start, top.span_end, x1, x2),
                        _overlap_1d(bottom.span_start, bottom.span_end, x1, x2),
                    )
                    vertical_overlap = min(
                        _overlap_1d(left.span_start, left.span_end, y1, y2),
                        _overlap_1d(right.span_start, right.span_end, y1, y2),
                    )
                    if horizontal_overlap < rect_width * 0.35 or vertical_overlap < rect_height * 0.35:
                        continue
                    contour = np.array(
                        [
                            [[int(round(x1)), int(round(y1))]],
                            [[int(round(x2)), int(round(y1))]],
                            [[int(round(x2)), int(round(y2))]],
                            [[int(round(x1)), int(round(y2))]],
                        ],
                        dtype=np.int32,
                    )
                    candidate = _candidate_from_contour(
                        contour,
                        width=width,
                        height=height,
                        image_area=image_area,
                        min_area_ratio=min_area_ratio,
                        source="line_pairs",
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates


def _find_contrast_candidates(image: np.ndarray, min_area_ratio: float) -> list[_ContourCandidate]:
    height, width = image.shape[:2]
    image_area = width * height
    candidates: list[_ContourCandidate] = []

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_width = max(6, min(width, height) // 24)
    background = np.median(_frame_pixels(lab, frame_width), axis=0)
    color_distance = np.linalg.norm(lab - background, axis=2)
    median_distance = float(np.median(color_distance))
    mad = float(np.median(np.abs(color_distance - median_distance)))
    distance_threshold = median_distance + max(12.0, 2.8 * mad)
    saturation = hsv[:, :, 1].astype(np.float32)
    saturation_threshold = float(np.median(saturation)) + 24.0

    mask = (
        (color_distance > distance_threshold)
        | ((saturation > saturation_threshold) & (color_distance > distance_threshold * 0.55))
    ).astype(np.uint8) * 255
    kernel_size = max(5, min(25, (min(width, height) // 55) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        candidate = _candidate_from_contour(
            contour,
            width=width,
            height=height,
            image_area=image_area,
            min_area_ratio=min_area_ratio,
            source="contrast_region",
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _find_contour_candidates(image: np.ndarray, min_area_ratio: float) -> list[_ContourCandidate]:
    height, width = image.shape[:2]
    image_area = width * height
    gray = _enhanced_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates: list[_ContourCandidate] = []

    # Faded album photos vary enough that one Canny setting either splits prints
    # by internal detail or misses low-contrast prints entirely.
    passes = [
        ((50, 150), 7, 2),
        ((80, 180), 7, 2),
        ((50, 150), 15, 2),
    ]
    for canny_thresholds, kernel_size, iterations in passes:
        edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            candidate = _candidate_from_contour(
                contour,
                width=width,
                height=height,
                image_area=image_area,
                min_area_ratio=min_area_ratio,
                source="edge_contour",
            )
            if candidate is not None:
                candidates.append(candidate)

    candidates.extend(_find_contrast_candidates(image, min_area_ratio))
    if not candidates:
        candidates.extend(_find_line_pair_candidates(image, min_area_ratio))
    return candidates


def _consolidate_candidates(candidates: list[_ContourCandidate]) -> list[_ContourCandidate]:
    candidates = _discard_container_candidates(candidates)
    selected: list[_ContourCandidate] = []
    for candidate in sorted(
        candidates, key=lambda item: item.area_ratio + (0.01 if item.source == "edge_contour" else 0.0), reverse=True
    ):
        keep = True
        for kept in selected:
            iou, overlap_min = _box_overlap_ratio(candidate.box, kept.box)
            if overlap_min > 0.45 or iou > 0.25:
                keep = False
                break
        if keep:
            selected.append(candidate)
    return selected


def _discard_container_candidates(candidates: list[_ContourCandidate]) -> list[_ContourCandidate]:
    filtered: list[_ContourCandidate] = []
    for candidate in candidates:
        distinct_children: list[_ContourCandidate] = []
        for other in sorted(candidates, key=lambda item: item.area_ratio, reverse=True):
            if other is candidate or other.area_ratio >= candidate.area_ratio * 0.75:
                continue
            if not _mostly_inside(other.box, candidate.box):
                continue
            if any(_box_overlap_ratio(other.box, child.box)[1] > 0.55 for child in distinct_children):
                continue
            distinct_children.append(other)

        child_area_ratio = sum(_box_area(child.box) for child in distinct_children) / max(1, _box_area(candidate.box))
        if candidate.area_ratio > 0.18 and len(distinct_children) >= 3 and child_area_ratio > 0.35:
            continue
        if candidate.area_ratio > 0.24 and len(distinct_children) >= 2 and child_area_ratio > 0.28:
            continue
        filtered.append(candidate)
    return filtered


def _normalized_box_area(box: dict[str, float]) -> float:
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def _normalized_box_overlap(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, float, float]:
    left = max(a["x1"], b["x1"])
    top = max(a["y1"], b["y1"])
    right = min(a["x2"], b["x2"])
    bottom = min(a["y2"], b["y2"])
    if right <= left or bottom <= top:
        return 0.0, 0.0, 0.0, 0.0
    intersection = (right - left) * (bottom - top)
    area_a = _normalized_box_area(a)
    area_b = _normalized_box_area(b)
    union = area_a + area_b - intersection
    return (
        intersection / union if union > 0 else 0.0,
        intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0,
        intersection / area_a if area_a > 0 else 0.0,
        intersection / area_b if area_b > 0 else 0.0,
    )


def _score_value(detection: SegmentationDetection, key: str, default: float = 0.0) -> float:
    scores = detection.mask.get("scores", {}) if detection.mask else {}
    try:
        return float(scores.get(key, default))
    except (TypeError, ValueError):
        return default


def _detection_polygon(detection: SegmentationDetection) -> np.ndarray:
    records = (detection.mask or {}).get("polygon") or []
    try:
        points = np.array([[float(point["x"]), float(point["y"])] for point in records], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        points = np.empty((0, 2), dtype=np.float32)
    if len(points) >= 3 and np.isfinite(points).all():
        return np.clip(points, 0.0, 1.0)
    box = detection.bounding_box
    return np.array(
        [[box["x1"], box["y1"]], [box["x2"], box["y1"]], [box["x2"], box["y2"]], [box["x1"], box["y2"]]],
        dtype=np.float32,
    )


def _boundary_inputs(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Give printed borders with two Canny responses a narrow scoring tolerance.
    tolerant_edges = cv2.dilate(_edge_support_map(image), np.ones((5, 5), dtype=np.uint8), iterations=1)
    return lab, _frame_background_lab_model(lab), tolerant_edges


def _detection_boundary_evidence(
    detection: SegmentationDetection,
    lab: np.ndarray,
    lab_background: tuple[np.ndarray, float],
    tolerant_edge_map: np.ndarray,
) -> dict[str, float]:
    """Score full-resolution physical-boundary evidence for any proposal source."""
    height, width = lab.shape[:2]
    try:
        polygon = normalized_vertices_to_pixels(_detection_polygon(detection), width, height)
        quad, _ = quad_from_polygon(polygon)
    except (cv2.error, ValueError):
        return {
            "score": 0.0,
            "edge_support": 0.0,
            "border_contrast": 0.0,
            "exterior_background_support": 0.0,
            "rectangularity": 0.0,
        }
    edge_support = _mean_score(_edge_support_scores(tolerant_edge_map, quad))
    border_contrast = _mean_score(_border_contrast_scores(lab, quad))
    exterior_background = _mean_score(_exterior_background_scores(lab, quad, lab_background))
    rectangularity = _rectangularity_score(quad)
    score = 0.35 * edge_support + 0.40 * border_contrast + 0.15 * exterior_background + 0.10 * rectangularity
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "edge_support": round(edge_support, 4),
        "border_contrast": round(border_contrast, 4),
        "exterior_background_support": round(exterior_background, 4),
        "rectangularity": round(rectangularity, 4),
    }


def _detection_geometry_evidence(detection: SegmentationDetection) -> dict:
    """Expose bounded, source-independent shape evidence for one proposal."""
    polygon = _detection_polygon(detection)
    try:
        quad, quad_method = quad_from_polygon(polygon)
        contour = quad.reshape(-1, 1, 2).astype(np.float32)
        convex = bool(cv2.isContourConvex(contour))
        rectangularity = _rectangularity_score(quad)
        area_ratio = abs(float(cv2.contourArea(contour)))
        quadrilateral_available = True
    except (cv2.error, ValueError):
        quad_method = "unavailable"
        convex = False
        rectangularity = 0.0
        area_ratio = _normalized_box_area(detection.bounding_box)
        quadrilateral_available = False

    validity = 1.0 if detection.geometry_valid else 0.0
    quad_score = 1.0 if quadrilateral_available and convex else 0.0
    score = 0.50 * validity + 0.30 * rectangularity + 0.20 * quad_score
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "valid": bool(detection.geometry_valid),
        "aspect_ratio": round(float(detection.aspect_ratio), 4),
        "area_ratio": round(float(np.clip(area_ratio, 0.0, 1.0)), 4),
        "rectangularity": round(float(np.clip(rectangularity, 0.0, 1.0)), 4),
        "quadrilateral_available": quadrilateral_available,
        "convex": convex,
        "quad_method": quad_method,
        "review_reasons": list(detection.review_reasons),
    }


def _model_set_risk_reasons(
    detections: list[SegmentationDetection],
    settings: Settings,
    evidence: dict[int, dict[str, float]],
) -> list[str]:
    if not detections:
        return ["MODEL_SET_EMPTY"]
    reasons: list[str] = []
    for index, detection in enumerate(detections):
        source = str((detection.mask or {}).get("source", ""))
        area_ratio = _score_value(detection, "area_ratio", _normalized_box_area(detection.bounding_box))
        if not detection.geometry_valid:
            reasons.append(f"MODEL_GEOMETRY_INVALID:{index}")
        if source != "yolo_seg_mask":
            reasons.append(f"MODEL_MASK_UNAVAILABLE:{index}")
        if area_ratio > 0.62:
            reasons.append(f"MODEL_PAGE_CONTAINER_RISK:{index}")
        if detection.confidence <= min(0.98, float(settings.yolo_confidence_threshold) + 0.05):
            reasons.append(f"MODEL_CONFIDENCE_MARGIN_LOW:{index}")
        if evidence[index]["score"] < 0.42:
            reasons.append(f"MODEL_BOUNDARY_EVIDENCE_WEAK:{index}")
    return list(dict.fromkeys(reasons))


def _cross_source_same_instance(
    first: SegmentationDetection,
    second: SegmentationDetection,
) -> tuple[bool, dict[str, float]]:
    box_iou, box_overlap_min, _, _ = _normalized_box_overlap(first.bounding_box, second.bounding_box)
    if box_iou < 0.45 and box_overlap_min < 0.70:
        return False, {"box_iou": box_iou, "polygon_iou": 0.0}
    overlap = normalized_polygon_overlap(_detection_polygon(first), _detection_polygon(second))
    same = (
        overlap.iou >= 0.82
        or (overlap.iou >= 0.62 and box_iou >= 0.72)
        or (min(overlap.first_containment, overlap.second_containment) >= 0.82 and box_overlap_min >= 0.88)
    )
    return same, {
        "box_iou": round(box_iou, 4),
        "polygon_iou": round(overlap.iou, 4),
        "first_containment": round(overlap.first_containment, 4),
        "second_containment": round(overlap.second_containment, 4),
    }


def _fusion_score(detection: SegmentationDetection, boundary: dict[str, float]) -> float:
    geometry_score = 1.0 if detection.geometry_valid else 0.0
    return round(0.62 * boundary["score"] + 0.28 * detection.confidence + 0.10 * geometry_score, 4)


def fuse_candidate_evidence(
    image: np.ndarray,
    model_detections: list[SegmentationDetection],
    classical_detections: list[SegmentationDetection],
    *,
    model_evidence: dict[int, dict[str, float]] | None = None,
) -> tuple[list[SegmentationDetection], dict]:
    """Fuse proposal sources after assessment, preserving real overlapping instances."""
    records: list[dict] = []
    inputs = None
    for family, detections in (("model", model_detections), ("classical", classical_detections)):
        for index, detection in enumerate(detections):
            if family == "model" and model_evidence is not None:
                boundary = model_evidence[index]
            else:
                if inputs is None:
                    inputs = _boundary_inputs(image)
                boundary = _detection_boundary_evidence(detection, *inputs)
            geometry = _detection_geometry_evidence(detection)
            records.append(
                {
                    "family": family,
                    "detection": detection,
                    "boundary": boundary,
                    "geometry": geometry,
                    "score": _fusion_score(detection, boundary),
                }
            )

    clusters: list[list[dict]] = []
    cluster_matches: list[list[dict[str, float]]] = []
    for record in records:
        matched_cluster = None
        matched_details: dict[str, float] | None = None
        for cluster_index, cluster in enumerate(clusters):
            for existing in cluster:
                # Never collapse two proposals from the same detector family:
                # overlapping physical prints are legitimate instances.
                if existing["family"] == record["family"]:
                    continue
                same, details = _cross_source_same_instance(existing["detection"], record["detection"])
                if same:
                    matched_cluster = cluster_index
                    matched_details = details
                    break
            if matched_cluster is not None:
                break
        if matched_cluster is None:
            clusters.append([record])
            cluster_matches.append([])
        else:
            clusters[matched_cluster].append(record)
            if matched_details is not None:
                cluster_matches[matched_cluster].append(matched_details)

    selected: list[SegmentationDetection] = []
    cluster_metadata: list[dict] = []
    replacements = 0
    for cluster_index, cluster in enumerate(clusters):
        ranked = sorted(
            cluster,
            key=lambda item: (item["score"], item["boundary"]["score"], item["detection"].confidence),
            reverse=True,
        )
        winner = ranked[0]
        model_alternatives = [item for item in cluster if item["family"] == "model"]
        if winner["family"] == "classical" and model_alternatives:
            replacements += 1
        reason = "single_source_proposal"
        if len(cluster) > 1:
            runner_up = ranked[1]
            reason = (
                "higher_boundary_evidence"
                if winner["boundary"]["score"] > runner_up["boundary"]["score"] + 0.04
                else "higher_fused_evidence"
            )
        provenance = {
            "cluster_id": cluster_index,
            "decision": "selected",
            "selection_reason": reason,
            "selected_family": winner["family"],
            "selected_score": winner["score"],
            "boundary_evidence": winner["boundary"],
            "proposal_count": len(cluster),
            "proposal_families": sorted({item["family"] for item in cluster}),
        }
        winner_detection = winner["detection"]
        selected.append(
            replace(
                winner_detection,
                mask={
                    **(winner_detection.mask or {}),
                    "evidence_schema": "segmentation_detection_evidence_v1",
                    "boundary_evidence": winner["boundary"],
                    "geometry_evidence": winner["geometry"],
                    "fusion": provenance,
                },
            )
        )
        cluster_metadata.append(
            {
                **provenance,
                "matches": cluster_matches[cluster_index],
                "alternatives": [
                    {
                        "family": item["family"],
                        "detector_source": (item["detection"].mask or {}).get("detector_source"),
                        "confidence": item["detection"].confidence,
                        "boundary_score": item["boundary"]["score"],
                        "fusion_score": item["score"],
                    }
                    for item in ranked
                ],
            }
        )

    selected.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    return selected, {
        "strategy": "mask_aware_staged_evidence_v1",
        "model_proposals": len(model_detections),
        "classical_proposals": len(classical_detections),
        "selected_instances": len(selected),
        "model_hypotheses_replaced": replacements,
        "clusters": cluster_metadata,
    }


def _distinct_nested_children(
    parent_index: int, detections: list[SegmentationDetection], areas: list[float]
) -> list[int]:
    parent = detections[parent_index]
    children: list[int] = []
    for index, detection in enumerate(detections):
        if index == parent_index or areas[index] >= areas[parent_index] * 0.72:
            continue
        _, _, child_inside_parent, _ = _normalized_box_overlap(detection.bounding_box, parent.bounding_box)
        if child_inside_parent < 0.82:
            continue
        if any(
            _normalized_box_overlap(detection.bounding_box, detections[child].bounding_box)[1] > 0.55
            for child in children
        ):
            continue
        children.append(index)
    return children


def _is_photo_like_parent(detection: SegmentationDetection) -> bool:
    area_ratio = _score_value(detection, "area_ratio", _normalized_box_area(detection.bounding_box))
    outline_sides = _score_value(detection, "outline_supported_sides")
    exterior_sides = _score_value(detection, "exterior_background_sides")
    return area_ratio <= 0.62 and (outline_sides >= 3 or exterior_sides >= 2 or detection.confidence >= 0.56)


def _filter_nested_classical_detections(
    detections: list[SegmentationDetection],
) -> tuple[list[SegmentationDetection], dict[str, int]]:
    if len(detections) < 2:
        return detections, {}

    rejected: dict[int, str] = {}
    areas = [_normalized_box_area(detection.bounding_box) for detection in detections]

    for index, detection in enumerate(detections):
        children = _distinct_nested_children(index, detections, areas)
        child_area_ratio = sum(areas[child] for child in children) / max(areas[index], 1e-9)
        if areas[index] > 0.18 and len(children) >= 2 and child_area_ratio > 0.24:
            rejected[index] = "container_detection_with_multiple_photo_regions"

    for index, detection in enumerate(detections):
        if index in rejected:
            continue
        for parent_index, parent in enumerate(detections):
            if parent_index == index or parent_index in rejected or areas[index] >= areas[parent_index] * 0.78:
                continue
            if not _is_photo_like_parent(parent):
                continue
            _, overlap_min, child_inside_parent, _ = _normalized_box_overlap(
                detection.bounding_box, parent.bounding_box
            )
            if child_inside_parent < 0.72 and overlap_min < 0.86:
                continue
            child_area_fraction = areas[index] / max(areas[parent_index], 1e-9)
            parent_outline = _score_value(parent, "outline_supported_sides")
            child_outline = _score_value(detection, "outline_supported_sides")
            child_exterior_sides = _score_value(detection, "exterior_background_sides")
            parent_is_at_least_as_plausible = (
                parent.confidence >= detection.confidence - 0.15 or parent_outline >= child_outline
            )
            if child_area_fraction < 0.78 and (parent_is_at_least_as_plausible or child_exterior_sides <= 1):
                rejected[index] = "partial_photo_inside_larger_candidate"
                break

    rejection_counts: dict[str, int] = {}
    for reason in rejected.values():
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    return [detection for index, detection in enumerate(detections) if index not in rejected], rejection_counts


def select_candidate_set(detections: list[SegmentationDetection]) -> tuple[list[SegmentationDetection], dict[str, int]]:
    """Perform duplicate suppression only after every proposal was assessed.

    This intentionally differs from the historical contour-area-first NMS:
    a large page/container can no longer suppress a smaller valid print before
    it has a chance to fail the geometry and boundary gates.
    """
    filtered, rejected = _filter_nested_classical_detections(detections)
    selected: list[SegmentationDetection] = []
    duplicate_count = 0
    for candidate in sorted(filtered, key=lambda item: item.confidence, reverse=True):
        if any(_normalized_box_overlap(candidate.bounding_box, kept.bounding_box)[0] > 0.82 for kept in selected):
            duplicate_count += 1
            continue
        selected.append(candidate)
    if duplicate_count:
        rejected["near_duplicate_after_scoring"] = duplicate_count
    return selected, rejected


def detect_photos_classical(
    image: np.ndarray, *, min_area_ratio: float = _CLASSICAL_MIN_AREA_RATIO,
    max_line_refinement_pixels: int | None = None,
) -> SegmentationResult:
    started = perf_counter()
    height, width = image.shape[:2]
    image_area = width * height
    edge_map = _edge_support_map(image)
    # Do not consolidate by area before candidate assessment.  Containers are
    # frequently larger than the true prints and used to erase them here.
    filtered_contours = _find_contour_candidates(image, min_area_ratio)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_background = _frame_background_lab_model(lab)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    detections: list[SegmentationDetection] = []
    source_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    line_cache: dict[tuple[int, int, int, int], np.ndarray | None] = {}
    line_pixels = 0
    unrefined_candidates = 0
    line_pad = 2 * int(max(12, min(width, height) * 0.025))
    for candidate in filtered_contours:
        polygon_points, polygon_source = _quadrilateral_from_contour(candidate.contour)
        _, _, candidate_width, candidate_height = candidate.box
        candidate_pixels = (candidate_width + line_pad) * (candidate_height + line_pad)
        refined_by_lines = False
        if max_line_refinement_pixels is None or line_pixels + candidate_pixels <= max_line_refinement_pixels:
            polygon_points, refined_by_lines = _refine_quadrilateral_with_lines(image, polygon_points, line_cache)
            line_pixels += candidate_pixels
        else:
            unrefined_candidates += 1
        polygon_points = _clamp_points(polygon_points, width, height)
        x, y, w, h = cv2.boundingRect(polygon_points.astype(np.int32))
        contour_width, contour_height = _quadrilateral_geometry_dimensions(polygon_points, w, h)
        quality = evaluate_segmentation_geometry(
            contour_width,
            contour_height,
            min_aspect_ratio=0.5,
            max_aspect_ratio=3.0,
        )
        quad_area_ratio = cv2.contourArea(polygon_points.reshape(-1, 1, 2)) / image_area
        assessment = _assess_candidate(
            lab,
            edge_map,
            lab_background,
            laplacian,
            polygon_points,
            area_ratio=max(candidate.area_ratio, quad_area_ratio),
            polygon_source=polygon_source,
            detector_source=candidate.source,
            refined_by_lines=refined_by_lines,
        )
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        if assessment.rejected or not quality.geometry_valid:
            for reason in assessment.rejection_reasons:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            for reason in quality.review_reasons:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            continue
        detections.append(
            SegmentationDetection(
                bounding_box=pixel_box_to_normalized(x, y, w, h, width, height),
                mask={
                    "polygon": _normalized_polygon(polygon_points, width, height),
                    "source": polygon_source,
                    "detector_source": candidate.source,
                    "corner_refinement": "line_segments" if refined_by_lines else "contour",
                    "scores": {
                        "edge_support": round(assessment.edge_support, 3),
                        "border_contrast": round(assessment.border_contrast, 3),
                        "area_ratio": round(float(assessment.area_ratio), 4),
                        "rectangularity": round(assessment.rectangularity, 3),
                        "interior_texture": round(assessment.interior_texture, 3),
                        "edge_supported_sides": float(assessment.edge_supported_sides),
                        "outline_supported_sides": float(assessment.outline_supported_sides),
                        "exterior_background_support": round(assessment.exterior_background_support, 3),
                        "exterior_background_sides": float(assessment.exterior_background_sides),
                    },
                },
                confidence=assessment.confidence,
                aspect_ratio=quality.aspect_ratio,
                geometry_valid=quality.geometry_valid,
                review_reasons=quality.review_reasons,
            )
        )

    detections, nested_rejections = select_candidate_set(detections)
    for reason, count in nested_rejections.items():
        rejected_counts[reason] = rejected_counts.get(reason, 0) + count
    detections.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    classical_time_ms = round((perf_counter() - started) * 1000, 2)
    return SegmentationResult(
        detections=detections,
        metadata={
            "model": "classical_hybrid_quad",
            "classical_time_ms": classical_time_ms,
            # Compatibility for callers that historically treated any
            # detector runtime as inference time. New code should consume the
            # explicit classical/model component fields instead.
            "inference_time_ms": classical_time_ms,
            "fallback_used": True,
            "candidate_sources": source_counts,
            "rejected_candidates": rejected_counts,
            "line_refinement_skipped": unrefined_candidates,
        },
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _find_proto_output(outputs: list[np.ndarray]) -> np.ndarray | None:
    for output in outputs:
        array = np.asarray(output)
        if array.ndim != 4:
            continue
        if array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            continue
        if array.shape[-1] <= 64 and array.shape[-1] <= min(array.shape[0], array.shape[1]):
            return array.transpose(2, 0, 1).astype(np.float32)
        if array.shape[0] <= 64:
            return array.astype(np.float32)
        if array.shape[0] <= 256 and array.shape[0] <= min(array.shape[1], array.shape[2]):
            return array.astype(np.float32)
    return None


def _prediction_rows(outputs: list[np.ndarray]) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for output in outputs:
        array = np.asarray(output)
        if array.ndim == 4:
            continue
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            continue
        if array.shape[0] <= 256 and array.shape[1] > 256 and array.shape[1] > array.shape[0]:
            array = array.T
        if array.shape[-1] >= 5:
            rows.append(array.astype(np.float32))
    return rows


def _clip_normalized_box(box: dict[str, float]) -> dict[str, float] | None:
    clipped = {
        "x1": max(0.0, min(1.0, box["x1"])),
        "y1": max(0.0, min(1.0, box["y1"])),
        "x2": max(0.0, min(1.0, box["x2"])),
        "y2": max(0.0, min(1.0, box["y2"])),
    }
    if clipped["x2"] <= clipped["x1"] or clipped["y2"] <= clipped["y1"]:
        return None
    return clipped


def _normalize_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    transform: _LetterboxTransform,
    native: bool = True,
) -> tuple[dict[str, float], dict[str, float]] | None:
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        input_box = {
            "x1": x1 / transform.input_width,
            "y1": y1 / transform.input_height,
            "x2": x2 / transform.input_width,
            "y2": y2 / transform.input_height,
        }
    else:
        input_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    clipped_input_box = _clip_normalized_box(input_box)
    if clipped_input_box is None:
        return None
    source_box = _source_box_from_input_box(clipped_input_box, transform, native=native)
    if source_box is None:
        return None
    return source_box, clipped_input_box


def _normalize_cxcywh(
    cx: float,
    cy: float,
    box_width: float,
    box_height: float,
    *,
    transform: _LetterboxTransform,
    native: bool = True,
) -> tuple[dict[str, float], dict[str, float]] | None:
    x1 = cx - box_width / 2.0
    y1 = cy - box_height / 2.0
    x2 = cx + box_width / 2.0
    y2 = cy + box_height / 2.0
    return _normalize_xyxy(x1, y1, x2, y2, transform=transform, native=native)


def _box_to_points(box: dict[str, float], width: int, height: int, *, native: bool) -> np.ndarray:
    if native:
        x1, y1, x2, y2 = normalized_box_to_pixels(box, width, height)
        x2, y2 = max(x1, x2 - 1), max(y1, y2 - 1)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return np.array(
        [
            [box["x1"] * width, box["y1"] * height],
            [box["x2"] * width, box["y1"] * height],
            [box["x2"] * width, box["y2"] * height],
            [box["x1"] * width, box["y2"] * height],
        ],
        dtype=np.float32,
    )


def _expand_mask_polygon_to_pixel_edges(points: np.ndarray, mask_width: int, mask_height: int) -> np.ndarray:
    expanded = points.astype(np.float32).copy()
    if expanded.size == 0:
        return expanded
    max_x = float(np.max(expanded[:, 0]))
    max_y = float(np.max(expanded[:, 1]))
    expanded[np.isclose(expanded[:, 0], max_x), 0] = np.minimum(
        mask_width, expanded[np.isclose(expanded[:, 0], max_x), 0] + 1.0
    )
    expanded[np.isclose(expanded[:, 1], max_y), 1] = np.minimum(
        mask_height, expanded[np.isclose(expanded[:, 1], max_y), 1] + 1.0
    )
    return expanded


def _polygon_from_binary_mask(binary_mask: np.ndarray, *, legacy_expansion: bool = False) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return None
    perimeter = cv2.arcLength(contour, True)
    for epsilon_ratio in (0.004, 0.008, 0.012, 0.02, 0.035, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if 4 <= len(approx) <= 48:
            break
    else:
        approx = cv2.convexHull(contour).reshape(-1, 2)
        if len(approx) < 4:
            return None
    if legacy_expansion:
        return _expand_mask_polygon_to_pixel_edges(approx, binary_mask.shape[1], binary_mask.shape[0])
    if cv2.isContourConvex(approx):
        native = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        # Preserve inward deficits erased by simplification, within the established silhouette.
        if 4 < len(native) <= 128 and not cv2.isContourConvex(native) and all(
            cv2.pointPolygonTest(approx, tuple(map(float, point)), True) >= -1.0 for point in native
        ):
            return native.astype(np.float32)
    return approx.astype(np.float32)


def _native_mask_roi(logits: np.ndarray, box: dict[str, float], transform: _LetterboxTransform) -> tuple[np.ndarray, int, int]:
    width, height = transform.source_width, transform.source_height
    x1, x2 = (max(0, min(width, int(np.ceil(box[key] * width)))) for key in ("x1", "x2"))
    y1, y2 = (max(0, min(height, int(np.ceil(box[key] * height)))) for key in ("y1", "y2"))
    binary = np.empty((y2 - y1, x2 - x1), dtype=np.uint8)
    if not binary.size:
        return binary, x1, y1
    left, top = int(transform.pad_x), int(transform.pad_y)
    content_width, content_height = transform.resized_width, transform.resized_height
    # Undo the *actual* rounded letterbox before scaling logits to source pixel centres.
    model_logits = cv2.resize(logits, (transform.input_width, transform.input_height), interpolation=cv2.INTER_LINEAR)
    active = model_logits[top : top + content_height, left : left + content_width]
    sx, sy = content_width / width, content_height / height
    # Keep full-resolution float intermediates bounded even for large page-sized proposals.
    tile_size = 2048
    for row in range(0, binary.shape[0], tile_size):
        for column in range(0, binary.shape[1], tile_size):
            tile = binary[row : row + tile_size, column : column + tile_size]
            mapping = np.array(
                [[sx, 0, (x1 + column + 0.5) * sx - 0.5], [0, sy, (y1 + row + 0.5) * sy - 0.5]],
                dtype=np.float64,
            )
            resized = cv2.warpAffine(
                active, mapping, (tile.shape[1], tile.shape[0]),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            tile[:] = resized > 0
    return binary, x1, y1


def _polygon_from_proto_mask(
    proto: np.ndarray,
    coefficients: np.ndarray,
    box: dict[str, float],
    *,
    transform: _LetterboxTransform,
    source_box: dict[str, float],
    native: bool,
) -> np.ndarray | None:
    channels, mask_height, mask_width = proto.shape
    if coefficients.size != channels:
        return None
    logits = np.matmul(coefficients.astype(np.float32), proto.reshape(channels, -1)).reshape(mask_height, mask_width)
    if native:
        binary, x, y = _native_mask_roi(logits, source_box, transform)
        points = _polygon_from_binary_mask(binary) if binary.size else None
        return points + np.array([x, y], dtype=np.float32) if points is not None else None

    input_width, input_height = transform.input_width, transform.input_height
    mask = _sigmoid(logits)
    resized = cv2.resize(mask, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    binary = (resized > 0.5).astype(np.uint8) * 255

    x1 = int(round(box["x1"] * input_width))
    y1 = int(round(box["y1"] * input_height))
    x2 = int(round(box["x2"] * input_width))
    y2 = int(round(box["y2"] * input_height))
    crop_mask = np.zeros_like(binary)
    crop_mask[max(0, y1) : min(input_height, y2), max(0, x1) : min(input_width, x2)] = binary[
        max(0, y1) : min(input_height, y2),
        max(0, x1) : min(input_width, x2),
    ]
    points = _polygon_from_binary_mask(crop_mask, legacy_expansion=True)
    return _model_points_to_source(points, transform) if points is not None else None


def _make_detection(
    *,
    box: dict[str, float],
    points: np.ndarray,
    confidence: float,
    image_width: int,
    image_height: int,
    settings: Settings,
    source: str,
    detector_source: str,
    scores: dict[str, float] | None = None,
) -> SegmentationDetection | None:
    points = _clamp_points(points, image_width, image_height)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    if w <= 0 or h <= 0:
        return None
    contour_width, contour_height = _quadrilateral_geometry_dimensions(points, w, h)
    quality = evaluate_segmentation_geometry(
        contour_width,
        contour_height,
        min_aspect_ratio=settings.segmentation_min_aspect_ratio,
        max_aspect_ratio=settings.segmentation_max_aspect_ratio,
    )
    normalized_box = pixel_box_to_normalized(x, y, w, h, image_width, image_height)
    clipped_box = _clip_normalized_box(normalized_box) or box
    mask_scores = dict(scores or {})
    mask_scores.setdefault(
        "area_ratio", round(float(cv2.contourArea(points.reshape(-1, 1, 2)) / max(1, image_width * image_height)), 4)
    )
    return SegmentationDetection(
        bounding_box=clipped_box,
        mask={
            "polygon": _normalized_polygon(points, image_width, image_height),
            "source": source,
            "detector_source": detector_source,
            "scores": mask_scores,
        },
        confidence=round(float(confidence), 3),
        aspect_ratio=quality.aspect_ratio,
        geometry_valid=quality.geometry_valid,
        review_reasons=quality.review_reasons,
    )


def _nms_detections(detections: list[SegmentationDetection], threshold: float = 0.80) -> list[SegmentationDetection]:
    """Suppress only mask-level duplicates, not merely overlapping boxes."""
    selected: list[SegmentationDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        candidate_polygon = _detection_polygon(detection)
        for kept in selected:
            box_iou, box_overlap_min, _, _ = _normalized_box_overlap(detection.bounding_box, kept.bounding_box)
            if box_iou < 0.35 and box_overlap_min < 0.70:
                continue
            overlap = normalized_polygon_overlap(candidate_polygon, _detection_polygon(kept))
            if overlap.iou > threshold or (
                min(overlap.first_containment, overlap.second_containment) > 0.92 and box_overlap_min > 0.92
            ):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(detection)
    return sorted(selected, key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))


def _parse_simple_box_outputs(
    outputs: list[np.ndarray],
    width: int,
    height: int,
    settings: Settings,
    *,
    input_width: int | None = None,
    input_height: int | None = None,
) -> list[SegmentationDetection]:
    input_width = input_width or width
    input_height = input_height or height
    transform = _build_letterbox_transform(width, height, input_width, input_height)
    proto = _find_proto_output(outputs)
    mask_channels = int(proto.shape[0]) if proto is not None else 0
    native = getattr(settings, "enable_precise_photo_crops", True)
    proposal_threshold = getattr(settings, "yolo_proposal_threshold", settings.yolo_confidence_threshold)
    detections: list[SegmentationDetection] = []
    for array in _prediction_rows(outputs):
        columns = array.shape[-1]
        if columns < 5:
            continue
        has_masks = proto is not None and columns >= 5 + mask_channels
        uses_centers = has_masks or columns > 7
        score_end = columns - mask_channels if has_masks else columns
        confidences = np.max(array[:, 4:score_end], axis=1) if uses_centers else array[:, 4]
        normalize = _normalize_cxcywh if uses_centers else _normalize_xyxy
        for index in np.flatnonzero(confidences >= proposal_threshold):
            row = array[index]
            normalized_boxes = normalize(*map(float, row[:4]), transform=transform, native=native)
            if normalized_boxes is None:
                continue
            source_box, input_box = normalized_boxes
            source = "yolo_box"
            points = None
            if has_masks:
                source = "yolo_seg_mask"
                points = _polygon_from_proto_mask(
                    proto,
                    row[score_end:],
                    input_box,
                    transform=transform,
                    source_box=source_box,
                    native=native,
                )
            if points is None:
                points = _box_to_points(source_box, width, height, native=native)
                if native:
                    source = "yolo_box"
            detection = _make_detection(
                box=source_box,
                points=points,
                confidence=float(confidences[index]),
                image_width=width,
                image_height=height,
                settings=settings,
                source=source,
                detector_source="yolo_onnx",
                scores={"mask_channels": float(mask_channels)} if mask_channels else None,
            )
            if detection is not None:
                detections.append(detection)
    return _nms_detections(detections)


def _tile_detections_on_page(
    detections: list[SegmentationDetection], bounds: tuple[int, int, int, int],
    width: int, height: int,
) -> list[SegmentationDetection]:
    left, top, right, bottom = bounds
    scale = np.array([right - left - 1, bottom - top - 1])
    page_scale = np.array([width - 1, height - 1])
    clearance = max(3, round(.015 * min(width, height)))
    mapped = []
    for detection in detections:
        if detection.mask.get("source") != "yolo_seg_mask" or detection.confidence < .60 or not detection.geometry_valid:
            continue
        points = _detection_polygon(detection) * scale
        lower, upper = points.min(axis=0), points.max(axis=0)
        if ((left > 0 and lower[0] < clearance) or (top > 0 and lower[1] < clearance)
                or (right < width and scale[0] - upper[0] < clearance)
                or (bottom < height and scale[1] - upper[1] < clearance)):
            continue
        page_points = points + [left, top]
        polygon = _normalized_polygon(page_points, width, height)
        low, high = page_points.min(axis=0) / page_scale, page_points.max(axis=0) / page_scale
        box = {"x1": float(low[0]), "y1": float(low[1]), "x2": float(high[0]), "y2": float(high[1])}
        scores = (detection.mask.get("scores") or {}) | {
            "area_ratio": cv2.contourArea(page_points.astype(np.float32)) / (width * height),
        }
        mapped.append(replace(detection, bounding_box=box, mask=detection.mask | {
            "polygon": polygon, "scores": scores, "detector_source": "yolo_onnx_tiled",
        }))
    return mapped


def _fuse_dense_views(
    views: list[list[SegmentationDetection]], evidence: list[list[dict[str, float]]],
    *, preserve_full_page: bool = False,
) -> list[tuple[SegmentationDetection, dict[str, float]]]:
    detections = [detection for view in views for detection in view]
    polygons = [_detection_polygon(detection) for detection in detections]
    boundaries = [item for view in evidence for item in view]
    view_ids = [index for index, view in enumerate(views) for _ in view]

    @lru_cache(maxsize=16_384)
    def comparison(first: int, second: int) -> tuple[float, bool]:
        box_iou, containment, *_ = _normalized_box_overlap(detections[first].bounding_box, detections[second].bounding_box)
        if box_iou < .35 and containment < .774:
            return box_iou, False
        overlap = normalized_polygon_overlap(polygons[first], polygons[second])
        return box_iou, overlap.iou >= .50 or max(overlap.first_containment, overlap.second_containment) >= .86

    agreements = [{view} for view in view_ids]
    for first, first_view in enumerate(view_ids):
        for second in range(first + 1, len(detections)):
            second_view = view_ids[second]
            if first_view == second_view:
                continue
            box_iou, same = comparison(first, second)
            if same and box_iou >= .60:
                agreements[first].add(second_view)
                agreements[second].add(first_view)
    ranking = sorted(range(len(detections)), key=lambda index: (
        .8 * detections[index].confidence + .2 * boundaries[index]["score"] + .07 * min(3, len(agreements[index]) - 1)
    ), reverse=True)

    def duplicate(first: int, second: int) -> bool:
        if preserve_full_page and view_ids[first] == view_ids[second] == 0:
            return False
        box_iou, same = comparison(min(first, second), max(first, second))
        # Two views that each resolve both overlapping prints support keeping both.
        separate = box_iou < .60 and len(agreements[first] & agreements[second]) >= 2
        return same and not separate

    selected = []
    for index in ranking:
        if any(duplicate(index, kept) for kept in selected):
            continue
        selected.append(index)
    return [(detections[index], boundaries[index]) for index in selected]


def _recover_dense_prints(
    image: np.ndarray, detections: list[SegmentationDetection], model_evidence: dict[int, dict[str, float]],
    infer: Callable[[np.ndarray], list[SegmentationDetection]],
    inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
) -> tuple[list[SegmentationDetection], dict[int, dict[str, float]], dict]:
    height, width = image.shape[:2]
    tile_width, tile_height = round(.70 * width), round(.70 * height)
    views = [detections]
    for top in (0, height - tile_height):
        for left in (0, width - tile_width):
            right, bottom = left + tile_width, top + tile_height
            views.append(_tile_detections_on_page(
                infer(image[top:bottom, left:right]), (left, top, right, bottom), width, height,
            ))
    tile_proposals = sum(len(view) for view in views[1:])
    if not tile_proposals:
        return detections, model_evidence, {"tiles": 4, "proposals": tile_proposals, "selected": 0}
    evidence = [[model_evidence[index] for index in range(len(detections))]] + [
        [_detection_boundary_evidence(detection, *inputs) for detection in view] for view in views[1:]
    ]
    # Partial tile coverage may improve individual prints, but cannot resolve pairs seen only on the full page.
    winners = _fuse_dense_views(views, evidence, preserve_full_page=tile_proposals < len(detections))
    selected = [item for item, _ in winners]
    selected_evidence = {index: evidence for index, (_, evidence) in enumerate(winners)}
    return selected, selected_evidence, {"tiles": 4, "proposals": tile_proposals, "selected": sum(
        item.mask.get("detector_source") == "yolo_onnx_tiled" for item in selected
    )}


def _project_deskewed_detection(
    detection: SegmentationDetection, view_shape: tuple[int, ...], image_shape: tuple[int, ...],
    inverse: np.ndarray, settings: Settings,
) -> SegmentationDetection | None:
    if detection.mask.get("source") != "yolo_seg_mask":
        return None
    height, width = image_shape[:2]
    points = normalized_vertices_to_pixels(_detection_polygon(detection), view_shape[1], view_shape[0])
    points = cv2.transform(points[None], inverse)[0]
    if np.any(points < -1) or np.any(points > np.array([width, height])):
        return None  # A proposal in the inference padding has no native source pixels.
    scores = {key: value for key, value in detection.mask.get("scores", {}).items() if key != "area_ratio"}
    return _make_detection(
        box=detection.bounding_box, points=points, confidence=detection.confidence,
        image_width=width, image_height=height, settings=settings,
        source="yolo_seg_mask", detector_source="yolo_onnx_deskewed", scores=scores,
    )


def _deskew_retry(
    image: np.ndarray, source: np.ndarray | None, angle: float, initial: list[SegmentationDetection],
    initial_evidence: dict[int, dict[str, float]], inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
    infer: Callable[[np.ndarray], list[SegmentationDetection]], settings: Settings,
) -> tuple[list[SegmentationDetection], dict[int, dict[str, float]], dict]:
    from app.pipeline.enhancement import prepare_for_cutout

    view, inverse = deskewed_view(source if source is not None else image, angle)
    if source is not None:
        view, _ = prepare_for_cutout(view)
    rotated = infer(view)
    view_inputs = _boundary_inputs(view)
    rotated_evidence = {index: _detection_boundary_evidence(item, *view_inputs) for index, item in enumerate(rotated)}
    rotated, _, tiles = _recover_dense_prints(view, rotated, rotated_evidence, infer, view_inputs)
    del view_inputs

    recovered = []
    for detection in rotated:
        item = _project_deskewed_detection(detection, view.shape, image.shape, inverse, settings)
        if item is not None:
            recovered.append(item)
    evidence = {index: _detection_boundary_evidence(item, *inputs) for index, item in enumerate(recovered)}

    def supported(item: SegmentationDetection, boundary: dict[str, float]) -> bool:
        return (item.geometry_valid and item.mask.get("source") == "yolo_seg_mask"
                and item.confidence >= .60 and boundary["score"] >= .60)

    base_support = sum(supported(item, initial_evidence[i]) for i, item in enumerate(initial))
    small_support = sum(
        supported(item, evidence[i]) and _score_value(item, "area_ratio") < _CLASSICAL_MIN_AREA_RATIO
        for i, item in enumerate(recovered)
    )
    metadata = {"angle_degrees": round(angle, 4), "base_supported": base_support, "small_supported": small_support,
                "view_width": view.shape[1], "view_height": view.shape[0], "tiles": tiles, "selected": False}
    if not tiles["selected"] or small_support < max(16, base_support + 4, 1.20 * base_support):
        return initial, initial_evidence, metadata

    # Keep independently supported full-page detections at places the retry does not cover.
    for index, item in enumerate(initial):
        if supported(item, initial_evidence[index]) and all(
            _normalized_box_overlap(item.bounding_box, candidate.bounding_box)[2] < .25 for candidate in recovered
        ):
            evidence[len(recovered)] = initial_evidence[index]
            recovered.append(item)
    return recovered, evidence, metadata | {"selected": True}


def refine_album_page_result(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    polygons = [detection.mask.get("polygon", []) for detection in result.detections]
    mount_rejections, quads = mounted_print_evidence(image, polygons)
    mount_polygons = [_normalized_polygon(order_quad_cyclic(quad), 1, 1) for quad in quads]
    paper_rejections = paper_and_edge_rejection_reasons(image, polygons + mount_polygons)
    rejected = {index: reason for index, reason in paper_rejections.items() if index < len(polygons)} | mount_rejections

    height, width = image.shape[:2]
    detections = [detection for index, detection in enumerate(result.detections) if index not in rejected]
    for offset, (quad, polygon) in enumerate(zip(quads, mount_polygons, strict=True)):
        if len(polygons) + offset in paper_rejections:
            continue
        left, top = quad.min(axis=0)
        right, bottom = quad.max(axis=0)
        bbox = {"x1": float(left), "y1": float(top), "x2": float(right), "y2": float(bottom)}
        quality = evaluate_segmentation_geometry((right - left) * width, (bottom - top) * height)
        detections.append(
            SegmentationDetection(
                bounding_box=bbox,
                mask={
                    "polygon": polygon,
                    "source": "chromatic_mount_boundary",
                    "detector_source": "chromatic_mount",
                },
                confidence=0.70,
                aspect_ratio=quality.aspect_ratio,
                geometry_valid=quality.geometry_valid,
                review_reasons=[*quality.review_reasons, "chromatic_mount_boundary_needs_review"],
            )
        )
    containers = _page_containers(detections, image)
    if not rejected and not quads and not containers:
        return result
    detections = [detection for index, detection in enumerate(detections) if index not in containers]
    detections.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    replacement_count = sum(len(polygons) + offset not in paper_rejections for offset in range(len(quads)))
    metadata = {
        **result.metadata,
        "mount_replacements": replacement_count,
        "page_proposal_rejections": rejected,
        "container_rejections": len(containers),
    }
    return SegmentationResult(detections=detections, metadata=metadata)


def _page_containers(detections: list[SegmentationDetection], image: np.ndarray) -> set[int]:
    if len(detections) < 3:
        return set()
    polygons = [_detection_polygon(detection) for detection in detections]
    areas = [float(cv2.contourArea(polygon)) for polygon in polygons]
    bounds = [(polygon.min(axis=0), polygon.max(axis=0)) for polygon in polygons]
    credible = []
    for index, (detection, polygon, area) in enumerate(zip(detections, polygons, areas, strict=True)):
        width, height = cv2.minAreaRect(polygon)[1]
        if detection.confidence >= 0.48 and area >= 0.012 and area >= 0.70 * width * height:
            credible.append(index)
    credible.sort(key=areas.__getitem__, reverse=True)

    @lru_cache(maxsize=None)
    def overlap(first: int, second: int):
        return normalized_polygon_overlap(polygons[first], polygons[second])

    rejected = set()
    for index in sorted(range(len(detections)), key=areas.__getitem__):
        area = areas[index]
        lower, upper = bounds[index]
        at_edge = np.any(lower <= 0.035) or np.any(upper >= 0.965)
        boundary = detections[index].mask.get("boundary_evidence", {})
        weak_outline = (
            area >= 0.22
            and detections[index].mask.get("source") == "yolo_seg_mask"
            and boundary.get("score", 1.0) < 0.42
            and boundary.get("edge_support", 1.0) < 0.12
        )
        broad_region = area > 0.40 or (area >= 0.16 and at_edge)
        if not (broad_region or weak_outline):
            continue
        children = []
        for child in credible:
            if child in rejected or areas[child] >= area * 0.68:
                continue
            child_lower, child_upper = bounds[child]
            intersection = np.maximum(0, np.minimum(upper, child_upper) - np.maximum(lower, child_lower)).prod()
            if intersection < areas[child] * 0.82 or overlap(child, index).first_containment < 0.82:
                continue
            if any(max(overlap(child, other).first_containment, overlap(child, other).second_containment) > 0.35 for other in children):
                continue
            children.append(child)
        if broad_region and len(children) >= 2 and sum(areas[child] for child in children) >= 0.25 * area:
            rejected.add(index)
        elif weak_outline and len(children) == 1:
            child = children[0]
            child_boundary = detections[child].mask.get("boundary_evidence", {})
            if (
                areas[child] >= 0.30 * area
                and child_boundary.get("border_contrast", 0.0) >= 0.52
                and paper_around_nested_print(image, polygons[index], polygons[child])
            ):
                rejected.add(index)
    return rejected


def _refine_album_windows(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    leaves = album_window_leaves(image)
    if not leaves:
        return result

    rejected = {}
    for index, detection in enumerate(result.detections):
        polygon = _detection_polygon(detection)
        extents = [normalized_polygon_overlap(polygon, leaf.outline) for leaf in leaves]
        if min(extent.second_containment for extent in extents) >= .68 and sum(
            extent.first_containment for extent in extents
        ) >= .85:
            rejected[index] = "album_spread"
            continue
        for leaf, extent in zip(leaves, extents, strict=True):
            if extent.first_containment < .85:
                continue
            if extent.second_containment >= .70:
                rejected[index] = "empty_album_leaf" if leaf.print_outline is None else "album_leaf"
            elif leaf.print_outline is not None:
                overlap = normalized_polygon_overlap(polygon, leaf.print_outline)
                if max(overlap.first_containment, overlap.second_containment) >= .80:
                    rejected[index] = "replaced_by_visible_album_window"
            elif leaf.decoration is not None:
                overlap = normalized_polygon_overlap(polygon, leaf.decoration)
                if overlap.first_containment >= .70 and overlap.second_containment >= .60:
                    rejected[index] = "empty_album_decoration"
    detections = [detection for index, detection in enumerate(result.detections) if index not in rejected]
    height, width = image.shape[:2]
    for leaf in leaves:
        if leaf.print_outline is None:
            continue
        lower, upper = leaf.print_outline.min(axis=0), leaf.print_outline.max(axis=0)
        extent = (upper - lower) * [width - 1, height - 1]
        quality = evaluate_segmentation_geometry(*extent)
        mask = {
            "polygon": _normalized_polygon(leaf.print_outline, 1, 1),
            "source": ALBUM_WINDOW_SOURCE,
            "detector_source": ALBUM_WINDOW_SOURCE,
        }
        reasons = list(quality.review_reasons)
        if leaf.aperture_quad is not None:
            mask["warp_quad"] = _normalized_polygon(leaf.aperture_quad, 1, 1)
            mask["album_window_occluded"] = True
            reasons.append(ALBUM_WINDOW_OCCLUDED)
        else:
            reasons.append(ALBUM_WINDOW_STOCK_UNCERTAIN)
        detections.append(SegmentationDetection(
            bounding_box={"x1": float(lower[0]), "y1": float(lower[1]), "x2": float(upper[0]), "y2": float(upper[1])},
            mask=mask, confidence=.70, aspect_ratio=quality.aspect_ratio,
            geometry_valid=quality.geometry_valid, review_reasons=reasons,
        ))
    detections.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    return SegmentationResult(detections, {
        **result.metadata, "album_window_rejections": rejected,
        "album_window_detections": sum(leaf.print_outline is not None for leaf in leaves),
    })


def segment_album_page(
    image: np.ndarray, settings: Settings, *, source_image: np.ndarray | None = None,
) -> SegmentationResult:
    source = source_image if source_image is not None and source_image.shape == image.shape else None
    result = _segment_album_page(image, settings, source)
    if settings.enable_classical_segmentation_fallback:
        result = refine_album_page_result(image, result)
        if settings.enable_precise_photo_crops:
            result = _refine_album_windows(source if source is not None else image, result)
    model_path = settings.visible_mask_model_path
    if model_path is None or not settings.enable_precise_photo_crops:
        return result

    from app.pipeline.visible_mask_refinement import UnresolvedMaskOwnership, refine_visible_masks, resolve_mask_ownership

    refinable = [index for index, detection in enumerate(result.detections) if detection.mask.get("source") != ALBUM_WINDOW_SOURCE]
    originals = [_detection_polygon(result.detections[index]) for index in refinable]
    fixed = [_detection_polygon(detection) for detection in result.detections if detection.mask.get("source") == ALBUM_WINDOW_SOURCE]
    try:
        source = source_image if source_image is not None else image
        if source.shape != image.shape:
            raise ValueError("source and detection images have different shapes")
        outlines, digest = refine_visible_masks(source, originals, model_path, settings.model_manifest_path)
        if fixed and outlines:
            resolved = resolve_mask_ownership(source.shape, originals + fixed, outlines + fixed)
            if any(not np.array_equal(protected, outline)
                   for protected, outline in zip(fixed, resolved[len(originals):], strict=True)):
                raise UnresolvedMaskOwnership
            outlines = resolved[:len(originals)]
        updated = result.detections.copy()
        for index, original, outline in zip(refinable, originals, outlines, strict=True):
            if not np.array_equal(outline, original):
                detection = updated[index]
                updated[index] = replace(detection, mask={
                    **detection.mask, "visible_polygon": [{"x": float(x), "y": float(y)} for x, y in outline],
                })
        details = {"model_sha256": digest, "refined_detections": sum(
            original is not refined for original, refined in zip(result.detections, updated, strict=True)
        )}
        return SegmentationResult(updated, {**result.metadata, "visible_mask_refinement": details})
    except UnresolvedMaskOwnership:
        return SegmentationResult(result.detections, {
            **result.metadata, "visible_mask_refinement": {"ownership_unresolved": True},
        })
    except Exception as exc:  # ONNX provider errors preserve the initial proposals for review.
        return SegmentationResult(result.detections, {
            **result.metadata, "visible_mask_refinement": {"error": str(exc)},
        })


def _segment_album_page(image: np.ndarray, settings: Settings, source: np.ndarray | None = None) -> SegmentationResult:
    model_path = Path(settings.yolo_model_path)
    line_budget = min(3 * image.shape[0] * image.shape[1], 12_000_000) if settings.enable_precise_photo_crops else None
    metadata: dict = {
        "model_path": str(model_path),
        "confidence_threshold": settings.yolo_confidence_threshold,
        "proposal_threshold": settings.yolo_proposal_threshold,
    }
    if model_path.exists():
        try:
            if settings.require_registered_models:
                from app.pipeline.model_registry import verify_model_artifact

                metadata["model_artifact"] = verify_model_artifact(
                    model_path,
                    settings.model_manifest_path,
                    model_id=model_path.name,
                )
            import onnxruntime as ort

            started = perf_counter()
            session = _cached_onnx_session(str(model_path), model_path.stat().st_mtime_ns)
            input_meta = session.get_inputs()[0]
            input_shape = input_meta.shape
            registered_input = metadata.get("model_artifact", {}).get("input", {})
            target_height = int(
                input_shape[2] if isinstance(input_shape[2], int) else registered_input.get("height", 640)
            )
            target_width = int(
                input_shape[3] if isinstance(input_shape[3], int) else registered_input.get("width", 640)
            )

            def infer(frame: np.ndarray):
                letterboxed, transform = _letterbox_image(frame, target_width, target_height)
                tensor = letterboxed[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                outputs = session.run(None, {input_meta.name: tensor[np.newaxis, ...]})
                detections = _parse_simple_box_outputs(
                    outputs, frame.shape[1], frame.shape[0], settings,
                    input_width=target_width, input_height=target_height,
                )
                return detections, transform, outputs

            detections, transform, outputs = infer(image)
            model_time_ms = round((perf_counter() - started) * 1000, 2)
            cache_info = _cached_onnx_session.cache_info()
            metadata.update(
                {
                    "model": "yolov8_seg_onnx",
                    "onnxruntime_version": ort.__version__,
                    "providers": session.get_providers(),
                    "session_cache": {"strategy": "path_mtime_lru", "size": cache_info.currsize},
                    "inference_time_ms": model_time_ms,
                    "model_time_ms": model_time_ms,
                    "input_size": {"width": target_width, "height": target_height},
                    "letterbox": {
                        "scale": round(transform.scale, 6),
                        "resized": {"width": transform.resized_width, "height": transform.resized_height},
                        "pad": {"x": transform.pad_x, "y": transform.pad_y},
                        "source": {"width": transform.source_width, "height": transform.source_height},
                    },
                    "output_tensors": [list(np.asarray(output).shape) for output in outputs],
                    "fallback_used": False,
                }
            )
            deskew_enabled = settings.enable_precise_photo_crops and settings.enable_classical_segmentation_fallback
            if detections or deskew_enabled:
                inputs = _boundary_inputs(image) if detections else None
                model_evidence = {
                    index: _detection_boundary_evidence(detection, *inputs) for index, detection in enumerate(detections)
                } if inputs is not None else {}
                risk_reasons = _model_set_risk_reasons(detections, settings, model_evidence)
                metadata["model_set_risk"] = {
                    "risky": bool(risk_reasons),
                    "reasons": risk_reasons,
                    "boundary_evidence": {str(index): value for index, value in model_evidence.items()},
                }
                # Classical geometry is an expensive evidence channel. Run it
                # for an empty/risky model set, not universally and not merely
                # after total model failure.
                dense_small_prints = (
                    settings.enable_precise_photo_crops
                    and len(detections) >= 16
                    and np.median([_score_value(d, "area_ratio", _normalized_box_area(d.bounding_box)) for d in detections])
                    < _CLASSICAL_MIN_AREA_RATIO
                )
                if deskew_enabled and risk_reasons and not dense_small_prints:
                    retry_started = perf_counter()
                    try:
                        angle = album_rotation(source if source is not None else image)
                        if angle is not None:
                            if inputs is None:
                                inputs = _boundary_inputs(image)
                            recovered, recovered_evidence, retry = _deskew_retry(
                                image, source, angle, detections, model_evidence, inputs, lambda frame: infer(frame)[0], settings,
                            )
                            metadata["deskew_inference"] = retry | {
                                "runtime_ms": round((perf_counter() - retry_started) * 1000, 2),
                            }
                            if retry["selected"]:
                                selected, fusion = fuse_candidate_evidence(image, recovered, [], model_evidence=recovered_evidence)
                                metadata["fusion"] = fusion
                                return SegmentationResult(detections=selected, metadata=metadata)
                    except Exception as exc:  # Keep the original inference path if deskew cannot be evaluated.
                        metadata["deskew_inference"] = {"error": str(exc)}
                if settings.enable_classical_segmentation_fallback and risk_reasons and dense_small_prints:
                    metadata["classical_evidence_skipped"] = "dense_small_prints"
                    tile_started = perf_counter()
                    try:
                        recovered, recovered_evidence, tile_metadata = _recover_dense_prints(
                            image, detections, model_evidence, lambda frame: infer(frame)[0], inputs,
                        )
                    except Exception as exc:  # Preserve full-page proposals if a tile cannot be processed.
                        tile_metadata = {"error": str(exc)}
                    else:
                        if tile_metadata["selected"]:
                            detections, model_evidence = recovered, recovered_evidence
                    metadata["dense_tile_inference"] = tile_metadata | {
                        "runtime_ms": round((perf_counter() - tile_started) * 1000, 2),
                    }
                del inputs
                if detections and settings.enable_classical_segmentation_fallback and risk_reasons and not dense_small_prints:
                    fallback = detect_photos_classical(image, max_line_refinement_pixels=line_budget)
                    merged, fusion = fuse_candidate_evidence(
                        image, detections, fallback.detections, model_evidence=model_evidence
                    )
                    metadata.update(
                        {
                            "fallback_used": True,
                            "classical_time_ms": fallback.metadata["classical_time_ms"],
                            "classical_evidence_trigger": risk_reasons,
                            "proposal_sources": {"yolo_onnx": len(detections), "classical": len(fallback.detections)},
                            "fusion": fusion,
                            "classical_metadata": fallback.metadata,
                        }
                    )
                    return SegmentationResult(detections=merged, metadata=metadata)
                if detections:
                    model_only, fusion = fuse_candidate_evidence(image, detections, [], model_evidence=model_evidence)
                    metadata["fusion"] = fusion
                    return SegmentationResult(detections=model_only, metadata=metadata)
            metadata["onnx_parse_warning"] = "No supported detections were parsed from model outputs."
        except Exception as exc:  # noqa: BLE001 - model loading should fail into reviewable metadata.
            metadata["onnx_error"] = str(exc)

    if not settings.enable_classical_segmentation_fallback:
        return SegmentationResult(detections=[], metadata={**metadata, "fallback_used": False})

    fallback = detect_photos_classical(image, max_line_refinement_pixels=line_budget)
    selected, fusion = fuse_candidate_evidence(image, [], fallback.detections)
    return SegmentationResult(
        detections=selected,
        metadata={
            **metadata,
            **fallback.metadata,
            "classical_evidence_trigger": ["MODEL_SET_EMPTY"],
            "fusion": fusion,
        },
    )
