from dataclasses import dataclass
from functools import lru_cache
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

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


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
    return [
        {
            "x": max(0.0, min(1.0, float(x) / width)),
            "y": max(0.0, min(1.0, float(y) / height)),
        }
        for x, y in points
    ]


def _build_letterbox_transform(source_width: int, source_height: int, input_width: int, input_height: int) -> _LetterboxTransform:
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


def _source_box_from_input_box(input_box: dict[str, float], transform: _LetterboxTransform) -> dict[str, float] | None:
    scale = max(transform.scale, 1e-6)
    source_box = {
        "x1": ((input_box["x1"] * transform.input_width) - transform.pad_x) / scale / transform.source_width,
        "y1": ((input_box["y1"] * transform.input_height) - transform.pad_y) / scale / transform.source_height,
        "x2": ((input_box["x2"] * transform.input_width) - transform.pad_x) / scale / transform.source_width,
        "y2": ((input_box["y2"] * transform.input_height) - transform.pad_y) / scale / transform.source_height,
    }
    return _clip_normalized_box(source_box)


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    summed = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(summed)]
    rect[2] = points[np.argmax(summed)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


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


def _quadrilateral_geometry_dimensions(points: np.ndarray, fallback_width: int, fallback_height: int) -> tuple[float, float]:
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


def _refine_quadrilateral_with_lines(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, bool]:
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

    roi_gray = _enhanced_gray(image[y1:y2, x1:x2])
    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except Exception:
        detector = cv2.createLineSegmentDetector()
    detected = detector.detect(roi_gray)[0]
    if detected is None or len(detected) < 4:
        return ordered, False

    lines = detected.reshape(-1, 4).astype(np.float32)
    lines[:, [0, 2]] += x1
    lines[:, [1, 3]] += y1

    fitted_sides: list[tuple[np.ndarray, np.ndarray] | None] = []
    for index in range(4):
        side_start = ordered[index]
        side_end = ordered[(index + 1) % 4]
        side_length = float(np.linalg.norm(side_end - side_start))
        distance_tolerance = max(8.0, side_length * 0.06, min(width, height) * 0.012)
        support_points: list[np.ndarray] = []
        for line in lines:
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


def _border_contrast_scores(image: np.ndarray, points: np.ndarray) -> list[float]:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
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
            distances.append(float(np.linalg.norm(lab[iy, ix] - lab[oy, ox])))
        if distances:
            side_scores.append(min(1.0, float(np.median(distances)) / 30.0))
        else:
            side_scores.append(0.0)
    return side_scores


def _frame_background_lab_model(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    frame_width = max(6, min(width, height) // 24)
    frame = _frame_pixels(lab, frame_width)
    background = np.median(frame, axis=0)
    distances = np.linalg.norm(frame - background, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = median_distance + max(12.0, 2.6 * mad)
    return background.astype(np.float32), threshold


def _exterior_background_scores(image: np.ndarray, points: np.ndarray) -> list[float]:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    background, threshold = _frame_background_lab_model(image)
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


def _interior_texture_score(image: np.ndarray, points: np.ndarray) -> float:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, _clamp_points(points, width, height).astype(np.int32), 255)
    if cv2.countNonZero(mask) < max(64, int(width * height * 0.002)):
        return 0.0
    _, _, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
    inset = max(3, min(21, int(round(min(box_width, box_height) * 0.06))))
    kernel = np.ones((inset | 1, inset | 1), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    if cv2.countNonZero(eroded) > 0:
        mask = eroded
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = float(np.std(laplacian[mask > 0]))
    return float(max(0.0, min(1.0, texture / 42.0)))


def _assess_candidate(
    image: np.ndarray,
    edge_map: np.ndarray,
    points: np.ndarray,
    *,
    area_ratio: float,
    polygon_source: str,
    detector_source: str,
    refined_by_lines: bool,
) -> _CandidateAssessment:
    edge_scores = _edge_support_scores(edge_map, points)
    border_scores = _border_contrast_scores(image, points)
    exterior_scores = _exterior_background_scores(image, points)
    edge_support = _mean_score(edge_scores)
    border_contrast = _mean_score(border_scores)
    exterior_background_support = _mean_score(exterior_scores)
    rectangularity = _rectangularity_score(points)
    interior_texture = _interior_texture_score(image, points)
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
    image_height, image_width = image.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
    touches_frame = sum(
        [
            x <= 2,
            y <= 2,
            x + box_width >= image_width - 2,
            y + box_height >= image_height - 2,
        ]
    )
    if area_ratio > 0.18 and touches_frame >= 1 and (box_width / image_width > 0.62 or box_height / image_height > 0.62):
        rejection_reasons.append("frame_touching_container_candidate")
    if edge_support < 0.08 and border_contrast < 0.16 and interior_texture < 0.32:
        rejection_reasons.append("weak_photo_boundary")
    if area_ratio < 0.16 and outline_supported_sides <= 2 and exterior_background_sides <= 1:
        rejection_reasons.append("incomplete_photo_outline")
    if detector_source == "contrast_region" and edge_support < 0.12 and border_contrast < 0.22 and interior_texture < 0.16:
        rejection_reasons.append("smooth_contrast_false_positive")
    contour_width, contour_height = _quadrilateral_geometry_dimensions(points, 0, 0)
    aspect_ratio = contour_width / contour_height if contour_height > 0 else 0.0
    if detector_source == "contrast_region" and interior_texture < 0.08 and (aspect_ratio < 0.55 or aspect_ratio > 2.1):
        rejection_reasons.append("smooth_extreme_aspect_false_positive")
    if polygon_source == "classical_min_area_rect" and edge_support < 0.16 and border_contrast < 0.24:
        rejection_reasons.append("unsupported_min_area_rectangle")
    if detector_source in {"edge_contour", "line_pairs"} and area_ratio < 0.12 and exterior_background_support < 0.18 and exterior_background_sides == 0:
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
            horizontal.append(_AxisLine(position=(y1 + y2) / 2.0, span_start=min(x1, x2), span_end=max(x1, x2), length=length))
        elif 66 <= angle <= 114:
            vertical.append(_AxisLine(position=(x1 + x2) / 2.0, span_start=min(y1, y2), span_end=max(y1, y2), length=length))

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

    mask = ((color_distance > distance_threshold) | ((saturation > saturation_threshold) & (color_distance > distance_threshold * 0.55))).astype(np.uint8) * 255
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
    for candidate in sorted(candidates, key=lambda item: item.area_ratio + (0.01 if item.source == "edge_contour" else 0.0), reverse=True):
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


def _distinct_nested_children(parent_index: int, detections: list[SegmentationDetection], areas: list[float]) -> list[int]:
    parent = detections[parent_index]
    children: list[int] = []
    for index, detection in enumerate(detections):
        if index == parent_index or areas[index] >= areas[parent_index] * 0.72:
            continue
        _, _, child_inside_parent, _ = _normalized_box_overlap(detection.bounding_box, parent.bounding_box)
        if child_inside_parent < 0.82:
            continue
        if any(_normalized_box_overlap(detection.bounding_box, detections[child].bounding_box)[1] > 0.55 for child in children):
            continue
        children.append(index)
    return children


def _is_photo_like_parent(detection: SegmentationDetection) -> bool:
    area_ratio = _score_value(detection, "area_ratio", _normalized_box_area(detection.bounding_box))
    outline_sides = _score_value(detection, "outline_supported_sides")
    exterior_sides = _score_value(detection, "exterior_background_sides")
    return area_ratio <= 0.62 and (outline_sides >= 3 or exterior_sides >= 2 or detection.confidence >= 0.56)


def _filter_nested_classical_detections(detections: list[SegmentationDetection]) -> tuple[list[SegmentationDetection], dict[str, int]]:
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
            _, overlap_min, child_inside_parent, _ = _normalized_box_overlap(detection.bounding_box, parent.bounding_box)
            if child_inside_parent < 0.72 and overlap_min < 0.86:
                continue
            child_area_fraction = areas[index] / max(areas[parent_index], 1e-9)
            parent_outline = _score_value(parent, "outline_supported_sides")
            child_outline = _score_value(detection, "outline_supported_sides")
            child_exterior_sides = _score_value(detection, "exterior_background_sides")
            parent_is_at_least_as_plausible = parent.confidence >= detection.confidence - 0.15 or parent_outline >= child_outline
            if child_area_fraction < 0.78 and (parent_is_at_least_as_plausible or child_exterior_sides <= 1):
                rejected[index] = "partial_photo_inside_larger_candidate"
                break

    rejection_counts: dict[str, int] = {}
    for reason in rejected.values():
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    return [detection for index, detection in enumerate(detections) if index not in rejected], rejection_counts


def detect_photos_classical(image: np.ndarray, *, min_area_ratio: float = 0.02) -> SegmentationResult:
    height, width = image.shape[:2]
    image_area = width * height
    edge_map = _edge_support_map(image)
    filtered_contours = _consolidate_candidates(_find_contour_candidates(image, min_area_ratio))

    detections: list[SegmentationDetection] = []
    source_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    for candidate in filtered_contours:
        polygon_points, polygon_source = _quadrilateral_from_contour(candidate.contour)
        polygon_points, refined_by_lines = _refine_quadrilateral_with_lines(image, polygon_points)
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
            image,
            edge_map,
            polygon_points,
            area_ratio=max(candidate.area_ratio, quad_area_ratio),
            polygon_source=polygon_source,
            detector_source=candidate.source,
            refined_by_lines=refined_by_lines,
        )
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        if assessment.rejected:
            for reason in assessment.rejection_reasons:
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

    detections, nested_rejections = _filter_nested_classical_detections(detections)
    for reason, count in nested_rejections.items():
        rejected_counts[reason] = rejected_counts.get(reason, 0) + count
    detections.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    return SegmentationResult(
        detections=detections,
        metadata={
            "model": "classical_hybrid_quad",
            "inference_time_ms": 0.0,
            "fallback_used": True,
            "candidate_sources": source_counts,
            "rejected_candidates": rejected_counts,
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
    source_box = _source_box_from_input_box(clipped_input_box, transform)
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
) -> tuple[dict[str, float], dict[str, float]] | None:
    x1 = cx - box_width / 2.0
    y1 = cy - box_height / 2.0
    x2 = cx + box_width / 2.0
    y2 = cy + box_height / 2.0
    return _normalize_xyxy(x1, y1, x2, y2, transform=transform)


def _box_to_points(box: dict[str, float], width: int, height: int) -> np.ndarray:
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
    expanded[np.isclose(expanded[:, 0], max_x), 0] = np.minimum(mask_width, expanded[np.isclose(expanded[:, 0], max_x), 0] + 1.0)
    expanded[np.isclose(expanded[:, 1], max_y), 1] = np.minimum(mask_height, expanded[np.isclose(expanded[:, 1], max_y), 1] + 1.0)
    return expanded


def _polygon_from_binary_mask(binary_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return None
    perimeter = cv2.arcLength(contour, True)
    for epsilon_ratio in (0.004, 0.008, 0.012, 0.02, 0.035, 0.05):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if 4 <= len(approx) <= 48:
            return _expand_mask_polygon_to_pixel_edges(approx, binary_mask.shape[1], binary_mask.shape[0])
    hull = cv2.convexHull(contour).reshape(-1, 2)
    if len(hull) >= 4:
        return _expand_mask_polygon_to_pixel_edges(hull, binary_mask.shape[1], binary_mask.shape[0])
    return None


def _polygon_from_proto_mask(
    proto: np.ndarray,
    coefficients: np.ndarray,
    box: dict[str, float],
    *,
    input_width: int,
    input_height: int,
) -> np.ndarray | None:
    channels, mask_height, mask_width = proto.shape
    if coefficients.size != channels:
        return None
    logits = np.matmul(coefficients.astype(np.float32), proto.reshape(channels, -1)).reshape(mask_height, mask_width)
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
    return _polygon_from_binary_mask(crop_mask)


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
    mask_scores.setdefault("area_ratio", round(float(cv2.contourArea(points.reshape(-1, 1, 2)) / max(1, image_width * image_height)), 4))
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


def _nms_detections(detections: list[SegmentationDetection], threshold: float = 0.45) -> list[SegmentationDetection]:
    selected: list[SegmentationDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        candidate_box = detection.bounding_box
        candidate_pixel = (
            int(candidate_box["x1"] * 10000),
            int(candidate_box["y1"] * 10000),
            int((candidate_box["x2"] - candidate_box["x1"]) * 10000),
            int((candidate_box["y2"] - candidate_box["y1"]) * 10000),
        )
        if any(
            _box_overlap_ratio(
                candidate_pixel,
                (
                    int(kept.bounding_box["x1"] * 10000),
                    int(kept.bounding_box["y1"] * 10000),
                    int((kept.bounding_box["x2"] - kept.bounding_box["x1"]) * 10000),
                    int((kept.bounding_box["y2"] - kept.bounding_box["y1"]) * 10000),
                ),
            )[0]
            > threshold
            for kept in selected
        ):
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
    detections: list[SegmentationDetection] = []
    for array in _prediction_rows(outputs):
        for row in array:
            row = row.reshape(-1)
            normalized_boxes: tuple[dict[str, float], dict[str, float]] | None = None
            mask_coefficients: np.ndarray | None = None
            source = "yolo_box"
            if proto is not None and row.size >= 4 + mask_channels + 1:
                class_count = int(row.size - 4 - mask_channels)
                if class_count < 1:
                    continue
                class_scores = row[4 : 4 + class_count]
                confidence = float(np.max(class_scores))
                normalized_boxes = _normalize_cxcywh(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    transform=transform,
                )
                mask_coefficients = row[4 + class_count : 4 + class_count + mask_channels]
                source = "yolo_seg_mask"
            elif row.size <= 7:
                confidence = float(row[4])
                normalized_boxes = _normalize_xyxy(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    transform=transform,
                )
            else:
                class_scores = row[4:]
                confidence = float(np.max(class_scores))
                normalized_boxes = _normalize_cxcywh(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    transform=transform,
                )
            if confidence < settings.yolo_confidence_threshold:
                continue
            if normalized_boxes is None:
                continue
            source_box, input_box = normalized_boxes
            points = None
            if proto is not None and mask_coefficients is not None:
                points = _polygon_from_proto_mask(
                    proto,
                    mask_coefficients,
                    input_box,
                    input_width=input_width,
                    input_height=input_height,
                )
                if points is not None:
                    points = _model_points_to_source(points, transform)
            if points is None:
                points = _box_to_points(source_box, width, height)
            detection = _make_detection(
                box=source_box,
                points=points,
                confidence=confidence,
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


def segment_album_page(image: np.ndarray, settings: Settings) -> SegmentationResult:
    model_path = Path(settings.yolo_model_path)
    metadata: dict = {
        "model_path": str(model_path),
        "confidence_threshold": settings.yolo_confidence_threshold,
    }
    if model_path.exists():
        try:
            import onnxruntime as ort

            started = perf_counter()
            session = _cached_onnx_session(str(model_path), model_path.stat().st_mtime_ns)
            input_meta = session.get_inputs()[0]
            input_shape = input_meta.shape
            target_height = int(input_shape[2] if isinstance(input_shape[2], int) else 640)
            target_width = int(input_shape[3] if isinstance(input_shape[3], int) else 640)
            letterboxed, transform = _letterbox_image(image, target_width, target_height)
            tensor = letterboxed[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            outputs = session.run(None, {input_meta.name: tensor[np.newaxis, ...]})
            detections = _parse_simple_box_outputs(
                outputs,
                image.shape[1],
                image.shape[0],
                settings,
                input_width=target_width,
                input_height=target_height,
            )
            cache_info = _cached_onnx_session.cache_info()
            metadata.update(
                {
                    "model": "yolov8_seg_onnx",
                    "onnxruntime_version": ort.__version__,
                    "providers": session.get_providers(),
                    "session_cache": {"strategy": "path_mtime_lru", "size": cache_info.currsize},
                    "inference_time_ms": round((perf_counter() - started) * 1000, 2),
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
            if detections:
                return SegmentationResult(detections=detections, metadata=metadata)
            metadata["onnx_parse_warning"] = "No supported detections were parsed from model outputs."
        except Exception as exc:  # noqa: BLE001 - model loading should fail into reviewable metadata.
            metadata["onnx_error"] = str(exc)

    if not settings.enable_classical_segmentation_fallback:
        return SegmentationResult(detections=[], metadata={**metadata, "fallback_used": False})

    fallback = detect_photos_classical(image)
    return SegmentationResult(detections=fallback.detections, metadata={**metadata, **fallback.metadata})
