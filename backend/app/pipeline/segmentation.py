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


def _edge_support_score(edge_map: np.ndarray, points: np.ndarray) -> float:
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
    return float(np.mean(side_scores)) if side_scores else 0.0


def _border_contrast_score(image: np.ndarray, points: np.ndarray) -> float:
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
    return float(np.mean(side_scores)) if side_scores else 0.0


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
        filtered.append(candidate)
    return filtered


def detect_photos_classical(image: np.ndarray, *, min_area_ratio: float = 0.02) -> SegmentationResult:
    height, width = image.shape[:2]
    image_area = width * height
    edge_map = _edge_support_map(image)
    filtered_contours = _consolidate_candidates(_find_contour_candidates(image, min_area_ratio))

    detections: list[SegmentationDetection] = []
    source_counts: dict[str, int] = {}
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
        edge_support = _edge_support_score(edge_map, polygon_points)
        border_contrast = _border_contrast_score(image, polygon_points)
        quad_area_ratio = cv2.contourArea(polygon_points.reshape(-1, 1, 2)) / image_area
        confidence = _candidate_confidence(
            edge_support=edge_support,
            border_contrast=border_contrast,
            area_ratio=max(candidate.area_ratio, quad_area_ratio),
            polygon_source=polygon_source,
            refined_by_lines=refined_by_lines,
        )
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        detections.append(
            SegmentationDetection(
                bounding_box=pixel_box_to_normalized(x, y, w, h, width, height),
                mask={
                    "polygon": _normalized_polygon(polygon_points, width, height),
                    "source": polygon_source,
                    "detector_source": candidate.source,
                    "corner_refinement": "line_segments" if refined_by_lines else "contour",
                    "scores": {
                        "edge_support": round(edge_support, 3),
                        "border_contrast": round(border_contrast, 3),
                        "area_ratio": round(float(max(candidate.area_ratio, quad_area_ratio)), 4),
                    },
                },
                confidence=confidence,
                aspect_ratio=quality.aspect_ratio,
                geometry_valid=quality.geometry_valid,
                review_reasons=quality.review_reasons,
            )
        )

    detections.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    return SegmentationResult(
        detections=detections,
        metadata={
            "model": "classical_hybrid_quad",
            "inference_time_ms": 0.0,
            "fallback_used": True,
            "candidate_sources": source_counts,
        },
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
