from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, islice
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.adjacent_prints import ADJACENT_PRINT_REVIEW, adjacent_album_edge_outlines
from app.pipeline.album_leaf_review import (
    ALBUM_LEAF_COUNT_REVIEW, ALBUM_LEAF_REVIEW_CHILD, ALBUM_LEAF_REVIEW_TRACES, album_leaf_review_traces,
)
from app.pipeline.album_windows import ALBUM_WINDOW_OCCLUDED, ALBUM_WINDOW_SOURCE, ALBUM_WINDOW_STOCK_UNCERTAIN, album_window_leaves
from app.pipeline.capture_background_review import ALBUM_BOUNDARY_TRACES, CAPTURE_BACKGROUND_COUNT_REVIEW, album_binding_traces
from app.pipeline.capture_fabric_review import CAPTURE_FABRIC_PROPOSAL, capture_fabric_traces
from app.pipeline.contested_scene_edge import CONTESTED_SCENE_EDGE_TRACES, contested_scene_edge_traces
from app.pipeline.contrast_crossing_review import CONTRAST_CROSSING_TRACES, contrast_crossing_traces, contrast_review_windows
from app.pipeline.crossing_prints import (
    CONTESTED_TILE_OUTLINE, CROSSING_PRINT_REVIEW, crossing_print_outlines, near_leaf_edge_traces, staggered_print_source_traces,
    tile_crosses_unsupported_frame_claim,
)
from app.pipeline.crossing_fallback_review import (
    CROSSING_FALLBACK_COUNT_REVIEW, CROSSING_FALLBACK_TRACES, crossing_fallback_review_traces,
)
from app.pipeline.dark_album_review import DARK_ALBUM_PROPOSAL, find_dark_mounted_print_review, inspect_dark_album_backing
from app.pipeline.edge_prints import (
    CAPTURE_EDGE_HOLD, CAPTURE_EDGE_SOURCE, CLIPPED_PRINT_OVERLAP_REVIEW, SCANNER_BACKING_REVIEW,
    clipped_fragment_needs_review, clipped_print_outlines, clipped_print_overlap_uncertain, scanner_backing_evidence,
    visible_capture_edge_traces,
)
from app.pipeline.embossed_album_pockets import (
    EMBOSSED_CARRIER_PROPOSAL, EMBOSSED_POCKET_PROPOSAL, embossed_outer_carrier_traces, embossed_pocket_traces,
)
from app.pipeline.film_carrier_review import refitted_film_carrier
from app.pipeline.frame_backing_review import FRAME_BACKING_COUNT_REVIEW, FRAME_BACKING_PROPOSAL, frame_backing_trace
from app.pipeline.fragmented_prints import (
    COMPACT_SCENE_CROSSING_TRACE, EXTENDED_PRINT_OUTLINE_REVIEW, FRAGMENT_OUTLINE_REVIEW, FRAGMENTED_PRINT_REVIEW,
    SOURCE_FRAGMENT_GROUP_REVIEW, SOURCE_FRAGMENT_PROPOSAL,
    compact_scene_crossing_trace, measure_extended_print_outline, measure_fragmented_print, nested_scene_fragment_continuities,
    nested_scene_fragment_with_shared_edges,
    suggest_dark_mounted_outline, suggest_fragmented_outline,
)
from app.pipeline.fusion_ownership import (
    BORROWED_FALLBACK_EDGE, NESTED_EDGE_CONTOUR, NESTED_EDGE_CONTOUR_REVIEW,
    NESTED_FALLBACK_CONTOUR, NESTED_FALLBACK_CONTOUR_REVIEW, _covered_area,
    borrowed_fallback_edge_traces, deferred_neighbor_masks, nested_fallback_contour_spans, unsupported_fallback_ownership,
)
from app.pipeline.geometry import (
    PolygonOverlap,
    normalized_polygon_overlap,
    normalized_vertices_to_pixels,
    order_quad_cyclic,
    pixel_vertices_to_normalized,
    quad_from_polygon,
)
from app.pipeline.image_ops import normalized_box_to_pixels, pixel_box_to_normalized
from app.pipeline.merged_print_review import (
    MERGED_PRINT_PAIR, MERGED_PRINT_PAIR_REVIEW, measured_contained_print_outline,
    measured_merged_print_pair, measured_merged_print_with_one_tile,
)
from app.pipeline.mounted_prints import mounted_print_evidence
from app.pipeline.mounted_window_review import MOUNTED_WINDOW_REVIEW_TRACES, mounted_window_traces
from app.pipeline.multipanel_prints import (
    FILM_GRID_REVIEW, GRID_OUTLINE_REVIEW_SOURCE, MULTIPANEL_PRINT_REVIEW, PAPER_GRID_REVIEW, PAPER_JOIN_REVIEW,
    RECOVERED_STOCK_SOURCE,
    film_grid_review_seam, grid_fragment_on_shared_stock, nested_print_on_shared_stock, paper_grid_review_seam,
    paper_join_review_traces, recover_shared_stock_outline, shared_stock_outline, transparent_film_rim,
)
from app.pipeline.near_capture_review import (
    NEAR_CAPTURE_PRINT_TRACES, near_capture_candidate_turn, near_capture_print_traces, quarter_turn,
)
from app.pipeline.neighboring_print_review import (
    FOLLOWUP_PRINT_EDGE_GUIDE, NEIGHBORING_PRINT_EDGE_TRACES, followup_print_edge_guide, followup_print_position,
    neighboring_layout_traces, neighboring_print_outline, next_print_position,
)
from app.pipeline.page_deskew import album_rotation, deskewed_view
from app.pipeline.partial_print_review import PARTIAL_PRINT_EXTENT_REVIEW, PARTIAL_PRINT_EXTENT_TRACES, partial_print_extent_traces
from app.pipeline.possible_print_regions import (
    POSSIBLE_PRINT_REGIONS, POSSIBLE_PRINT_REGIONS_REVIEW, measure_possible_print_regions, possible_print_region_window,
)
from app.pipeline.print_edges import (
    PRINT_EDGE_CONFIRMATION, _backing_color, _is_pale_stock, line_fit_omits_print_stock, refine_print_boundary,
)
from app.pipeline.print_outline_suggestion import suggest_outer_print_outline
from app.pipeline.printed_illustration_review import (
    PRINTED_ILLUSTRATION_COUNT_REVIEW, PRINTED_ILLUSTRATION_REVIEW_FRAME, PRINTED_ILLUSTRATION_REVIEW_ID,
    PRINTED_ILLUSTRATION_REVIEW_SIZE, find_illustration_frame, illustration_evidence_id, valid_illustration_frame,
)
from app.pipeline.projective_prints import PROJECTIVE_PRINT_PROPOSAL, projective_print_review
from app.pipeline.protected_regions import validate_protected_regions
from app.pipeline.quality import evaluate_segmentation_geometry
from app.pipeline.scalloped_prints import (
    SCALLOPED_PRINT_CONFLICT, SCALLOPED_PRINT_REVIEW, measure_scalloped_print, possible_layer_edge, suggest_scalloped_motif,
)
from app.pipeline.segmentation_filter import (
    CONTRAST_COMPONENT_SOURCE,
    SOURCE_CONTOUR_REVISION,
    paper_and_edge_rejection_reasons,
    paper_around_nested_print,
)
from app.pipeline.shared_motif_review import (
    SHARED_MOTIF_REVIEW, measure_shared_motif, motif_review_id, near_nested_model_masks,
)
from app.pipeline.source_fragment_review import source_fragment_role, suggest_source_fragment_group
from app.pipeline.tile_print_review import (
    CONTESTED_TILE_FOLLOWUP, TILED_PRINT_PROPOSAL, contested_tile_followup_outline,
    measured_adjacent_tile_outline, measured_grid_tile_outline, measured_recentered_ribbed_outline,
    opposing_view_print_outlines,
    ribbed_backing_tile_outline, tile_print_outline, tiled_outline_conflicts,
)

_CLASSICAL_MIN_AREA_RATIO = 0.02
_CLASSICAL_DUPLICATE_CHECK_LIMIT = 12
_MULTIPANEL_PAIR_LIMIT = 12
_MULTIPANEL_GROUP_LIMIT = 12
_MULTIPANEL_MAX_GROUP_SIZE = 6


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
class _TileReviewSearch:
    infer: Callable[[np.ndarray], list[SegmentationDetection]]
    initial_claims: list[dict]
    unfitted_seed: np.ndarray | None
    contested_scene_scouts: list[SegmentationDetection]


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


def _merge_axis_lines(lines: list[_AxisLine], tolerance: float) -> list[_AxisLine]:
    if not lines:
        return []
    merged: list[_AxisLine] = []
    current = sorted(lines, key=lambda line: line.position)
    group: list[_AxisLine] = [current[0]]
    position_sum = current[0].position
    for line in current[1:]:
        if abs(line.position - position_sum / len(group)) <= tolerance:
            group.append(line)
            position_sum += line.position
            continue
        merged.extend(_merge_overlapping_axis_lines(group))
        group = [line]
        position_sum = line.position
    merged.extend(_merge_overlapping_axis_lines(group))
    return sorted(merged, key=lambda line: line.position)


def _merge_overlapping_axis_lines(group: list[_AxisLine]) -> list[_AxisLine]:
    ordered = sorted(group, key=lambda line: line.span_start)
    merged: list[_AxisLine] = []
    overlapping = [ordered[0]]
    end = ordered[0].span_end
    for line in ordered[1:]:
        if line.span_start > end:
            merged.append(_merge_axis_line_group(overlapping))
            overlapping = []
        overlapping.append(line)
        end = max(end, line.span_end)
    merged.append(_merge_axis_line_group(overlapping))
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


def _find_line_pair_candidates(
    image: np.ndarray, min_area_ratio: float, *, edge_map: np.ndarray | None = None,
) -> list[_ContourCandidate]:
    height, width = image.shape[:2]
    image_area = width * height
    if edge_map is None:
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
    if len(vertical) < 2:
        return []

    positions = np.array([line.position for line in vertical])
    starts = np.array([line.span_start for line in vertical])
    ends = np.array([line.span_end for line in vertical])
    candidates: list[_ContourCandidate] = []
    for top_index, top in enumerate(horizontal):
        for bottom in horizontal[top_index + 1 :]:
            y1, y2 = top.position, bottom.position
            rect_height = y2 - y1
            if rect_height < min_dimension * 0.10:
                continue
            estimated_width = (top.span_end - top.span_start + bottom.span_end - bottom.span_start) / 2
            x_tolerance = min(min_dimension * 0.045, max(8.0, estimated_width * 0.16))
            y_tolerance = min(min_dimension * 0.045, max(8.0, rect_height * 0.16))
            if max(abs(top.span_start - bottom.span_start), abs(top.span_end - bottom.span_end)) > 2 * x_tolerance:
                continue

            # Each side must end near both corners. Long, crossing lines in a texture
            # can cover a rectangle without supporting any of its corners.
            vertical_error = np.maximum(abs(starts - y1), abs(ends - y2)) / y_tolerance
            left_error = np.maximum(abs(positions - top.span_start), abs(positions - bottom.span_start)) / x_tolerance
            right_error = np.maximum(abs(positions - top.span_end), abs(positions - bottom.span_end)) / x_tolerance
            left_scores = np.maximum(left_error, vertical_error)
            right_scores = np.maximum(right_error, vertical_error)
            left_index, right_index = int(left_scores.argmin()), int(right_scores.argmin())
            if max(left_scores[left_index], right_scores[right_index]) > 1.0:
                continue
            x1, x2 = positions[left_index], positions[right_index]
            rect_width = x2 - x1
            if rect_width < min_dimension * 0.10 or rect_width * rect_height / image_area > 0.35:
                continue
            contour = np.rint([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).astype(np.int32).reshape(-1, 1, 2)
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


def _component_has_print_stock(lab: np.ndarray, quad: np.ndarray, background: np.ndarray, threshold: float) -> bool:
    edges = np.roll(quad, -1, axis=0) - quad
    inward = np.column_stack((-edges[:, 1], edges[:, 0]))
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    points = quad[:, None] + np.linspace(.12, .88, 30, dtype=np.float32)[None, :, None] * edges[:, None]
    offsets = points[:, None] + np.float32([2, -2])[None, :, None, None] * inward[:, None, None]
    coords = offsets.reshape(8, -1, 2)
    if np.any((coords < 0) | (coords > np.array(lab.shape[1::-1]) - 1)):
        return False
    colors = cv2.remap(lab, coords[:, :, 0].copy(), coords[:, :, 1].copy(), cv2.INTER_LINEAR).reshape(4, 2, -1, 3)
    inside, outside = colors[:, 0], colors[:, 1]
    stock = np.median(inside, axis=(0, 1))
    if (not _is_pale_stock(stock) or np.linalg.norm(stock - background) <= threshold
            or np.any(np.mean(np.linalg.norm(inside - stock, axis=2) <= 10, axis=1) < .75)
            or np.any(np.mean(np.linalg.norm(outside - background, axis=2) <= threshold, axis=1) < .58)):
        return False

    target = np.float32([[0, 0], [63, 0], [63, 63], [0, 63]])
    interior = cv2.warpPerspective(lab, cv2.getPerspectiveTransform(quad, target), (64, 64))[6:-6, 6:-6]
    if np.mean(np.linalg.norm(interior - background, axis=2) > threshold) < .55:
        return False
    # Pictures and plain labels can both have a bright rectangular border.
    bins = (interior // 4).astype(np.int32).reshape(-1, 3) @ [4096, 64, 1]
    return bool(np.unique(bins, return_counts=True)[1].max() < .65 * len(bins))


def _find_separated_contrast_prints(image: np.ndarray, settings: Settings) -> list[SegmentationDetection]:
    height, width = image.shape[:2]
    area = width * height
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    background, _ = _frame_background_lab_model(lab)
    squared_distance = np.zeros((height, width), np.float32)
    for index, tone in enumerate(background):
        difference = lab[:, :, index].astype(np.float32) - tone
        np.square(difference, out=difference)
        squared_distance += difference
    threshold = max(12., np.sqrt(np.quantile(squared_distance, .25)) + 9.)
    foreground = np.uint8(squared_distance > threshold**2)
    # Closing these components can bridge the visible album paper between prints.
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        if not .008 * area <= cv2.contourArea(contour) <= .10 * area:
            continue
        seed = cv2.approxPolyDP(contour, .024 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(seed) != 4 or not cv2.isContourConvex(seed):
            continue
        seed = order_quad_cyclic(seed)
        if not _component_has_print_stock(lab, seed, background, threshold):
            continue
        quad, sides, reasons = refine_print_boundary(image, seed)
        if quad is None or reasons or any(
            side["support"] < .80 or (side["corner_support"] or 0) < .60 for side in sides.values()
        ) or not _component_has_print_stock(lab, quad, background, threshold):
            continue
        detection = _make_detection(
            box={}, points=quad, confidence=min(.98, min(side["support"] for side in sides.values())),
            image_width=width, image_height=height, settings=settings,
            source="classical_border_quad", detector_source=CONTRAST_COMPONENT_SOURCE,
        )
        if detection is not None and detection.geometry_valid:
            candidates.append(detection)
    return candidates


def _closed_edge_contours(gray: np.ndarray) -> Iterator[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
        yield from contours


def _find_contour_candidates(image: np.ndarray, min_area_ratio: float) -> list[_ContourCandidate]:
    height, width = image.shape[:2]
    candidates = []
    for contour in _closed_edge_contours(_enhanced_gray(image)):
        candidate = _candidate_from_contour(
            contour, width=width, height=height, image_area=width * height,
            min_area_ratio=min_area_ratio, source="edge_contour",
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(_find_contrast_candidates(image, min_area_ratio))
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


def _defer_nested_fallback_contours(
    source: np.ndarray, selected: list[SegmentationDetection],
) -> tuple[list[SegmentationDetection], list[dict]]:
    models = []
    contours = []
    for index, item in enumerate(selected):
        if not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad"):
            continue
        fusion = item.mask["fusion"]
        detector = item.mask.get("detector_source")
        score = fusion["boundary_evidence"]["score"]
        if (fusion["proposal_families"] == ["model"] and detector == "yolo_onnx"
                and item.mask.get("source") == "yolo_seg_mask" and item.confidence >= .86 and score >= .78):
            models.append(index)
        if (fusion["proposal_count"] == 1 and fusion["proposal_families"] == ["classical"]
                and detector == "edge_contour" and item.confidence >= .60 and score >= .65):
            contours.append(index)
    if not models or len(contours) < 2:
        return selected, []

    groups: dict[int, list[tuple[int, float]]] = {index: [] for index in models}
    polygons = [_detection_polygon(item) for item in selected]
    modeled = [index for index, item in enumerate(selected) if "model" in item.mask["fusion"]["proposal_families"]]
    for index in contours:
        owners = []
        for parent in models:
            if _normalized_box_overlap(selected[index].bounding_box, selected[parent].bounding_box)[2] < .95:
                continue
            overlap = normalized_polygon_overlap(polygons[index], polygons[parent])
            if overlap.first_containment >= .975:
                owners.append((parent, overlap.second_containment))
        if len(owners) != 1:
            continue
        parent, fraction = owners[0]
        if not .25 <= fraction <= .58 or any(
            _normalized_box_overlap(selected[index].bounding_box, selected[peer].bounding_box)[2] >= .15
            and normalized_polygon_overlap(polygons[index], polygons[peer]).first_containment >= .15
            for peer in modeled if peer != parent
        ):
            continue
        groups[parent].append((index, fraction))

    height, width = source.shape[:2]
    deferred = {}
    held = set()
    for parent, children in groups.items():
        if len(children) != 2:
            continue
        (first, first_fraction), (second, second_fraction) = children
        if not .65 <= first_fraction + second_fraction <= .95:
            continue
        if normalized_polygon_overlap(polygons[first], polygons[second]).iou > .035:
            continue
        try:
            spans = nested_fallback_contour_spans(source, polygons[parent], [polygons[first], polygons[second]])
        except (cv2.error, ValueError):
            continue
        if not spans:
            continue
        held.add(parent)
        parent_id = selected[parent].mask["fusion"]["cluster_id"]
        for index, span in zip((first, second), spans, strict=True):
            deferred[index] = {
                "id": selected[index].mask["fusion"]["cluster_id"],
                "reason": NESTED_FALLBACK_CONTOUR,
                "parent_cluster_id": parent_id,
                "polygon": _normalized_polygon(
                    order_quad_cyclic(normalized_vertices_to_pixels(polygons[index], width, height)), width, height,
                ),
                "interior_edge_traces": [_normalized_polygon(span, width, height)],
            }
    return [
        replace(item, review_reasons=[NESTED_FALLBACK_CONTOUR_REVIEW]) if index in held else item
        for index, item in enumerate(selected) if index not in deferred
    ], list(deferred.values())


def _defer_frame_backing_contours(
    source: np.ndarray, selected: list[SegmentationDetection],
) -> tuple[list[SegmentationDetection], list[dict]]:
    height, width = source.shape[:2]
    polygons = [_detection_polygon(item) for item in selected]
    owners = [index for index, item in enumerate(selected)
              if "model" in item.mask["fusion"]["proposal_families"]]
    deferred = {}
    held = set()
    for index, item in enumerate(selected):
        fusion = item.mask["fusion"]
        if (item.mask.get("detector_source") != "contrast_region" or fusion["proposal_families"] != ["classical"]
                or fusion["proposal_count"] != 1 or not item.geometry_valid or item.review_reasons
                or item.mask.get("warp_quad") or item.confidence > .68 or fusion["boundary_evidence"]["score"] >= .55):
            continue
        polygon = polygons[index]
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        proposal = normalized_vertices_to_pixels(polygon, width, height)
        distance = np.minimum(proposal, np.float32([width - 1, height - 1]) - proposal)
        if np.count_nonzero(distance.min(axis=1) <= 2) != 2:
            continue
        overlaps = [(peer, normalized_polygon_overlap(polygon, polygons[peer])) for peer in owners
                    if _normalized_box_overlap(item.bounding_box, selected[peer].bounding_box)[2] >= .12]
        overlaps = [(peer, overlap) for peer, overlap in overlaps if overlap.first_containment >= .12]
        if len(overlaps) != 1:
            continue
        (peer, overlap), = overlaps
        neighbor = selected[peer]
        if (not .55 <= overlap.first_containment <= .84 or overlap.second_containment < .86
                or not neighbor.geometry_valid or neighbor.review_reasons or neighbor.mask.get("warp_quad")
                or neighbor.confidence < .78 or neighbor.mask["fusion"]["boundary_evidence"]["score"] < .84):
            continue
        if len(polygons[peer]) != 4 or not cv2.isContourConvex(polygons[peer]):
            continue
        try:
            surface = frame_backing_trace(source, proposal, normalized_vertices_to_pixels(polygons[peer], width, height))
        except (cv2.error, ValueError):
            continue
        if surface is None:
            continue
        held.add(peer)
        deferred[index] = {
            "id": fusion["cluster_id"],
            "reason": FRAME_BACKING_PROPOSAL,
            "parent_cluster_id": neighbor.mask["fusion"]["cluster_id"],
            "polygon": _normalized_polygon(order_quad_cyclic(proposal), width, height),
            "surface_edge_traces": [_normalized_polygon(surface, width, height)],
        }
    return [
        replace(item, review_reasons=[FRAME_BACKING_COUNT_REVIEW]) if index in held else item
        for index, item in enumerate(selected) if index not in deferred
    ], list(deferred.values())


def _defer_borrowed_fallback_contours(
    source: np.ndarray, selected: list[SegmentationDetection],
) -> tuple[list[SegmentationDetection], list[dict]]:
    models = [index for index, item in enumerate(selected) if "model" in item.mask["fusion"]["proposal_families"]]
    if not any(.40 <= selected[index].confidence < .70 for index in models):
        return selected, []
    height, width = source.shape[:2]
    polygons = [_detection_polygon(item) for item in selected]
    deferred = {}
    for index, item in enumerate(selected):
        fusion = item.mask["fusion"]
        if (item.mask.get("detector_source") != "contrast_region" or fusion["proposal_families"] != ["classical"]
                or fusion["proposal_count"] != 1 or not item.geometry_valid or item.review_reasons
                or item.mask.get("warp_quad") or item.confidence > .68 or fusion["boundary_evidence"]["score"] >= .55):
            continue
        polygon = polygons[index]
        overlaps = [(peer, normalized_polygon_overlap(polygon, polygons[peer])) for peer in models
                    if _normalized_box_overlap(item.bounding_box, selected[peer].bounding_box)[2] >= .12]
        overlaps = [(peer, overlap) for peer, overlap in overlaps if overlap.first_containment >= .12]
        if len(overlaps) != 1:
            continue
        (peer, overlap), = overlaps
        anchor = selected[peer]
        boundary = anchor.mask["fusion"]["boundary_evidence"]
        if (not .70 <= overlap.first_containment <= .90 or not .55 <= overlap.second_containment <= .82
                or anchor.mask.get("source") != "yolo_seg_mask" or anchor.mask.get("detector_source") != "yolo_onnx"
                or not anchor.geometry_valid or anchor.review_reasons or anchor.mask.get("warp_quad")
                or not .40 <= anchor.confidence < .70 or boundary["score"] < .65 or boundary.get("border_contrast", 0) < .70):
            continue
        proposal = normalized_vertices_to_pixels(polygon, width, height)
        owner = normalized_vertices_to_pixels(polygons[peer], width, height)
        try:
            traces = borrowed_fallback_edge_traces(source, proposal, owner)
        except (cv2.error, ValueError):
            continue
        if traces is None:
            continue
        shared, interior, exposed = traces
        deferred[index] = {
            "id": fusion["cluster_id"],
            "reason": BORROWED_FALLBACK_EDGE,
            "parent_cluster_id": anchor.mask["fusion"]["cluster_id"],
            "polygon": item.mask["polygon"],
            "borrowed_edge_traces": [_normalized_polygon(shared, width, height)],
            "interior_edge_traces": [_normalized_polygon(interior, width, height)],
            "exposed_edge_traces": [_normalized_polygon(exposed, width, height)],
            "source_mask": {key: item.mask[key] for key in ("source", "detector_source", "polygon", "visible_polygon",
                            "visible_polygon_source", "proposal_visible_polygon") if key in item.mask},
        }
    return [item for index, item in enumerate(selected) if index not in deferred], list(deferred.values())


def fuse_candidate_evidence(
    image: np.ndarray,
    model_detections: list[SegmentationDetection],
    classical_detections: list[SegmentationDetection],
    *,
    model_evidence: dict[int, dict[str, float]] | None = None,
    ownership_source: np.ndarray | None = None,
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

    model_outlines = [
        _detection_polygon(item["detection"]) for item in records
        if ownership_source is not None and item["family"] == "model"
        and item["detection"].mask.get("source") == "yolo_seg_mask"
        and item["detection"].geometry_valid and not item["detection"].review_reasons
        and item["detection"].confidence >= .70 and item["boundary"]["score"] >= .65
        and item["boundary"].get("border_contrast", 0) >= .70
    ]
    selected: list[SegmentationDetection] = []
    cluster_metadata: list[dict] = []
    deferred = []
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
        if (ownership_source is not None and model_outlines and len(cluster) == 1
                and winner["family"] == "classical" and not winner_detection.review_reasons
                and winner_detection.mask.get("detector_source") in {"edge_contour", "contrast_region"}):
            proposal = _detection_polygon(winner_detection)
            try:
                conflict = unsupported_fallback_ownership(ownership_source, proposal, model_outlines)
            except (cv2.error, ValueError):
                conflict = None
            if conflict:
                deferred.append({"id": cluster_index, "reason": conflict, "polygon": [
                    {"x": float(x), "y": float(y)} for x, y in order_quad_cyclic(proposal)
                ]})
                cluster_metadata.append({**provenance, "decision": "deferred_for_count_review", "selection_reason": conflict})
                continue
        shared_stock = next((item["detection"] for item in ranked
                             if MULTIPANEL_PRINT_REVIEW in item["detection"].review_reasons
                             and item["detection"].mask.get("visible_polygon")), None)
        if shared_stock is not None and shared_stock is not winner_detection:
            winner_detection = replace(winner_detection, mask={
                **winner_detection.mask, "visible_polygon": shared_stock.mask["visible_polygon"],
            }, review_reasons=[*winner_detection.review_reasons, MULTIPANEL_PRINT_REVIEW])
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

    if ownership_source is not None:
        selected, nested = _defer_nested_fallback_contours(ownership_source, selected)
        selected, backing = _defer_frame_backing_contours(ownership_source, selected)
        selected, borrowed = _defer_borrowed_fallback_contours(ownership_source, selected)
        for offer in (*nested, *backing, *borrowed):
            cluster_metadata[offer["id"]].update(decision="deferred_for_count_review", selection_reason=offer["reason"])
        deferred.extend((*nested, *backing, *borrowed))
        deferred.sort(key=lambda offer: offer["id"])
    selected.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    return selected, {
        "strategy": "mask_aware_staged_evidence_v1",
        "model_proposals": len(model_detections),
        "classical_proposals": len(classical_detections),
        "selected_instances": len(selected),
        "model_hypotheses_replaced": replacements,
        "clusters": cluster_metadata,
        **({"deferred_print_proposals": deferred} if deferred else {}),
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


def _group_multipanel_prints(
    image: np.ndarray, detections: list[SegmentationDetection],
) -> tuple[list[SegmentationDetection], int]:
    if len(detections) < 3:
        return detections, 0
    height, width = image.shape[:2]
    polygons = [_detection_polygon(item) for item in detections]
    areas = [float(cv2.contourArea(polygon)) for polygon in polygons]
    removed = set()
    grouped = {}
    order = sorted(range(len(detections)), key=areas.__getitem__, reverse=True)
    for index in order:
        if index in removed or not .025 <= areas[index] <= .75:
            continue
        parent = detections[index]
        children = _distinct_nested_children(index, detections, areas)
        children = [child for child in children if child not in removed
                    and normalized_polygon_overlap(polygons[child], polygons[index]).first_containment >= .96]
        if len(children) < 2 or sum(areas[child] for child in children) < .27 * areas[index]:
            continue
        if any(detections[child].mask.get("detector_source") == "chromatic_mount" for child in children):
            continue
        panels = [normalized_vertices_to_pixels(polygons[child], width, height) for child in children]
        try:
            seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygons[index], width, height))
            quad = shared_stock_outline(image, seed, panels)
        except (cv2.error, ValueError):
            continue
        if quad is None:
            continue
        proposed = _normalized_polygon(quad, width, height)
        measured = pixel_vertices_to_normalized(quad, width, height)
        if normalized_polygon_overlap(polygons[index], measured).iou < .90 or any(
            normalized_polygon_overlap(polygons[child], measured).first_containment < .96 for child in children
        ):
            continue
        # Keep the detector's outline for scoring and seed both crop modes with
        # the measured physical perimeter. Confirmation remains a review action.
        grouped[index] = replace(parent, mask={**parent.mask, "visible_polygon": proposed},
                                 review_reasons=list(dict.fromkeys([*parent.review_reasons, MULTIPANEL_PRINT_REVIEW])))
        removed.update(children)
    return [grouped.get(index, item) for index, item in enumerate(detections) if index not in removed], len(removed)


def _adjacent_panel_boxes(first: dict[str, float], second: dict[str, float]) -> bool:
    for axis, across in (("x", "y"), ("y", "x")):
        shared = min(first[f"{across}2"], second[f"{across}2"]) - max(first[f"{across}1"], second[f"{across}1"])
        shorter = min(first[f"{across}2"] - first[f"{across}1"], second[f"{across}2"] - second[f"{across}1"])
        extent = max(first[f"{axis}2"], second[f"{axis}2"]) - min(first[f"{axis}1"], second[f"{axis}1"])
        gap = max(first[f"{axis}1"], second[f"{axis}1"]) - min(first[f"{axis}2"], second[f"{axis}2"])
        if shorter > 0 and shared >= .45 * shorter and -.40 * extent <= gap <= .22 * extent:
            return True
    return False


def _connected_panel_indices(seed: int, allowed: set[int], neighbors: dict[int, set[int]]) -> set[int]:
    connected = {seed}
    frontier = [seed]
    while frontier:
        unseen = (neighbors[frontier.pop()] & allowed) - connected
        connected.update(unseen)
        frontier.extend(unseen)
    return connected


def _adjacent_panel_groups(pairs: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    """Try neighboring layouts and their subsets when one neighbor may be a separate print."""
    neighbors: dict[int, set[int]] = {}
    for first, second in pairs:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    remaining = set(neighbors)
    groups = []
    while remaining:
        component = _connected_panel_indices(min(remaining), remaining, neighbors)
        remaining.difference_update(component)
        if 3 <= len(component) <= _MULTIPANEL_MAX_GROUP_SIZE:
            groups.append(tuple(sorted(component)))
        if 4 <= len(component) <= _MULTIPANEL_MAX_GROUP_SIZE + 1:
            for index in sorted(component):
                subset = component - {index}
                if _connected_panel_indices(min(subset), subset, neighbors) == subset:
                    groups.append(tuple(sorted(subset)))
    if len(groups) > _MULTIPANEL_GROUP_LIMIT:
        return pairs
    return sorted(groups, key=lambda group: (-len(group), group)) + pairs


def _can_seed_shared_stock(item: SegmentationDetection) -> bool:
    if ({MULTIPANEL_PRINT_REVIEW, FRAGMENTED_PRINT_REVIEW}.intersection(item.review_reasons)
            or item.mask.get("detector_source") in {"chromatic_mount", ALBUM_WINDOW_SOURCE, CONTRAST_COMPONENT_SOURCE}):
        return False
    # An interior image can legitimately be much narrower than a physical
    # print. The recovered outer outline must still pass the normal geometry check.
    return item.geometry_valid or (
        item.mask.get("source") == "yolo_seg_mask"
        and item.review_reasons in (["aspect_ratio_too_tall"], ["aspect_ratio_too_wide"])
    )


def _recover_multipanel_prints(
    image: np.ndarray, result: SegmentationResult, settings: Settings,
) -> SegmentationResult:
    detections = result.detections
    eligible = [index for index, item in enumerate(detections) if _can_seed_shared_stock(item)]
    possible_pairs = (
        (first, second) for first, second in combinations(eligible, 2)
        if _adjacent_panel_boxes(detections[first].bounding_box, detections[second].bounding_box)
    )
    pairs = list(islice(possible_pairs, _MULTIPANEL_PAIR_LIMIT + 1))
    if not pairs or len(pairs) > _MULTIPANEL_PAIR_LIMIT:
        # An untested pair could compete for ownership. Keep the original proposals.
        return result
    height, width = image.shape[:2]
    polygons = [_detection_polygon(item) for item in detections]
    native = [normalized_vertices_to_pixels(polygon, width, height) for polygon in polygons]
    recovered = []
    for indices in _adjacent_panel_groups(pairs):
        try:
            quad = recover_shared_stock_outline(image, [native[index] for index in indices])
        except (cv2.error, ValueError):
            continue
        if quad is None:
            continue
        area = cv2.contourArea(quad)
        if not .025 * width * height <= area <= .75 * width * height:
            continue
        item = _make_detection(
            box={}, points=quad, confidence=min(detections[index].confidence for index in indices),
            image_width=width, image_height=height, settings=settings,
            source=RECOVERED_STOCK_SOURCE, detector_source=RECOVERED_STOCK_SOURCE,
        )
        if item is None or not item.geometry_valid:
            continue
        outline = _detection_polygon(item)
        if any(
            _normalized_box_overlap(item.bounding_box, other.bounding_box)[0] > 0
            and normalized_polygon_overlap(outline, polygons[index]).second_containment > .03
            for index, other in enumerate(detections) if index not in indices
        ):
            continue
        recovered.append((indices, replace(item, review_reasons=[MULTIPANEL_PRINT_REVIEW])))
    if not recovered:
        return result
    used = {index for indices, _ in recovered for index in indices}
    if len(used) != sum(len(indices) for indices, _ in recovered) or any(
        _normalized_box_overlap(first.bounding_box, second.bounding_box)[0] > 0
        and normalized_polygon_overlap(_detection_polygon(first), _detection_polygon(second)).iou > .01
        for (_, first), (_, second) in combinations(recovered, 2)
    ):
        return result
    updated = [item for index, item in enumerate(detections) if index not in used]
    updated.extend(item for _, item in recovered)
    updated.sort(key=lambda item: (item.bounding_box["y1"], item.bounding_box["x1"]))
    return SegmentationResult(updated, {**result.metadata, "multipanel_outlines_recovered": len(recovered)})


def _unrotate_capture_quad(quad: np.ndarray, turns: int, width: int, height: int) -> np.ndarray:
    x, y = quad.T
    choices = ((x, y), (width - 1 - y, x), (width - 1 - x, height - 1 - y), (y, height - 1 - x))
    return order_quad_cyclic(np.column_stack(choices[turns]))


def _capture_split_claimants(
    quad: np.ndarray, on_current_side: dict[int, PolygonOverlap], polygons: dict[int, np.ndarray],
    detections: list[SegmentationDetection], width: int, height: int,
) -> list[int]:
    """Find rival pieces on different sides of the same measured capture corner."""
    if not on_current_side or max(match.second_containment for match in on_current_side.values()) >= .85:
        return []
    frame = [
        [(f"{axis}{suffix}", value) for suffix, value, edge in (("1", 0., 0), ("2", 1., limit))
         if np.count_nonzero(quad[:, coordinate] == edge) == 2]
        for coordinate, (axis, limit) in enumerate((("x", width - 1), ("y", height - 1)))
    ]
    if any(len(sides) != 1 for sides in frame):
        return []
    (x_side,), (y_side,) = frame
    outline = pixel_vertices_to_normalized(quad, width, height)
    left, top = outline.min(axis=0)
    right, bottom = outline.max(axis=0)
    outer_box = {"x1": left, "y1": top, "x2": right, "y2": bottom}
    claims = {index: normalized_polygon_overlap(polygon, outline) for index, polygon in polygons.items()
              if _normalized_box_overlap(outer_box, detections[index].bounding_box)[0] > 0}
    if any(match.first_containment >= .90 and match.second_containment >= .85 for match in claims.values()):
        return []
    parts = [index for index, match in claims.items()
             if .38 <= match.second_containment <= .83 and match.first_containment >= .96
             and detections[index].geometry_valid and not detections[index].review_reasons]
    models = [index for index in parts if detections[index].mask.get("source") == "yolo_seg_mask"]
    if len(models) != 2 or not any(index in on_current_side for index in models):
        return []
    first, second = models
    boxes = [detections[index].bounding_box for index in models]
    x_key, x_end = x_side
    y_key, y_end = y_side
    across_corner = ((abs(boxes[0][x_key] - x_end) < .008 and abs(boxes[1][y_key] - y_end) < .008)
                     or (abs(boxes[1][x_key] - x_end) < .008 and abs(boxes[0][y_key] - y_end) < .008))
    if not across_corner or normalized_polygon_overlap(polygons[first], polygons[second]).iou > .04:
        return []
    if _covered_area(quad, [normalized_vertices_to_pixels(polygons[index], width, height) for index in models]) < .85:
        return []
    classical = [index for index in parts if index in on_current_side
                 and detections[index].mask.get("detector_source") == "edge_contour"
                 and all(normalized_polygon_overlap(polygons[index], polygons[model]).second_containment >= .10
                         for model in models)]
    return [*models, *classical]


def _recover_capture_edge_prints(
    image: np.ndarray, result: SegmentationResult, settings: Settings,
) -> SegmentationResult:
    height, width = image.shape[:2]
    detections = result.detections
    eligible = [
        index for index, item in enumerate(detections)
        if item.mask.get("detector_source") not in {"chromatic_mount", ALBUM_WINDOW_SOURCE, CONTRAST_COMPONENT_SOURCE}
        and MULTIPANEL_PRINT_REVIEW not in item.review_reasons
    ]
    if len(eligible) < 2:
        return result
    polygons = {index: _detection_polygon(detections[index]) for index in eligible}
    proposals = []
    for turns, (key, position) in enumerate((("x2", 1.), ("y2", 1.), ("x1", 0.), ("y1", 0.))):
        edge = [index for index in eligible if abs(detections[index].bounding_box[key] - position) < .008]
        if len(edge) < 2:
            continue
        try:
            quads = [_unrotate_capture_quad(quad, turns, width, height)
                     for quad in clipped_print_outlines(np.rot90(image, turns))]
        except (cv2.error, ValueError):
            continue
        outline_owners = []
        for quad in quads:
            outline = pixel_vertices_to_normalized(quad, width, height)
            comparisons = {index: normalized_polygon_overlap(polygons[index], outline) for index in edge}
            matches = {index: overlap for index, overlap in comparisons.items()
                       if overlap.first_containment >= .88 and overlap.second_containment >= .42}
            outline_owners.append((quad, matches))
        degree = Counter(index for _, matches in outline_owners for index in matches)
        ambiguous = {index for _, matches in outline_owners for index in matches
                     if len(matches) != 1 or degree[index] != 1}
        supported_conflicts = []
        splits = []
        replacements = {}
        for quad, matches in outline_owners:
            split = _capture_split_claimants(quad, matches, polygons, detections, width, height)
            if split and all(other is quad or cv2.intersectConvexConvex(quad, other)[0] <= .5 for other in quads):
                ambiguous.update(split)
                splits.append((quad, split))
                continue
            # An ambiguous match leaves its contenders intact. It cannot
            # invalidate a disjoint outline with a unique owner.
            if len(matches) > 1:
                # The measured extent of a disjoint print can still mask its
                # pixels during the independent check of the exposed scanner.
                if (all(degree[index] == 1 and overlap.first_containment >= .97 for index, overlap in matches.items())
                        and any(overlap.second_containment >= .97 for overlap in matches.values())
                        and all(other is quad or cv2.intersectConvexConvex(quad, other)[0] <= 0.5 for other in quads)):
                    supported_conflicts.append((quad, matches))
                continue
            if not matches:
                continue
            index = next(iter(matches))
            if index in ambiguous:
                continue
            item = _make_detection(
                box={}, points=quad, confidence=detections[index].confidence,
                image_width=width, image_height=height, settings=settings,
                source=CAPTURE_EDGE_SOURCE, detector_source=CAPTURE_EDGE_SOURCE,
            )
            if item is None:
                ambiguous.add(index)
                continue
            mask = {**item.mask, "warp_quad": item.mask["polygon"], "capture_edge_clipped": True}
            replacements[index] = replace(item, mask=mask, review_reasons=[CAPTURE_EDGE_HOLD, *item.review_reasons])
        if len(replacements) >= 2:
            proposals.append((replacements, ambiguous, supported_conflicts, splits))
    # A corner can produce two distinct edge interpretations. Keep the
    # original suggestions in this case.
    if len(proposals) != 1:
        return result
    replacements, ambiguous, supported_conflicts, splits = proposals[0]
    updated = [replacements.get(index, item) for index, item in enumerate(detections)]
    metadata = {**result.metadata, "capture_edge_outlines_recovered": len(replacements)}
    recovered = [_detection_polygon(item) for item in replacements.values()]
    clipped = [normalized_vertices_to_pixels(outline, width, height) for outline in recovered]
    additional = [(quad, pixel_vertices_to_normalized(quad, width, height)) for quad, _ in supported_conflicts]
    rejected = set()
    overlap_reviews = 0
    for visible_quad, claimants in splits:
        traces = [_normalized_polygon(side, width, height)
                  for side in visible_capture_edge_traces(image, visible_quad)]
        for index in claimants:
            item = updated[index]
            updated[index] = replace(item, mask={**item.mask, "capture_split_review_traces": traces},
                                     review_reasons=[CAPTURE_EDGE_HOLD, *item.review_reasons])
    if splits:
        metadata["capture_edge_split_reviews"] = len(splits)
    for visible_quad, matches in supported_conflicts:
        if len(matches) != 2:
            continue
        fragment, full = sorted(matches, key=lambda index: matches[index].second_containment)
        item = detections[fragment]
        families = (item.mask.get("fusion") or {}).get("proposal_families", [])
        full_families = (detections[full].mask.get("fusion") or {}).get("proposal_families", [])
        outline = polygons[fragment]
        if (item.mask.get("detector_source") != "edge_contour" or "model" in families or "model" not in full_families
                or len(outline) != 4 or matches[fragment].first_containment < .985
                or matches[fragment].second_containment > .65
                or normalized_polygon_overlap(outline, polygons[full]).first_containment < .985):
            continue
        try:
            quad = order_quad_cyclic(normalized_vertices_to_pixels(outline, width, height))
            if clipped_fragment_needs_review(image, quad):
                updated[fragment] = replace(item, review_reasons=[CLIPPED_PRINT_OVERLAP_REVIEW, *item.review_reasons])
                parent = updated[full]
                traces = [_normalized_polygon(side, width, height)
                          for side in visible_capture_edge_traces(image, visible_quad)]
                updated[full] = replace(parent, mask={**parent.mask, "capture_edge_review_traces": traces})
                overlap_reviews += 1
        except (cv2.error, ValueError):
            continue
    for index, item in enumerate(detections):
        if (index in replacements or index in ambiguous or item.mask.get("detector_source") != "edge_contour"
                or "model" in (item.mask.get("fusion") or {}).get("proposal_families", [])
                or MULTIPANEL_PRINT_REVIEW in item.review_reasons):
            continue
        outline = _detection_polygon(item)
        if len(outline) != 4:
            continue
        overlaps = [normalized_polygon_overlap(outline, other) for other in recovered]
        shared = [clip for clip, overlap in zip(clipped, overlaps, strict=True)
                  if overlap.first_containment >= .985 and overlap.second_containment <= .45]
        crossing_occlusions = [quad for quad, other in additional
                               if normalized_polygon_overlap(outline, other).first_containment >= .02]
        recovered_crossings = sum(overlap.first_containment >= .02 for overlap in overlaps)
        crosses_multiple = recovered_crossings + len(crossing_occlusions) >= 3
        if len(shared) != 1 and not crosses_multiple:
            continue
        try:
            quad = order_quad_cyclic(normalized_vertices_to_pixels(outline, width, height))
            if len(shared) == 1 and clipped_print_overlap_uncertain(image, quad, shared[0]):
                updated[index] = replace(item, review_reasons=[CLIPPED_PRINT_OVERLAP_REVIEW, *item.review_reasons])
                overlap_reviews += 1
            elif crosses_multiple:
                evidence = scanner_backing_evidence(image, quad, clipped) if recovered_crossings >= 3 else None
                if evidence == "backing":
                    rejected.add(index)
                    continue
                # A disputed outline can also cover a separate dark print.
                if evidence is None and crossing_occlusions:
                    evidence = scanner_backing_evidence(image, quad, clipped + crossing_occlusions, allow_rejection=False)
                if evidence is not None:
                    updated[index] = replace(item, review_reasons=[SCANNER_BACKING_REVIEW, *item.review_reasons])
        except (cv2.error, ValueError):
            continue
    if overlap_reviews:
        metadata["capture_edge_overlap_reviews"] = overlap_reviews
    if rejected:
        metadata["scanner_backing_outlines_rejected"] = len(rejected)
        updated = [item for index, item in enumerate(updated) if index not in rejected]
    return SegmentationResult(updated, metadata)


def _verified_fragment_parent(
    image: np.ndarray, index: int, children: list[int],
    detections: list[SegmentationDetection], containers: dict[int, list[int]],
) -> SegmentationDetection | None:
    parent = detections[index]
    if (len(children) != 2 or parent.mask.get("detector_source") != "edge_contour"
            or _score_value(parent, "rectangularity") < .92 or _score_value(parent, "exterior_background_sides") != 4
            or any(detections[child].mask.get("detector_source") != "contrast_region" for child in children)
            or min(_score_value(detections[child], "rectangularity") for child in children) > .80):
        return None
    height, width = image.shape[:2]
    outer = _detection_polygon(parent)
    parts = [_detection_polygon(detections[child]) for child in children]
    if len(outer) != 4 or any(len(part) != 4 for part in parts):
        return None
    overlaps = [normalized_polygon_overlap(part, outer) for part in parts]
    if (min(overlap.first_containment for overlap in overlaps) < .94
            or not .58 <= sum(overlap.second_containment for overlap in overlaps) <= 1.02
            or normalized_polygon_overlap(*parts).iou > .10):
        return None
    try:
        measured = measure_fragmented_print(
            image, normalized_vertices_to_pixels(outer, width, height),
            [normalized_vertices_to_pixels(part, width, height) for part in parts],
        )
    except (cv2.error, ValueError):
        return None
    if measured is None:
        return None
    outline = pixel_vertices_to_normalized(measured, width, height)
    for other_index, other in enumerate(detections):
        if (other_index in containers or other_index in children
                or _normalized_box_overlap(parent.bounding_box, other.bounding_box)[0] == 0):
            continue
        overlap = normalized_polygon_overlap(outline, _detection_polygon(other))
        if overlap.second_containment > .04 and overlap.iou < .82:
            return None
    polygon = _normalized_polygon(measured, width, height)
    return replace(parent, mask={**parent.mask, "visible_polygon": polygon, "warp_quad": polygon, "fragmented_print": True},
                   review_reasons=[FRAGMENTED_PRINT_REVIEW, *parent.review_reasons])


def _filter_nested_classical_detections(
    detections: list[SegmentationDetection], *, source_image: np.ndarray | None = None,
    unresolved_containers: list[SegmentationDetection] | None = None,
) -> tuple[list[SegmentationDetection], dict[str, int]]:
    if len(detections) < 2:
        return detections, {}

    rejected: dict[int, str] = {}
    areas = [_normalized_box_area(detection.bounding_box) for detection in detections]

    containers = {}
    for index in range(len(detections)):
        children = _distinct_nested_children(index, detections, areas)
        child_area_ratio = sum(areas[child] for child in children) / max(areas[index], 1e-9)
        if areas[index] > 0.18 and len(children) >= 2 and child_area_ratio > 0.24:
            containers[index] = children
    verified = {}
    for index, children in containers.items():
        parent = (_verified_fragment_parent(source_image, index, children, detections, containers)
                  if source_image is not None else None)
        if parent is None:
            rejected[index] = "container_detection_with_multiple_photo_regions"
            if unresolved_containers is not None:
                unresolved_containers.append(detections[index])
        else:
            verified[index] = parent

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
    return [verified.get(index, detection) for index, detection in enumerate(detections) if index not in rejected], rejection_counts


def _attach_fragment_outline_suggestions(
    image: np.ndarray, unresolved: list[SegmentationDetection], selected: list[SegmentationDetection],
) -> list[SegmentationDetection]:
    height, width = image.shape[:2]
    polygons = [_detection_polygon(item) for item in selected]
    suggestions = {}
    conflicts = set()
    for outer in sorted(unresolved, key=lambda item: item.confidence, reverse=True):
        outline = _detection_polygon(outer)
        if (len(outline) != 4 or outer.mask.get("detector_source") != "edge_contour"
                or _score_value(outer, "rectangularity") < .92
                or _score_value(outer, "exterior_background_sides") != 4):
            continue
        overlapping = [index for index, item in enumerate(selected)
                       if _normalized_box_overlap(outer.bounding_box, item.bounding_box)[0] > 0]
        overlaps = {index: normalized_polygon_overlap(polygons[index], outline) for index in overlapping}
        children = [index for index, match in overlaps.items()
                    if len(polygons[index]) == 4 and .12 <= match.second_containment <= .65
                    and match.first_containment >= .94
                    and selected[index].mask.get("detector_source") in {"contrast_region", "edge_contour"}]
        if (not 2 <= len(children) <= 4 or any(
            match.second_containment > .04 and index not in children for index, match in overlaps.items()
        )):
            continue
        pairs = [(first, second) for first, second in combinations(range(len(children)), 2)
                 if .55 <= sum(overlaps[children[index]].second_containment for index in (first, second)) <= 1.02
                 and normalized_polygon_overlap(polygons[children[first]], polygons[children[second]]).iou <= .10]
        if not pairs:
            continue
        try:
            hypothesis = suggest_fragmented_outline(
                image, normalized_vertices_to_pixels(outline, width, height),
                [normalized_vertices_to_pixels(polygons[index], width, height) for index in children], pairs,
            )
        except (cv2.error, ValueError):
            continue
        if hypothesis is None:
            continue
        quad, uncertain = hypothesis
        normalized = pixel_vertices_to_normalized(quad, width, height)
        if (normalized_polygon_overlap(outline, normalized).iou < .96 or any(
            normalized_polygon_overlap(polygons[index], normalized).first_containment < .93 for index in children
        )):
            continue
        for index in children:
            previous = suggestions.get(index)
            if previous is None:
                suggestions[index] = normalized, uncertain
            elif previous[1] != uncertain or normalized_polygon_overlap(previous[0], normalized).iou < .96:
                conflicts.add(index)
    for index, (outline, uncertain) in suggestions.items():
        item = selected[index]
        hint = {
            "quad": [{"x": float(x), "y": float(y)} for x, y in outline],
            "uncertain_edge": uncertain,
        }
        mask = item.mask if index in conflicts else {**item.mask, "fragment_outline_suggestion": hint}
        selected[index] = replace(item, mask=mask, review_reasons=[FRAGMENT_OUTLINE_REVIEW, *item.review_reasons])
    return selected


def _restore_classical_outline(
    image: np.ndarray, fragment: SegmentationDetection, outer: SegmentationDetection, neighbors: list[SegmentationDetection],
) -> SegmentationDetection | None:
    inner_polygon, outer_polygon = _detection_polygon(fragment), _detection_polygon(outer)
    if len(inner_polygon) != 4 or len(outer_polygon) != 4:
        return None
    scope = normalized_polygon_overlap(inner_polygon, outer_polygon)
    if scope.first_containment < .98 or not .72 <= scope.second_containment <= .95:
        return None
    height, width = image.shape[:2]
    inner = order_quad_cyclic(normalized_vertices_to_pixels(inner_polygon, width, height))
    seed = order_quad_cyclic(normalized_vertices_to_pixels(outer_polygon, width, height))
    if not cv2.isContourConvex(inner) or not cv2.isContourConvex(seed):
        return None
    try:
        measured = measure_extended_print_outline(image, inner, seed)
    except (ValueError, cv2.error):
        return None
    if measured is None:
        return None
    outline = pixel_vertices_to_normalized(measured, width, height)
    if normalized_polygon_overlap(outer_polygon, outline).iou < .90:
        return None
    for neighbor in neighbors:
        other = _detection_polygon(neighbor)
        for candidate in (outer_polygon, outline):
            conflict = normalized_polygon_overlap(candidate, other)
            if max(conflict.first_containment, conflict.second_containment) > .04:
                return None
    x, y, w, h = cv2.boundingRect(measured)
    dimensions = _quadrilateral_geometry_dimensions(measured, w, h)
    quality = evaluate_segmentation_geometry(*dimensions, min_aspect_ratio=.5, max_aspect_ratio=3.)
    if not quality.geometry_valid:
        return None
    mask = {
        **outer.mask, "polygon": _normalized_polygon(measured, width, height),
        "corner_refinement": "native_boundary", "extended_print_outline": True,
    }
    return replace(
        outer, mask=mask, bounding_box=pixel_box_to_normalized(x, y, w, h, width, height),
        aspect_ratio=quality.aspect_ratio, review_reasons=[EXTENDED_PRINT_OUTLINE_REVIEW, *outer.review_reasons],
    )


def select_candidate_set(
    detections: list[SegmentationDetection], *, source_image: np.ndarray | None = None,
) -> tuple[list[SegmentationDetection], dict[str, int]]:
    """Perform duplicate suppression only after every proposal was assessed.

    This intentionally differs from the historical contour-area-first NMS:
    a large page/container can no longer suppress a smaller valid print before
    it has a chance to fail the geometry and boundary gates.
    """
    unresolved: list[SegmentationDetection] = []
    filtered, rejected = _filter_nested_classical_detections(
        detections, source_image=source_image, unresolved_containers=unresolved if source_image is not None else None,
    )
    selected: list[SegmentationDetection] = []
    alternatives: dict[int, list[SegmentationDetection]] = {}
    duplicate_count = 0
    for candidate in sorted(filtered, key=lambda item: item.confidence, reverse=True):
        collisions = [index for index, kept in enumerate(selected)
                      if _normalized_box_overlap(candidate.bounding_box, kept.bounding_box)[0] > .82]
        if collisions:
            duplicate_count += 1
            if (source_image is not None and len(collisions) == 1
                    and candidate.mask.get("detector_source") == "contrast_region"
                    and selected[collisions[0]].mask.get("detector_source") == "edge_contour"):
                alternatives.setdefault(collisions[0], []).append(candidate)
            continue
        selected.append(candidate)
    if duplicate_count:
        rejected["near_duplicate_after_scoring"] = duplicate_count
    if source_image is not None and len(alternatives) <= _CLASSICAL_DUPLICATE_CHECK_LIMIT:
        for index, rivals in alternatives.items():
            if len(rivals) != 1:
                continue
            recovered = _restore_classical_outline(
                source_image, selected[index], rivals[0], selected[:index] + selected[index + 1:],
            )
            if recovered is not None:
                selected[index] = recovered
    if source_image is not None and unresolved:
        selected = _attach_fragment_outline_suggestions(source_image, unresolved, selected)
    return selected, rejected


def detect_photos_classical(
    image: np.ndarray, *, min_area_ratio: float = _CLASSICAL_MIN_AREA_RATIO,
    max_line_refinement_pixels: int | None = None,
    multipanel_source: np.ndarray | None = None,
) -> SegmentationResult:
    started = perf_counter()
    height, width = image.shape[:2]
    image_area = width * height
    edge_map = _edge_support_map(image)
    # Do not consolidate by area before candidate assessment.  Containers are
    # frequently larger than the true prints and used to erase them here.
    contours = _find_contour_candidates(image, min_area_ratio)
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

    def proposals():
        yield from contours
        # A raw fragment or album outline can fail assessment. Try the
        # existing line search if the precise path still has no print.
        if not contours or (multipanel_source is not None and not detections):
            yield from _find_line_pair_candidates(image, min_area_ratio, edge_map=edge_map)

    for candidate in proposals():
        polygon_points, polygon_source = _quadrilateral_from_contour(candidate.contour)
        _, _, candidate_width, candidate_height = candidate.box
        candidate_pixels = (candidate_width + line_pad) * (candidate_height + line_pad)
        refined_by_lines = False
        retained_stock = False
        if max_line_refinement_pixels is None or line_pixels + candidate_pixels <= max_line_refinement_pixels:
            contour = polygon_points
            fitted, refined_by_lines = _refine_quadrilateral_with_lines(image, contour, line_cache)
            retained_stock = (refined_by_lines and candidate.source == "edge_contour"
                              and polygon_source == "classical_border_quad" and multipanel_source is not None
                              and line_fit_omits_print_stock(multipanel_source, contour, fitted))
            polygon_points = contour if retained_stock else fitted
            refined_by_lines = refined_by_lines and not retained_stock
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
        polygon = _normalized_polygon(polygon_points, width, height)
        review = {"warp_quad": polygon, "paper_edge_confirmation": True} if retained_stock else {}
        detections.append(
            SegmentationDetection(
                bounding_box=pixel_box_to_normalized(x, y, w, h, width, height),
                mask={
                    "polygon": polygon,
                    "source": polygon_source,
                    "detector_source": candidate.source,
                    "corner_refinement": "line_segments" if refined_by_lines else "contour",
                    **review,
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
                review_reasons=[*quality.review_reasons, PRINT_EDGE_CONFIRMATION] if retained_stock else quality.review_reasons,
            )
        )

    if multipanel_source is not None:
        detections, panels = _group_multipanel_prints(multipanel_source, detections)
        if panels:
            rejected_counts["panels_on_shared_print_stock"] = panels
    detections, nested_rejections = select_candidate_set(detections, source_image=multipanel_source)
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
            "fragmented_prints_recovered": sum(FRAGMENTED_PRINT_REVIEW in item.review_reasons for item in detections),
            "fragment_outline_reviews": sum(FRAGMENT_OUTLINE_REVIEW in item.review_reasons for item in detections),
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
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    points = _clamp_points(points, image_width, image_height)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    area = cv2.contourArea(points)
    if w < 2 or h < 2 or area <= 0:
        return None
    polygon_reasons = []
    try:
        contour_width, contour_height = _quadrilateral_geometry_dimensions(points, w, h)
    except ValueError:
        # A visible mask can have positive area without four usable perspective corners.
        contour_width, contour_height = w, h
        polygon_reasons.append("PRINT_CORNERS_UNCERTAIN")
    quality = evaluate_segmentation_geometry(
        contour_width,
        contour_height,
        min_aspect_ratio=settings.segmentation_min_aspect_ratio,
        max_aspect_ratio=settings.segmentation_max_aspect_ratio,
    )
    normalized_box = pixel_box_to_normalized(x, y, w, h, image_width, image_height)
    clipped_box = _clip_normalized_box(normalized_box) or box
    mask_scores = dict(scores or {})
    mask_scores.setdefault("area_ratio", round(area / max(1, image_width * image_height), 4))
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
        geometry_valid=quality.geometry_valid and not polygon_reasons,
        review_reasons=quality.review_reasons + polygon_reasons,
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
    rejected: Counter[str] | None = None,
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
            if not np.isfinite(row[:4]).all() or not np.isfinite(confidences[index]):
                if rejected is not None:
                    rejected["nonfinite_prediction"] += 1
                continue
            normalized_boxes = normalize(*map(float, row[:4]), transform=transform, native=native)
            if normalized_boxes is None:
                continue
            source_box, input_box = normalized_boxes
            source = "yolo_box"
            points = None
            if has_masks:
                coefficients = row[score_end:]
                if not np.isfinite(coefficients).all():
                    if rejected is not None:
                        rejected["nonfinite_mask_coefficients"] += 1
                    continue
                source = "yolo_seg_mask"
                points = _polygon_from_proto_mask(
                    proto,
                    coefficients,
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
            elif rejected is not None:
                rejected["collapsed_source_polygon"] += 1
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


def refine_album_page_result(
    image: np.ndarray, result: SegmentationResult, *, source_image: np.ndarray | None = None,
) -> SegmentationResult:
    polygons = [detection.mask.get("polygon", []) for detection in result.detections]
    mount_rejections, quads = mounted_print_evidence(image, polygons)
    mount_polygons = [_normalized_polygon(order_quad_cyclic(quad), 1, 1) for quad in quads]
    paper_rejections = paper_and_edge_rejection_reasons(image, polygons + mount_polygons, source_image=source_image)
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
    panels = 0
    if source_image is not None:
        detections, panels = _group_multipanel_prints(source_image, detections)
    containers = _page_containers(detections, image)
    if not rejected and not quads and not containers and not panels:
        return result
    replacement_count = sum(len(polygons) + offset not in paper_rejections for offset in range(len(quads)))
    metadata = {
        **result.metadata,
        "mount_replacements": replacement_count,
        "page_proposal_rejections": rejected,
        "container_rejections": len(containers),
        "multipanel_fragments": panels,
    }
    if source_image is not None and containers:
        recovery = _mounted_container_review(source_image, detections, containers)
        if recovery is not None:
            fusion = dict(metadata.get("fusion") or {})
            pending = list(fusion.get("deferred_print_proposals") or [])
            if not any(proposal["polygon"] == recovery["polygon"] for proposal in pending):
                next_id = max((proposal["id"] for proposal in pending), default=-1) + 1
                pending.append({"id": next_id, **recovery})
                metadata["fusion"] = {**fusion, "deferred_print_proposals": pending}
    detections = [detection for index, detection in enumerate(detections) if index not in containers]
    detections.sort(key=lambda detection: (detection.bounding_box["y1"], detection.bounding_box["x1"]))
    return SegmentationResult(detections=detections, metadata=metadata)


def _mounted_container_review(
    image: np.ndarray, detections: list[SegmentationDetection], containers: set[int],
) -> dict | None:
    """Offer a possible whole print in count review without assigning the fragments."""
    height, width = image.shape[:2]
    polygons = [_detection_polygon(item) for item in detections]
    candidates = []
    for index in sorted(containers):
        item, polygon = detections[index], polygons[index]
        boundary = item.mask.get("boundary_evidence") or {}
        if (item.mask.get("source") != "yolo_seg_mask" or not item.geometry_valid or item.review_reasons
                or item.confidence < .80 or boundary.get("score", 0) < .70
                or np.any((polygon < .05) | (polygon > .95))):
            continue
        conflicts = {other: normalized_polygon_overlap(shape, polygon) for other, shape in enumerate(polygons)
                     if other != index}
        claims = [other for other, match in conflicts.items() if match.first_containment >= .98 and .05 <= match.iou <= .67]
        if (not 3 <= len(claims) <= _MULTIPANEL_MAX_GROUP_SIZE or any(other in containers for other in claims)
                or any(match.first_containment >= .08 for other, match in conflicts.items() if other not in claims)):
            continue
        try:
            fragments = [quad_from_polygon(normalized_vertices_to_pixels(polygons[child], width, height))[0]
                         for child in claims]
            distinct = {fragment.tobytes(): fragment for fragment in fragments}
            if len(distinct) < 3:
                continue
            native = normalized_vertices_to_pixels(polygon, width, height)
            quad = suggest_dark_mounted_outline(image, native, list(distinct.values()))
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        if quad is None or normalized_polygon_overlap(polygon, pixel_vertices_to_normalized(quad, width, height)).iou < .90:
            continue
        parts = [detections[child].mask.get("polygon") for child in claims]
        candidates.append({
            "reason": "possible_mounted_print",
            "polygon": _normalized_polygon(quad, width, height),
            "visible_polygon": item.mask.get("visible_polygon") or item.mask["polygon"],
            **({"fragment_polygons": parts} if all(parts) else {}),
        })
    return candidates[0] if len(candidates) == 1 else None


def _page_containers(detections: list[SegmentationDetection], image: np.ndarray) -> set[int]:
    if len(detections) < 2:
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


def _recover_separated_contrast_prints(
    image: np.ndarray, result: SegmentationResult, settings: Settings,
) -> SegmentationResult:
    candidates = _find_separated_contrast_prints(image, settings)
    if len(candidates) < 2:
        return result
    originals = result.detections
    original_polygons = [_detection_polygon(item) for item in originals]
    verified_polygons = [_detection_polygon(item) for item in candidates]
    no_overlap = PolygonOverlap(0., 0., 0.)
    comparisons = [
        [normalized_polygon_overlap(original, verified)
         if _normalized_box_overlap(item.bounding_box, candidate.bounding_box)[0] > 0 else no_overlap
         for candidate, verified in zip(candidates, verified_polygons, strict=True)]
        for item, original in zip(originals, original_polygons, strict=True)
    ]
    matches = [{index for index, overlap in enumerate(row)
                if overlap.first_containment >= .94 and overlap.second_containment >= .70}
               for row in comparisons]
    containers = {}
    height, width = image.shape[:2]
    for index, (item, row) in enumerate(zip(originals, comparisons, strict=True)):
        score = (item.mask.get("boundary_evidence") or {}).get("score", 1.)
        children = {child for child, overlap in enumerate(row) if overlap.second_containment >= .28}
        if matches[index] or score >= .55 or len(children) < 2 or sum(row[j].first_containment for j in children) < .24:
            continue
        try:
            seed, _ = quad_from_polygon(normalized_vertices_to_pixels(original_polygons[index], width, height))
            outline, _, reasons = refine_print_boundary(image, seed)
        except (ValueError, cv2.error):
            outline, reasons = None, []
        if outline is None or reasons:
            containers[index] = children

    adopted = {}
    replaced = set()
    for index, candidate in enumerate(candidates):
        duplicates = {old for old, group in enumerate(matches) if index in group}
        if any(comparisons[old][index].second_containment >= .94 for old in duplicates):
            continue
        if any(row[index].second_containment > .20 for old, row in enumerate(comparisons)
               if old not in duplicates and old not in containers):
            continue
        adopted[index] = candidate
        replaced.update(duplicates)
    if not adopted:
        return result

    # Keep a weak enclosing proposal if the separate prints do not survive the overlap checks.
    for old, children in containers.items():
        represented = [child for child in children if child in adopted
                       or any(child in group and other not in replaced for other, group in enumerate(matches))]
        if len(represented) < 2 or sum(comparisons[old][child].first_containment for child in represented) < .24:
            return result
    kept = [item for i, item in enumerate(originals) if i not in replaced and i not in containers]
    inputs = _boundary_inputs(image)
    for candidate in adopted.values():
        kept.append(replace(candidate, mask={**candidate.mask,
            "evidence_schema": "segmentation_detection_evidence_v1",
            "boundary_evidence": _detection_boundary_evidence(candidate, *inputs),
            "geometry_evidence": _detection_geometry_evidence(candidate),
        }))
    kept.sort(key=lambda item: (item.bounding_box["y1"], item.bounding_box["x1"]))
    return SegmentationResult(kept, {**result.metadata, "contrast_components": {
        "verified": len(candidates), "selected": len(adopted),
        "replaced": len(replaced), "containers_removed": len(containers),
    }})


def _shared_source_growth(
    first_raw: np.ndarray, first_measured: np.ndarray, second_raw: np.ndarray, second_measured: np.ndarray,
) -> bool:
    lower = np.maximum(first_measured.min(axis=0), second_measured.min(axis=0))
    upper = np.minimum(first_measured.max(axis=0), second_measured.max(axis=0))
    if np.any(upper - lower < 2):
        return False
    origin = np.floor(lower).astype(int)
    extent = np.ceil(upper).astype(int) - origin + 1

    def raster(points: np.ndarray) -> np.ndarray:
        vertices = np.rint(points - origin).astype(np.int32)
        return cv2.fillPoly(np.zeros(tuple(extent[::-1]), np.uint8), [vertices], 1)

    shared = raster(first_measured) & raster(second_measured) & (1 - (raster(first_raw) | raster(second_raw)))
    # A shared rasterized edge can be one pixel wide. Competing interior
    # extensions need review before assigning the same stock to two prints.
    return bool(cv2.erode(shared, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).any())


def _refine_source_contours(
    image: np.ndarray, result: SegmentationResult, settings: Settings, protected_regions: list[dict] | None,
) -> SegmentationResult:
    from app.pipeline.perspective import apply_refined_visibility, measure_detected_visibility
    from app.pipeline.protected_regions import detect_protected_regions

    height, width = image.shape[:2]
    regions = [] if not settings.enable_protected_region_guard else protected_regions
    if regions is None:
        regions = detect_protected_regions(image) if result.detections else []
    originals = [normalized_vertices_to_pixels(_detection_polygon(item), width, height) for item in result.detections]
    candidates = {}
    for index, (item, own) in enumerate(zip(result.detections, originals, strict=True)):
        if (item.mask.get("detector_source") in {CONTRAST_COMPONENT_SOURCE, RECOVERED_STOCK_SOURCE, CAPTURE_EDGE_SOURCE}
                or item.mask.get("nested_print_suggestion")
                or item.mask.get("visible_polygon_source") == GRID_OUTLINE_REVIEW_SOURCE
                or SCALLOPED_PRINT_CONFLICT in item.review_reasons):
            continue
        measured = measure_detected_visibility(
            image, item.mask, own, [other for i, other in enumerate(originals) if i != index], regions,
            geometry_valid=item.geometry_valid, protect_regions=settings.enable_protected_region_guard,
            enable_classical_retry=settings.enable_classical_segmentation_fallback,
        )
        if measured is not None:
            candidates[index] = measured
    conflicts = set()
    for (i, (first, _)), (j, (second, _)) in combinations(candidates.items(), 2):
        if _shared_source_growth(originals[i], first, originals[j], second):
            conflicts.update((i, j))
    updated = result.detections.copy()
    for index, (outline, source) in candidates.items():
        item = updated[index]
        if index in conflicts:
            updated[index] = replace(item, review_reasons=[*item.review_reasons, "IMAGE_VISIBILITY_UNCERTAIN"])
        else:
            mask = apply_refined_visibility(item.mask, {
                "refined_visible_polygon": _normalized_polygon(outline, width, height),
                "refined_visibility_source": source,
            })
            updated[index] = replace(item, mask=mask)
    return SegmentationResult(updated, {**result.metadata, "source_contour_refinement": {
        "revision": SOURCE_CONTOUR_REVISION,
        "refined_detections": len(candidates) - len(conflicts),
        "ownership_conflicts": len(conflicts),
    }})


def _recover_scalloped_prints(image: np.ndarray, result: SegmentationResult, *, trim_border: bool = True) -> SegmentationResult:
    backing = _backing_color(image)
    if backing[0] < 245 or np.linalg.norm(backing[1:] - 128) > 5:
        return result
    height, width = image.shape[:2]
    detections = result.detections
    polygons = [_detection_polygon(item) for item in detections]
    measures = {}
    for index, item in enumerate(detections):
        if item.review_reasons or item.mask.get("warp_quad") or not item.geometry_valid:
            continue
        try:
            seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygons[index], width, height))
            measured = measure_scalloped_print(image, seed, backing)
        except (cv2.error, ValueError):
            continue
        if measured is not None:
            measures[index] = measured
    if not measures:
        return result

    outlines = {index: pixel_vertices_to_normalized(item.outline, width, height) for index, item in measures.items()}
    alternatives = {}
    removed = set()
    for index in measures:
        if index in removed:
            continue
        competing = [other for other in measures if other != index and other not in removed
                     and _normalized_box_overlap(detections[index].bounding_box, detections[other].bounding_box)[0] >= .60
                     and normalized_polygon_overlap(outlines[index], outlines[other]).iou >= .985]
        if len(competing) != 1:
            continue
        other = competing[0]
        overlap = normalized_polygon_overlap(polygons[index], polygons[other])
        if max(overlap.first_containment, overlap.second_containment) < .96 or not .65 <= overlap.iou <= .94:
            continue
        if any(j not in (index, other) and normalized_polygon_overlap(outlines[index], polygon).second_containment >= .08
               for j, polygon in enumerate(polygons)):
            continue
        inner = index if overlap.first_containment > overlap.second_containment else other
        try:
            inner_quad, _ = quad_from_polygon(normalized_vertices_to_pixels(polygons[inner], width, height))
            if possible_layer_edge(image, inner_quad):
                continue
        except (cv2.error, ValueError):
            continue
        winner, alternate = sorted((index, other), key=lambda j: normalized_polygon_overlap(outlines[j], polygons[j]).iou,
                                   reverse=True)
        alternatives[winner] = (detections[alternate].mask["polygon"], alternate == inner)
        removed.add(alternate)

    claims = {}
    pairs = []
    for index, outline in outlines.items():
        if index in removed:
            continue
        claimants = {}
        for other, polygon in enumerate(polygons):
            if other == index or other in removed:
                continue
            overlap = normalized_polygon_overlap(outline, polygon)
            if overlap.second_containment >= .08:
                claimants[other] = overlap
        claims[index] = bool(claimants)
        if len(claimants) == 1:
            other, overlap = next(iter(claimants.items()))
            # A measured perimeter can guide review of a partial claimant
            # without deciding whether there is a separate sheet on top.
            if other not in measures and overlap.second_containment >= .96 and overlap.first_containment <= .70:
                pairs.append((index, other))
    degree = Counter(index for pair in pairs for index in pair)
    groups = {}
    hints = {}
    for index, other in pairs:
        if degree[index] == degree[other] == 1:
            groups[index] = groups[other] = _normalized_polygon(measures[index].quad, width, height)
            hints[index] = _normalized_polygon(measures[index].outline, width, height)

    updated = []
    for index, item in enumerate(detections):
        if index in removed:
            continue
        if index in groups:
            mask = {**item.mask, "scalloped_ownership_group": groups[index]}
            if index in hints:
                mask["scalloped_ownership_outline"] = hints[index]
            item = replace(item, mask=mask)
        measured = measures.get(index)
        if measured is not None:
            if claims[index]:
                updated.append(replace(item, review_reasons=[SCALLOPED_PRINT_CONFLICT, *item.review_reasons]))
                continue
            mask = {**item.mask, "scalloped_print": True,
                    "visible_polygon": _normalized_polygon(measured.outline, width, height),
                    "warp_quad": _normalized_polygon(measured.quad, width, height)}
            if trim_border:
                motif = suggest_scalloped_motif(image, measured.quad, measured.outline, measured.stock_colors)
                if motif is not None:
                    mask["scalloped_motif_suggestion"] = _normalized_polygon(motif, width, height)
            if index in alternatives:
                mask["scalloped_alternate_polygon"], mask["scalloped_alternate_is_inner"] = alternatives[index]
            item = replace(item, mask=mask, review_reasons=[SCALLOPED_PRINT_REVIEW, *item.review_reasons])
        updated.append(item)
    return SegmentationResult(updated, {**result.metadata, "scalloped_print_outlines": sum(
        item.mask.get("scalloped_print") is True for item in updated),
                                        "scalloped_print_alternates": len(removed),
                                        "scalloped_ownership_hints": len(hints)})


def _suggest_adjacent_album_edge_print(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    if not result.metadata.get("fallback_used") or len(result.detections) != 1:
        return result
    anchor, = result.detections
    if (not anchor.geometry_valid or (anchor.mask.get("boundary_evidence") or {}).get("score", 0) < .75
            or set(anchor.review_reasons) & {MULTIPANEL_PRINT_REVIEW, FRAGMENTED_PRINT_REVIEW}):
        return result
    height, width = image.shape[:2]
    outline = _detection_polygon(anchor)
    if len(outline) != 4:
        return result
    neighbor = order_quad_cyclic(normalized_vertices_to_pixels(outline, width, height))
    try:
        alternatives = adjacent_album_edge_outlines(image, neighbor)
        if not alternatives and anchor.mask.get("detector_source") == "line_pairs":
            traces = staggered_print_source_traces(image, neighbor)
            if traces:
                return SegmentationResult(
                    result.detections, {**result.metadata, "missing_print_edge_traces": traces},
                )
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    if not alternatives:
        return result
    # Multiple source interpretations are ambiguous, even when all have three strong edges.
    quad = alternatives[0]
    if any(np.max(np.linalg.norm(quad - alternative, axis=1)) > 3 for alternative in alternatives[1:]):
        return result
    # A narrow bright band can be album paper, a print border or shared
    # stock. Keep the observed instance and offer the outline in review.
    mask = {**anchor.mask, "adjacent_print_suggestion": _normalized_polygon(quad, width, height)}
    return SegmentationResult(
        [replace(anchor, mask=mask, review_reasons=[*anchor.review_reasons, ADJACENT_PRINT_REVIEW])], result.metadata,
    )


def _grid_fragment_candidates(
    image: np.ndarray, detections: list[SegmentationDetection], quads: dict[int, np.ndarray],
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray | None]]:
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    outlines = [_detection_polygon(detection) for detection in detections]
    recovered = {}
    for child, item in enumerate(detections):
        child_score = (item.mask.get("boundary_evidence") or {}).get("score", 1)
        if (not item.geometry_valid or item.review_reasons or item.mask.get("source") != "yolo_seg_mask"
                or child_score > .66):
            continue
        polygon = outlines[child]
        if len(polygon) < 4:
            continue
        for parent, outer_quad in quads.items():
            anchor = detections[parent]
            source = anchor.mask.get("detector_source")
            score = (anchor.mask.get("boundary_evidence") or {}).get("score", 0)
            needs_outline = (source == "contrast_region" and .45 <= score < .75 and not anchor.mask.get("warp_quad"))
            native_outline = source == "edge_contour" and score >= .75 and child_score <= .62
            if anchor.confidence < .75 or not (needs_outline or native_outline):
                continue
            iou, _, _, contained = _normalized_box_overlap(anchor.bounding_box, item.bounding_box)
            if not .28 <= iou <= .65 or contained < .96:
                continue
            overlap = normalized_polygon_overlap(outlines[parent], polygon)
            if not .28 <= overlap.iou < .60 or overlap.second_containment < (.96 if needs_outline else .99):
                continue
            try:
                inner_quad, method = quad_from_polygon(polygon * scale)
                if method == "minimum_area_rect":
                    continue
                outer = _order_points(outer_quad)
                alternative = None
                if needs_outline:
                    if np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1).min() < 480:
                        continue
                    if parent not in recovered:
                        neighbors = [poly * scale for i, poly in enumerate(outlines) if i != parent]
                        recovered[parent] = suggest_outer_print_outline(image, outer, outer, neighbors)
                    alternative = recovered[parent]
                    if alternative is None or normalized_polygon_overlap(
                        alternative / scale, polygon,
                    ).second_containment < .99:
                        continue
                    outer = alternative
                if grid_fragment_on_shared_stock(image, outer, inner_quad, check_seam_edge=needs_outline):
                    yield parent, child, inner_quad, alternative
            except (cv2.error, ValueError):
                continue


def _defer_nested_print_candidates(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    if len(detections) < 2:
        return result
    height, width = image.shape[:2]
    quads = {}
    for index, item in enumerate(detections):
        held_stock = item.mask.get("paper_edge_confirmation") is True and item.review_reasons == [PRINT_EDGE_CONFIRMATION]
        if not item.geometry_valid or (item.review_reasons and not held_stock):
            continue
        polygon = _detection_polygon(item)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            quads[index] = normalized_vertices_to_pixels(polygon, width, height)
    pairs = []
    for parent, child in combinations(quads, 2):
        outer, inner = detections[parent].bounding_box, detections[child].bounding_box
        iou, _, first, second = _normalized_box_overlap(outer, inner)
        if not .54 <= iou <= .92 or max(first, second) < .95:
            continue
        if first > second:
            parent, child = child, parent
        if detections[child].review_reasons:
            continue
        outer_quad, inner_quad = _order_points(quads[parent]), _order_points(quads[child])
        if nested_print_on_shared_stock(image, outer_quad, inner_quad):
            quads[child] = inner_quad
            pairs.append((parent, child))
    suggestions = quads.copy()
    outer_suggestions = {}
    for parent, child, inner_quad, alternative in _grid_fragment_candidates(image, detections, quads):
        suggestions[child] = inner_quad
        pairs.append((parent, child))
        if alternative is not None:
            outer_suggestions[parent] = _normalized_polygon(alternative, width, height)
    degree = Counter(index for pair in pairs for index in pair)
    deferred = {
        parent: child for parent, child in pairs
        if degree[parent] == 1 and degree[child] == 1
    }
    if not deferred:
        return result
    updated = detections.copy()
    metadata = {**result.metadata}
    fusion = {**(metadata.get("fusion") or {})}
    pending = list(fusion.get("deferred_print_proposals") or [])
    next_id = max((proposal["id"] for proposal in pending), default=-1) + 1
    outer_reviews = 0
    for parent, child in deferred.items():
        item = updated[parent]
        outline = _normalized_polygon(suggestions[child], width, height)
        mask = {**item.mask}
        if parent in outer_suggestions:
            mask.update(visible_polygon=outer_suggestions[parent], visible_polygon_source=GRID_OUTLINE_REVIEW_SOURCE)
            if prior := item.mask.get("visible_polygon"):
                mask["proposal_visible_polygon"] = prior
            # Count review is a page decision and survives a separate crop edit.
            pending.append({"id": next_id, "reason": "shared_print_grid", "polygon": outline})
            next_id += 1
            outer_reviews += 1
        else:
            mask["nested_print_suggestion"] = outline
        updated[parent] = replace(item, mask=mask, review_reasons=[MULTIPANEL_PRINT_REVIEW, *item.review_reasons])
    uncounted = set(deferred.values())
    metadata["nested_print_count_suggestions"] = len(uncounted) - outer_reviews
    if outer_reviews:
        metadata["fusion"] = {**fusion, "deferred_print_proposals": pending}
        metadata["grid_outline_count_suggestions"] = outer_reviews
    return SegmentationResult(
        [item for index, item in enumerate(updated) if index not in uncounted],
        metadata,
    )


def _suggest_crossing_print_outline(
    image: np.ndarray, result: SegmentationResult, *, prepared: np.ndarray | None = None,
) -> SegmentationResult:
    rejections = result.metadata.get("rejected_candidates") or {}
    if (not result.metadata.get("fallback_used") or not rejections.get("frame_touching_container_candidate")
            or not 2 <= len(result.detections) <= 4):
        return result
    # This search is for a weak contour that crosses a paper edge. Keeping a
    # well-supported or explicitly ambiguous print intact is more important.
    weak = {index for index, item in enumerate(result.detections)
            if item.geometry_valid and not item.review_reasons and not item.mask.get("warp_quad")
            and item.mask.get("detector_source") == "edge_contour"
            and (item.mask.get("boundary_evidence") or {}).get("score", 1) < .65
            and _score_value(item, "exterior_background_sides") <= 2}
    if not weak:
        return result
    try:
        outlines = crossing_print_outlines(image, _quadrilateral_from_contour)
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    if len(outlines) != 1:
        return result
    height, width = image.shape[:2]
    quad = outlines[0]
    outline = pixel_vertices_to_normalized(quad, width, height)
    overlaps = [normalized_polygon_overlap(_detection_polygon(item), outline) for item in result.detections]
    competing = [index for index, match in enumerate(overlaps)
                 if max(match.first_containment, match.second_containment) >= .06]
    if len(competing) != 1:
        return result
    index, = competing
    match = overlaps[index]
    if (index not in weak or not .10 <= match.first_containment <= .45
            or not .20 <= match.second_containment <= .55):
        return result
    item = result.detections[index]
    # Dark content, bright stock and shadows can also be printed together.
    # The alternative is visible in review and never participates in instance
    # counts, mask ownership, warp selection or automatic exports.
    detections = result.detections.copy()
    detections[index] = replace(
        item, mask={**item.mask, "crossing_print_suggestion": _normalized_polygon(quad, width, height)},
        review_reasons=[CROSSING_PRINT_REVIEW],
    )
    metadata = result.metadata
    if prepared is not None:
        others = [_detection_polygon(photo) for j, photo in enumerate(result.detections) if j != index]
        try:
            traces = near_leaf_edge_traces(image, prepared, quad, others, _quadrilateral_from_contour)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            traces = []
        if traces:
            metadata = {**metadata, "missing_print_edge_traces": traces}
    return SegmentationResult(detections, metadata)


def _flag_disputed_paper_grids(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    if not result.metadata.get("fallback_used") or not 2 <= len(detections) <= _MULTIPANEL_GROUP_LIMIT:
        return result
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    outlines = [_detection_polygon(item) for item in detections]
    eligible = [index for index, item in enumerate(detections) if item.geometry_valid and not item.review_reasons]
    children = [index for index in eligible if detections[index].mask.get("source") == "yolo_seg_mask"
                and len(outlines[index]) >= 4
                and (detections[index].mask.get("boundary_evidence") or {}).get("score", 1) <= .66]
    if not children:
        return result
    disputes = []
    for parent in eligible:
        item, outline = detections[parent], outlines[parent]
        score = (item.mask.get("boundary_evidence") or {}).get("score", 0)
        if (item.mask.get("detector_source") != "contrast_region" or not .45 <= score < .75
                or item.confidence < .75 or item.mask.get("warp_quad") or len(outline) != 4
                or not cv2.isContourConvex(outline)):
            continue
        nested = [index for index in children
                  if (match := normalized_polygon_overlap(outline, outlines[index])).second_containment >= .95
                  and .28 <= match.iou <= .75]
        if not 1 <= len(nested) <= 2:
            continue
        if len(nested) == 2:
            overlap = normalized_polygon_overlap(outlines[nested[0]], outlines[nested[1]])
            if max(overlap.first_containment, overlap.second_containment) < .95:
                continue
        try:
            seed = order_quad_cyclic(outline * scale)
            if np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min() < 480:
                continue
            neighbors = [(index, poly) for index, poly in enumerate(outlines) if index != parent]
            measured = suggest_outer_print_outline(image, seed, seed, [poly * scale for _, poly in neighbors])
            if measured is None:
                continue
            claims = {index for index, poly in neighbors
                      if (overlap := normalized_polygon_overlap(measured / scale, poly)).second_containment >= .99
                      and overlap.first_containment >= .15}
            if claims != set(nested):
                continue
            cuts = {}
            for index in nested:
                quad, method = quad_from_polygon(outlines[index] * scale)
                if method != "minimum_area_rect" and (cut := paper_grid_review_seam(image, measured, quad)) is not None:
                    cuts[index] = _normalized_polygon(cut, width, height)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        if cuts.keys() == claims:
            disputes.append((parent, _normalized_polygon(measured, width, height), cuts))
    owners = Counter(index for parent, _, cuts in disputes for index in (parent, *cuts))
    updated = detections.copy()
    for parent, outline, cuts in disputes:
        group = {parent: {"outline": outline}, **{index: {"outline": outline, "cut": cut} for index, cut in cuts.items()}}
        if any(owners[index] != 1 for index in group):
            continue
        # The exterior can surround a separate inset. Keep every proposed
        # photo and attach the measured outline and proposed cuts for review.
        for index, review in group.items():
            item = updated[index]
            updated[index] = replace(item, mask={**item.mask, "paper_grid_review": review},
                                     review_reasons=[PAPER_GRID_REVIEW, *item.review_reasons])
    return SegmentationResult(updated, result.metadata) if disputes else result


def _flag_conflicting_film_grids(
    image: np.ndarray, result: SegmentationResult, *, require_clear_rim: bool = False,
) -> SegmentationResult:
    detections = result.detections
    if not result.metadata.get("fallback_used") or not 3 <= len(detections) <= _MULTIPANEL_GROUP_LIMIT:
        return result
    minimum_model_edge = .65 if require_clear_rim else .70
    model_indices = {index for index, item in enumerate(detections)
                     if item.geometry_valid and not item.review_reasons and item.mask.get("source") == "yolo_seg_mask"
                     and (item.mask.get("boundary_evidence") or {}).get("score", 0) >= minimum_model_edge}
    if len(model_indices) < 2:
        return result
    height, width = image.shape[:2]
    outlines = [_detection_polygon(item) for item in detections]
    disputes = {}
    for index, item in enumerate(detections):
        outline = outlines[index]
        source = item.mask.get("detector_source")
        edge = item.mask.get("boundary_evidence") or {}
        if require_clear_rim:
            carrier = (source == "edge_contour" and item.mask.get("source") == "classical_border_quad"
                       and edge.get("score", 0) >= .90)
        else:
            carrier = source == "contrast_region" and edge.get("score", 1) < .70
        if (not item.geometry_valid or item.review_reasons or not carrier
                or len(outline) != 4 or not cv2.isContourConvex(outline)):
            continue
        owners = {other: overlap for other, shape in enumerate(outlines) if other != index
                  and (overlap := normalized_polygon_overlap(outline, shape)).first_containment >= .10}
        if (len(owners) != 2 or not owners.keys() <= model_indices
                or min(match.second_containment for match in owners.values()) < (.985 if require_clear_rim else .80)
                or sum(match.first_containment for match in owners.values()) < (.78 if require_clear_rim else .88)):
            continue
        first, second = owners
        if normalized_polygon_overlap(outlines[first], outlines[second]).iou > .005:
            continue
        try:
            first_quad, second_quad = (quad_from_polygon(normalized_vertices_to_pixels(outlines[i], width, height))[0]
                                       for i in (first, second))
            seam = film_grid_review_seam(image, first_quad, second_quad)
            if seam is None:
                continue
            broad = order_quad_cyclic(normalized_vertices_to_pixels(outline, width, height))
            markers = {"film_grid_review_seam": _normalized_polygon(seam, width, height)}
            if require_clear_rim:
                peers = [normalized_vertices_to_pixels(shape, width, height) for other, shape in enumerate(outlines)
                         if other not in {index, first, second} and len(shape) >= 3]
                rim = transparent_film_rim(image, broad, seam, peers)
                if rim is None:
                    continue
                markers["film_grid_review_rim"] = _normalized_polygon(rim, width, height)
            else:
                _, sides, _ = refine_print_boundary(image, broad)
                if sum(side["support"] < .40 for side in sides.values()) < 2:
                    continue
        except (cv2.error, ValueError):
            continue
        disputes[index, first, second] = markers
    degree = Counter(i for group in disputes for i in group)
    markers = {}
    for group, evidence in disputes.items():
        if all(degree[index] == 1 for index in group):
            markers.update(dict.fromkeys(group, evidence))
    if not markers:
        return result
    updated = detections.copy()
    for index, evidence in markers.items():
        item = updated[index]
        updated[index] = replace(item, mask={**item.mask, **evidence},
                                 review_reasons=[FILM_GRID_REVIEW, *item.review_reasons])
    return SegmentationResult(updated, result.metadata)


def _flag_partial_film_grid(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    if not result.metadata.get("fallback_used") or not 3 <= len(detections) <= _MULTIPANEL_GROUP_LIMIT:
        return result
    fragments = [index for index, item in enumerate(detections)
                 if not item.geometry_valid and item.review_reasons in (["aspect_ratio_too_tall"], ["aspect_ratio_too_wide"])
                 and item.mask.get("source") == "yolo_seg_mask" and item.mask.get("detector_source") == "yolo_onnx"
                 and not item.mask.get("warp_quad") and item.confidence >= .45
                 and (item.mask.get("boundary_evidence") or {}).get("score", 0) >= .67]
    if len(fragments) != 2:
        return result
    outlines = [_detection_polygon(item) for item in detections]
    first, second = (outlines[index] for index in fragments)
    if normalized_polygon_overlap(first, second).iou > .005:
        return result
    carriers = []
    for index, (item, outline) in enumerate(zip(detections, outlines, strict=True)):
        if (not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad")
                or item.mask.get("source") != "classical_border_quad" or item.mask.get("detector_source") != "contrast_region"
                or not .40 <= (item.mask.get("boundary_evidence") or {}).get("score", 0) < .70
                or len(outline) != 4 or not cv2.isContourConvex(outline)):
            continue
        overlaps = [normalized_polygon_overlap(outline, child) for child in (first, second)]
        if (min(match.second_containment for match in overlaps) >= .85
                and min(match.first_containment for match in overlaps) >= .12
                and .46 <= sum(match.first_containment for match in overlaps) < .77):
            carriers.append(index)
    if len(carriers) != 1:
        return result
    carrier, = carriers
    members = {carrier, *fragments}
    height, width = image.shape[:2]
    peers = []
    claims = _tiled_review_claims(result) + deferred_neighbor_masks(result.metadata)
    for index, claim in enumerate(claims):
        if index in members:
            continue
        for key in ("polygon", "visible_polygon", "warp_quad"):
            points = claim.get(key) or []
            if len(points) >= 3:
                peers.append(normalized_vertices_to_pixels([(p["x"], p["y"]) for p in points], width, height))
    try:
        seed = order_quad_cyclic(normalized_vertices_to_pixels(outlines[carrier], width, height))
        children = [normalized_vertices_to_pixels(child, width, height) for child in (first, second)]
        carrier_geometry = refitted_film_carrier(image, seed, children, peers)
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    if carrier_geometry is None:
        return result
    _, *traces = carrier_geometry
    seam, rim = (_normalized_polygon(trace, width, height) for trace in traces)
    return SegmentationResult([
        replace(item, mask={**item.mask, "film_grid_review_seam": seam, "film_grid_review_rim": rim},
                review_reasons=[FILM_GRID_REVIEW, *item.review_reasons]) if index in members else item
        for index, item in enumerate(detections)
    ], result.metadata)


def _flag_printed_illustration(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    items = result.detections
    if (len(items) != PRINTED_ILLUSTRATION_REVIEW_SIZE or not result.metadata.get("fallback_used")
            or (result.metadata.get("fusion") or {}).get("deferred_print_proposals")
            or any(not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad") for item in items)
            or Counter(item.mask.get("detector_source") for item in items) != {"yolo_onnx": 3, "edge_contour": 1}):
        return result
    normalized = [_detection_polygon(item) for item in items]
    if any(normalized_polygon_overlap(first, second).iou < .28 for first, second in combinations(normalized, 2)):
        return result
    height, width = image.shape[:2]
    polygons = [normalized_vertices_to_pixels(polygon, width, height) for polygon in normalized]
    try:
        frame = find_illustration_frame(image, polygons)
        if frame is None:
            return result
        for polygon in polygons:
            quad, method = quad_from_polygon(polygon)
            if method == "minimum_area_rect":
                return result
            # A second supported print side is enough to leave this claim out
            # of the shared-illustration decision.
            _, evidence, _ = refine_print_boundary(image, order_quad_cyclic(quad))
            supports = sorted((side.get("support", 0) for side in evidence.values()), reverse=True)
            if len(supports) != 4 or supports[0] < .80 or supports[1] > .46:
                return result
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    outline = _normalized_polygon(frame, width, height)
    if not valid_illustration_frame(outline):
        return result
    source_id = illustration_evidence_id(outline, [item.mask["polygon"] for item in items])
    detections = [replace(item, mask={**item.mask, PRINTED_ILLUSTRATION_REVIEW_FRAME: [outline],
                                    PRINTED_ILLUSTRATION_REVIEW_ID: source_id},
                          review_reasons=[PRINTED_ILLUSTRATION_COUNT_REVIEW]) for item in items]
    return SegmentationResult(detections, result.metadata)


def _flag_shared_motif(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    fusion = result.metadata.get("fusion") or {}
    if not result.metadata.get("fallback_used") or fusion.get("deferred_print_proposals"):
        return result
    candidates = [index for index, item in enumerate(detections) if item.geometry_valid and not item.review_reasons
                  and item.mask.get("source") == "yolo_seg_mask" and item.mask.get("detector_source") == "yolo_onnx"
                  and not any(item.mask.get(name) for name in ("warp_quad", "visible_polygon", MERGED_PRINT_PAIR))]
    if len(candidates) < 2:
        return result
    polygons = [_detection_polygon(item) for item in detections]
    pairs = [(first, second) for first, second in combinations(candidates, 2)
             if _normalized_box_overlap(detections[first].bounding_box, detections[second].bounding_box)[0] >= .65
             and near_nested_model_masks(polygons[first], polygons[second])]
    counts = Counter(index for pair in pairs for index in pair)
    height, width = image.shape[:2]
    updated = detections.copy()
    for first, second in pairs:
        if counts[first] != 1 or counts[second] != 1:
            continue
        others = [polygon for index, polygon in enumerate(polygons) if index not in (first, second)]
        try:
            outline = measure_shared_motif(image, (polygons[first], polygons[second]), others)
        except (cv2.error, ValueError):
            continue
        if outline is None:
            continue
        guide = _normalized_polygon(outline, width, height)
        evidence = {"outline": guide, "id": motif_review_id(guide, [detections[i].mask["polygon"] for i in (first, second)])}
        for index in (first, second):
            item = detections[index]
            updated[index] = replace(item, mask={**item.mask, SHARED_MOTIF_REVIEW: evidence})
    return SegmentationResult(updated, result.metadata)


def _flag_compact_scene_crossings(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    fusion = result.metadata.get("fusion") or {}
    if not result.metadata.get("fallback_used") or fusion.get("deferred_print_proposals") or not 2 <= len(detections) <= 12:
        return result
    candidates = [i for i, item in enumerate(detections) if item.geometry_valid and not item.review_reasons
                  and not any(item.mask.get(key) for key in ("warp_quad", "visible_polygon", MERGED_PRINT_PAIR))]
    parents = [i for i in candidates if detections[i].confidence >= .50
               and detections[i].mask.get("source") == "yolo_seg_mask"
               and detections[i].mask.get("detector_source") == "yolo_onnx"]
    if not parents:
        return result
    polygons = [_detection_polygon(item) for item in detections]
    height, width = image.shape[:2]
    updated = detections.copy()
    for index in candidates:
        item, polygon = detections[index], polygons[index]
        provenance = item.mask.get("fusion") or {}
        if (item.confidence < .60 or item.mask.get("source") != "classical_border_quad"
                or provenance.get("proposal_families") != ["classical"]
                or len(polygon) != 4 or not cv2.isContourConvex(polygon)):
            continue
        matches = [parent for parent in parents if _normalized_box_overlap(detections[parent].bounding_box, item.bounding_box)[3] >= .90
                   and (overlap := normalized_polygon_overlap(polygons[parent], polygon, raster_size=1024)).second_containment >= .975
                   and .20 <= overlap.iou <= .42]
        if len(matches) != 1:
            continue
        parent, = matches
        others = [normalized_vertices_to_pixels(peer, width, height) for peer_index, peer in enumerate(polygons)
                  if peer_index not in (index, parent)]
        try:
            trace = compact_scene_crossing_trace(
                image, order_quad_cyclic(normalized_vertices_to_pixels(polygon, width, height)),
                normalized_vertices_to_pixels(polygons[parent], width, height), others,
            )
        except (ValueError, cv2.error):
            continue
        if trace is not None:
            updated[index] = replace(item, mask={**item.mask, COMPACT_SCENE_CROSSING_TRACE: _normalized_polygon(trace, width, height)})
    return SegmentationResult(updated, result.metadata)


def _flag_disputed_paper_joins(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    if not result.metadata.get("fallback_used") or not 2 <= len(detections) <= _MULTIPANEL_GROUP_LIMIT:
        return result
    height, width = image.shape[:2]
    polygons = [_detection_polygon(item) for item in detections]
    suitable = [
        index for index, item in enumerate(detections)
        if item.geometry_valid and not item.review_reasons and not item.mask.get("warp_quad")
        and .35 <= (item.mask.get("boundary_evidence") or {}).get("exterior_background_support", 0) <= .85
    ]
    groups = {}
    for left, right in combinations(suitable, 2):
        if (not _adjacent_panel_boxes(detections[left].bounding_box, detections[right].bounding_box)
                or normalized_polygon_overlap(polygons[left], polygons[right]).iou > .018):
            continue
        try:
            first, second = (quad_from_polygon(normalized_vertices_to_pixels(polygons[index], width, height))[0]
                             for index in (left, right))
            traces = paper_join_review_traces(image, first, second)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        if not traces:
            continue
        hull = cv2.convexHull(np.concatenate((polygons[left], polygons[right])))[:, 0]
        if any(normalized_polygon_overlap(hull, polygon).second_containment > .06
               for index, polygon in enumerate(polygons) if index not in (left, right)):
            continue
        groups[left, right] = [_normalized_polygon(trace, width, height) for trace in traces]
    claims = Counter(index for pair in groups for index in pair)
    if not groups:
        return result
    updated = detections.copy()
    for pair, traces in groups.items():
        if all(claims[index] == 1 for index in pair):
            for index in pair:
                item = updated[index]
                updated[index] = replace(item, mask={**item.mask, "paper_join_review_traces": traces},
                                         review_reasons=[PAPER_JOIN_REVIEW])
    return SegmentationResult(updated, result.metadata)


def _flag_crossing_fallback_contours(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    if not result.metadata.get("fallback_used") or not 4 <= len(detections) <= 12:
        return result
    models = [index for index, item in enumerate(detections) if item.geometry_valid and not item.review_reasons
              and not item.mask.get("warp_quad") and item.mask.get("source") == "yolo_seg_mask"
              and item.mask.get("detector_source") == "yolo_onnx" and item.confidence >= .40
              and (item.mask.get("boundary_evidence") or {}).get("score", 0) >= .65]
    if len(models) < 3:
        return result
    height, width = image.shape[:2]
    polygons = [_detection_polygon(item) for item in detections]
    native = [normalized_vertices_to_pixels(polygon, width, height) for polygon in polygons]
    peers = [native[model] for model in models]
    uncounted = []
    for offer in (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []:
        outline = offer.get("visible_polygon") or offer.get("polygon") or []
        if len(outline) >= 3:
            uncounted.append(normalized_vertices_to_pixels([(point["x"], point["y"]) for point in outline], width, height))
    updated = detections.copy()
    for index, item in enumerate(detections):
        fusion = item.mask.get("fusion") or {}
        boundary = item.mask.get("boundary_evidence") or {}
        if (not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad")
                or item.mask.get("source") != "classical_border_quad" or item.mask.get("detector_source") != "edge_contour"
                or fusion.get("proposal_count") != 1 or fusion.get("proposal_families") != ["classical"]
                or not .55 <= item.confidence <= .86 or not .45 <= boundary.get("score", 0) <= .71):
            continue
        outline = polygons[index]
        if len(outline) != 4 or not cv2.isContourConvex(outline):
            continue
        ownership = [normalized_polygon_overlap(outline, polygons[model]).first_containment for model in models]
        inside = [part for part in ownership if part >= .10]
        if len(inside) != 2 or not .42 <= sum(inside) <= .72:
            continue
        quad = native[index]
        gap_to_frame = np.minimum(quad, [width - 1, height - 1] - quad)
        if np.any(gap_to_frame < 4):
            continue
        try:
            other_claims = [polygon for peer, polygon in enumerate(native)
                            if peer != index and peer not in models and len(polygon) >= 3]
            traces = crossing_fallback_review_traces(image, quad, peers, extension_claims=[*other_claims, *uncounted])
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        if traces:
            updated[index] = replace(
                item, mask={**item.mask, CROSSING_FALLBACK_TRACES: [
                    _normalized_polygon(trace, width, height) for trace in traces
                ]}, review_reasons=[CROSSING_FALLBACK_COUNT_REVIEW],
            )
    return SegmentationResult(updated, result.metadata)


def _flag_album_leaf_count_review(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    detections = result.detections
    if not result.metadata.get("fallback_used") or len(detections) < 2:
        return result
    height, width = image.shape[:2]
    polygons = [_detection_polygon(item) for item in detections]
    pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
    pending_outlines = [offer.get("visible_polygon") or offer.get("polygon") or [] for offer in pending]
    updated = detections.copy()
    for index, item in enumerate(detections):
        outline = polygons[index]
        fusion = item.mask.get("fusion") or {}
        if (not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad")
                or item.mask.get("source") != "classical_border_quad" or item.mask.get("detector_source") != "edge_contour"
                or fusion.get("proposal_count") != 1 or fusion.get("proposal_families") != ["classical"]
                or len(outline) != 4 or not cv2.isContourConvex(outline) or not .18 <= cv2.contourArea(outline) <= .48):
            continue
        contained = [peer for peer, polygon in enumerate(polygons) if peer != index
                     and normalized_polygon_overlap(outline, polygon).second_containment >= .08]
        if len(contained) != 1:
            continue
        child_index, = contained
        child = detections[child_index]
        proof = child.mask.get("boundary_evidence") or {}
        overlap = normalized_polygon_overlap(outline, polygons[child_index])
        if (not child.geometry_valid or child.review_reasons or child.confidence < .50
                or len(child.mask.get("polygon") or []) < 4
                or child.mask.get("source") != "yolo_seg_mask" or child.mask.get("detector_source") != "yolo_onnx"
                or proof.get("score", 0.) < .58 or proof.get("border_contrast", 0.) < .66
                or overlap.second_containment < .97 or not .28 <= overlap.iou <= .66):
            continue
        # Leave this choice to the general count editor when another visible
        # proposal in the area still needs to be accounted for.
        if any(len(points) >= 3 and normalized_polygon_overlap(
            outline, [(point["x"], point["y"]) for point in points],
        ).second_containment >= .08 for points in pending_outlines):
            continue
        try:
            traces = album_leaf_review_traces(image, normalized_vertices_to_pixels(outline, width, height))
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue
        if traces:
            updated[index] = replace(item, mask={
                **item.mask,
                ALBUM_LEAF_REVIEW_TRACES: [_normalized_polygon(trace, width, height) for trace in traces],
                ALBUM_LEAF_REVIEW_CHILD: [point.copy() for point in child.mask["polygon"]],
            }, review_reasons=[ALBUM_LEAF_COUNT_REVIEW])
    return SegmentationResult(updated, result.metadata)


def _offer_mounted_window_review(source: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    if not result.metadata.get("fallback_used") or not 4 <= len(result.detections) <= 8:
        return result
    leaves = []
    for item in result.detections:
        fusion = item.mask.get("fusion") or {}
        if (not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad")
                or item.mask.get("source") != "classical_border_quad" or item.mask.get("detector_source") != "edge_contour"
                or "classical" not in fusion.get("proposal_families", ()) or fusion.get("proposal_count") not in (1, 2)):
            continue
        outline = _detection_polygon(item)
        if len(outline) == 4 and cv2.isContourConvex(outline) and .18 <= cv2.contourArea(outline) <= .48:
            leaves.append((item, order_quad_cyclic(outline)))
    if len(leaves) != 2:
        return result
    leaves.sort(key=lambda leaf: leaf[1][:, 0].mean())
    (_, left), (_, right) = leaves
    gap = right[[0, 3]] - left[[1, 2]]
    if np.any(gap[:, 0] < 0) or np.any(gap[:, 0] > .16) or np.any(np.abs(gap[:, 1]) > .15):
        return result

    candidates = []
    for index, (leaf, outline) in enumerate(leaves):
        overlapping = []
        for item in result.detections:
            if item.mask.get("detector_source") != "yolo_onnx" or not item.geometry_valid or not .25 <= item.confidence < .50:
                continue
            polygon = _detection_polygon(item)
            if (normalized_polygon_overlap(outline, polygon, raster_size=512).second_containment >= .97
                    and np.ptp(polygon[:, 0]) >= .72 * np.ptp(outline[:, 0])):
                overlapping.append(item)
        if len(overlapping) == 2 and normalized_polygon_overlap(*map(_detection_polygon, overlapping), raster_size=512).iou >= .55:
            candidates.append((index, leaf, outline, overlapping))
    if len(candidates) != 1:
        return result
    (index, leaf, outline, overlapping), = candidates
    height, width = source.shape[:2]
    binding_side = leaves[1 - index][1]
    try:
        if not album_leaf_review_traces(source, normalized_vertices_to_pixels(binding_side, width, height)):
            return result
        peers = [item for item in result.detections if item is not leaf and all(item is not peer for peer in overlapping)]
        claims = _tiled_review_claims(SegmentationResult(peers, result.metadata))
        traces = mounted_window_traces(source, normalized_vertices_to_pixels(outline, width, height),
                                       1 if index else 3, claims, [item.mask for item in overlapping])
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return result
    if not traces:
        return result
    return SegmentationResult(result.detections, {**result.metadata, MOUNTED_WINDOW_REVIEW_TRACES: [
        _normalized_polygon(trace, width, height) for trace in traces
    ]})


def _offer_projective_print_review(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    metadata = result.metadata
    fusion = metadata.get("fusion") or {}
    rejections = metadata.get("rejected_candidates") or {}
    if (result.detections or not metadata.get("fallback_used") or fusion.get("model_proposals")
            or fusion.get("classical_proposals") or fusion.get("deferred_print_proposals")
            or metadata.get("missing_print_edge_traces")
            or not {"aspect_ratio_too_tall", "aspect_ratio_too_wide", "frame_touching_container_candidate"} & rejections.keys()):
        return result
    try:
        outline, traces = projective_print_review(image)
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    if not traces:
        return result
    if outline is None:
        return SegmentationResult(result.detections, {**metadata, "missing_print_edge_traces": traces})
    # A common support can depict two separate items. Keep both sources of
    # evidence in count review until the user decides what is physically visible.
    proposal = {"id": 0, "reason": PROJECTIVE_PRINT_PROPOSAL,
                "polygon": _normalized_polygon(outline, *image.shape[1::-1])}
    return SegmentationResult(result.detections, {
        **metadata, "missing_print_edge_traces": traces,
        "fusion": {**fusion, "deferred_print_proposals": [proposal]},
    })


def _native_fragment_seeds(image: np.ndarray) -> Iterator[np.ndarray]:
    height, width = image.shape[:2]
    seen = set()
    for contour in _closed_edge_contours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)):
        candidate = _candidate_from_contour(
            contour, width=width, height=height, image_area=width * height,
            min_area_ratio=.05, source="edge_contour",
        )
        if candidate is None:
            continue
        quad, source = _quadrilateral_from_contour(contour)
        if source != "classical_border_quad":
            continue
        try:
            ordered = order_quad_cyclic(quad)
        except ValueError:
            continue
        key = ordered.tobytes()
        if cv2.isContourConvex(ordered) and cv2.contourArea(ordered) >= 1 and key not in seen:
            seen.add(key)
            yield ordered


def _offer_source_fragment_group_review(
    image: np.ndarray, result: SegmentationResult, regions: list[dict] | None,
) -> SegmentationResult:
    metadata = result.metadata
    fusion = metadata.get("fusion") or {}
    if (not metadata.get("fallback_used") or fusion.get("model_proposals") != 0
            or not 2 <= len(result.detections) <= 8
            or not (metadata.get("rejected_candidates") or {}).get("frame_touching_container_candidate")):
        return result
    height, width = image.shape[:2]
    quads = []
    partials = []
    crossings = []
    for index, item in enumerate(result.detections):
        mask = item.mask
        records = mask.get("visible_polygon") or mask.get("polygon") or []
        quad = None
        if len(records) == 4:
            try:
                points = normalized_vertices_to_pixels([(p["x"], p["y"]) for p in records], width, height)
                quad = order_quad_cyclic(points)
            except (KeyError, TypeError, ValueError):
                pass
        if quad is not None and (not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 1):
            quad = None
        quads.append(quad)
        if quad is None or item.review_reasons or not item.geometry_valid:
            continue
        role = source_fragment_role(mask, item.confidence, quad, width, height)
        if role == "partial":
            partials.append(index)
        elif role == "frame_crossing":
            crossings.append(index)
    if not 1 <= len(partials) <= 3 or len(crossings) != 1:
        return result
    group = suggest_source_fragment_group(image, quads, partials, crossings, _native_fragment_seeds)
    if group is None:
        return result
    first, second, outline, weak = group
    if regions and not validate_protected_regions(outline, regions, width=width, height=height)["passed"]:
        return result
    pending = list(fusion.get("deferred_print_proposals") or [])
    normalized = _normalized_polygon(outline, width, height)
    if any(normalized_polygon_overlap(
        [(point["x"], point["y"]) for point in proposal["polygon"]],
        pixel_vertices_to_normalized(outline, width, height),
    ).iou > .10 for proposal in pending):
        return result
    proposal_id = max((proposal["id"] for proposal in pending), default=-1) + 1
    detections = result.detections.copy()
    pending.append({
        "id": proposal_id, "reason": SOURCE_FRAGMENT_PROPOSAL, "polygon": normalized, "uncertain_edge": weak,
        "fragment_polygons": [detections[index].mask["polygon"] for index in (first, second)],
    })
    for index in (first, second):
        item = detections[index]
        detections[index] = replace(item, mask={**item.mask, "source_fragment_group_review": proposal_id},
                                    review_reasons=[SOURCE_FRAGMENT_GROUP_REVIEW, *item.review_reasons])
    return SegmentationResult(detections, {**metadata, "fusion": {**fusion, "deferred_print_proposals": pending}})


def _defer_nested_scene_fragments(
    image: np.ndarray, result: SegmentationResult, *, edge_contours: bool = False,
) -> SegmentationResult:
    """Hold the count when a fallback outline ends inside a larger photograph."""
    if not result.metadata.get("fallback_used"):
        return result
    source = "edge_contour" if edge_contours else "contrast_region"
    parents, fragments = [], []
    for index, item in enumerate(result.detections):
        if not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad"):
            continue
        fusion = item.mask.get("fusion") or {}
        families = fusion.get("proposal_families") or []
        if ("model" in families and item.confidence >= .85 and (not edge_contours
                or (families == ["model"] and item.mask.get("detector_source") == "yolo_onnx"
                    and item.mask.get("source") == "yolo_seg_mask"))):
            parents.append(index)
        if (families == ["classical"] and fusion.get("proposal_count") == 1
                and item.mask.get("detector_source") == source and .60 <= item.confidence <= .82
                and (not edge_contours or item.mask.get("source") == "classical_border_quad")):
            fragments.append(index)
    if not parents or not fragments:
        return result

    height, width = image.shape[:2]
    detections = result.detections.copy()
    polygons = [_detection_polygon(item) for item in detections]
    fusion = {**(result.metadata.get("fusion") or {})}
    pending = list(fusion.get("deferred_print_proposals") or [])
    pending_masks = [records for offer in pending for records in (
        offer.get("polygon"), offer.get("visible_polygon"), (offer.get("source_mask") or {}).get("polygon"),
        (offer.get("source_mask") or {}).get("visible_polygon"),
    ) if records and len(records) >= 3]
    next_id = max((proposal["id"] for proposal in pending), default=-1) + 1
    deferred = set()
    for child in fragments:
        fragment = polygons[child]
        if len(fragment) != 4 or not cv2.isContourConvex(fragment):
            continue
        containing = {index: normalized_polygon_overlap(polygon, fragment)
                      for index, polygon in enumerate(polygons) if index != child
                      and _normalized_box_overlap(detections[index].bounding_box, detections[child].bounding_box)[3] >= .08}
        owners = [(index, overlap) for index, overlap in containing.items() if overlap.second_containment >= .08]
        if len(owners) != 1:
            continue
        (parent, overlap), = owners
        if parent not in parents or detections[parent].review_reasons:
            continue
        minimum, maximum, containment = (.14, .26, .99) if edge_contours else (.55, .78, .96)
        if not minimum <= overlap.iou <= maximum or overlap.second_containment < containment:
            continue
        quad = order_quad_cyclic(normalized_vertices_to_pixels(fragment, width, height))
        outline = normalized_vertices_to_pixels(polygons[parent], width, height)
        extra = {}
        if edge_contours:
            alternate = [records for index, item in enumerate(detections) if index not in (parent, child)
                         for records in (item.mask.get("polygon"), item.mask.get("visible_polygon"))
                         if records and len(records) >= 3]
            overlaps = (normalized_polygon_overlap([(p["x"], p["y"]) for p in other], fragment)
                        for other in [*pending_masks, *alternate])
            if any(max(hit.first_containment, hit.second_containment) >= .08 for hit in overlaps):
                continue
            traces, shared = nested_scene_fragment_with_shared_edges(image, outline, quad)
            mask = detections[child].mask
            extra = {
                "shared_paper_edge_traces": [_normalized_polygon(trace, width, height) for trace in shared],
                "source_mask": {key: mask[key] for key in ("source", "detector_source", "polygon", "visible_polygon",
                                "visible_polygon_source", "proposal_visible_polygon") if key in mask},
                "fusion_cluster_id": mask["fusion"]["cluster_id"],
            }
        else:
            traces = nested_scene_fragment_continuities(image, outline, quad)
        if not traces:
            continue

        pending.append({
            "id": next_id,
            "reason": NESTED_EDGE_CONTOUR if edge_contours else NESTED_FALLBACK_CONTOUR,
            "parent_cluster_id": detections[parent].mask["fusion"]["cluster_id"],
            "polygon": _normalized_polygon(quad, width, height),
            "interior_edge_traces": [_normalized_polygon(trace, width, height) for trace in traces],
            **extra,
        })
        next_id += 1
        deferred.add(child)
        reason = NESTED_EDGE_CONTOUR_REVIEW if edge_contours else NESTED_FALLBACK_CONTOUR_REVIEW
        detections[parent] = replace(detections[parent], review_reasons=[reason])
    if not deferred:
        return result
    fusion["deferred_print_proposals"] = pending
    return SegmentationResult(
        [item for index, item in enumerate(detections) if index not in deferred],
        {**result.metadata, "fusion": fusion},
    )


def _defer_dark_album_claims(image: np.ndarray, result: SegmentationResult, acceptance: float) -> SegmentationResult:
    fusion = result.metadata.get("fusion") or {}
    if not result.metadata.get("fallback_used") or not fusion.get("model_proposals") or not fusion.get("classical_proposals"):
        return result
    if not 2 <= len(result.detections) <= 12:
        return result
    height, width = image.shape[:2]
    peers = []
    candidates = []
    for index, item in enumerate(result.detections):
        if not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad"):
            continue
        source = item.mask.get("source")
        provenance = item.mask.get("fusion") or {}
        families = provenance.get("proposal_families")
        polygon = _detection_polygon(item)
        if source == "classical_border_quad" and families == ["classical", "model"] and item.confidence >= .70:
            if len(polygon) == 4:
                peers.append(normalized_vertices_to_pixels(polygon, width, height))
        elif (source == "yolo_seg_mask" and families == ["model"] and provenance.get("proposal_count") == 1
              and item.mask.get("detector_source") == "yolo_onnx" and .30 <= item.confidence < min(.70, acceptance)):
            candidates.append((index, polygon))
    if not peers or not 1 <= len(candidates) <= 6:
        return result

    pending = list(fusion.get("deferred_print_proposals") or [])
    deferred = set()
    next_id = max((proposal["id"] for proposal in pending), default=-1) + 1
    for index, polygon in candidates:
        try:
            quad, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
            evidence = inspect_dark_album_backing(image, order_quad_cyclic(quad), peers)
        except (ValueError, cv2.error):
            continue
        if evidence is None:
            continue
        item = result.detections[index]
        pending.append({
            "id": next_id, "reason": DARK_ALBUM_PROPOSAL, "polygon": _normalized_polygon(quad, width, height),
            "visible_polygon": item.mask.get("visible_polygon") or item.mask["polygon"],
            "dark_backing_traces": [_normalized_polygon(span, width, height) for span in evidence.backing_traces],
            "borrowed_edge_traces": [_normalized_polygon(span, width, height) for span in evidence.borrowed_traces],
            "source_mask": {key: item.mask[key] for key in ("source", "detector_source", "polygon", "visible_polygon",
                            "visible_polygon_source", "proposal_visible_polygon") if key in item.mask},
            "fusion_cluster_id": item.mask["fusion"]["cluster_id"],
            "proposal_confidence": item.confidence,
        })
        next_id += 1
        deferred.add(index)
    if not deferred:
        return result
    kept = [item for index, item in enumerate(result.detections) if index not in deferred]
    return SegmentationResult(kept, {**result.metadata, "fusion": {
        **fusion, "deferred_print_proposals": pending, "dark_backing_review_count": len(deferred),
    }})


def _adjacent_pocket_peer(box: dict[str, float], peer: dict[str, float]) -> bool:
    width, depth = box["x2"] - box["x1"], box["y2"] - box["y1"]
    x_overlap = min(box["x2"], peer["x2"]) - max(box["x1"], peer["x1"])
    gap = max(peer["y1"] - box["y2"], box["y1"] - peer["y2"])
    return x_overlap >= .72 * width and .03 * depth <= gap <= .38 * depth


def _embossed_carrier_offers(
    image: np.ndarray, result: SegmentationResult, models: list[tuple[int, SegmentationDetection]], acceptance: float,
) -> Iterator[tuple[int, np.ndarray, list[np.ndarray], list[dict]]]:
    width, height = image.shape[1::-1]
    outlines = [_detection_polygon(item) for item in result.detections]
    strong = {index for index, item in models if item.confidence >= max(.85, acceptance)
              and (item.mask.get("boundary_evidence") or {}).get("border_contrast", 0.) >= .72}
    if not strong:
        return
    pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
    unresolved = [points for offer in pending for points in (
        offer.get("polygon"), offer.get("visible_polygon"), (offer.get("source_mask") or {}).get("polygon"),
        (offer.get("source_mask") or {}).get("visible_polygon"),
    ) if points and len(points) >= 3]
    for index, item in enumerate(result.detections):
        provenance = item.mask.get("fusion") or {}
        if (not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad") or item.mask.get("visible_polygon")
                or item.mask.get("source") != "classical_border_quad" or item.mask.get("detector_source") != "edge_contour"
                or provenance.get("proposal_families") != ["classical"] or provenance.get("proposal_count") != 1
                or not .58 <= item.confidence <= .75):
            continue
        shell = outlines[index]
        if len(shell) != 4 or not cv2.isContourConvex(shell) or not .15 <= cv2.contourArea(shell) <= .43:
            continue
        contained = [(peer, overlap) for peer, outline in enumerate(outlines) if peer != index
                     and (overlap := normalized_polygon_overlap(shell, outline)).second_containment >= .08]
        if len(contained) != 1:
            continue
        (peer, overlap), = contained
        if peer not in strong or overlap.second_containment < .95 or not .45 <= overlap.iou <= .68:
            continue
        other_visible = [candidate.mask["visible_polygon"] for other, candidate in enumerate(result.detections)
                         if other not in (peer, index) and candidate.mask.get("visible_polygon")]
        if any(normalized_polygon_overlap(shell, [(p["x"], p["y"]) for p in polygon]).second_containment >= .08
               for polygon in [*unresolved, *other_visible]):
            continue
        child = normalized_vertices_to_pixels(outlines[peer], width, height)
        try:
            outer = order_quad_cyclic(normalized_vertices_to_pixels(shell, width, height))
            seed, _ = quad_from_polygon(child)
            traces = embossed_outer_carrier_traces(image, outer, order_quad_cyclic(seed), child)
        except (ValueError, cv2.error):
            continue
        if traces:
            yield index, outer, traces, result.detections[peer].mask["polygon"]


def _defer_embossed_pocket_claims(image: np.ndarray, result: SegmentationResult, acceptance: float) -> SegmentationResult:
    fusion = result.metadata.get("fusion") or {}
    if not result.metadata.get("fallback_used") or not fusion.get("model_proposals") or not 2 <= len(result.detections) <= 12:
        return result
    models = [
        (index, item) for index, item in enumerate(result.detections)
        if item.geometry_valid and not item.review_reasons and not item.mask.get("warp_quad")
        and item.mask.get("source") == "yolo_seg_mask" and item.mask.get("detector_source") == "yolo_onnx"
        and (item.mask.get("fusion") or {}).get("proposal_families") == ["model"]
    ]
    peer_boxes = [item.bounding_box for _, item in models if item.confidence >= max(.85, acceptance)]
    if not peer_boxes:
        return result
    width, height = image.shape[1::-1]
    pending = list(fusion.get("deferred_print_proposals") or [])
    next_id = max((proposal["id"] for proposal in pending), default=-1) + 1
    offers = [(index, quad, traces, EMBOSSED_CARRIER_PROPOSAL, {"contained_print_polygon": child})
              for index, quad, traces, child in _embossed_carrier_offers(image, result, models, acceptance)]
    for index, item in models:
        provenance = item.mask["fusion"]
        if (not .30 <= item.confidence < min(.70, acceptance) or provenance.get("proposal_count") != 1
                or (provenance.get("boundary_evidence") or {}).get("border_contrast", 1.) > .53
                or sum(_adjacent_pocket_peer(item.bounding_box, peer) for peer in peer_boxes) != 1):
            continue
        source_outline = normalized_vertices_to_pixels(_detection_polygon(item), width, height)
        try:
            quad, _ = quad_from_polygon(source_outline)
            quad = order_quad_cyclic(quad)
            traces = embossed_pocket_traces(image, quad)
        except (ValueError, cv2.error):
            continue
        if traces:
            offers.append((index, quad, traces, EMBOSSED_POCKET_PROPOSAL, {}))
    if not offers:
        return result
    deferred = set()
    # Keep original offer IDs first for pages with both kinds of proposal.
    for index, quad, traces, reason, extra in sorted(offers, key=lambda offer: offer[3] != EMBOSSED_POCKET_PROPOSAL):
        item = result.detections[index]
        # The same continuous cover could hide a genuine paper print. Preserve
        # the proposal and its original ownership until explicit count review.
        pending.append({
            "id": next_id, "reason": reason, "polygon": _normalized_polygon(quad, width, height),
            "visible_polygon": item.mask.get("visible_polygon") or item.mask["polygon"],
            "embossing_edge_traces": [_normalized_polygon(trace, width, height) for trace in traces],
            "source_mask": {key: item.mask[key] for key in ("source", "detector_source", "polygon", "visible_polygon",
                            "visible_polygon_source", "proposal_visible_polygon") if key in item.mask},
            "fusion_cluster_id": item.mask["fusion"]["cluster_id"],
            "proposal_confidence": item.confidence,
            **extra,
        })
        next_id += 1
        deferred.add(index)
    return SegmentationResult(
        [item for index, item in enumerate(result.detections) if index not in deferred],
        {**result.metadata, "fusion": {**fusion, "deferred_print_proposals": pending}},
    )


def _offer_dark_mounted_print(image: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    metadata = result.metadata
    fusion = metadata.get("fusion") or {}
    if (not metadata.get("fallback_used") or not fusion.get("model_proposals")
            or not 2 <= len(result.detections) <= 12):
        return result
    pending = list(fusion.get("deferred_print_proposals") or [])
    claims = [_detection_polygon(item) for item in result.detections]
    for item in result.detections:
        if (visible := item.mask.get("visible_polygon")) and len(visible) >= 3:
            claims.append(np.float32([(point["x"], point["y"]) for point in visible]))
    for proposal in pending:
        if (polygon := proposal.get("visible_polygon") or proposal.get("polygon")) and len(polygon) >= 3:
            claims.append(np.float32([(point["x"], point["y"]) for point in polygon]))
    suggestion = find_dark_mounted_print_review(image, claims)
    if suggestion is None:
        return _review_truncated_dark_mount(image, result, claims)
    outline, traces = suggestion
    height, width = image.shape[:2]
    pending.append({
        "id": max((prior["id"] for prior in pending), default=-1) + 1,
        "reason": PROJECTIVE_PRINT_PROPOSAL,
        "polygon": _normalized_polygon(outline, width, height),
        "mounted_paper_edge_traces": [_normalized_polygon(trace, width, height) for trace in traces],
    })
    return SegmentationResult(result.detections, {
        **metadata, "fusion": {**fusion, "deferred_print_proposals": pending},
    })


def _unverified_frame_proposals(result: SegmentationResult) -> list[int]:
    return [
        index for index, item in enumerate(result.detections)
        if item.mask.get("detector_source") == "yolo_onnx" and not item.geometry_valid
        and set(item.review_reasons) <= {"aspect_ratio_too_wide", "aspect_ratio_too_tall"}
        and item.review_reasons and not item.mask.get("warp_quad")
        and len(item.mask.get("polygon") or []) >= 4
        and not item.mask.get("visible_polygon")
        and (item.mask.get("fusion") or {}).get("proposal_families") == ["model"]
    ]


def _flag_capture_background(source: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    candidates = _unverified_frame_proposals(result)
    if len(candidates) != 1:
        return result
    index, = candidates
    item = result.detections[index]
    peers = [_detection_polygon(peer) for peer in result.detections if peer.geometry_valid]
    try:
        traces = album_binding_traces(source, _detection_polygon(item), peers)
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    if not traces:
        return result
    updated = result.detections.copy()
    updated[index] = replace(item, mask={**item.mask, ALBUM_BOUNDARY_TRACES: [_normalized_polygon(span, 1, 1) for span in traces]},
                             review_reasons=[*item.review_reasons, CAPTURE_BACKGROUND_COUNT_REVIEW])
    return SegmentationResult(updated, result.metadata)


def _defer_capture_fabric_claims(source: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    height, width = source.shape[:2]
    candidates = _unverified_frame_proposals(result)
    contours = [normalized_vertices_to_pixels(_detection_polygon(result.detections[index]), width, height)
                for index in candidates]
    try:
        traces = capture_fabric_traces(source, contours)
    except (cv2.error, ValueError):
        return result
    if not traces:
        return result
    fusion = result.metadata.get("fusion") or {}
    pending = list(fusion.get("deferred_print_proposals") or [])
    next_id = max((offer["id"] for offer in pending), default=-1) + 1
    deferred = set()
    for local, spans in traces.items():
        index = candidates[local]
        item = result.detections[index]
        try:
            quad, _ = quad_from_polygon(contours[local])
            quad = order_quad_cyclic(np.clip(quad, 0, [width - 1, height - 1]))
        except (cv2.error, ValueError):
            continue
        pending.append({
            "id": next_id, "reason": CAPTURE_FABRIC_PROPOSAL,
            "polygon": _normalized_polygon(quad, width, height),
            "visible_polygon": item.mask["polygon"],
            "source_mask": {key: item.mask[key] for key in ("source", "detector_source", "polygon")},
            "fabric_similarity_traces": [_normalized_polygon(span, width, height) for span in spans],
            "fusion_cluster_id": item.mask["fusion"]["cluster_id"],
            "proposal_confidence": item.confidence,
        })
        next_id += 1
        deferred.add(index)
    if not deferred:
        return result
    return SegmentationResult(
        [item for index, item in enumerate(result.detections) if index not in deferred],
        {**result.metadata, "fusion": {**fusion, "deferred_print_proposals": pending}},
    )


def _review_truncated_dark_mount(
    image: np.ndarray, result: SegmentationResult, claims: list[np.ndarray],
) -> SegmentationResult:
    candidates = [
        index for index, item in enumerate(result.detections)
        if item.geometry_valid and not item.review_reasons and len(claims[index]) == 4
        and not (item.mask.get("visible_polygon") or item.mask.get("warp_quad"))
        and item.mask.get("source") == "classical_border_quad" and item.mask.get("detector_source") == "edge_contour"
        and .55 <= item.confidence <= .74
        and (item.mask.get("fusion") or {}).get("proposal_families") == ["classical"]
        and (item.mask.get("boundary_evidence") or {}).get("score", 1.) <= .58
    ]
    if len(candidates) != 1:
        return result
    index, = candidates
    evidence = find_dark_mounted_print_review(image, [*claims[:index], *claims[index + 1:]], partial_claim=claims[index])
    if evidence is None:
        return result
    _, spans = evidence
    height, width = image.shape[:2]
    detections = result.detections.copy()
    original = detections[index]
    detections[index] = replace(original, mask={
        **original.mask,
        "projective_outline_review": True,
        "mounted_paper_edge_traces": [_normalized_polygon(span, width, height) for span in spans],
    })
    return SegmentationResult(detections, result.metadata)


def segment_album_page(
    image: np.ndarray, settings: Settings, *, source_image: np.ndarray | None = None,
    protected_regions: list[dict] | None = None,
) -> SegmentationResult:
    source = source_image if source_image is not None and source_image.shape == image.shape else None
    evidence_source = source if source is not None else image
    tile_offers = []
    partial_tile_seeds = []
    merged_tile_pairs = []
    ribbed_tile_seeds = []
    contrast_review = []
    adjacent_view_pairs = []
    grid_view_pairs = []
    recentered_rotated_tile = []
    near_capture_views = []
    tile_review_search = []
    possible_region_inference = []
    contained_tile_pairs = []
    result = _segment_album_page(
        image, settings, source, tile_offers, partial_tile_seeds, merged_tile_pairs, ribbed_tile_seeds, contrast_review,
        adjacent_view_pairs, grid_view_pairs, recentered_rotated_tile, near_capture_views, tile_review_search,
        possible_region_inference, contained_tile_pairs,
    )
    if settings.enable_classical_segmentation_fallback:
        if settings.enable_precise_photo_crops and (
            result.metadata.get("fallback_used") or (result.metadata.get("model_set_risk") or {}).get("risky")
        ):
            result = _recover_separated_contrast_prints(evidence_source, result, settings)
        result = refine_album_page_result(image, result, source_image=evidence_source if settings.enable_precise_photo_crops else None)
        if settings.enable_precise_photo_crops:
            result = _refine_album_windows(evidence_source, result)
            result = _recover_multipanel_prints(evidence_source, result, settings)
            result = _recover_capture_edge_prints(evidence_source, result, settings)
            result = _recover_scalloped_prints(evidence_source, result, trim_border=settings.enable_border_trim)
            result = _suggest_adjacent_album_edge_print(evidence_source, result)
            result = _suggest_crossing_print_outline(evidence_source, result, prepared=image)
            result = _defer_nested_print_candidates(evidence_source, result)
            result = _flag_disputed_paper_grids(evidence_source, result)
            result = _flag_conflicting_film_grids(evidence_source, result)
            result = _flag_disputed_paper_joins(evidence_source, result)
            result = _offer_projective_print_review(evidence_source, result)
            if source is not None and settings.visible_mask_model_path is None:
                result = _offer_source_fragment_group_review(source, result, protected_regions)
    model_path = settings.visible_mask_model_path
    if not settings.enable_precise_photo_crops:
        return result
    if model_path is None:
        result = _refine_source_contours(evidence_source, result, settings, protected_regions)
        if settings.enable_classical_segmentation_fallback:
            if source is not None:
                result = _defer_nested_scene_fragments(source, result)
                result = _defer_dark_album_claims(source, result, settings.yolo_confidence_threshold)
                result = _defer_embossed_pocket_claims(source, result, settings.yolo_confidence_threshold)
                result = _defer_nested_scene_fragments(source, result, edge_contours=True)
            # The rim is a count-review annotation. Add it after ownership
            # and backing checks that use review reasons to skip proposals.
            result = _flag_conflicting_film_grids(evidence_source, result, require_clear_rim=True)
            if source is not None:
                result = _flag_crossing_fallback_contours(source, result)
                result = _flag_album_leaf_count_review(source, result)
                result = _offer_dark_mounted_print(source, result)
                result = _defer_capture_fabric_claims(source, result)
                result = _flag_capture_background(source, result)
        if tile_offers:
            result = _offer_unclaimed_tiled_prints(evidence_source, result, tile_offers)
        if partial_tile_seeds:
            result = _review_partial_tile_extent(evidence_source, result, partial_tile_seeds)
        if len(merged_tile_pairs) == 1:
            result = _review_merged_tile_pair(evidence_source, result, *merged_tile_pairs[0])
        if ribbed_tile_seeds:
            result = _offer_ribbed_backing_tile(evidence_source, result, ribbed_tile_seeds[0])
        if adjacent_view_pairs:
            result = _offer_paired_tile_prints(
                evidence_source, result, adjacent_view_pairs, measured_adjacent_tile_outline,
                "recovered_from_adjacent_views",
            )
        if grid_view_pairs:
            result = _offer_paired_tile_prints(
                evidence_source, result, grid_view_pairs, measured_grid_tile_outline, "recovered_from_grid_views",
            )
        if recentered_rotated_tile:
            started = perf_counter()
            try:
                outline = measured_recentered_ribbed_outline(
                    evidence_source, recentered_rotated_tile, _tiled_review_claims(result),
                )
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                outline = None
            review = dict(result.metadata["tiled_print_review"])
            review["runtime_ms"] = round(review["runtime_ms"] + (perf_counter() - started) * 1000, 2)
            if outline is not None:
                height, width = evidence_source.shape[:2]
                result = _offer_unclaimed_tiled_prints(evidence_source, result, [_normalized_polygon(outline, width, height)])
                review["candidates"] += 1
                review["recovered_from_recentered_rotation"] = True
            result = SegmentationResult(result.detections, {**result.metadata, "tiled_print_review": review})
        if near_capture_views and not result.metadata.get("missing_print_edge_traces"):
            try:
                traces = near_capture_print_traces(evidence_source, near_capture_views, _tiled_review_claims(result))
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                traces = []
            if traces:
                result = SegmentationResult(result.detections, {
                    **result.metadata, NEAR_CAPTURE_PRINT_TRACES: [_normalized_polygon(trace, 1, 1) for trace in traces],
                })
        if tile_review_search and (result.metadata.get("tiled_print_review") or {}).get("recovered_from_grid_views"):
            result = _offer_neighboring_print(evidence_source, image, result, tile_review_search[0])
        if contrast_review:
            (seed, traces), = contrast_review
            matching = [i for i, item in enumerate(result.detections) if _contrast_review_candidate(item)
                        and np.array_equal(_detection_polygon(item), seed)]
            if len(matching) == 1:
                index, = matching
                height, width = evidence_source.shape[:2]
                detections = result.detections.copy()
                item = detections[index]
                detections[index] = replace(item, mask={
                    **item.mask, CONTRAST_CROSSING_TRACES: [_normalized_polygon(span, width, height) for span in traces],
                })
                result = SegmentationResult(detections, result.metadata)
        if settings.enable_classical_segmentation_fallback:
            result = _flag_partial_film_grid(evidence_source, result)
            if source is not None:
                result = _flag_printed_illustration(source, result)
                result = _flag_shared_motif(source, result)
                result = _flag_compact_scene_crossings(source, result)
        if source is not None and possible_region_inference and any(item.mask.get(MERGED_PRINT_PAIR) for item in result.detections):
            result = _review_possible_print_regions(source, image, result, possible_region_inference[0],
                                                   settings.yolo_confidence_threshold)
        if source is not None and len(contained_tile_pairs) == 1:
            result = _review_contained_tile_pair(source, result, *contained_tile_pairs[0])
        if source is not None and (result.metadata.get("tiled_print_review") or {}).get("recovered_from_recentered_rotation"):
            result = _offer_neighboring_layout_traces(source, result)
        if source is not None and tile_review_search:
            result = _record_contested_tile_followup(source, image, result, tile_review_search[0])
            result = _offer_contested_scene_edge(source, image, result, tile_review_search[0], settings.yolo_confidence_threshold)
        if source is not None:
            result = _offer_mounted_window_review(source, result)
        return result

    from app.pipeline.visible_mask_refinement import UnresolvedMaskOwnership, refine_visible_masks, resolve_mask_ownership

    fixed_indices = {index for index, detection in enumerate(result.detections)
                     if detection.mask.get("source") == ALBUM_WINDOW_SOURCE
                     or detection.mask.get("detector_source") in {CONTRAST_COMPONENT_SOURCE, CAPTURE_EDGE_SOURCE}
                     or MULTIPANEL_PRINT_REVIEW in detection.review_reasons
                     or FRAGMENTED_PRINT_REVIEW in detection.review_reasons
                     or SCALLOPED_PRINT_REVIEW in detection.review_reasons
                     or SCALLOPED_PRINT_CONFLICT in detection.review_reasons}
    refinable = [index for index in range(len(result.detections)) if index not in fixed_indices]
    originals = [_detection_polygon(result.detections[index]) for index in refinable]
    fixed = [_detection_polygon(result.detections[index]) for index in sorted(fixed_indices)]
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
        result = SegmentationResult(updated, {**result.metadata, "visible_mask_refinement": details})
    except UnresolvedMaskOwnership:
        result = SegmentationResult(result.detections, {
            **result.metadata, "visible_mask_refinement": {"ownership_unresolved": True},
        })
    except Exception as exc:  # ONNX provider errors preserve the initial proposals for review.
        result = SegmentationResult(result.detections, {
            **result.metadata, "visible_mask_refinement": {"error": str(exc)},
        })
    if settings.enable_classical_segmentation_fallback:
        result = _flag_conflicting_film_grids(evidence_source, result, require_clear_rim=True)
        result = _flag_partial_film_grid(evidence_source, result)
    return result


def _contested_tile_outlines(
    source: np.ndarray, detections: list[SegmentationDetection], active: list[dict],
    pending: list[dict], blocked: list[list[dict]],
) -> list[SegmentationDetection]:
    """Attach an exclusive outline choice to an unsupported classical proposal."""
    height, width = source.shape[:2]
    alternatives: dict[int, list[list[dict]]] = {}
    for outline in blocked:
        if tiled_outline_conflicts(outline, pending, width, height):
            continue
        owners = [i for i, claim in enumerate(active) if tiled_outline_conflicts(outline, [claim], width, height)]
        if len(owners) != 1:
            continue
        index, = owners
        item = detections[index]
        fusion = item.mask.get("fusion") or {}
        if (not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad")
                or item.mask.get("source") != "classical_min_area_rect"
                or item.mask.get("detector_source") != "contrast_region"
                or fusion.get("proposal_count") != 1 or fusion.get("proposal_families") != ["classical"]):
            continue
        points = _detection_polygon(item)
        tile = np.float32([(point["x"], point["y"]) for point in outline])
        overlap = normalized_polygon_overlap(tile, points, raster_size=512)
        if len(points) != 4 or not .05 <= overlap.first_containment <= .30 or not .02 <= overlap.second_containment <= .30:
            continue
        try:
            claim = order_quad_cyclic(normalized_vertices_to_pixels(points, width, height))
            target = order_quad_cyclic(normalized_vertices_to_pixels(tile, width, height))
            if tile_crosses_unsupported_frame_claim(source, target, claim):
                alternatives.setdefault(index, []).append(outline)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
    if not alternatives:
        return detections
    updated = detections.copy()
    for index, outlines in alternatives.items():
        if len(outlines) == 1:
            item = detections[index]
            updated[index] = replace(item, mask={
                **item.mask, "crossing_print_suggestion": outlines[0], "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE,
            }, review_reasons=[CROSSING_PRINT_REVIEW])
    return updated


def _tiled_paper_claims(detections: list[SegmentationDetection]) -> list[dict]:
    return [
        {**item.mask, "polygon": item.mask.get("polygon") or _normalized_polygon(_detection_polygon(item), 1, 1)}
        for item in detections
    ]


def _offer_unclaimed_tiled_prints(
    source: np.ndarray, result: SegmentationResult, offers: list[list[dict]],
) -> SegmentationResult:
    height, width = source.shape[:2]
    fusion = result.metadata.get("fusion") or {}
    pending = list(fusion.get("deferred_print_proposals") or [])
    active = _tiled_paper_claims(result.detections)
    next_id = max((item["id"] for item in pending), default=-1) + 1
    accepted = []
    blocked = []
    for outline in offers:
        if tiled_outline_conflicts(outline, [*active, *pending, *accepted], width, height):
            blocked.append(outline)
        else:
            offer = {"id": next_id + len(accepted), "reason": TILED_PRINT_PROPOSAL, "polygon": outline}
            accepted.append(offer)
    detections = result.detections
    if blocked and result.metadata.get("fallback_used"):
        detections = _contested_tile_outlines(source, detections, active, [*pending, *accepted], blocked)
    if not accepted:
        return result if detections is result.detections else SegmentationResult(detections, result.metadata)
    return SegmentationResult(detections, {
        **result.metadata, "fusion": {**fusion, "deferred_print_proposals": [*pending, *accepted]},
    })


def _tiled_review_claims(result: SegmentationResult) -> list[dict]:
    pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
    claims = _tiled_paper_claims(result.detections) + pending
    for item in result.detections:
        if crossing := item.mask.get("crossing_print_suggestion"):
            claims.append({"polygon": crossing})
        for outline in (item.mask.get(MERGED_PRINT_PAIR) or {}).get("outlines", ()):
            claims.append({"polygon": outline})
    return claims


def _offer_contested_scene_edge(
    source: np.ndarray, image: np.ndarray, result: SegmentationResult, search: _TileReviewSearch, acceptance: float,
) -> SegmentationResult:
    owners = [index for index, item in enumerate(result.detections)
              if item.mask.get("crossing_print_suggestion_source") == CONTESTED_TILE_OUTLINE
              and item.mask.get("crossing_print_suggestion")]
    if len(owners) != 1 or not search.contested_scene_scouts:
        return result
    index, = owners
    owner = result.detections[index]
    height, width = source.shape[:2]
    peers = result.detections[:index] + result.detections[index + 1:]
    if any(item.mask.get(POSSIBLE_PRINT_REGIONS) for item in result.detections):
        return result
    claims = _tiled_review_claims(SegmentationResult(peers, result.metadata))
    claims.append({"polygon": owner.mask["crossing_print_suggestion"]})
    seeds = [_detection_polygon(item) for item in search.contested_scene_scouts
             if item.confidence < acceptance and not tiled_outline_conflicts(item.mask["polygon"], claims, width, height)
             and 0 < normalized_polygon_overlap(_detection_polygon(item), _detection_polygon(owner),
                                                raster_size=512).first_containment <= .05]
    if len(seeds) != 1:
        return result
    seed, = seeds
    try:
        inputs = _boundary_inputs(image)
        windows = ((.56, .60), (.52, .50)) if width >= height else ((.60, .56), (.50, .52))
        views = [_centered_tile_print_mask(image, seed, [], search.infer, inputs, window=window) for window in windows]
        if any(view is None for view in views):
            return result
        traces = contested_scene_edge_traces(source, views, [*claims, owner.mask])
    except Exception:  # Optional source advice must preserve the ordinary detections if a local view fails.
        return result
    if not traces:
        return result
    return SegmentationResult(result.detections, {**result.metadata, CONTESTED_SCENE_EDGE_TRACES: [
        _normalized_polygon(trace, width, height) for trace in traces
    ]})


def _offer_neighboring_layout_traces(source: np.ndarray, result: SegmentationResult) -> SegmentationResult:
    if any(item.mask.get(POSSIBLE_PRINT_REGIONS) for item in result.detections):
        return result
    pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
    anchors = [_detection_polygon(item) for item in result.detections
               if item.geometry_valid and item.mask.get("detector_source") == "yolo_onnx"
               and not item.mask.get(MERGED_PRINT_PAIR)]
    try:
        traces = neighboring_layout_traces(source, anchors, pending, _tiled_review_claims(result))
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    if not traces:
        return result
    height, width = source.shape[:2]
    return SegmentationResult(result.detections, {
        **result.metadata, NEIGHBORING_PRINT_EDGE_TRACES: [_normalized_polygon(trace, width, height) for trace in traces],
    })


def _record_contested_tile_followup(
    source: np.ndarray, image: np.ndarray, result: SegmentationResult, search: _TileReviewSearch,
) -> SegmentationResult:
    """Keep a measured neighbor dormant while the original proposal owns its pixels."""
    seed = search.unfitted_seed
    if seed is None:
        return result
    owners = [index for index, item in enumerate(result.detections)
              if item.mask.get("crossing_print_suggestion_source") == CONTESTED_TILE_OUTLINE
              and item.mask.get("crossing_print_suggestion")
              and item.mask.get("detector_source") == "contrast_region"]
    if len(owners) != 1:
        return result
    index, = owners
    item = result.detections[index]
    replacement = item.mask["crossing_print_suggestion"]
    overlap = normalized_polygon_overlap(seed, _detection_polygon(item), raster_size=1024)
    if overlap.first_containment < .80 or overlap.second_containment > .60:
        return result
    replacement_points = np.float32([(p["x"], p["y"]) for p in replacement])
    if normalized_polygon_overlap(seed, replacement_points, raster_size=1024).first_containment > .012:
        return result
    height, width = source.shape[:2]
    try:
        center = seed.mean(axis=0) * np.float32([width - 1, height - 1])
        observation = _centered_tile_print_mask(
            image, seed, [], search.infer, _boundary_inputs(image), window=(.44, .49), center=center,
        )
        if observation is None:
            return result
        outline = contested_tile_followup_outline(source, seed, observation, replacement)
        if outline is None:
            return result
        polygon = _normalized_polygon(outline, width, height)
        peers = result.detections[:index] + result.detections[index + 1:]
        claims = _tiled_review_claims(SegmentationResult(peers, result.metadata)) + [{"polygon": replacement}]
        if tiled_outline_conflicts(polygon, claims, width, height):
            return result
    except Exception:  # A failed optional view must preserve the earlier proposals.
        return result
    mask = {**item.mask, CONTESTED_TILE_FOLLOWUP: polygon}
    try:
        scale = np.float32([width - 1, height - 1])
        active = [quad_from_polygon(_detection_polygon(other) * scale)[0] for other in peers
                  if other.geometry_valid and other.mask.get("detector_source") == "yolo_onnx"
                  and not any(other.mask.get(key) for key in (POSSIBLE_PRINT_REGIONS, MERGED_PRINT_PAIR))]
        pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
        held = [order_quad_cyclic(np.float32([(p["x"], p["y"]) for p in row["polygon"]]) * scale)
                for row in pending if row.get("reason") == TILED_PRINT_PROPOSAL and len(row.get("polygon") or []) == 4]
        source_peer = order_quad_cyclic(outline)
        projected = followup_print_position(order_quad_cyclic(replacement_points * scale), source_peer, active, held)
        if projected is not None and (guide := followup_print_edge_guide(
            source, projected, source_peer, [*claims, {"polygon": polygon}],
        )):
            mask[FOLLOWUP_PRINT_EDGE_GUIDE] = guide
    except (cv2.error, ValueError, KeyError, TypeError, np.linalg.LinAlgError):
        pass  # The measured follow-up offer remains available independently of the open guide.
    detections = result.detections.copy()
    detections[index] = replace(item, mask=mask)
    return SegmentationResult(detections, result.metadata)


def _measure_neighboring_print(
    source: np.ndarray, image: np.ndarray, projection: np.ndarray, direction: np.ndarray,
    claims: list[dict], infer: Callable[[np.ndarray], list[SegmentationDetection]],
) -> np.ndarray | None:
    height, width = source.shape[:2]
    center = projection.mean(axis=0)
    projected = pixel_vertices_to_normalized(projection, width, height)
    inputs = _boundary_inputs(image)
    masks = []
    for extent in ((.36, .42), (.44, .49)):
        left, top, right, bottom = window = _tile_window(center, width, height, extent)
        mapped = _tile_detections_on_page(infer(image[top:bottom, left:right]), window, width, height)
        matches = [(item, polygon) for item in mapped
                   if normalized_polygon_overlap((polygon := _detection_polygon(item)), projected, raster_size=1024).iou >= .75]
        if len(matches) != 1:
            return None
        (item, polygon), = matches
        if (not item.geometry_valid or item.confidence < .80 or _detection_boundary_evidence(item, *inputs)["score"] < .80
                or tiled_outline_conflicts(item.mask["polygon"], claims, width, height)):
            return None
        masks.append(polygon)
    return neighboring_print_outline(source, masks, direction, claims)


def _offer_neighboring_print(
    source: np.ndarray, image: np.ndarray, result: SegmentationResult, search: _TileReviewSearch,
) -> SegmentationResult:
    started = perf_counter()
    review = dict(result.metadata["tiled_print_review"])
    pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
    claims = _tiled_review_claims(result) + search.initial_claims
    height, width = source.shape[:2]
    try:
        position = next_print_position(pending, claims, width, height)
        if position is None:
            return result
        outline = _measure_neighboring_print(source, image, *position, claims, search.infer)
        if outline is not None:
            result = _offer_unclaimed_tiled_prints(source, result, [_normalized_polygon(outline, width, height)])
            review["candidates"] += 1
            review["recovered_from_neighboring_prints"] = True
    except Exception as exc:  # An optional review view must not affect existing crop decisions.
        review["neighboring_print_error"] = str(exc)
    review["runtime_ms"] = round(review["runtime_ms"] + (perf_counter() - started) * 1000, 2)
    return SegmentationResult(result.detections, {**result.metadata, "tiled_print_review": review})


def _offer_ribbed_backing_tile(source: np.ndarray, result: SegmentationResult, seed: np.ndarray) -> SegmentationResult:
    started = perf_counter()
    fusion = result.metadata.get("fusion") or {}
    pending = list(fusion.get("deferred_print_proposals") or [])
    try:
        outline = ribbed_backing_tile_outline(source, seed, _tiled_review_claims(result))
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        outline = None
    review = dict(result.metadata["tiled_print_review"])
    review["runtime_ms"] = round(review["runtime_ms"] + (perf_counter() - started) * 1000, 2)
    metadata = {**result.metadata, "tiled_print_review": review}
    if outline is not None:
        next_id = max((item["id"] for item in pending), default=-1) + 1
        pending.append({"id": next_id, "reason": TILED_PRINT_PROPOSAL, "polygon": outline})
        metadata["fusion"] = {**fusion, "deferred_print_proposals": pending}
        review["recovered_from_ribbed_backing"] = True
        review["candidates"] += 1
    return SegmentationResult(result.detections, metadata)


def _offer_paired_tile_prints(
    source: np.ndarray, result: SegmentationResult, pairs: list[list[tuple[np.ndarray, float]]],
    measure: Callable[[np.ndarray, list[tuple[np.ndarray, float]], list[dict]], np.ndarray | None],
    statistic: str,
) -> SegmentationResult:
    started = perf_counter()
    height, width = source.shape[:2]
    fusion = result.metadata.get("fusion") or {}
    pending = list(fusion.get("deferred_print_proposals") or [])
    claims = _tiled_review_claims(result)
    fits = []
    for pair in pairs:
        try:
            if (outline := measure(source, pair, claims)) is not None:
                fits.append(_normalized_polygon(outline, width, height))
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
    # An unresolved competition for the same physical region yields no offer.
    offers = [outline for i, outline in enumerate(fits)
              if not tiled_outline_conflicts(outline, [{"polygon": fit} for j, fit in enumerate(fits) if i != j],
                                             width, height)]
    review = dict(result.metadata["tiled_print_review"])
    review["runtime_ms"] = round(review["runtime_ms"] + (perf_counter() - started) * 1000, 2)
    metadata = {**result.metadata, "tiled_print_review": review}
    if offers:
        next_id = max((item["id"] for item in pending), default=-1) + 1
        pending.extend({"id": next_id + i, "reason": TILED_PRINT_PROPOSAL, "polygon": outline}
                       for i, outline in enumerate(offers))
        metadata["fusion"] = {**fusion, "deferred_print_proposals": pending}
        review["candidates"] += len(offers)
        review[statistic] = len(offers)
    return SegmentationResult(result.detections, metadata)


def _tile_window(
    center: np.ndarray, width: int, height: int, window: tuple[float, float],
) -> tuple[int, int, int, int]:
    tile_width, tile_height = round(window[0] * width), round(window[1] * height)
    left = min(max(round(center[0] - tile_width / 2), 0), width - tile_width)
    top = min(max(round(center[1] - tile_height / 2), 0), height - tile_height)
    return left, top, left + tile_width, top + tile_height


def _adjacent_tile_search_locations(
    detections: list[SegmentationDetection], evidence: dict[int, dict[str, float]], width: int, height: int,
) -> list[tuple[np.ndarray, bool]]:
    """Project a compact print one row away along the axis shared by the long proposals."""
    anchors, axes = [], []
    for index, item in enumerate(detections):
        if item.review_reasons or not item.geometry_valid or evidence[index]["score"] < .76:
            continue
        points = normalized_vertices_to_pixels(_detection_polygon(item), width, height)
        quad, _ = quad_from_polygon(points)
        if quad is None:
            continue
        quad = order_quad_cyclic(quad)
        sides = np.float32([quad[1] - quad[0] + quad[2] - quad[3], quad[2] - quad[1] + quad[3] - quad[0]])
        lengths = np.linalg.norm(sides, axis=1)
        long_side = lengths.max()
        if long_side <= 0:
            continue
        ratio = lengths.min() / long_side
        area = cv2.contourArea(quad) / (width * height)
        if .42 <= ratio <= .70 and .055 <= area <= .16:
            axes.append(sides[lengths.argmax()] / long_side)
        elif ratio >= .80 and .035 <= area <= .09:
            anchors.append((quad, sides, lengths))
    if len(anchors) != 1 or len(axes) < 2 or any(abs(axis @ axes[0]) < .97 for axis in axes[1:]):
        return []
    (quad, sides, lengths), = anchors
    parallel = np.abs(sides @ axes[0]) / lengths
    if parallel.max() < .96:
        return []
    axis = sides[parallel.argmax()]
    horizontal = abs(axis[0]) > abs(axis[1])
    return [(quad + sign * .58 * axis, horizontal) for sign in (1, -1)]


def _collect_adjacent_view_pairs(
    image: np.ndarray, detections: list[SegmentationDetection], evidence: dict[int, dict[str, float]],
    offers: list[list[dict]], infer: Callable[[np.ndarray], list[SegmentationDetection]],
    inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray], acceptance: float,
) -> list[list[tuple[np.ndarray, float]]]:
    height, width = image.shape[:2]
    dims = np.float32([width - 1, height - 1])
    claims = _tiled_paper_claims(detections) + [{"polygon": outline} for outline in offers]
    pairs = []
    for projected, horizontal in _adjacent_tile_search_locations(detections, evidence, width, height):
        if (np.any(np.minimum(projected, dims - projected) < 4)
                or tiled_outline_conflicts(_normalized_polygon(projected, width, height), claims, width, height)):
            continue
        observations = []
        for window in ((.56, .60), (.48, .53)):
            left, top, right, bottom = bounds = _tile_window(
                projected.mean(axis=0), width, height, window[::-1] if horizontal else window,
            )
            matches = []
            for candidate in _tile_detections_on_page(infer(image[top:bottom, left:right]), bounds, width, height):
                polygon = _detection_polygon(candidate)
                if normalized_polygon_overlap(polygon, projected / dims, raster_size=512).iou >= .55:
                    matches.append((candidate, polygon))
            if len(matches) != 1:
                break
            (candidate, polygon), = matches
            if (tiled_outline_conflicts(candidate.mask["polygon"], claims, width, height)
                    or _detection_boundary_evidence(candidate, *inputs)["score"] < .80):
                break
            observations.append((polygon, candidate.confidence))
        if (len(observations) == 2 and max(confidence for _, confidence in observations) >= max(.80, acceptance)
                and normalized_polygon_overlap(observations[0][0], observations[1][0], raster_size=512).iou >= .90):
            pairs.append(observations)
    return pairs


def _collect_grid_view_pairs(
    image: np.ndarray, scouts: list[SegmentationDetection], claims: list[dict],
    infer: Callable[[np.ndarray], list[SegmentationDetection]],
    inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray], acceptance: float,
) -> list[list[tuple[np.ndarray, float]]]:
    """Revisit complete unclaimed masks in smaller contexts that contain them."""
    height, width = image.shape[:2]
    dims = np.float32([width - 1, height - 1])
    tile_width, tile_height = round(.52 * width), round(.52 * height)
    remaining_x, remaining_y = width - tile_width, height - tile_height
    clearance = max(3, round(.015 * min(width, height)))
    windows = [
        (x, y, x + tile_width, y + tile_height)
        for y in (0, round(remaining_y / 2), remaining_y)
        for x in (0, round(remaining_x / 2), remaining_x)
    ]
    cached = {}

    def candidates(bounds: tuple[int, int, int, int]) -> list[tuple[SegmentationDetection, np.ndarray]]:
        if bounds not in cached:
            left, top, right, bottom = bounds
            mapped = _tile_detections_on_page(infer(image[top:bottom, left:right]), bounds, width, height)
            cached[bounds] = [(item, _detection_polygon(item)) for item in mapped]
        return cached[bounds]

    def supported(item: SegmentationDetection) -> bool:
        return (item.confidence >= .80 and _detection_boundary_evidence(item, *inputs)["score"] >= .80
                and not tiled_outline_conflicts(item.mask["polygon"], claims, width, height))

    pairs = []
    unclaimed = (scout for scout in sorted(scouts, key=lambda item: item.confidence, reverse=True)
                 if not tiled_outline_conflicts(scout.mask["polygon"], claims, width, height))
    for scout in islice(unclaimed, 3):
        seed = _detection_polygon(scout)
        low, high = seed.min(axis=0) * dims, seed.max(axis=0) * dims
        observations = {}
        ambiguous = False
        for bounds in windows:
            left, top, right, bottom = bounds
            inner_min = [left + clearance if left else 0, top + clearance if top else 0]
            inner_max = [right - 1 - (clearance if right < width else 0),
                         bottom - 1 - (clearance if bottom < height else 0)]
            if np.any(low < inner_min) or np.any(high > inner_max):
                continue
            matches = [(item, mask) for item, mask in candidates(bounds)
                       if normalized_polygon_overlap(mask, seed, raster_size=512).iou >= .65]
            if len(matches) > 1:
                ambiguous = True
                break
            if matches:
                item, mask = matches[0]
                if supported(item):
                    observations[bounds] = (mask, item.confidence)
        if ambiguous or not observations:
            continue
        if len(observations) == 1:
            # One smaller grid view can still have a distinct centred check.
            first, _ = next(iter(observations.values()))
            center = (first.min(axis=0) + first.max(axis=0)) / 2 * dims
            bounds = _tile_window(center, width, height, (.52, .50))
            matches = [(item, mask) for item, mask in candidates(bounds)
                       if normalized_polygon_overlap(mask, first, raster_size=512).iou >= .65]
            if len(matches) != 1:
                continue
            (item, mask), = matches
            if not supported(item):
                continue
            observations[bounds] = (mask, item.confidence)
        ranked = sorted(observations.values(), key=lambda row: row[1], reverse=True)
        if (len(ranked) >= 2 and ranked[0][1] >= acceptance
                and all(normalized_polygon_overlap(first[0], second[0], raster_size=512).iou >= .90
                        for first, second in combinations(ranked, 2))):
            pairs.append(ranked[:2])
    return pairs


def _centered_tile_print_mask(
    image: np.ndarray, original: np.ndarray,
    claims: list[dict], infer: Callable[[np.ndarray], list[SegmentationDetection]],
    inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
    *, window: tuple[float, float] = (.56, .60), center: np.ndarray | None = None,
) -> np.ndarray | None:
    height, width = image.shape[:2]
    if center is None:
        center = (original.min(axis=0) + original.max(axis=0)) / 2 * [width - 1, height - 1]
    left, top, right, bottom = bounds = _tile_window(center, width, height, window)
    candidates = _tile_detections_on_page(infer(image[top:bottom, left:right]), bounds, width, height)
    matches = []
    for candidate in candidates:
        polygon = _detection_polygon(candidate)
        agreement = normalized_polygon_overlap(polygon, original, raster_size=512).iou
        if agreement >= .65:
            matches.append((candidate, polygon, agreement))
    if len(matches) != 1:
        return None
    (candidate, polygon, agreement), = matches
    if (candidate.confidence < .80 or agreement < .90 or _detection_boundary_evidence(candidate, *inputs)["score"] < .80
            or tiled_outline_conflicts(candidate.mask["polygon"], claims, width, height)):
        return None
    return polygon


def _centered_tile_print_offer(
    source: np.ndarray, image: np.ndarray, seed: SegmentationDetection, original_outline: np.ndarray,
    claims: list[dict], infer: Callable[[np.ndarray], list[SegmentationDetection]],
    inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
) -> list[dict] | None:
    """Try one closer view of an unclaimed print with an existing native fit."""
    height, width = source.shape[:2]
    original = _detection_polygon(seed)
    polygon = _centered_tile_print_mask(image, original, claims, infer, inputs)
    if polygon is None:
        return None
    outline = tile_print_outline(source, polygon, require_clear_sides=True)
    if outline is None:
        return None
    outline = order_quad_cyclic(outline)
    if np.max(np.linalg.norm(outline - order_quad_cyclic(original_outline), axis=1)) > 4:
        return None
    fit = outline / np.float32([width - 1, height - 1])
    if any(normalized_polygon_overlap(fit, observed, raster_size=512).iou < .85 for observed in (original, polygon)):
        return None
    normalized = _normalized_polygon(outline, width, height)
    return None if tiled_outline_conflicts(normalized, claims, width, height) else normalized


def _recover_unfitted_tile_print_offer(
    source: np.ndarray, image: np.ndarray, seed: SegmentationDetection, claims: list[dict],
    infer: Callable[[np.ndarray], list[SegmentationDetection]],
    inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
) -> list[dict] | None:
    """Require two close views with matching original-pixel fits when the first fit failed."""
    height, width = source.shape[:2]
    original = _detection_polygon(seed)
    fitted = []
    observed = [original]
    for window in ((.56, .60), (.48, .53)):
        polygon = _centered_tile_print_mask(image, original, claims, infer, inputs, window=window)
        if polygon is None:
            return None
        outline = tile_print_outline(source, polygon, minimum_side_support=.90)
        if outline is None:
            return None
        observed.append(polygon)
        fitted.append(order_quad_cyclic(outline))
    if (normalized_polygon_overlap(observed[1], observed[2], raster_size=512).iou < .90
            or np.max(np.linalg.norm(fitted[0] - fitted[1], axis=1)) > 4):
        return None
    for outline in fitted:
        vertices = pixel_vertices_to_normalized(outline, width, height)
        if (any(normalized_polygon_overlap(vertices, polygon, raster_size=512).iou < .85 for polygon in observed)
                or tiled_outline_conflicts(_normalized_polygon(outline, width, height), claims, width, height)):
            return None
    return _normalized_polygon(fitted[0], width, height)


def _review_partial_tile_extent(source: np.ndarray, result: SegmentationResult, seeds: list[list[dict]]) -> SegmentationResult:
    if not result.metadata.get("fallback_used"):
        return result
    height, width = source.shape[:2]
    pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
    updated = result.detections.copy()
    for outline in seeds:
        if tiled_outline_conflicts(outline, pending, width, height):
            continue
        owners = [i for i, item in enumerate(updated) if tiled_outline_conflicts(outline, [item.mask], width, height)]
        if len(owners) != 1:
            continue
        index, = owners
        item = updated[index]
        fusion = item.mask.get("fusion") or {}
        if (not item.geometry_valid or item.review_reasons or item.mask.get("warp_quad")
                or item.mask.get("source") != "classical_min_area_rect" or item.mask.get("detector_source") != "edge_contour"
                or fusion.get("proposal_count") != 1 or fusion.get("proposal_families") != ["classical"]):
            continue
        points = _detection_polygon(item)
        tile = np.float32([(point["x"], point["y"]) for point in outline])
        overlap = normalized_polygon_overlap(tile, points, raster_size=512)
        if len(points) != 4 or overlap.second_containment < .95 or not .35 <= overlap.first_containment <= .70:
            continue
        try:
            claim = order_quad_cyclic(normalized_vertices_to_pixels(points, width, height))
            target = order_quad_cyclic(normalized_vertices_to_pixels(tile, width, height))
            traces = partial_print_extent_traces(source, claim, target)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        if traces:
            updated[index] = replace(item, mask={
                **item.mask, PARTIAL_PRINT_EXTENT_TRACES: [_normalized_polygon(span, width, height) for span in traces],
            }, review_reasons=[PARTIAL_PRINT_EXTENT_REVIEW])
    return SegmentationResult(updated, result.metadata)


def _nominate_merged_tile_pair(
    parents: list[SegmentationDetection], candidates: list[SegmentationDetection], evidence: dict[int, dict[str, float]],
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    tiles = [(item, _detection_polygon(item)) for i, item in enumerate(candidates)
             if item.mask.get("detector_source") == "yolo_onnx_tiled" and item.confidence >= .65 and evidence[i]["score"] >= .80]
    nominees = []
    for parent in parents:
        if (not parent.geometry_valid or not .40 <= parent.confidence < .65
                or parent.mask.get("detector_source") != "yolo_onnx" or parent.mask.get("warp_quad")):
            continue
        original = _detection_polygon(parent)
        matching = []
        for tile, polygon in tiles:
            overlap = normalized_polygon_overlap(polygon, original, raster_size=512)
            if overlap.first_containment >= .94 and .30 <= overlap.second_containment <= .65:
                matching.append((tile, polygon, overlap.second_containment))
        if len(matching) != 2:
            continue
        (first, a, part_a), (second, b, part_b) = matching
        if (.80 <= part_a + part_b <= .97 and max(first.confidence, second.confidence) >= .80
                and normalized_polygon_overlap(a, b, raster_size=512).iou <= .01):
            nominees.append((original, [a, b]))
    return nominees


def _nominate_merged_parent_with_one_tile(
    parents: list[SegmentationDetection], candidates: list[SegmentationDetection], evidence: dict[int, dict[str, float]],
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    tiles = [_detection_polygon(item) for i, item in enumerate(candidates)
             if item.mask.get("detector_source") == "yolo_onnx_tiled" and item.confidence >= .80 and evidence[i]["score"] >= .80]
    nominees = []
    for parent in parents:
        if (not parent.geometry_valid or not .25 <= parent.confidence <= .38
                or parent.mask.get("detector_source") != "yolo_onnx" or parent.mask.get("warp_quad")
                or not .045 <= _score_value(parent, "area_ratio") <= .12):
            continue
        original = _detection_polygon(parent)
        matching = []
        for polygon in tiles:
            overlap = normalized_polygon_overlap(polygon, original, raster_size=512)
            if overlap.first_containment >= .90 and .35 <= overlap.second_containment <= .62:
                matching.append(polygon)
        if len(matching) == 1:
            nominees.append((original, matching))
    return nominees


def _nominate_contained_tile_pairs(
    parents: list[SegmentationDetection], candidates: list[SegmentationDetection], evidence: dict[int, dict[str, float]],
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    tiles = [_detection_polygon(item) for i, item in enumerate(candidates)
             if item.mask.get("detector_source") == "yolo_onnx_tiled" and item.confidence >= .80 and evidence[i]["score"] >= .80]
    proposals = []
    for parent in parents:
        if (parent.geometry_valid or not .25 <= parent.confidence < .50
                or parent.mask.get("detector_source") != "yolo_onnx" or parent.mask.get("warp_quad")
                or not .07 <= _score_value(parent, "area_ratio") <= .16):
            continue
        polygon = _detection_polygon(parent)
        overlaps = [(tile, normalized_polygon_overlap(tile, polygon, raster_size=512)) for tile in tiles]
        matching = [(tile, overlap.second_containment) for tile, overlap in overlaps
                    if overlap.first_containment >= .94 and .34 <= overlap.second_containment <= .58]
        if len(matching) == 2:
            (a, part_a), (b, part_b) = matching
            if .83 <= part_a + part_b <= .97 and normalized_polygon_overlap(a, b, raster_size=512).iou <= .01:
                proposals.append((polygon, [a, b]))
    return proposals


def _review_contained_tile_pair(
    source: np.ndarray, result: SegmentationResult, parent: np.ndarray, tiles: list[np.ndarray],
) -> SegmentationResult:
    owners = [i for i, item in enumerate(result.detections) if np.array_equal(_detection_polygon(item), parent)]
    if len(owners) != 1:
        return result
    owner, = owners
    item = result.detections[owner]
    geometry_reasons = {"PRINT_CORNERS_UNCERTAIN", "aspect_ratio_too_tall", "aspect_ratio_too_wide"}
    if (item.geometry_valid or item.mask.get("warp_quad") or item.mask.get("crossing_print_suggestion")
            or set(item.review_reasons) - geometry_reasons):
        return result
    peer_polygons = [_detection_polygon(peer) for i, peer in enumerate(result.detections)
                     if i != owner and peer.geometry_valid and peer.mask.get("detector_source") == "yolo_onnx"]
    peers = [(tile_index, peer) for tile_index, tile in enumerate(tiles) for peer in peer_polygons
             if normalized_polygon_overlap(tile, peer, raster_size=512).iou >= .85]
    if len(peers) != 1:
        return result
    (tile_index, peer), = peers
    if normalized_polygon_overlap(peer, parent, raster_size=512).first_containment < .94:
        return result
    candidate = tiles[1 - tile_index]
    height, width = source.shape[:2]
    other = result.detections[:owner] + result.detections[owner + 1:]
    claims = _tiled_review_claims(SegmentationResult(other, result.metadata))
    raw = _normalized_polygon(candidate, 1, 1)
    if tiled_outline_conflicts(raw, claims, width, height):
        return result
    try:
        seed, peer_seed = [quad_from_polygon(normalized_vertices_to_pixels(poly, width, height))[0]
                           for poly in (candidate, peer)]
        if seed is None or peer_seed is None:
            return result
        outline = measured_contained_print_outline(source, order_quad_cyclic(seed), order_quad_cyclic(peer_seed))
        if outline is None:
            return result
        normalized = _normalized_polygon(outline, width, height)
        footprint = pixel_vertices_to_normalized(outline, width, height)
        if (normalized_polygon_overlap(footprint, candidate, raster_size=512).iou < .85
                or normalized_polygon_overlap(footprint, parent, raster_size=512).first_containment < .95
                or tiled_outline_conflicts(normalized, claims, width, height)):
            return result
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    detections = result.detections.copy()
    detections[owner] = replace(item, mask={
        **item.mask, "crossing_print_suggestion": normalized, "crossing_print_suggestion_source": CONTESTED_TILE_OUTLINE,
    }, review_reasons=[*item.review_reasons, CROSSING_PRINT_REVIEW])
    return SegmentationResult(detections, result.metadata)


def _review_merged_tile_pair(
    source: np.ndarray, result: SegmentationResult, parent: np.ndarray, polygons: list[np.ndarray],
) -> SegmentationResult:
    owners = [i for i, item in enumerate(result.detections) if np.array_equal(_detection_polygon(item), parent)]
    if len(owners) != 1:
        return result
    index, = owners
    item = result.detections[index]
    if item.review_reasons or item.mask.get("warp_quad") or item.mask.get("detector_source") != "yolo_onnx":
        return result
    height, width = source.shape[:2]
    try:
        seeds = [quad_from_polygon(normalized_vertices_to_pixels(poly, width, height))[0] for poly in polygons]
        if any(seed is None for seed in seeds):
            return result
        seeds = [order_quad_cyclic(seed) for seed in seeds]
        if len(seeds) == 1:
            parent_seed, _ = quad_from_polygon(normalized_vertices_to_pixels(parent, width, height))
            if parent_seed is None:
                return result
            pair = measured_merged_print_with_one_tile(source, order_quad_cyclic(parent_seed), seeds[0])
        else:
            pair = measured_merged_print_pair(source, seeds)
        if pair is None:
            return result
        outlines, lane = pair
        footprints = [pixel_vertices_to_normalized(quad, width, height) for quad in outlines]
        if len(seeds) == 1 and normalized_polygon_overlap(footprints[1], polygons[0], raster_size=512).iou < .84:
            return result
        overlaps = [normalized_polygon_overlap(poly, parent, raster_size=512) for poly in footprints]
        if (min(overlap.first_containment for overlap in overlaps) < .96
                or not .78 <= sum(overlap.second_containment for overlap in overlaps) <= .95):
            return result
        shown = [_normalized_polygon(quad, width, height) for quad in outlines]
        pending = (result.metadata.get("fusion") or {}).get("deferred_print_proposals") or []
        other_claims = [peer.mask for i, peer in enumerate(result.detections) if i != index] + pending
        direction = lane[1] - lane[0]
        normal = 2 * np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
        lane_band = _normalized_polygon(np.vstack((lane - normal, lane[::-1] + normal)), width, height)
        if any(tiled_outline_conflicts(outline, other_claims, width, height) for outline in (*shown, lane_band)):
            return result
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return result
    detections = result.detections.copy()
    detections[index] = replace(item, mask={
        **item.mask, MERGED_PRINT_PAIR: {"outlines": shown, "separator": _normalized_polygon(lane, width, height)},
    }, review_reasons=[MERGED_PRINT_PAIR_REVIEW])
    return SegmentationResult(detections, result.metadata)


def _review_possible_print_regions(
    source: np.ndarray, prepared: np.ndarray, result: SegmentationResult,
    infer: Callable[[np.ndarray], list[SegmentationDetection]], confidence_threshold: float,
) -> SegmentationResult:
    candidates = []
    for index, item in enumerate(result.detections):
        mask = item.mask
        fusion = mask.get("fusion") or {}
        if (not item.geometry_valid or item.review_reasons or not .55 <= item.confidence < min(.70, confidence_threshold)
                or mask.get("source") != "yolo_seg_mask" or mask.get("detector_source") != "yolo_onnx"
                or fusion.get("proposal_count") != 1 or fusion.get("proposal_families") != ["model"]
                or any(mask.get(key) for key in ("warp_quad", "visible_polygon", "crossing_print_suggestion", MERGED_PRINT_PAIR))):
            continue
        parent = _detection_polygon(item)
        if (window := possible_print_region_window(parent, source.shape[:2])) is not None:
            candidates.append((index, parent, window))
    if len(candidates) != 1:
        return result
    (index, parent, window), = candidates
    height, width = source.shape[:2]
    left, top, right, bottom = window
    started = perf_counter()
    try:
        local = _tile_detections_on_page(infer(prepared[top:bottom, left:right]), window, width, height)
        inputs = _boundary_inputs(prepared)
        observations = [(_detection_polygon(item), item.confidence, _detection_boundary_evidence(item, *inputs)["score"])
                        for item in local]
        peers = result.detections[:index] + result.detections[index + 1:]
        claims = _tiled_review_claims(SegmentationResult(peers, result.metadata))
        outlines = measure_possible_print_regions(source, parent, observations, claims)
    except Exception:  # Optional comparison never replaces detection or existing review evidence.
        outlines = None
    metadata = {**result.metadata, "possible_print_region_review": {
        "runtime_ms": round((perf_counter() - started) * 1000, 2), "found": outlines is not None,
    }}
    if outlines is None:
        return SegmentationResult(result.detections, metadata)
    detections = result.detections.copy()
    item = detections[index]
    detections[index] = replace(item, mask={
        **item.mask, POSSIBLE_PRINT_REGIONS: {"outlines": [_normalized_polygon(quad, width, height) for quad in outlines]},
    }, review_reasons=[POSSIBLE_PRINT_REGIONS_REVIEW])
    return SegmentationResult(detections, metadata)


def _collect_tiled_print_offers(
    source: np.ndarray, image: np.ndarray, detections: list[SegmentationDetection],
    model_evidence: dict[int, dict[str, float]], infer: Callable[[np.ndarray], list[SegmentationDetection]],
    inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
    partial_tile_seeds: list[list[dict]] | None,
    merged_tile_pairs: list[tuple[np.ndarray, list[np.ndarray]]] | None,
    grid_scouts: list[SegmentationDetection] | None = None,
    near_capture_scouts: list[tuple[np.ndarray, int]] | None = None,
    contained_tile_pairs: list[tuple[np.ndarray, list[np.ndarray]]] | None = None,
    contested_scene_scouts: list[SegmentationDetection] | None = None,
) -> tuple[
    list[list[dict]], tuple[SegmentationDetection, np.ndarray] | None,
    SegmentationDetection | None, SegmentationDetection | None,
]:
    candidates, evidence, _ = _recover_dense_prints(image, detections, model_evidence, infer, inputs)
    if contained_tile_pairs is not None:
        try:
            contained_tile_pairs.extend(_nominate_contained_tile_pairs(detections, candidates, evidence))
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            pass  # The held comparison must preserve existing offers.
    if merged_tile_pairs is not None:
        try:
            merged_tile_pairs.extend(_nominate_merged_tile_pair(detections, candidates, evidence))
            if not merged_tile_pairs:
                merged_tile_pairs.extend(_nominate_merged_parent_with_one_tile(detections, candidates, evidence))
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            pass  # Optional count guides must preserve ordinary tile offers.
    height, width = source.shape[:2]
    claims = [item.mask for item in detections]
    offers = []
    retry_seed = None
    unfitted_seed = None
    ribbed_seed = None
    for index, candidate in enumerate(candidates):
        if (grid_scouts is not None and candidate.mask.get("detector_source") == "yolo_onnx_tiled"
                and candidate.confidence >= .60 and evidence[index]["score"] >= .80
                and .025 <= _score_value(candidate, "area_ratio") <= .12):
            grid_scouts.append(candidate)
        if (near_capture_scouts is not None and candidate.geometry_valid
                and candidate.mask.get("detector_source") == "yolo_onnx_tiled"
                and candidate.confidence >= .80 and evidence[index]["score"] >= .70
                and .025 <= _score_value(candidate, "area_ratio") <= .12):
            polygon = _detection_polygon(candidate)
            turn = near_capture_candidate_turn(polygon, width, height)
            if turn is not None:
                near_capture_scouts.append((polygon, turn))
        strong = candidate.confidence >= .80
        if (candidate.mask.get("detector_source") != "yolo_onnx_tiled"
                or candidate.confidence < .65 or evidence[index]["score"] < .80
                or (not strong and retry_seed is not None)):
            continue
        conflicted = tiled_outline_conflicts(candidate.mask["polygon"], claims, width, height)
        polygon = _detection_polygon(candidate)
        if conflicted:
            if contested_scene_scouts is not None and not strong and candidate.geometry_valid:
                contested_scene_scouts.append(candidate)
            if not strong or partial_tile_seeds is None or partial_tile_seeds:
                continue
            # A small brush against one raw neighbor can seed a review aid.
            # This path cannot create a count offer.
            nearby = [item for item in detections
                      if tiled_outline_conflicts(candidate.mask["polygon"], [item.mask], width, height)]
            if len(nearby) != 1 or normalized_polygon_overlap(
                polygon, _detection_polygon(nearby[0]), raster_size=512,
            ).first_containment > .012:
                continue
        try:
            outline = tile_print_outline(source, polygon)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        if outline is None:
            if not conflicted:
                if strong and (ribbed_seed is None or candidate.confidence > ribbed_seed.confidence):
                    ribbed_seed = candidate
                elif not strong and (unfitted_seed is None or candidate.confidence > unfitted_seed.confidence):
                    unfitted_seed = candidate
            continue
        normalized = _normalized_polygon(outline, width, height)
        if tiled_outline_conflicts(normalized, claims, width, height):
            continue
        if conflicted:
            fit = pixel_vertices_to_normalized(outline, width, height)
            if partial_tile_seeds is not None and normalized_polygon_overlap(polygon, fit, raster_size=512).iou >= .85:
                partial_tile_seeds.append(normalized)
        elif strong:
            offers.append(normalized)
        else:
            retry_seed = (candidate, outline)
    return offers, retry_seed, unfitted_seed, ribbed_seed


def _collect_rotated_print_offers(
    source: np.ndarray, image: np.ndarray, detections: list[SegmentationDetection],
    existing_offers: list[list[dict]], infer: Callable[[np.ndarray], list[SegmentationDetection]], acceptance: float,
    *, scouts: list[np.ndarray], inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
) -> list[list[dict]]:
    height, width = source.shape[:2]
    claims = [item.mask for item in detections] + [{"polygon": outline} for outline in existing_offers]
    observations = []
    for turn in (1, 3):
        shapes = []
        for item in infer(np.ascontiguousarray(np.rot90(image, turn))):
            if not item.geometry_valid or item.confidence < .55 or item.mask.get("source") != "yolo_seg_mask":
                continue
            x, y = _detection_polygon(item).T
            mapped = np.column_stack((1 - y, x) if turn == 1 else (y, 1 - x))
            polygon = _normalized_polygon(mapped, 1, 1)
            if tiled_outline_conflicts(polygon, claims, width, height):
                continue
            shapes.append((mapped, item.confidence))
            if item.confidence >= .60 and .025 <= _score_value(item, "area_ratio") <= .12:
                native = replace(item, mask={**item.mask, "polygon": polygon})
                if _detection_boundary_evidence(native, *inputs)["score"] >= .80:
                    scouts.append(mapped)
        observations.append(shapes)
    offers = []
    for outline in opposing_view_print_outlines(source, *observations, acceptance=acceptance):
        normalized = _normalized_polygon(outline, width, height)
        if not tiled_outline_conflicts(normalized, claims, width, height):
            offers.append(normalized)
            claims.append({"polygon": normalized})
    return offers


def _recenter_rotated_print(
    source: np.ndarray, image: np.ndarray, scouts: list[np.ndarray], claims: list[dict],
    infer: Callable[[np.ndarray], list[SegmentationDetection]], inputs: tuple[np.ndarray, tuple[np.ndarray, float], np.ndarray],
) -> list[np.ndarray]:
    height, width = source.shape[:2]
    unclaimed = [seed for seed in scouts if not tiled_outline_conflicts(_normalized_polygon(seed, 1, 1), claims, width, height)]
    if len(unclaimed) != 1:
        return []
    seed, = unclaimed
    fitted = ribbed_backing_tile_outline(source, seed, claims)
    if fitted is None:
        return []
    points = normalized_vertices_to_pixels([(point["x"], point["y"]) for point in fitted], width, height)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    # Shift towards the page center to keep context on both sides of the backing.
    center[0] += .025 * width * np.sign((width - 1) / 2 - center[0])
    match = _centered_tile_print_mask(image, seed, claims, infer, inputs, window=(.60, .60), center=center)
    return [] if match is None else [seed, match]


def _near_capture_tile_views(
    image: np.ndarray, scouts: list[tuple[np.ndarray, int]],
    infer: Callable[[np.ndarray], list[SegmentationDetection]],
) -> list[np.ndarray]:
    if len(scouts) != 1:
        return []
    (polygon, turn), = scouts
    prepared = np.ascontiguousarray(np.rot90(image, turn))
    height, width = prepared.shape[:2]
    polygon = quarter_turn(polygon, turn)
    center = polygon.mean(axis=0) * np.float32([width - 1, height - 1])
    offset = np.float32([.02 * width, .025 * height * np.sign(center[1] - (height - 1) / 2)])
    result = []
    for position, extent in ((center, (.56, .60)), (center + offset, (.55, .50))):
        left, top, right, bottom = window = _tile_window(position, width, height, extent)
        detections = _tile_detections_on_page(infer(prepared[top:bottom, left:right]), window, width, height)
        matches = [raw for item in detections if item.geometry_valid and item.confidence >= .80
                   and normalized_polygon_overlap((raw := _detection_polygon(item)), polygon, raster_size=1024).iou >= .80]
        if len(matches) != 1:
            return []
        result.append(quarter_turn(matches[0], (-turn) % 4))
    return result


def _contrast_review_candidate(item: SegmentationDetection) -> bool:
    fusion = item.mask.get("fusion") or {}
    return (
        item.geometry_valid and not item.review_reasons and not item.mask.get("warp_quad")
        and item.mask.get("source") == "classical_min_area_rect" and item.mask.get("detector_source") == "contrast_region"
        and fusion.get("proposal_count") == 1 and fusion.get("proposal_families") == ["classical"]
    )


def _contrast_review_observations(
    image: np.ndarray, quad: np.ndarray, infer: Callable[[np.ndarray], list[SegmentationDetection]],
) -> Iterator[tuple[int, np.ndarray, float]]:
    height, width = image.shape[:2]
    inputs = _boundary_inputs(image)
    observations = {}
    for side, window in contrast_review_windows(quad, width, height):
        if window not in observations:
            left, top, right, bottom = window
            detections = _tile_detections_on_page(infer(image[top:bottom, left:right]), window, width, height)
            observations[window] = [
                (normalized_vertices_to_pixels(_detection_polygon(item), width, height), item.confidence)
                for item in detections if item.geometry_valid and item.confidence >= .76
                and item.mask.get("source") == "yolo_seg_mask" and _detection_boundary_evidence(item, *inputs)["score"] >= .68
            ]
        for polygon, confidence in observations[window]:
            yield side, polygon, confidence


def _segment_album_page(
    image: np.ndarray, settings: Settings, source: np.ndarray | None = None,
    tile_offers: list[list[dict]] | None = None,
    partial_tile_seeds: list[list[dict]] | None = None,
    merged_tile_pairs: list[tuple[np.ndarray, list[np.ndarray]]] | None = None,
    ribbed_tile_seeds: list[np.ndarray] | None = None,
    contrast_review: list[tuple[np.ndarray, list[np.ndarray]]] | None = None,
    adjacent_view_pairs: list[list[tuple[np.ndarray, float]]] | None = None,
    grid_view_pairs: list[list[tuple[np.ndarray, float]]] | None = None,
    recentered_rotated_tile: list[np.ndarray] | None = None,
    near_capture_views: list[np.ndarray] | None = None,
    tile_review_search: list[_TileReviewSearch] | None = None,
    possible_region_inference: list[Callable[[np.ndarray], list[SegmentationDetection]]] | None = None,
    contained_tile_pairs: list[tuple[np.ndarray, list[np.ndarray]]] | None = None,
) -> SegmentationResult:
    model_path = Path(settings.yolo_model_path)
    line_budget = min(3 * image.shape[0] * image.shape[1], 12_000_000) if settings.enable_precise_photo_crops else None
    multipanel_source = (source if source is not None else image) if settings.enable_precise_photo_crops else None
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
            rejected_proposals: Counter[str] = Counter()

            def infer(frame: np.ndarray, *, record_rejections: bool = True):
                letterboxed, transform = _letterbox_image(frame, target_width, target_height)
                tensor = letterboxed[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                outputs = session.run(None, {input_meta.name: tensor[np.newaxis, ...]})
                detections = _parse_simple_box_outputs(
                    outputs, frame.shape[1], frame.shape[0], settings,
                    input_width=target_width, input_height=target_height,
                    rejected=rejected_proposals if record_rejections else None,
                )
                if record_rejections and rejected_proposals:
                    metadata["rejected_model_proposals"] = dict(rejected_proposals)
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
                if (source is not None and tile_offers is not None and deskew_enabled and risk_reasons
                        and settings.visible_mask_model_path is None and 2 <= len(detections) <= 8
                        and sum(item.geometry_valid and .02 <= _score_value(item, "area_ratio") <= .20
                                for item in detections) >= 2
                        and sum(item.confidence < settings.yolo_confidence_threshold for item in detections) >= 2):
                    review_started = perf_counter()
                    tile_review = {}
                    unfitted_seed = None
                    try:
                        grid_scouts = []
                        rotated_scouts = []
                        near_capture_scouts = []
                        contested_scene_scouts = []
                        direct, retry_seed, unfitted_seed, ribbed_seed = _collect_tiled_print_offers(
                            source, image, detections, model_evidence, lambda frame: infer(frame)[0], inputs,
                            partial_tile_seeds, merged_tile_pairs, grid_scouts, near_capture_scouts, contained_tile_pairs,
                            contested_scene_scouts,
                        )
                        tile_offers.extend(direct)
                        if tile_offers:
                            # A successful tile pass justifies two extra full-page views.
                            try:
                                rotated = _collect_rotated_print_offers(
                                    source, image, detections, tile_offers, lambda frame: infer(frame)[0],
                                    settings.yolo_confidence_threshold,
                                    scouts=rotated_scouts, inputs=inputs,
                                )
                            except Exception as exc:  # The centered view can still corroborate its seed.
                                tile_review["rotated_error"] = str(exc)
                            else:
                                tile_offers.extend(rotated)
                                tile_review["rotated_candidates"] = len(rotated)
                        if retry_seed is not None:
                            claims = [item.mask for item in detections] + [{"polygon": outline} for outline in tile_offers]
                            centered = _centered_tile_print_offer(
                                source, image, *retry_seed, claims, lambda frame: infer(frame)[0], inputs,
                            )
                            if centered is not None:
                                tile_offers.append(centered)
                                tile_review["centered_candidate"] = True
                        if not tile_offers and unfitted_seed is not None:
                            recovered = _recover_unfitted_tile_print_offer(
                                source, image, unfitted_seed, [item.mask for item in detections], lambda frame: infer(frame)[0], inputs,
                            )
                            if recovered is not None:
                                tile_offers.append(recovered)
                                tile_review["recovered_from_centered_pair"] = True
                        if ribbed_seed is not None and ribbed_tile_seeds is not None:
                            ribbed_tile_seeds.append(_detection_polygon(ribbed_seed))
                        if tile_offers and adjacent_view_pairs is not None:
                            adjacent_view_pairs.extend(_collect_adjacent_view_pairs(
                                image, detections, model_evidence, tile_offers, lambda frame: infer(frame)[0], inputs,
                                settings.yolo_confidence_threshold,
                            ))
                        if tile_offers and grid_scouts and grid_view_pairs is not None:
                            claims = _tiled_paper_claims(detections) + [{"polygon": outline} for outline in tile_offers]
                            grid_view_pairs.extend(_collect_grid_view_pairs(
                                image, grid_scouts, claims, lambda frame: infer(frame)[0], inputs,
                                settings.yolo_confidence_threshold,
                            ))
                        if rotated_scouts and recentered_rotated_tile is not None:
                            claims = _tiled_paper_claims(detections) + [{"polygon": outline} for outline in tile_offers]
                            recentered_rotated_tile.extend(_recenter_rotated_print(
                                source, image, rotated_scouts, claims, lambda frame: infer(frame, record_rejections=False)[0], inputs,
                            ))
                        if near_capture_scouts and near_capture_views is not None:
                            try:
                                near_capture_views.extend(_near_capture_tile_views(
                                    image, near_capture_scouts, lambda frame: infer(frame, record_rejections=False)[0],
                                ))
                            except Exception as exc:  # Advice is independent of the detector and crop decisions.
                                tile_review["near_capture_error"] = str(exc)
                    except Exception as exc:  # Keep earlier proposals if an optional view fails.
                        tile_review["error"] = str(exc)
                    metadata["tiled_print_review"] = {
                        **tile_review, "candidates": len(tile_offers),
                        "runtime_ms": round((perf_counter() - review_started) * 1000, 2),
                    }
                    if tile_review_search is not None and (grid_view_pairs or unfitted_seed is not None or contested_scene_scouts):
                        tile_review_search.append(_TileReviewSearch(
                            lambda frame: infer(frame, record_rejections=False)[0], _tiled_paper_claims(detections),
                            _detection_polygon(unfitted_seed) if unfitted_seed is not None else None,
                            contested_scene_scouts,
                        ))
                    if possible_region_inference is not None and merged_tile_pairs:
                        possible_region_inference.append(lambda frame: infer(frame, record_rejections=False)[0])
                del inputs
                if detections and settings.enable_classical_segmentation_fallback and risk_reasons and not dense_small_prints:
                    fallback = detect_photos_classical(
                        image, max_line_refinement_pixels=line_budget, multipanel_source=multipanel_source,
                    )
                    merged, fusion = fuse_candidate_evidence(
                        image, detections, fallback.detections, model_evidence=model_evidence,
                        ownership_source=(image if source is None else source) if settings.enable_precise_photo_crops else None,
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
                    if (source is not None and contrast_review is not None
                            and "tiled_print_review" in metadata and 2 <= len(merged) <= 8):
                        candidates = [item for item in merged if _contrast_review_candidate(item)]
                        if len(candidates) == 1:
                            seed = _detection_polygon(candidates[0])
                            if len(seed) == 4 and cv2.isContourConvex(seed):
                                started = perf_counter()
                                quad = order_quad_cyclic(normalized_vertices_to_pixels(seed, *source.shape[1::-1]))
                                try:
                                    traces = contrast_crossing_traces(source, quad, _contrast_review_observations(
                                        image, quad, lambda frame: infer(frame, record_rejections=False)[0],
                                    ))
                                except Exception:  # A failed optional view leaves the existing proposals intact.
                                    traces = []
                                if traces:
                                    contrast_review.append((seed, traces))
                                    metadata["contrast_crossing_review"] = {
                                        "runtime_ms": round((perf_counter() - started) * 1000, 2),
                                    }
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

    fallback = detect_photos_classical(image, max_line_refinement_pixels=line_budget, multipanel_source=multipanel_source)
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
