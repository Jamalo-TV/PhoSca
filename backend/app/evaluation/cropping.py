"""Assignment-based cropping metrics.

The production detector is deliberately not imported here.  Keeping the
metrics independent prevents a detector change from silently changing the
definition of success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from typing import Sequence

import cv2
import numpy as np

Polygon = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class PageCroppingMetrics:
    expected: int
    detected: int
    true_positive: int
    false_positive: int
    false_negative: int
    duplicates: int
    polygon_iou: float
    mask_iou: float
    boundary_f_score: float
    corner_error: float | None
    content_cutoff: float
    background_retained: float
    serious_failures: int
    automatic_acceptances: int
    review_count: int
    matches: tuple[dict, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CroppingDatasetMetrics:
    page_count: int
    expected_photos: int
    detected_photos: int
    detection_precision: float
    detection_recall: float
    detection_f1: float
    polygon_iou: float
    mask_iou: float
    boundary_f_score: float
    corner_localization_error: float | None
    missing_photo_rate: float
    duplicate_detection_rate: float
    false_positive_rate: float
    content_cutoff: float
    background_retained: float
    automatic_acceptance_rate: float
    human_review_rate: float
    serious_failure_rate: float
    page_perfect_rate: float
    totals: dict[str, int]

    def as_dict(self) -> dict:
        return asdict(self)


def is_locked_album_manifest(records: Sequence[dict], split: dict) -> bool:
    """Check that the declared independent split pins distinct, reviewed album pages."""
    groups = {str(record.get("source_group") or "").strip().casefold() for record in records}
    hashes = {str(record.get("image_sha256") or "").strip().lower() for record in records}
    return (
        split.get("album_independent") is True and "" not in groups and len(groups) >= 2
        and len(hashes) == len(records)
        and all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in hashes)
        and all(record.get("annotation_provenance") == "manually_verified" for record in records)
    )


def _safe_polygon(polygon: Polygon) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        return np.empty((0, 2), dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0)
    return points


def _mask(polygon: Polygon, width: int, height: int) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.uint8)
    points = _safe_polygon(polygon)
    if len(points) < 3:
        return result
    scaled = np.rint(points * np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(result, [scaled], 1)
    return result


def _mask_overlap(
    truth: Polygon, prediction: Polygon, width: int, height: int, *, truth_mask: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    if truth_mask is None:
        truth_mask = _mask(truth, width, height)
    prediction_mask = _mask(prediction, width, height)
    truth_area = int(truth_mask.sum())
    prediction_area = int(prediction_mask.sum())
    intersection = int(np.logical_and(truth_mask, prediction_mask).sum())
    union = truth_area + prediction_area - intersection
    iou = intersection / union if union else 0.0
    cutoff = (truth_area - intersection) / truth_area if truth_area else 1.0
    background = (prediction_area - intersection) / prediction_area if prediction_area else 1.0
    return iou, iou, cutoff, background


def _pixel_bounds(
    polygon: Polygon, width: int, height: int, *, truth_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int] | None:
    if truth_mask is not None:
        left, top, span_x, span_y = cv2.boundingRect(truth_mask)
        return left, top, left + span_x - 1, top + span_y - 1
    points = _safe_polygon(polygon)
    if not len(points):
        return None
    pixels = np.rint(points * np.float32([max(width - 1, 1), max(height - 1, 1)])).astype(np.int32)
    left, top = pixels.min(axis=0)
    right, bottom = pixels.max(axis=0)
    return left, top, right, bottom


def _bounds_intersect(first, second) -> bool:
    if first is None or second is None:
        return False
    left, top, right, bottom = first
    x1, y1, x2, y2 = second
    return left <= x2 and x1 <= right and top <= y2 and y1 <= bottom


def _boundary_f_score(
    truth: Polygon, prediction: Polygon, width: int, height: int, tolerance_ratio: float = 0.0025,
    *, truth_mask: np.ndarray | None = None,
) -> float:
    if truth_mask is None:
        truth_mask = _mask(truth, width, height)
    prediction_mask = _mask(prediction, width, height)
    kernel = np.ones((3, 3), dtype=np.uint8)
    truth_boundary = cv2.morphologyEx(truth_mask, cv2.MORPH_GRADIENT, kernel)
    prediction_boundary = cv2.morphologyEx(prediction_mask, cv2.MORPH_GRADIENT, kernel)
    truth_pixels = int(truth_boundary.sum())
    prediction_pixels = int(prediction_boundary.sum())
    if not truth_pixels or not prediction_pixels:
        return 0.0
    tolerance = max(1, int(round(hypot(width, height) * tolerance_ratio)))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    truth_band = cv2.dilate(truth_boundary, dilation_kernel)
    prediction_band = cv2.dilate(prediction_boundary, dilation_kernel)
    precision = float(np.logical_and(prediction_boundary, truth_band).sum() / prediction_pixels)
    recall = float(np.logical_and(truth_boundary, prediction_band).sum() / truth_pixels)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _cyclic_quad(polygon: Polygon) -> np.ndarray | None:
    points = _safe_polygon(polygon)
    if len(points) < 4:
        return None
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
    perimeter = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    quad = None
    for epsilon in (0.005, 0.01, 0.02, 0.04, 0.08):
        approximation = cv2.approxPolyDP(hull.reshape(-1, 1, 2), epsilon * perimeter, True).reshape(-1, 2)
        if len(approximation) == 4:
            quad = approximation
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull.reshape(-1, 1, 2)))
    center = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0]))]
    start = int(np.argmin(quad[:, 0] + quad[:, 1]))
    return np.roll(quad, -start, axis=0).astype(np.float32)


def _aligned_corner_errors(truth_quad: np.ndarray, prediction_quad: np.ndarray) -> np.ndarray:
    alignments = (
        np.linalg.norm(truth_quad - np.roll(candidate, offset, axis=0), axis=1)
        for candidate in (prediction_quad, prediction_quad[::-1])
        for offset in range(4)
    )
    return min(alignments, key=np.mean)


def _corner_error(truth: Polygon, prediction: Polygon, width: int, height: int) -> float | None:
    truth_quad = _cyclic_quad(truth)
    prediction_quad = _cyclic_quad(prediction)
    if truth_quad is None or prediction_quad is None:
        return None
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    errors = _aligned_corner_errors(truth_quad * scale, prediction_quad * scale)
    return float(errors.mean() / np.linalg.norm(scale))


def _source_center_footprint(
    source_shape: tuple[int, int], matrix: np.ndarray, output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = source_shape
    out_height, out_width = output_shape
    corners = np.float64([[0, 0, 1], [out_width - 1, 0, 1], [out_width - 1, out_height - 1, 1], [0, out_height - 1, 1]])
    mapped = corners @ np.linalg.inv(matrix).T
    weights = mapped[:, 2]
    if not np.isfinite(mapped).all() or not (np.all(weights > 0) or np.all(weights < 0)):
        raise ValueError("Crop footprint crosses the projective horizon")
    quad = mapped[:, :2] / weights[:, None]
    orientation = np.sign(cv2.contourArea(quad.astype(np.float32), oriented=True))
    if not np.isfinite(quad).all() or orientation not in (-1, 1):
        raise ValueError("Invalid crop footprint")

    # Scan source pixel centers between the inverse-mapped output centers, with
    # no half-output-pixel extension at the perimeter.
    tolerance = 1e-5
    first_y = max(0, int(np.ceil(quad[:, 1].min() - tolerance)))
    last_y = min(height - 1, int(np.floor(quad[:, 1].max() + tolerance)))
    rows = np.arange(first_y, last_y + 1)
    left = np.zeros(len(rows))
    right = np.full(len(rows), width - 1, dtype=np.float64)
    valid = np.ones(len(rows), dtype=bool)
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        dx, dy = (end - start) * orientation
        bound = dx * (rows - start[1]) + tolerance * np.hypot(dx, dy)
        if dy > 1e-12:
            right = np.minimum(right, start[0] + bound / dy)
        elif dy < -1e-12:
            left = np.maximum(left, start[0] + bound / dy)
        else:
            valid &= bound >= 0
    valid &= left <= right
    footprint = np.zeros(source_shape, dtype=bool)
    for row, first, last in zip(rows[valid], np.ceil(left[valid]).astype(int), np.floor(right[valid]).astype(int), strict=True):
        footprint[row, first:last + 1] = True
    return footprint, quad


def evaluate_rendered_crop(
    expected_mask: np.ndarray,
    source_quad: Polygon,
    page_to_output: np.ndarray,
    output_shape: tuple[int, int],
    *,
    expected_corners: np.ndarray | None = None,
    expected_shape: tuple[int, int] | None = None,
    expected_aspect_ratio: float | None = None,
    source_landmarks: np.ndarray | None = None,
    expected_landmarks: np.ndarray | None = None,
    sampling: str | None = None,
) -> dict:
    """Score the actual linear sampler against a source-space ownership mask.

    Source and output landmarks use pixel centers. A one-source-pixel band is
    excluded from the strict cutoff and contamination counts.
    """
    height, width = expected_mask.shape
    output_height, output_width = output_shape
    matrix = np.asarray(page_to_output, dtype=np.float64)
    expected = np.asarray(expected_mask != 0, dtype=np.uint8)
    if (
        not expected.any()
        or not np.isfinite(matrix).all()
        or matrix.shape != (3, 3)
        or min(output_height, output_width) < 2
        or np.linalg.matrix_rank(matrix) != 3
    ):
        raise ValueError("Invalid crop measurement geometry")
    footprint, sampled_corners = _source_center_footprint((height, width), matrix, output_shape)
    footprint_area = int(footprint.sum())
    intersection = int((footprint & expected).sum())
    expected_area = int(expected.sum())
    kernel = np.ones((3, 3), dtype=np.uint8)
    expected_core = cv2.erode(expected, kernel, borderType=cv2.BORDER_REPLICATE)
    outside_band = 1 - cv2.dilate(expected, kernel)
    from app.pipeline.crop_sampling import OWNED_BILINEAR, warp_owned_crop

    def replay(source):
        size = (output_width, output_height)
        if sampling == OWNED_BILINEAR:
            return warp_owned_crop(source.astype(np.float32), matrix, size)[0]
        return cv2.warpPerspective(source.astype(np.float32), matrix, size)

    strict_contaminated = replay(outside_band) > 1e-6
    source_support = replay(np.ones((height, width), dtype=np.uint8))
    # Fractional contributions are scored against independent pixel ownership.
    ownership = replay(expected)
    spans = np.linalg.norm(np.roll(sampled_corners, -1, axis=0) - sampled_corners, axis=1)
    vertices = np.asarray(source_quad, dtype=np.float64) * [width - 1, height - 1]
    report = {
        "footprint_iou": intersection / (footprint_area + expected_area - intersection),
        "content_cutoff": (expected_area - intersection) / expected_area,
        "foreign_fraction": float(np.mean(1.0 - ownership)),
        "foreign_source_samples": int((ownership < 1 - 1e-6).sum()),
        "outside_source_samples": int((source_support < 1 - 1e-6).sum()),
        "cutoff_beyond_1px": int((expected_core & ~footprint).sum()),
        "foreign_beyond_1px": int(strict_contaminated.sum()),
        "sampling_shortfall_px": max(
            0.0, float(spans[::2].min()) - (output_width - 1), float(spans[1::2].min()) - (output_height - 1),
        ),
        "warp_metadata_error_max_px": float(_aligned_corner_errors(sampled_corners, vertices).max()),
    }
    if expected_corners is not None:
        errors = _aligned_corner_errors(expected_corners, sampled_corners)
        report["corner_errors_px"] = errors.tolist()
        report["corner_error_max_px"] = float(errors.max())
        ordered = np.asarray(expected_corners, dtype=np.float64).reshape(1, 4, 2)
        mapped = cv2.perspectiveTransform(ordered, matrix)[0]
        output_corners = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1],
                                   [0, output_height - 1]])
        nearest = np.argmin(np.linalg.norm(mapped[:, None] - output_corners, axis=2), axis=1)
        report["orientation_matches_reference"] = bool(np.array_equal(nearest, np.arange(4)))
    if expected_aspect_ratio is None and expected_shape is not None:
        expected_aspect_ratio = (expected_shape[1] - 1) / (expected_shape[0] - 1)
    if expected_aspect_ratio is not None:
        if not np.isfinite(expected_aspect_ratio) or expected_aspect_ratio <= 0:
            raise ValueError("Invalid reference aspect ratio")
        actual_aspect = (output_width - 1) / (output_height - 1)
        report["reference_aspect_ratio"] = expected_aspect_ratio
        report["aspect_ratio_relative_error"] = abs(actual_aspect / expected_aspect_ratio - 1)
    if source_landmarks is not None and expected_landmarks is not None and expected_shape is not None:
        expected_height, expected_width = expected_shape
        actual = cv2.perspectiveTransform(source_landmarks.astype(np.float64).reshape(1, -1, 2), matrix)[0]
        actual *= np.array([expected_width - 1, expected_height - 1]) / np.array(
            [output_width - 1, output_height - 1]
        )
        landmark_errors = np.linalg.norm(actual - expected_landmarks, axis=1)
        report["landmark_error_p95_px"] = float(np.percentile(landmark_errors, 95))
    return report


def _hungarian_assignment(cost: list[list[float]]) -> list[int]:
    size = len(cost)
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    parent = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        parent[0] = row
        column = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = parent[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                value = cost[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if value < minimum[candidate]:
                    minimum[candidate] = value
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for index in range(size + 1):
                if used[index]:
                    u[parent[index]] += delta
                    v[index] -= delta
                else:
                    minimum[index] -= delta
            column = next_column
            if parent[column] == 0:
                break
        while column:
            previous = way[column]
            parent[column] = parent[previous]
            column = previous
    assignment = [-1] * size
    for column in range(1, size + 1):
        if parent[column]:
            assignment[parent[column] - 1] = column - 1
    return assignment


def evaluate_page_crops(
    truth: Sequence[Polygon],
    predictions: Sequence[Polygon],
    *,
    width: int,
    height: int,
    iou_threshold: float = 0.85,
    accepted: Sequence[bool] | None = None,
    truth_masks: Sequence[np.ndarray | None] | None = None,
) -> PageCroppingMetrics:
    """Match each instance once, using its native truth mask when provided."""
    size = max(len(truth), len(predictions))
    accepted = list(accepted) if accepted is not None else [True] * len(predictions)
    native = [None] * len(truth) if truth_masks is None else [
        None if mask is None else np.asarray(mask != 0, dtype=np.uint8) for mask in truth_masks
    ]
    if len(native) != len(truth) or any(
        mask is not None and (mask.shape != (height, width) or not mask.any()) for mask in native
    ):
        raise ValueError("Native instance masks must match the page size and follow the truth instance order")
    truth_bounds = [_pixel_bounds(polygon, width, height, truth_mask=mask)
                    for polygon, mask in zip(truth, native, strict=True)]
    prediction_bounds = [_pixel_bounds(polygon, width, height) for polygon in predictions]
    overlaps: list[list[tuple[float, float, float, float]]] = [
        [
            _mask_overlap(expected, detected, width, height, truth_mask=mask)
            if _bounds_intersect(expected_bounds, detected_bounds) else (0.0, 0.0, 1.0, 1.0)
            for detected, detected_bounds in zip(predictions, prediction_bounds)
        ]
        for expected, expected_bounds, mask in zip(truth, truth_bounds, native, strict=True)
    ]
    cost = [[1.0] * size for _ in range(size)]
    for truth_index, row in enumerate(overlaps):
        for prediction_index, values in enumerate(row):
            iou = values[0]
            # Prefer the maximum number of valid matches before comparing their IoU.
            cost[truth_index][prediction_index] = 1.0 - iou - (size + 1 if iou >= iou_threshold else 0)
    assignment = _hungarian_assignment(cost) if size else []

    match_records: list[dict] = []
    valid_truth: set[int] = set()
    valid_predictions: set[int] = set()
    assigned_predictions: set[int] = set()
    for truth_index, prediction_index in enumerate(assignment[: len(truth)]):
        if prediction_index < 0 or prediction_index >= len(predictions):
            continue
        assigned_predictions.add(prediction_index)
        polygon_iou, mask_iou, cutoff, background = overlaps[truth_index][prediction_index]
        mask = native[truth_index]
        boundary = _boundary_f_score(truth[truth_index], predictions[prediction_index], width, height, truth_mask=mask)
        corner = _corner_error(truth[truth_index], predictions[prediction_index], width, height) if mask is None else None
        is_valid = polygon_iou >= iou_threshold
        if is_valid:
            valid_truth.add(truth_index)
            valid_predictions.add(prediction_index)
        serious = polygon_iou < 0.5 or cutoff > 0.08 or background > 0.20
        match_records.append(
            {
                "truth_index": truth_index,
                "prediction_index": prediction_index,
                "valid_detection": is_valid,
                "polygon_iou": polygon_iou,
                "mask_iou": mask_iou,
                "boundary_f_score": boundary,
                "corner_error": corner,
                "content_cutoff": cutoff,
                "background_retained": background,
                "serious_failure": serious,
            }
        )

    eligible_predictions = {
        prediction_index
        for prediction_index in range(len(predictions))
        if any(row[prediction_index][0] >= iou_threshold for row in overlaps)
    }
    true_positive = len(valid_truth)
    false_negative = len(truth) - true_positive
    false_positive = len(predictions) - true_positive
    duplicates = len(eligible_predictions - valid_predictions)
    # A poor assigned pair is one physical failure, not simultaneously a
    # missing truth, a false positive, and a semantic crop failure.
    unmatched_serious = (len(truth) - len(match_records)) + len(set(range(len(predictions))) - assigned_predictions)
    serious_failures = unmatched_serious + sum(bool(item["serious_failure"]) for item in match_records)

    def mean(name: str) -> float:
        values = [float(item[name]) for item in match_records]
        return float(np.mean(values)) if values else 0.0

    corner_values = [float(item["corner_error"]) for item in match_records if item["corner_error"] is not None]
    return PageCroppingMetrics(
        expected=len(truth),
        detected=len(predictions),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        duplicates=duplicates,
        polygon_iou=mean("polygon_iou"),
        mask_iou=mean("mask_iou"),
        boundary_f_score=mean("boundary_f_score"),
        corner_error=float(np.mean(corner_values)) if corner_values else None,
        content_cutoff=mean("content_cutoff"),
        background_retained=mean("background_retained"),
        serious_failures=serious_failures,
        automatic_acceptances=sum(bool(value) for value in accepted),
        review_count=len(predictions) - sum(bool(value) for value in accepted),
        matches=tuple(match_records),
    )


def evaluate_cropping_dataset(pages: Sequence[PageCroppingMetrics]) -> CroppingDatasetMetrics:
    totals = {
        "true_positive": sum(page.true_positive for page in pages),
        "false_positive": sum(page.false_positive for page in pages),
        "false_negative": sum(page.false_negative for page in pages),
        "duplicates": sum(page.duplicates for page in pages),
        "serious_failures": sum(page.serious_failures for page in pages),
    }
    expected = sum(page.expected for page in pages)
    detected = sum(page.detected for page in pages)
    precision_denominator = totals["true_positive"] + totals["false_positive"]
    recall_denominator = totals["true_positive"] + totals["false_negative"]
    precision = totals["true_positive"] / precision_denominator if precision_denominator else 0.0
    recall = totals["true_positive"] / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_weight = sum(len(page.matches) for page in pages)

    def weighted(name: str) -> float:
        if not matched_weight:
            return 0.0
        return sum(float(getattr(page, name)) * len(page.matches) for page in pages) / matched_weight

    corner_weight = sum(len([item for item in page.matches if item["corner_error"] is not None]) for page in pages)
    corner_total = sum(
        sum(float(item["corner_error"]) for item in page.matches if item["corner_error"] is not None) for page in pages
    )
    accepted = sum(page.automatic_acceptances for page in pages)
    reviews = sum(page.review_count for page in pages)
    perfect = sum(
        page.false_positive == 0 and page.false_negative == 0 and page.serious_failures == 0 for page in pages
    )
    return CroppingDatasetMetrics(
        page_count=len(pages),
        expected_photos=expected,
        detected_photos=detected,
        detection_precision=precision,
        detection_recall=recall,
        detection_f1=f1,
        polygon_iou=weighted("polygon_iou"),
        mask_iou=weighted("mask_iou"),
        boundary_f_score=weighted("boundary_f_score"),
        corner_localization_error=corner_total / corner_weight if corner_weight else None,
        missing_photo_rate=totals["false_negative"] / expected if expected else 0.0,
        duplicate_detection_rate=totals["duplicates"] / detected if detected else 0.0,
        false_positive_rate=totals["false_positive"] / detected if detected else 0.0,
        content_cutoff=weighted("content_cutoff"),
        background_retained=weighted("background_retained"),
        automatic_acceptance_rate=accepted / detected if detected else 0.0,
        human_review_rate=reviews / detected if detected else 0.0,
        serious_failure_rate=totals["serious_failures"] / max(expected, 1),
        page_perfect_rate=perfect / len(pages) if pages else 0.0,
        totals=totals,
    )
