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


def _mask_overlap(truth: Polygon, prediction: Polygon, width: int, height: int) -> tuple[float, float, float, float]:
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


def _boundary_f_score(
    truth: Polygon, prediction: Polygon, width: int, height: int, tolerance_ratio: float = 0.0025
) -> float:
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


def _corner_error(truth: Polygon, prediction: Polygon) -> float | None:
    truth_quad = _cyclic_quad(truth)
    prediction_quad = _cyclic_quad(prediction)
    if truth_quad is None or prediction_quad is None:
        return None
    candidates: list[float] = []
    for reverse in (False, True):
        candidate = prediction_quad[::-1] if reverse else prediction_quad
        for offset in range(4):
            aligned = np.roll(candidate, offset, axis=0)
            candidates.append(float(np.mean(np.linalg.norm(truth_quad - aligned, axis=1)) / np.sqrt(2.0)))
    return min(candidates)


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
) -> PageCroppingMetrics:
    """Evaluate a page with one-to-one matching and content-safety metrics."""
    size = max(len(truth), len(predictions))
    accepted = list(accepted) if accepted is not None else [True] * len(predictions)
    overlaps: list[list[tuple[float, float, float, float]]] = [
        [_mask_overlap(expected, detected, width, height) for detected in predictions] for expected in truth
    ]
    cost = [[1.0] * size for _ in range(size)]
    for truth_index, row in enumerate(overlaps):
        for prediction_index, values in enumerate(row):
            cost[truth_index][prediction_index] = 1.0 - values[0]
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
        boundary = _boundary_f_score(truth[truth_index], predictions[prediction_index], width, height)
        corner = _corner_error(truth[truth_index], predictions[prediction_index])
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
