"""Find open visible edges inside a decorated album leaf for count review."""

import cv2
import numpy as np

from app.pipeline.contested_scene_edge import _common_spans
from app.pipeline.edge_prints import _photo_texture
from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.near_capture_review import _claimed_contours
from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line


MOUNTED_WINDOW_REVIEW_TRACES = "mounted_window_review_traces"
MOUNTED_LOWER_WINDOW_REVIEW = "mounted_lower_window_review"
MOUNTED_WINDOW_SOURCE_MARGIN = 15.


def _contrast_components(contrast: np.ndarray, threshold: int, close: int, opening: int) -> list[tuple[np.ndarray, np.ndarray]]:
    height, width = contrast.shape
    # Ignore the leaf frame and the lower half covered by the overlapping proposals.
    binary = np.zeros_like(contrast, dtype=np.uint8)
    top, bottom = round(.03 * height), round(.48 * height)
    left, right = round(.03 * width), round(.97 * width)
    binary[top:bottom, left:right] = contrast[top:bottom, left:right] >= threshold
    for operation, size in ((cv2.MORPH_CLOSE, close), (cv2.MORPH_OPEN, opening)):
        binary = cv2.morphologyEx(binary, operation, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, span_x, span_y = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (not (.045 * width * height <= area <= .25 * width * height
                 and .13 * width <= span_x <= .47 * width and .20 * height <= span_y <= .43 * height
                 and area >= .66 * span_x * span_y) or y + span_y >= bottom - 2):
            continue
        polygon = cv2.approxPolyDP(contour, 1.8, True).reshape(-1, 2)
        box = [[x, y], [x + span_x, y], [x + span_x, y + span_y], [x, y + span_y]]
        candidates.append((np.float32(polygon), np.float32(box)))
    return candidates


def _station_spans(page: np.ndarray, a: np.ndarray, b: np.ndarray, radius: int) -> list[np.ndarray]:
    line, evidence = _side_line(page, a, b, radius)
    if line is None or evidence.get("ambiguous") or evidence.get("support", 0) < .85:
        return []
    start, end = np.float32(_line_endpoints(a, b, line))
    stations = np.linspace(.10, .90, 96, dtype=np.float32)
    band = _edge_band(page, start, end, [-14, -11, -8, -5, -2, 2, 5, 8, 11, 14], fractions=stations)
    if band is None:
        return []
    outside = np.median(band[:, :4], axis=1)[:, 0]
    inside = np.median(band[:, 6:], axis=1)[:, 0]
    immediate = np.linalg.norm(band[:, 4] - band[:, 5], axis=1)
    signed = (outside >= 107) & (outside - inside >= 13) & (immediate >= 6)
    if signed.mean() < .90 or min(chunk.mean() for chunk in np.array_split(signed, 4)) < .80:
        return []
    groups = np.flatnonzero(signed)
    spans = np.split(groups, np.flatnonzero(np.diff(groups) > 1) + 1)
    return [start + stations[group[[0, -1]], None] * (end - start) for group in spans if len(group) >= 6]


def _clear_pieces(trace: np.ndarray, claims: list[np.ndarray]) -> list[np.ndarray]:
    start, end = trace
    stations = np.linspace(start, end, int(np.linalg.norm(end - start)) + 1, dtype=np.float32)
    clear = np.ones(len(stations), dtype=bool)
    for contour in claims:
        clear &= [cv2.pointPolygonTest(contour, tuple(point), True) < -MOUNTED_WINDOW_SOURCE_MARGIN for point in stations]
    hits = np.flatnonzero(clear)
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)
    return [stations[group[[0, -1]]] for group in groups if len(group) >= 21]


def _lower_window_traces(
    page: np.ndarray, sample: np.ndarray, transform: np.ndarray, inverse: np.ndarray,
    upper: tuple[np.ndarray, np.ndarray], occupied: list[np.ndarray], claims: list[np.ndarray],
    fit_radius: int, outside_side: int,
) -> dict[str, list[np.ndarray]]:
    if len(occupied) != 2:
        return {}
    height, width = sample.shape[:2]
    # The lowest broad hypothesis sets the search depth, not an opening edge.
    depths = [cv2.perspectiveTransform(contour, transform)[:, 0, 1] for contour in occupied]
    lower = max(depths, key=np.min)
    top, bottom = float(lower.min()), float(lower.max())
    depth = bottom - top
    upper_bottom = max(box[2, 1] for box in upper)
    if not (.20 * height <= depth <= .48 * height and top - upper_bottom >= .08 * height):
        return {}

    observations = []
    scopes = []
    for box in upper:
        left, right = box[0, 0], box[1, 0]
        span = right - left
        bottom_seed = bottom - .07 * depth
        start_y, end_y = top + .13 * depth, bottom - .14 * depth
        polygon = np.float32([[left, start_y], [right, start_y], [right, bottom_seed], [left, bottom_seed]])
        if (left < .05 * width or right > .95 * width or bottom_seed >= height - fit_radius
                or not _photo_texture(sample, polygon)):
            return {}
        native = cv2.perspectiveTransform(polygon[None], inverse)[0]
        area = cv2.contourArea(native)
        if (sum(cv2.intersectConvexConvex(native, contour)[0] >= .95 * area for contour in occupied) < 2
                or any(cv2.intersectConvexConvex(native, contour)[0] > .01 for contour in claims)):
            return {}
        scopes.append(native)
        seeds = np.float32([[[right, start_y], [right, end_y]],
                            [[right - .09 * span, bottom_seed], [left + .09 * span, bottom_seed]],
                            [[left, end_y], [left, start_y]]])
        observations.append([_station_spans(page, *seed, fit_radius)
                             for seed in cv2.perspectiveTransform(seeds.reshape(1, -1, 2), inverse).reshape(3, 2, 2)])

    traces = []
    first_observation, second_observation = observations
    for side, (first, second) in enumerate(zip(first_observation, second_observation, strict=True), start=1):
        shared = _common_spans(first, second, claims)
        if sum(np.linalg.norm(end - start) for start, end in shared) < 90:
            if side in (outside_side, 2):
                return {}
            continue
        traces.extend(shared)
    # Publish only independent side segments. The top and the paper outline stay unknown.
    return {"traces": traces, "scopes": scopes}


def mounted_window_traces(
    page: np.ndarray, leaf: np.ndarray, outside_side: int, claims: list[dict], overlapping: list[dict],
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    """Measure the outer and lower window sides without assigning a paper outline."""
    quad = order_quad_cyclic(leaf)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    width, height = (round(max(lengths[::2])), round(max(lengths[1::2])))
    if min(width, height) < 400:
        return [], {}
    rectangle = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(quad, rectangle)
    sample = cv2.warpPerspective(page, transform, (width, height))
    light = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    radius = max(21, round(.065 * min(width, height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    background = cv2.GaussianBlur(cv2.dilate(light, kernel), (0, 0), .42 * radius)
    candidates = [_contrast_components(background - light, *settings) for settings in ((12, 5, 9), (16, 9, 5))]
    scale = np.float32([width - 1, height - 1])
    pairs = [(first_box, second_box) for first, first_box in candidates[0] for second, second_box in candidates[1]
             if normalized_polygon_overlap(first / scale, second / scale, raster_size=768).iou >= .90]
    if len(pairs) != 1:
        return [], {}
    first, second = pairs[0]
    if not _photo_texture(sample, first):
        return [], {}
    inverse = np.linalg.inv(transform)
    enclosures = [cv2.perspectiveTransform(box[None], inverse)[0] for box in (first, second)]
    contours = _claimed_contours(claims, *page.shape[1::-1], 0)
    if any(cv2.intersectConvexConvex(enclosures[0], claim)[0] > .01 for claim in contours):
        return [], {}
    occupied = _claimed_contours(overlapping, *page.shape[1::-1], 0)
    traces = []
    fit_radius = min(28, max(10, round(.0285 * min(width, height))))
    for side in (outside_side, 2):
        observations = [_station_spans(page, enclosure[side], enclosure[(side + 1) % 4], fit_radius) for enclosure in enclosures]
        shared = _common_spans(*observations, contours)
        if sum(np.linalg.norm(end - start) for start, end in shared) < 90:
            return [], {}
        traces.extend(piece for trace in shared for piece in _clear_pieces(trace, occupied))
    if sum(np.linalg.norm(end - start) for start, end in traces) < 90:
        return [], {}
    try:
        weak = _claimed_contours([{"polygon": item.get("polygon")} for item in overlapping], *page.shape[1::-1], 0)
        lower = _lower_window_traces(page, sample, transform, inverse, (first, second), weak, contours, fit_radius, outside_side)
    except (ValueError, cv2.error):
        lower = {}
    return traces, lower
