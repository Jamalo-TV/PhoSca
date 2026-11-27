"""Original-pixel evidence for count review of overlapping illustration proposals."""

import json
from dataclasses import dataclass
from hashlib import sha256

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic

PRINTED_ILLUSTRATION_COUNT_REVIEW = "PRINTED_ILLUSTRATION_COUNT_REVIEW"
PRINTED_ILLUSTRATION_REVIEW_FRAME = "printed_illustration_review_frame"
PRINTED_ILLUSTRATION_REVIEW_ID = "printed_illustration_review_id"
PRINTED_ILLUSTRATION_REVIEW_SIZE = 4
PRINTED_ILLUSTRATION_REVIEW_REVISION = 1

_CHART_WIDTH = 600
_CHART_HEIGHT = 700
_BOARD_SAMPLE_RADIUS = 30
_CORNERS = ((False, False), (True, False), (True, True), (False, True))


def valid_illustration_frame(frame: object) -> bool:
    return isinstance(frame, list) and 20 <= len(frame) <= 160 and all(
        isinstance(point, dict)
        and all(type(value := point.get(axis)) in (float, int) and 0 <= value <= 1 for axis in ("x", "y"))
        for point in frame
    )


def illustration_evidence_id(frame: list, polygons: list) -> str:
    encoded = sorted(json.dumps(polygon, sort_keys=True, separators=(",", ":")) for polygon in polygons)
    return sha256(json.dumps([frame, encoded], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RibbonSamples:
    chart_positions: np.ndarray
    native_peaks: np.ndarray
    supported: np.ndarray

    @property
    def coverage(self) -> float:
        return float(self.supported.mean())


def _project(points: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(1, -1, 2), mapping)[0]


def _sample(lab: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return cv2.remap(
        lab, np.ascontiguousarray(coordinates[..., 0], np.float32),
        np.ascontiguousarray(coordinates[..., 1], np.float32), cv2.INTER_LINEAR,
    )


def _warmth(lab: np.ndarray) -> np.ndarray:
    return lab[..., 2] - lab[..., 1]


def _candidate_boards(lab: np.ndarray) -> list[np.ndarray]:
    warmth = cv2.GaussianBlur(_warmth(lab), (0, 0), 7.)
    height, width = lab.shape[:2]
    upper_bound = np.array([width - 1, height - 1]) - _BOARD_SAMPLE_RADIUS
    proposals = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for threshold in (3, 5):
        mask = cv2.morphologyEx(np.uint8(warmth > threshold), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if not .48 <= cv2.contourArea(contour) / (height * width) <= .92:
            continue
        quad = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        if not ((quad >= _BOARD_SAMPLE_RADIUS) & (quad <= upper_bound)).all():
            continue
        quad = order_quad_cyclic(quad.astype(np.float32))
        horizontal = np.mean(np.linalg.norm(quad[[0, 2]] - quad[[1, 3]], axis=1))
        vertical = np.mean(np.linalg.norm(quad[[0, 2]] - quad[[3, 1]], axis=1))
        if min(horizontal, vertical) < 350 or max(horizontal, vertical) / min(horizontal, vertical) > 1.75:
            continue
        if horizontal > vertical:
            quad = np.roll(quad, -1, axis=0)
        if proposals and np.max(np.linalg.norm(proposals[-1] - quad, axis=1)) < 14:
            continue
        proposals.append(quad)
    return proposals


def _board_edges_supported(lab: np.ndarray, quad: np.ndarray) -> bool:
    fractions = np.linspace(.15, .85, 128)
    offsets = np.arange(-_BOARD_SAMPLE_RADIUS, _BOARD_SAMPLE_RADIUS + 1, dtype=np.float32)
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        along = end - start
        inward = np.array([-along[1], along[0]]) / np.linalg.norm(along)
        points = start + fractions[:, None, None] * along + offsets[None, :, None] * inward
        bands = _sample(lab, points)
        median = np.median(np.lib.stride_tricks.sliding_window_view(bands, 7, axis=1), axis=-1)
        warmth = _warmth(median)
        rise = warmth[:, 14:] - warmth[:, :-14]
        light = median[:, 14:, 0] - median[:, :-14, 0]
        support = (rise >= 6) | ((rise >= 2) & (light >= 20))
        octiles = support.reshape(8, -1, support.shape[1]).mean(axis=1)
        best = int(np.argmax(octiles.mean(axis=0) + .5 * octiles.min(axis=0)))
        if octiles[:, best].mean() < .84 or octiles[:, best].min() < .50:
            return False
    return True


def _ribbon_candidates(chart: np.ndarray) -> list[np.ndarray]:
    lightness = chart[..., 0]
    pale = (lightness >= 175) & (_warmth(chart) <= 12)
    blurred = cv2.GaussianBlur(lightness, (0, 0), 1.)
    height, width = lightness.shape
    margin_y, margin_x = int(.075 * height), int(.07 * width)
    results = []
    for filter_size, threshold in ((9, 10), (13, 12), (17, 14)):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        ridge = (blurred - opened >= threshold) & pale
        ridge[:margin_y] = ridge[-margin_y:] = False
        ridge[:, :margin_x] = ridge[:, -margin_x:] = False
        closed = cv2.morphologyEx(np.uint8(ridge), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        contenders = []
        for index, contour in enumerate(contours):
            parent_index = hierarchy[0, index, 3]
            if parent_index < 0:
                continue
            _, _, box_width, box_height = cv2.boundingRect(contour)
            hole_area = cv2.contourArea(contour)
            band_area = cv2.contourArea(contours[parent_index])
            if (.37 <= hole_area / (height * width) <= .72 and min(box_width, box_height) >= 300
                    and .89 <= hole_area / max(1., band_area) <= .985):
                contenders.append((hole_area, index))
        if contenders:
            results.append(contours[max(contenders)[1]])
    return results


def _native_ribbon(lab: np.ndarray, contour: np.ndarray, reverse: np.ndarray) -> RibbonSamples | None:
    closed = np.r_[contour[:, 0], contour[:1, 0]].astype(np.float32)
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    stations = np.linspace(0., distances[-1], max(32, round(distances[-1] / 4.)), endpoint=False)
    line = np.float32(np.column_stack([np.interp(stations, distances, coordinate) for coordinate in closed.T]))
    smoothed = sum(np.roll(line, step, axis=0) for step in range(-2, 3)) / 5.
    tangent = np.roll(smoothed, -2, axis=0) - np.roll(smoothed, 2, axis=0)
    tangent_lengths = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_lengths <= 0):
        return None
    winding = np.sign(cv2.contourArea(contour, oriented=True))
    outward = winding * np.column_stack([tangent[:, 1], -tangent[:, 0]]) / tangent_lengths[:, None]
    offsets = np.arange(-6, 13.1, .5, dtype=np.float32)
    points = smoothed[:, None, :] + offsets[None, :, None] * outward[:, None, :]
    source = _project(points, reverse).reshape(points.shape)
    if not ((source >= 0) & (source <= np.array(lab.shape[1::-1]) - 1)).all():
        return None
    bands = _sample(lab, source)
    peak_choices = np.flatnonzero((offsets >= -.5) & (offsets <= 5.))
    gains, drops = [], []
    for index in peak_choices:
        center = bands[:, index]
        inner = bands[:, index - 10:index - 4].mean(axis=1)
        outer = bands[:, index + 6:index + 12].mean(axis=1)
        gains.append(center[:, 0] - np.maximum(inner[:, 0], outer[:, 0]))
        drops.append(np.minimum(_warmth(inner), _warmth(outer)) - _warmth(center))
    ranks = np.minimum(gains, np.asarray(drops) * 4.)
    columns = np.arange(len(line))
    peak = ranks.argmax(axis=0)
    supported = ranks[peak, columns] >= 8.
    if (supported.mean() < .86 or np.median(np.asarray(gains)[peak, columns]) < 24
            or np.median(np.asarray(drops)[peak, columns]) < 3):
        return None
    # Consecutive supporting stations bound the gaps, including across the join.
    seen = np.flatnonzero(supported)
    largest_gap = np.diff(np.r_[seen, seen[:1] + len(line)]).max() - 1
    if largest_gap * cv2.arcLength(contour, True) / len(line) > 25:
        return None
    return RibbonSamples(smoothed, source[columns, peak_choices[peak]], supported)


def _has_inward_corners(contour: np.ndarray, ribbon: RibbonSamples) -> bool:
    outline = cv2.approxPolyDP(contour, 1., True)
    hull = cv2.convexHull(outline, returnPoints=False)
    defects = cv2.convexityDefects(outline, hull)
    if defects is None:
        return False
    x, y, width, height = cv2.boundingRect(contour)
    short_side = min(width, height)
    scale = np.float32([width - 1, height - 1])
    depths = np.zeros((2, 2))
    for first, second, deepest, fixed_depth in defects.reshape(-1, 4):
        endcaps = outline[[first, second], 0]
        length = np.linalg.norm(endcaps[0] - endcaps[1])
        if not .08 * short_side <= length <= .22 * short_side:
            continue
        u, v = (outline[deepest, 0] - [x, y]) / scale
        if .19 < u < .81 or .19 < v < .81:
            continue
        corner = (int(u > .5), int(v > .5))
        depths[corner] = max(depths[corner], fixed_depth / 256.)
    if not ((depths >= 9) & (depths <= 27)).all():
        return False

    u, v = ((ribbon.chart_positions - [x, y]) / scale).T
    for flip_x, flip_y in _CORNERS:
        across = 1 - u if flip_x else u
        down = 1 - v if flip_y else v
        in_corner = (.012 <= across) & (across < .15) & (.010 <= down) & (down < .14)
        if np.count_nonzero(in_corner) < 8 or ribbon.supported[in_corner].mean() < .75:
            return False
        order = np.argsort(across[in_corner])
        points = ribbon.chart_positions[in_corner][order]
        chord = points[-1] - points[0]
        length = np.linalg.norm(chord)
        if length <= 0:
            return False
        perpendicular = np.array([chord[1], -chord[0]]) / length
        deviations = np.abs((points - points[0]) @ perpendicular)
        if np.quantile(deviations, .75) < 5.2 or (np.diff(down[in_corner][order]) > .002).mean() > .13:
            return False
    return True


def find_illustration_frame(image: np.ndarray, native_polygons: list[np.ndarray]) -> np.ndarray | None:
    """Return a decorative outline for count review, with bounded gaps interpolated.

    A fixed chart suggests contours. Both the exterior and the chart's proposed
    narrow ribbon must be independently supported by the decoded native pixels.
    The resulting line describes visible contrast and cannot establish paper ownership.
    """
    height, width = image.shape[:2]
    if not 400 <= min(height, width) <= max(height, width) <= 2200:
        return None
    polygons = [np.asarray(polygon, np.float32).reshape(-1, 2) for polygon in native_polygons]
    if len(polygons) != PRINTED_ILLUSTRATION_REVIEW_SIZE or any(
        len(polygon) < 3 or not np.isfinite(polygon).all() for polygon in polygons
    ):
        return None
    scale = np.float32([width - 1, height - 1])
    normalized_polygons = [polygon / scale for polygon in polygons]
    chart = np.float32([[0, 0], [_CHART_WIDTH - 1, 0], [_CHART_WIDTH - 1, _CHART_HEIGHT - 1], [0, _CHART_HEIGHT - 1]])
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    frames = []
    for exterior in _candidate_boards(lab):
        if any(normalized_polygon_overlap(exterior / scale, polygon, raster_size=1024).second_containment < .985
               for polygon in normalized_polygons) or not _board_edges_supported(lab, exterior):
            continue
        projection = cv2.getPerspectiveTransform(exterior, chart)
        reverse = np.linalg.inv(projection)
        rectified = cv2.warpPerspective(lab, projection, (_CHART_WIDTH, _CHART_HEIGHT))
        candidates = []
        for contour in _ribbon_candidates(rectified):
            hole = _project(cv2.approxPolyDP(contour, 1.6, True), reverse) / scale
            if any(normalized_polygon_overlap(hole, polygon, raster_size=1024).second_containment < .90
                   for polygon in normalized_polygons):
                continue
            ribbon = _native_ribbon(lab, contour, reverse)
            if ribbon is not None and _has_inward_corners(contour, ribbon):
                candidates.append((hole, ribbon))
        if len(candidates) < 2:
            continue
        best_hole, ribbon = max(candidates, key=lambda candidate: candidate[1].coverage)
        if any(normalized_polygon_overlap(best_hole, hole, raster_size=1024).iou < .985 for hole, _ in candidates):
            continue
        support = ribbon.supported
        points = ribbon.native_peaks
        stations = np.arange(len(points))
        filled = np.column_stack([np.interp(stations, stations[support], coordinate[support], period=len(points))
                                  for coordinate in points.T])
        traced = cv2.approxPolyDP(filled.astype(np.float32), 2., True).reshape(-1, 2)
        if 20 <= len(traced) <= 160 and ((traced >= 0) & (traced <= scale)).all():
            frames.append(traced)
    return frames[0] if len(frames) == 1 else None
