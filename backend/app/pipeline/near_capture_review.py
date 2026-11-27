"""Measure open paper-edge intervals near the original capture boundary."""

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from app.pipeline.album_backing import _backing_reference, _repeating_profile
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, quad_from_polygon
from app.pipeline.print_edges import _line_endpoints, _quad_from_lines, _sample_lab, _side_line


NEAR_CAPTURE_PRINT_TRACES = "near_capture_print_edge_traces"
_CLAIM_MARGIN = 9


def quarter_turn(points: np.ndarray, turns: int) -> np.ndarray:
    x, y = points.T
    return np.column_stack(((x, y, 1 - x, 1 - y)[turns], (y, 1 - x, 1 - y, x)[turns])).astype(np.float32)


def _left_frame_turn(quads: list[np.ndarray], width: int, height: int) -> int | None:
    turns = []
    for turn in range(4):
        scale = np.float32([height - 1, width - 1] if turn % 2 else [width - 1, height - 1])
        sides = [order_quad_cyclic(quarter_turn(quad, turn) * scale)[[0, 3]] for quad in quads]
        top, bottom = np.mean(sides, axis=0)
        if bottom[0] <= 7 and top[0] > 19 and top[0] - bottom[0] > 21:
            turns.append(turn)
    return turns[0] if len(turns) == 1 else None


def near_capture_candidate_turn(polygon: np.ndarray, width: int, height: int) -> int | None:
    scale = np.float32([width - 1, height - 1])
    clearance = np.r_[polygon.min(axis=0) * scale, (1 - polygon.max(axis=0)) * scale]
    turns = np.flatnonzero(clearance <= 2)
    return int(turns[0]) if len(turns) == 1 else None


def _claimed_contours(claims: list[dict], width: int, height: int, turns: int) -> list[np.ndarray]:
    contours = []
    for claim in claims:
        nested = claim.get("source_mask")
        if isinstance(nested, dict):
            contours.extend(_claimed_contours([nested], width, height, turns))
        for key in ("polygon", "visible_polygon", "warp_quad"):
            points = claim.get(key) or []
            if len(points) >= 3:
                normalized = np.float32([(point["x"], point["y"]) for point in points])
                contours.append(cv2.convexHull(normalized_vertices_to_pixels(quarter_turn(normalized, turns), width, height)))
    return contours


@dataclass(frozen=True)
class _Rail:
    line: np.ndarray
    span: np.ndarray
    support: float
    signed: np.ndarray


def _signed_stations(
    image: np.ndarray, span: np.ndarray, palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a, b = span
    direction = b - a
    normal = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
    stations = a + np.linspace(.01, .99, 128, dtype=np.float32)[:, None] * direction
    offsets = np.float32([-9, -6, -4, -2, -1, 1, 2, 4, 6, 9])
    coords = stations[:, None] + offsets[None, :, None] * normal
    usable = ((coords >= .02) & (coords <= np.float32(image.shape[1::-1]) - 1.02)).all(axis=(1, 2))
    lab = _sample_lab(image, coords[usable].reshape(-1, 2)).reshape(-1, len(offsets), 3) if usable.any() else np.empty((0, 10, 3))
    outside, inside = lab[:, 1:4].mean(axis=1), lab[:, 6:9].mean(axis=1)
    out_distance = np.linalg.norm(outside[:, 1:] - palette, axis=1)
    in_distance = np.linalg.norm(inside[:, 1:] - palette, axis=1)
    change = np.linalg.norm(lab[:, 4] - lab[:, 5], axis=1)
    signed = np.zeros(len(stations), dtype=bool)
    signed[usable] = ((out_distance <= 8) & (outside[:, 0] >= 128) & (change >= 4)
                      & ((outside[:, 0] - inside[:, 0] >= 14) | (in_distance - out_distance >= 5)))
    return stations, signed


def _measure_open_rail(image: np.ndarray, seeds: list[np.ndarray], side: int, palette: np.ndarray) -> _Rail | None:
    extents = ((.08, .92), (.08, .92), (0., 1.), (.38, .92))
    options = []
    for quad, radius in product(seeds, (8, 11, 14)):
        a, b = quad[[side, (side + 1) % 4]]
        endpoints = a + np.float32(extents[side])[:, None] * (b - a)
        line, evidence = _side_line(image, *endpoints, radius)
        if line is None:
            continue
        span = np.float32(_line_endpoints(*endpoints, line))
        _, stations = _signed_stations(image, span, palette)
        rail = _Rail(line, span, evidence["support"], stations)
        options.append((.65 * stations.mean() + .35 * rail.support, rail))
    if not options:
        return None
    best_quality, best = max(options, key=lambda pair: pair[0])
    if any(rail.signed.mean() >= .68 and quality >= best_quality - .05
           and np.abs(best.span @ rail.line[:2] + rail.line[2]).mean() > 3
           for quality, rail in options):
        return None
    return best


def _backing_at_frame(image: np.ndarray, bottom: _Rail, palette: np.ndarray) -> bool:
    line = bottom.line
    if abs(line[1]) < .2:
        return False
    stop = round(float(-line[2] / line[1]))
    if not 103 <= stop < image.shape[0] - 1:
        return False
    rows = np.arange(stop - 103, stop + 1, dtype=np.float32)
    lab = _sample_lab(image, np.column_stack((np.zeros_like(rows), rows)))
    for strip in (lab, lab[-31:-1]):
        profile, repetition = _repeating_profile(strip[:, 0])
        if ((np.linalg.norm(strip[:, 1:] - palette, axis=1) <= 8).mean() < .93
                or repetition < .74 or profile.std() < 8):
            return False
    return True


def _trace_intervals(
    image: np.ndarray, rail: _Rail, palette: np.ndarray, contours: list[np.ndarray], *, check_midgap: bool = False,
) -> list[np.ndarray]:
    if rail.support < .75 or rail.signed.mean() < .65 or sum(part.mean() >= .60 for part in np.array_split(rail.signed, 8)) < 5:
        return []
    stations, signed = _signed_stations(image, rail.span, palette)
    if check_midgap:
        middle = np.flatnonzero(~signed[15:-15]) + 15
        gaps = np.split(middle, np.flatnonzero(np.diff(middle) > 1) + 1)
        if any(len(gap) and np.linalg.norm(stations[gap[-1]] - stations[gap[0]]) > 41 for gap in gaps):
            return []
    for contour in contours:
        signed &= [cv2.pointPolygonTest(contour, tuple(point), True) < -_CLAIM_MARGIN for point in stations]
    if signed.mean() < .40:
        return []
    hits = np.flatnonzero(signed)
    return [stations[run[[0, -1]]] for run in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1) if len(run) >= 14]


def near_capture_print_traces(image: np.ndarray, polygons: list[np.ndarray], claims: list[dict]) -> list[np.ndarray]:
    """Return observed source spans only. An extrapolated corner cannot enter a crop."""
    if len(polygons) != 2 or normalized_polygon_overlap(*polygons, raster_size=1024).iou < .90:
        return []
    height, width = image.shape[:2]
    dimensions = np.float32([width - 1, height - 1])
    quads = [quad_from_polygon(polygon * dimensions)[0] / dimensions for polygon in polygons]
    turns = _left_frame_turn(quads, width, height)
    if turns is None:
        return []
    source = np.ascontiguousarray(np.rot90(image, turns))
    height, width = source.shape[:2]
    scale = np.float32([width - 1, height - 1])
    seeds = [order_quad_cyclic(quarter_turn(quad, turns) * scale) for quad in quads]
    contours = _claimed_contours(claims, width, height, turns)

    def claimed(quad):
        return any(cv2.intersectConvexConvex(quad, contour)[0] > 1 for contour in contours)

    mean_quad = np.mean(seeds, axis=0)
    area = cv2.contourArea(mean_quad)
    if area <= 0:
        return []
    overlaps = [cv2.intersectConvexConvex(mean_quad, contour)[0] / area for contour in contours]
    if sum(value > .00005 for value in overlaps) > 1 or max(overlaps, default=0.) > .005:
        return []
    reference = _backing_reference(source, mean_quad, claimed)
    if reference is None:
        reference = next((evidence for seed in seeds if (evidence := _backing_reference(source, seed, claimed)) is not None), None)
    if reference is None:
        return []
    palette, _ = reference
    rails = [_measure_open_rail(source, seeds, side, palette) for side in range(4)]
    if any(rail is None for rail in rails) or not _backing_at_frame(source, rails[2], palette):
        return []
    envelope = _quad_from_lines(np.stack([rail.line for rail in rails]))
    if envelope is None or claimed(cv2.convexHull(np.clip(envelope, 0, scale))):
        return []
    sides = [_trace_intervals(source, rail, palette, contours, check_midgap=side == 2) for side, rail in enumerate(rails)]
    if not all(sides[index] for index in (1, 2, 3)):
        return []
    return [quarter_turn(trace / scale, (-turns) % 4) for side in sides for trace in side]


def near_capture_guide_conflicts(quad: list[dict], traces: list, width: int, height: int) -> bool:
    """Retire an advisory when a saved photo materially occupies its source witness band."""
    if not traces:
        return False
    outline = cv2.convexHull(normalized_vertices_to_pixels([(p["x"], p["y"]) for p in quad], width, height))
    for trace in traces:
        if len(trace) != 2:
            continue
        start, stop = normalized_vertices_to_pixels([(p["x"], p["y"]) for p in trace], width, height)
        direction = stop - start
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        normal = np.float32([-direction[1], direction[0]]) * (_CLAIM_MARGIN / length)
        band = np.float32([start - normal, stop - normal, stop + normal, start + normal])
        if cv2.intersectConvexConvex(outline, band)[0] >= 2 * _CLAIM_MARGIN * min(12., length / 4):
            return True
    return False
