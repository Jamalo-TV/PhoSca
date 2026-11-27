"""Compare a saved paper contour with a phase-supported source measurement."""

import cv2
import numpy as np

from app.pipeline.album_backing import _backing_reference, _repeating_profile
from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _edge_band, _line_endpoints, _quad_from_lines, _side_line


MEASURED_RIBBED_PRINT_OUTLINE = "measured_ribbed_print_outline"


def _line(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return np.cross(np.r_[start, 1.], np.r_[end, 1.])


def _stable_rails(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    rails = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        endpoints = []
        for radius in (3, 5, 7):
            line, evidence = _side_line(image, start, end, radius)
            if (line is not None and not evidence["ambiguous"] and evidence["support"] >= .73
                    and (evidence["corner_support"] or 0) >= .52):
                endpoints.append(_line_endpoints(start, end, line))
        if len(endpoints) < 2:
            return None
        positions = np.float32(endpoints)
        consensus = np.median(positions, axis=0)
        if np.linalg.norm(positions - consensus, axis=2).max() > .85:
            return None
        rails.append(_line(*consensus))
    return _quad_from_lines(np.stack(rails))


def _half_transition(image: np.ndarray, start: np.ndarray, end: np.ndarray, palette: np.ndarray) -> np.ndarray | None:
    edge = end - start
    length = np.linalg.norm(edge)
    along = np.linspace(.09, .91, round(.82 * length), dtype=np.float32)
    offsets = np.arange(-3.5, 2.25, .25, dtype=np.float32)
    band = _edge_band(image, start, end, offsets, fractions=along)
    if band is None:
        return None
    outside = np.median(band[:, offsets <= -2.25], axis=1)
    inside = np.median(band[:, offsets >= 1.], axis=1)
    direction = inside - outside
    contrast = np.sum(direction ** 2, axis=1)
    projection = ((band - outside[:, None, :]) * direction[:, None, :]).sum(axis=2) / np.maximum(1., contrast[:, None])
    before, after = projection[:, :-1], projection[:, 1:]
    rising = (before <= .5) & (after > .5) & (offsets[:-1] >= -2.25) & (offsets[:-1] <= .8)
    crossings = offsets[:-1] + .25 * (.5 - before) / np.maximum(after - before, 1.e-5)
    best = np.where(rising, np.abs(crossings + .45), np.inf).argmin(axis=1)
    rows = np.arange(len(along))
    supported = ((contrast >= 7 ** 2) & (np.linalg.norm(outside[:, 1:] - palette, axis=1) <= 10) & rising[rows, best])
    if supported.mean() < .88 or min(block.mean() for block in np.array_split(supported, 8)) < .58:
        return None
    station = along[supported] * length
    depth = crossings[rows[supported], best[supported]]
    vx, vy, x, y = cv2.fitLine(np.column_stack((station, depth)).astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    if abs(vx) < .99 or np.quantile(np.abs(depth - y - (station - x) * vy / vx), .90) > 1.2:
        return None
    inward = np.float32([-edge[1], edge[0]]) / length
    endpoints = np.float32([start, end]) + (y + (np.float32([0., length]) - x) * vy / vx)[:, None] * inward
    return _line(*endpoints)


def _same_exterior_phase(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    fractions = np.linspace(.12, .88, round(.76 * np.linalg.norm(end - start)), dtype=np.float32)
    band = _edge_band(image, start, end, [-9, -8, -7, -2], fractions=fractions)
    if band is None:
        return False
    profiles, repeats = zip(*(_repeating_profile(values) for values in band[..., 0].T), strict=True)
    reference = np.median(profiles[:3], axis=0)
    near = profiles[3]
    return bool(min(np.median(repeats[:3]), repeats[3]) >= .52
                and reference.std() >= 12. and near.std() >= .90 * reference.std()
                and reference @ near >= .72 * np.linalg.norm(reference) * np.linalg.norm(near))


def _enclosing_outline_veto(image: np.ndarray, quad: np.ndarray) -> bool:
    """Withhold an inner comparison when a larger closed perimeter is visible."""
    origin = np.maximum(0, np.floor(quad.min(axis=0)).astype(int) - 64)
    limit = np.minimum(image.shape[1::-1], np.ceil(quad.max(axis=0)).astype(int) + 65)
    if (limit - origin).prod() > 1_200_000:
        return True
    left, top = origin
    right, bottom = limit
    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), .65), 28, 64)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(quad)
    for contour in contours:
        if not 1.12 * area <= cv2.contourArea(contour) <= 2.6 * area:
            continue
        polygon = cv2.approxPolyDP(contour, .014 * cv2.arcLength(contour, True), True)[:, 0]
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        for candidate in (polygon, cv2.boxPoints(cv2.minAreaRect(contour))):
            enclosing = order_quad_cyclic(np.float32(candidate + origin))
            if any(cv2.pointPolygonTest(enclosing, tuple(point), True) < 4 for point in quad):
                continue
            evidence = [_side_line(image, a, b, 6)[1]
                        for a, b in zip(enclosing, np.roll(enclosing, -1, axis=0), strict=True)]
            if all(edge["support"] >= .88 and (edge["corner_support"] or 0) >= .66 and not edge["ambiguous"]
                   for edge in evidence):
                return True
    return False


def measure_ribbed_outline_comparison(
    image: np.ndarray, quad: np.ndarray, neighbors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a comparison only when four observed edges agree with the backing."""
    height, width = image.shape[:2]
    if np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() < 180:
        return None
    hulls = [cv2.convexHull(neighbor) for neighbor in neighbors]
    sampled = []

    def overlaps_known_paper(outline: np.ndarray) -> bool:
        return any(cv2.intersectConvexConvex(cv2.convexHull(outline), other)[0] > 1 for other in hulls)

    def claimed(outline: np.ndarray) -> bool:
        if overlaps_known_paper(outline):
            return True
        sampled.append(outline)
        return False

    stable = _stable_rails(image, quad)
    if stable is None or not cv2.isContourConvex(stable) or np.linalg.norm(stable - quad, axis=1).max() > 6:
        return None
    reference = _backing_reference(image, stable, claimed)
    if reference is None:
        return None
    palette, rib_side = reference
    lines = []
    for start, end in zip(stable, np.roll(stable, -1, axis=0), strict=True):
        line = _half_transition(image, start, end, palette)
        if line is None:
            return None
        lines.append(line)
    measured = _quad_from_lines(np.stack(lines))
    if (measured is None or not cv2.isContourConvex(measured)
            or np.any(np.minimum(measured, np.float32([width - 1, height - 1]) - measured) < 12)
            or not .75 < np.linalg.norm(measured - quad, axis=1).max() <= 6):
        return None
    for i in (rib_side, (rib_side + 2) % 4):
        if not _same_exterior_phase(image, measured[i], measured[(i + 1) % 4]):
            return None
    compared = np.vstack((quad, stable, measured))
    offsets = np.float32([[-12, -12], [12, -12], [12, 12], [-12, 12]])
    collars = (compared[:, None] + offsets).reshape(-1, 2)
    extent = cv2.convexHull(np.vstack([*sampled, collars]))[:, 0]
    if (np.any((extent < 0) | (extent > np.float32([width - 1, height - 1])))
            or overlaps_known_paper(extent) or _enclosing_outline_veto(image, measured)):
        return None
    return measured, extent
