"""Measure an open image transition inside an unsupported detection."""

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _line_endpoints, _sample_lab, _side_line


VISIBLE_SCENE_EDGE = "visible_scene_edge"
_UNIT_QUAD = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def _find_interior_step(
    image: np.ndarray, quad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Nominate a single continuous transition out of a broad, quiet field."""
    stations = np.linspace(.08, .92, 120, dtype=np.float32)
    depths = np.linspace(.055, .48, 256, dtype=np.float32)
    grid = np.stack(np.meshgrid(stations, depths, indexing="ij"), axis=-1)
    matrix = cv2.getPerspectiveTransform(_UNIT_QUAD, quad)
    points = cv2.perspectiveTransform(grid.reshape(1, -1, 2), matrix).reshape(*grid.shape)
    if np.any((points < 0) | (points > np.float32(image.shape[1::-1]) - 1)):
        return None
    lab = _sample_lab(image, points.reshape(-1, 2)).reshape(*grid.shape[:2], 3)
    field = np.median(lab[:, :24], axis=1)
    palette = np.median(field, axis=0)
    if palette[0] < 180:
        return None
    cols = np.arange(50, len(depths) - 12)
    outside = np.median(lab[:, cols[:, None] - [10, 6, 3, 2]], axis=2)
    inside = np.median(lab[:, cols[:, None] + [2, 3, 5, 8]], axis=2)
    differs = np.linalg.norm(lab - field[:, None], axis=2) > 16
    field_exceptions = np.cumsum(differs, axis=1)[:, cols - 8] / (cols - 7)
    valid = ((np.linalg.norm(outside - field[:, None], axis=2) <= 12)
             & (outside[..., 0] - inside[..., 0] >= 22)
             & (lab[:, cols - 1, 0] - lab[:, cols + 1, 0] >= 11)
             & (field_exceptions <= .16))
    supported = valid.any(axis=1)
    if supported.mean() < .87:
        return None
    locations = cols[valid.argmax(axis=1)]
    found = points[np.arange(len(stations)), locations]
    vx, vy, x, y = cv2.fitLine(found[supported], cv2.DIST_HUBER, 0, .01, .01).ravel()
    normal = np.float32([-vy, vx])
    residual = (found - np.float32([x, y])) @ normal
    supported &= np.abs(residual) < .012 * np.linalg.norm(quad[1] - quad[0])
    if (supported.mean() < .87 or min(part.mean() for part in np.array_split(supported, 6)) < .70
            or np.ptp(np.quantile(depths[locations[supported]], [.05, .95])) > .065):
        return None
    first, last = (found - residual[:, None] * normal)[np.flatnonzero(supported)[[0, -1]]]
    trim = .045 * (last - first)
    return np.float32([first + trim, last - trim]), palette


def _native_segments(image: np.ndarray, anchors: np.ndarray, palette: np.ndarray) -> list[np.ndarray]:
    length = np.linalg.norm(anchors[1] - anchors[0])
    radii = sorted({max(4, min(36, round(factor * length))) for factor in (.0065, .0087, .013)})
    fits = []
    for radius in radii:
        line, evidence = _side_line(image, *anchors, radius)
        if line is not None and evidence["support"] >= .85 and not evidence["ambiguous"]:
            fits.append(_line_endpoints(*anchors, line))
    if len(fits) < 2:
        return []
    measured = np.median(np.float32(fits), axis=0)
    if np.max(np.linalg.norm(fits - measured, axis=2)) > max(1., .0018 * length):
        return []
    fractions = np.linspace(.023, .977, 128, dtype=np.float32)
    offsets = np.float32([-39, -27, -16, -5, -2, 2, 5, 11, 21, 33]) * (length / 1680)
    band = _edge_band(image, *measured, offsets, fractions=fractions)
    if band is None:
        return []
    field = band[:, :3]
    stock = np.median(field, axis=1)
    step = np.median(band[:, 3:5], axis=1) - np.median(band[:, 5:7], axis=1)
    support = ((np.linalg.norm(field - stock[:, None], axis=2) < 14).all(axis=1)
               & (np.linalg.norm(stock - palette, axis=1) < 19)
               & (stock[:, 0] - np.median(band[:, 7:, 0], axis=1) >= 22)
               & (np.linalg.norm(step, axis=1) >= 18))
    if support.mean() < .81 or min(part.mean() for part in np.array_split(support, 6)) < .63:
        return []
    stations = measured[0] + fractions[:, None] * (measured[1] - measured[0])
    runs = np.flatnonzero(np.diff(np.r_[False, support, False])).reshape(-1, 2)
    return [stations[[first, last - 1]] for first, last in runs if last - first >= 12]


def _quiet_exterior(image: np.ndarray, spans: list[np.ndarray], inward: np.ndarray, width: float) -> bool:
    start, end = spans[0][0], spans[-1][-1]
    fractions = np.linspace(.05, .95, 340, dtype=np.float32)
    offsets = np.linspace(.08, .72, 204, dtype=np.float32) * width
    stations = start + fractions[:, None] * (end - start)
    points = stations[:, None] - offsets[None, :, None] * inward
    if np.any((points < 0) | (points > np.float32(image.shape[1::-1]) - 1)):
        return False
    levels = _sample_lab(image, points.reshape(-1, 2))[:, 0].reshape(points.shape[:2])
    edges = cv2.Canny(cv2.GaussianBlur(levels.astype(np.uint8), (0, 0), 1.2), 16, 32) > 0
    return max(part.mean() for part in np.array_split(edges, 8)) <= .005


def _enclosing_perimeter(image: np.ndarray, quad: np.ndarray, inside: np.ndarray, short_side: float) -> bool:
    """Withhold an interior guide when the outside has its own closed perimeter."""
    low = np.maximum(0, np.floor(quad.min(axis=0) - .10 * short_side).astype(int))
    high = np.minimum(image.shape[1::-1], np.ceil(quad.max(axis=0) + .10 * short_side).astype(int) + 1)
    left, top = low
    right, bottom = high
    scale = min(1., 1100 / max(right - left, bottom - top))
    patch = cv2.resize(image[top:bottom, left:right], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), .65)
    area = cv2.contourArea(quad)
    radius = max(6, min(24, round(.009 * short_side)))
    for low_gradient, high_gradient, kernel in ((22, 56, 3), (16, 36, 5)):
        edges = cv2.Canny(gray, low_gradient, high_gradient)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            extent = cv2.contourArea(contour) / scale ** 2
            if not .55 * area <= extent <= 1.38 * area:
                continue
            polygon = cv2.approxPolyDP(contour, .013 * cv2.arcLength(contour, True), True)[:, 0]
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            candidate = order_quad_cyclic(np.float32(polygon) / scale + np.float32(low))
            if (any(cv2.pointPolygonTest(candidate, tuple(point), True) < 8 for point in inside)
                    or cv2.intersectConvexConvex(candidate, quad)[0] < .53 * area):
                continue
            facts = [_side_line(image, a, b, radius)[1] for a, b in zip(candidate, np.roll(candidate, -1, axis=0), strict=True)]
            if all(edge["support"] >= .87 and (edge["corner_support"] or 0) >= .64 and not edge["ambiguous"] for edge in facts):
                return True
    return False


def suggest_visible_scene_edge(
    image: np.ndarray, quad: np.ndarray, boundary: dict, neighbors: list[np.ndarray],
) -> list[np.ndarray]:
    """Return only measured spans, without inferring the physical paper sides."""
    short_side = min(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1))
    if short_side < 420 or any(cv2.intersectConvexConvex(quad, cv2.convexHull(peer))[0] > 1 for peer in neighbors):
        return []
    candidates = []
    for side, name in enumerate(_PRINT_SIDES):
        facts = boundary.get(name)
        if not facts or facts["support"] >= .45 or facts["ambiguous"]:
            continue
        ordered = np.roll(quad, -side, axis=0)
        proposal = _find_interior_step(image, ordered)
        if proposal is None:
            continue
        spans = _native_segments(image, *proposal)
        if not spans:
            continue
        a, b = ordered[:2]
        direction = (b - a) / np.linalg.norm(b - a)
        inward = np.float32([-direction[1], direction[0]])
        points = np.vstack(spans)
        if not _quiet_exterior(image, spans, inward, float(np.min((points - a) @ inward))):
            continue
        context = np.vstack((points, points - .13 * short_side * inward, points + .17 * short_side * inward))
        if not _enclosing_perimeter(image, quad, context, short_side):
            candidates.append(spans)
    return candidates[0] if len(candidates) == 1 else []
