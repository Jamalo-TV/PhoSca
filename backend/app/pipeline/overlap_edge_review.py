"""Measure exposed paper segments beside a disputed overlapping claim."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap
from app.pipeline.print_edges import _edge_band, _is_pale_stock, _quad_from_lines, _side_line


OVERLAP_EDGE_REVIEW_REVISION = 1
_SIDES = ("top", "right", "bottom", "left")


def _reach_outside(old: np.ndarray, fitted: np.ndarray, side: int, shortest: float) -> bool:
    endpoints = [side, (side + 1) % 4]
    dx, dy = old[endpoints[1]] - old[endpoints[0]]
    outward = np.float32([dy, -dx]) / np.hypot(dx, dy)
    depth = (fitted[endpoints] - old[endpoints]) @ outward
    return bool(depth.min() >= max(3.5, .018 * shortest) and depth.max() <= min(26., .085 * shortest))


def _exposed_stock_spans(
    image: np.ndarray, start: np.ndarray, stop: np.ndarray, stock: np.ndarray, neighbors: list[np.ndarray],
) -> list[np.ndarray]:
    length = float(np.linalg.norm(stop - start))
    fractions = np.linspace(.035, .965, int(np.clip(round(length), 72, 512)), dtype=np.float32)
    band = _edge_band(image, start, stop, list(range(-9, 7)), fractions=fractions)
    if band is None:
        return []
    exterior = np.median(band[:, [0, 2, 4, 6], 0], axis=1)
    collar = band[:, 11:]
    supported = (np.linalg.norm(collar - stock, axis=2).max(axis=1) <= 10) & (
        np.median(collar[..., 0], axis=1) - exterior >= 18
    )
    steps = np.diff(band[..., 0], axis=1)
    peak = np.argmax(steps, axis=1)
    # The strongest lightness step must lie within 1.5 source pixels of the fitted line.
    supported &= (peak >= 7) & (peak <= 10) & (steps.max(axis=1) >= 5)
    direction = stop - start
    stations = start + fractions[:, None] * direction
    # Leave space for the sampled strip on both sides of the measured line.
    for neighbor in neighbors:
        lower, upper = neighbor.min(axis=0) - 9, neighbor.max(axis=0) + 9
        nearby = np.flatnonzero(supported & ((stations > lower) & (stations < upper)).all(axis=1))
        for index in nearby:
            if cv2.pointPolygonTest(neighbor, tuple(stations[index]), True) > -9:
                supported[index] = False
    if supported.mean() < .48:
        return []
    intervals = np.flatnonzero(np.diff(np.r_[False, supported, False])).reshape(-1, 2)
    inset = direction / length
    return [np.stack((stations[first] + inset, stations[last - 1] - inset)) for first, last in intervals
            if length * (fractions[last - 1] - fractions[first]) >= max(14., .06 * length + 2)]


def suggest_exposed_overlap_edges(
    image: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray],
) -> list[np.ndarray]:
    """Return observed spans only. A shared paper claim can have identical pixels."""
    if not 1 <= len(neighbors) <= 6:
        return []
    sides = [evidence.get(name) or {} for name in _SIDES]
    uncertain = [index for index, side in enumerate(sides) if side.get("ambiguous")]
    if len(uncertain) != 2 or uncertain[1] - uncertain[0] != 2:
        return []
    trusted = [index for index in range(4) if index not in uncertain]
    if any(sides[index].get("support", 0) < .88 or (sides[index].get("corner_support") or 0) < .80 for index in trusted):
        return []
    shortest = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
    if shortest < 155:
        return []
    scale = np.float32(image.shape[1::-1]) - 1
    area = cv2.contourArea(quad)
    candidates = []
    for side in trusted:
        opposite = (side + 2) % 4
        collar = _edge_band(image, quad[opposite], quad[(opposite + 1) % 4], [2, 3, 4, 5])
        if collar is None:
            continue
        stock = np.median(collar, axis=(0, 1))
        if not _is_pale_stock(stock) or np.mean(np.linalg.norm(collar - stock, axis=2) <= 10) < .87:
            continue

        lines = []
        for index in range(4):
            radius = int(np.clip(sides[index].get("search_radius_px") or 8, 6, 20))
            if index == side:
                radius = max(radius, int(np.clip(round(.065 * shortest), 12, 24)))
            line, support = _side_line(image, quad[index], quad[(index + 1) % 4], radius, rim_color=stock)
            if (line is None or support["ambiguous"] or support["support"] < (.76 if index == side else .88)
                    or (index != side and (support["corner_support"] or 0) < .78)):
                break
            lines.append(line)
        if len(lines) != 4:
            continue
        fitted = _quad_from_lines(np.asarray(lines))
        if (fitted is None or not cv2.isContourConvex(fitted) or not _reach_outside(quad, fitted, side, shortest)
                or not 1.014 * area <= cv2.contourArea(fitted) <= 1.17 * area
                or any(cv2.pointPolygonTest(fitted, tuple(point), True) < -1.5 for point in quad)
                or np.any(fitted < 9) or np.any(fitted > scale - 9)):
            continue
        fitted_area = cv2.contourArea(fitted)
        added_claims = []
        for peer in neighbors:
            proposed = normalized_polygon_overlap(fitted / scale, peer / scale, raster_size=1024).first_containment
            previous = normalized_polygon_overlap(quad / scale, peer / scale, raster_size=1024).first_containment
            if proposed * fitted_area - previous * area >= max(10., .002 * area):
                added_claims.append(proposed)
        if len(added_claims) != 1 or not .003 <= added_claims[0] <= .12:
            continue
        spans = _exposed_stock_spans(image, fitted[side], fitted[(side + 1) % 4], stock, neighbors)
        if spans:
            candidates.append(spans)
    return candidates[0] if len(candidates) == 1 else []
