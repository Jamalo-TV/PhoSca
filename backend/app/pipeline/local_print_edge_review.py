"""Compare a short supported extension on an otherwise uncertain print."""

import cv2
import numpy as np

from app.pipeline.bowed_edge_review import MeasuredPrintPreview, _displaced_edge
from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _sample_lab


def _dark_detail_continues(
    image: np.ndarray, stations: np.ndarray, outward: np.ndarray, phase: np.ndarray, supported: np.ndarray,
) -> bool:
    offsets = np.float32([-6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2])
    points = stations[:, None] + (phase[:, None, None] + offsets[None, :, None]) * outward
    lab = _sample_lab(image, points.reshape(-1, 2).astype(np.float32)).reshape(len(stations), len(offsets), 3)
    inner = np.median(lab[:, :5], axis=1)
    near = np.median(lab[:, 5:], axis=1)
    if np.median(np.linalg.norm(inner[supported] - near[supported], axis=1)) > 10:
        return False
    strips = np.column_stack((inner[:, 0], near[:, 0]))
    detail = (strips - cv2.GaussianBlur(strips.T, (0, 1), 1.8).T)[supported]
    if np.min(np.median(np.abs(detail), axis=0)) < .65:
        return False
    first, second = detail.T
    return bool(first @ second >= .58 * np.linalg.norm(first) * np.linalg.norm(second))


def _continuous_dark_depth(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, fractions: np.ndarray, allowed: np.ndarray,
) -> np.ndarray | None:
    depths = np.arange(-.5, 15.5, .5, dtype=np.float32)
    band = _edge_band(image, start, end, (-depths).tolist(), fractions=fractions)
    if band is None:
        return None
    light = band[..., 0]
    interruption = light > 104
    first = depths[interruption.argmax(axis=1)]
    return np.minimum(allowed, np.where(interruption.any(axis=1), np.maximum(0, first - 1), allowed))


def suggest_local_print_edge_preview(
    image: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray],
) -> MeasuredPrintPreview | None:
    sides = [evidence.get(name) or {} for name in _PRINT_SIDES]
    if sum(side.get("support", 0) >= .94 for side in sides) < 2:
        return None
    results = []
    for index, side in enumerate(sides):
        if (side.get("ambiguous") or not .58 <= side.get("support", 0) <= .82
                or (side.get("outward_search") or {}).get("print_rim_supported")):
            continue
        initial = side.get("start_support") or 0
        final = side.get("end_support") or 0
        weak_start = initial < final
        near_corner = sides[index - 1] if weak_start else sides[(index + 1) % 4]
        if min(initial, final) > .20 or max(initial, final) < .90 or not near_corner.get("ambiguous"):
            continue
        start, end = quad[[index, (index + 1) % 4]]
        tangent = end - start
        length = float(np.linalg.norm(tangent))
        if length < 580:
            continue
        outward = np.float32([tangent[1], -tangent[0]]) / length
        fractions = np.linspace(.026, .974, int(np.clip(round(length / 3.7), 180, 358)), dtype=np.float32)
        distance = fractions if weak_start else 1 - fractions
        limits = np.float32([.015, .54] if weak_start else [.46, .985])
        first, last = start + limits[:, None] * tangent
        band = np.float32([first - 9 * outward, last - 9 * outward, last + 27 * outward, first + 27 * outward])
        if any(cv2.intersectConvexConvex(band, cv2.convexHull(peer))[0] > 1 for peer in neighbors):
            continue

        observed = _displaced_edge(image, start, end, fractions, (-4, 14))
        if observed is None:
            continue
        fit, measured, supported = observed
        fitted = fit(fractions)
        weak = supported & (distance >= .033) & (distance <= .46)
        anchored = supported & (distance >= .64) & (distance <= .94)
        if (weak.mean() < .40 or (measured[weak] >= 2).mean() < .65
                or anchored.mean() < .27 or np.quantile(np.abs(measured[anchored]), .90) > 2):
            continue
        stations = start + fractions[:, None] * tangent
        texture = weak & (distance >= .044) & (distance <= .32) & (measured > 2.5)
        if texture.mean() < .24 or not _dark_detail_continues(image, stations, outward, fitted, texture):
            continue
        depth = np.where(weak, np.maximum(0, np.minimum(measured, fitted) - 1.25), 0)
        depth = _continuous_dark_depth(image, start, end, fractions, depth)
        if depth is None:
            continue
        depth = np.minimum.reduce([depth, np.r_[0., depth[:-1]], np.r_[depth[1:], 0.]])
        if depth.sum() * length * (fractions[1] - fractions[0]) < 500:
            continue
        expanded = cv2.approxPolyDP(np.float32(stations + depth[:, None] * outward), .15, False)[:, 0]
        outline = np.vstack([quad[:index + 1], expanded, quad[index + 1:]])
        positions = np.flatnonzero(weak & (depth > 0))
        paths = np.split(positions, np.flatnonzero((np.diff(positions) != 1)
                                                 | (np.abs(np.diff(measured[positions])) > 2)) + 1)
        traces = [cv2.approxPolyDP(np.float32(stations[path] + measured[path, None] * outward), .35, False)[:, 0]
                  for path in paths if len(path) >= 6]
        if traces and sum(np.linalg.norm(trace[-1] - trace[0]) for trace in traces) >= .18 * length:
            results.append(MeasuredPrintPreview(index, traces, outline, band))
    return results[0] if len(results) == 1 else None
