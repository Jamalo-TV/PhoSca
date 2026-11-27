"""Measure a visible image transition where a neighbouring rail confuses a straight fit."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _PRINT_SIDES, _edge_band


BOWED_SCENE_PREVIEW = "bowed_scene_preview"


@dataclass(frozen=True)
class BowedScene:
    outline: np.ndarray
    trace: np.ndarray
    scope: np.ndarray


def _connected_depths(score: np.ndarray) -> np.ndarray:
    # Stations are in source order. A large across-edge jump is more likely
    # to switch rails than to follow a bowed material boundary.
    shifts = np.arange(-5, 6)
    columns = np.arange(score.shape[1])
    previous = np.clip(columns[:, None] + shifts, 0, len(columns) - 1)
    cost = 3 * np.abs(shifts) + .28 * shifts ** 2
    current = score[0]
    parents = []
    for row in score[1:]:
        choices = current[previous] - cost
        best = choices.argmax(axis=1)
        current = row + choices[columns, best]
        parents.append(previous[columns, best])
    indices = [current.argmax()]
    for parent in reversed(parents):
        indices.append(parent[indices[-1]])
    return np.asarray(indices[::-1])


def _signed_step(image: np.ndarray, start: np.ndarray, end: np.ndarray, radius: int) -> np.ndarray | None:
    length = float(np.linalg.norm(end - start))
    fractions = np.linspace(0, 1, int(np.clip(round(length / 4.4), 118, 260)), dtype=np.float32)
    offsets = np.arange(-radius, radius + .25, .5, dtype=np.float32)
    band = _edge_band(image, start, end, offsets, fractions=fractions)
    if band is None:
        return None
    levels = band[..., 0]
    columns = np.arange(24, len(offsets) - 14)
    near = np.median(levels[:, columns[:, None] - np.arange(4, 9)], axis=2)
    inside = np.median(levels[:, columns[:, None] + np.arange(4, 9)], axis=2)
    far = np.median(levels[:, columns[:, None] - np.arange(19, 24)], axis=2)
    local = levels[:, columns - 2] - levels[:, columns + 2]
    contrast = near - inside
    pale = near >= 171
    eligible = pale & (local >= 6) & (contrast >= 14)
    score = (.8 * np.clip(local, 0, 48) + .4 * np.clip(contrast, 0, 48)
             + 1.25 * np.clip(near - 159, 0, 43) + .8 * np.clip(near - far, 0, 35))
    peaks = _connected_depths(np.where(eligible, score, -18))
    rows = np.arange(len(fractions))
    measured = offsets[columns[peaks]]
    supported = eligible[rows, peaks]
    independent_rim = (near[rows, peaks] - far[rows, peaks]) >= 10
    if (supported.mean() < .95 or min(part.mean() for part in np.array_split(supported, 8)) < .80
            or independent_rim.mean() < .72 or np.median(local[rows, peaks]) < 24):
        return None

    # Use only a connected observed span, capped by the first and last
    # measurements. No extrapolation is used at either end of the curve.
    intervals = np.flatnonzero(np.diff(np.r_[False, supported, False])).reshape(-1, 2)
    first, last = max(intervals, key=lambda run: run[1] - run[0])
    if first > .04 * len(rows) or last < .95 * len(rows) or last - first < .87 * len(rows):
        return None
    positions = fractions[first:last]
    depths = measured[first:last]
    if depths.min() >= -5 or depths.max() <= 5:
        return None
    trend = np.polynomial.Polynomial.fit(positions, depths, 3)
    residual = np.abs(depths - trend(positions))
    if np.quantile(residual, .90) > 1.65 or residual.max() > 4.:
        return None
    axis = end - start
    inward = np.float32([-axis[1], axis[0]]) / length
    points = start + positions[:, None] * axis + depths[:, None] * inward
    return np.float32(points)


def measure_bowed_scene(
    image: np.ndarray, quad: np.ndarray, boundary: dict, neighbors: list[np.ndarray],
) -> BowedScene | None:
    sides = [boundary.get(side) or {} for side in _PRINT_SIDES]
    weak = [i for i, side in enumerate(sides) if side.get("corner_support") is None]
    if (len(weak) != 1 or any(side.get("ambiguous") for side in sides)
            or any(side.get("support", 0) < .90 for i, side in enumerate(sides) if i not in weak)):
        return None
    index, = weak
    length = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if length.min() < 420 or sides[index].get("support", 0) >= .50:
        return None
    ordered = np.roll(quad, -index, axis=0)
    start, end = ordered[:2]
    radius = int(np.clip(round(.063 * length.min()), 30, 58))
    trace = _signed_step(image, start, end, radius)
    if trace is None:
        return None
    edge = end - start
    inward = np.float32([-edge[1], edge[0]]) / length[index]
    fractions = np.clip(((trace - start) @ edge) / (edge @ edge), 0, 1)
    depths = (trace - start) @ inward
    # Grow only into the directly observed image side of the transition.
    # A bright rim may still belong to the print, so do not trim the old view.
    additions = np.minimum(depths + 1.35, 0.)
    stations = start + fractions[:, None] * edge
    extension = stations + additions[:, None] * inward
    # Keep an unmodified corner where two different material steps meet.
    inside = (np.minimum(fractions, 1 - fractions) * length[index] >= 12) & (additions < -.05)
    measured = np.flatnonzero(inside)
    if len(measured) < 4:
        return None
    first, last = measured[[0, -1]]
    added_edge = cv2.approxPolyDP(extension[first:last + 1], .27, False)[:, 0]
    outline = np.vstack((start, stations[first], added_edge, stations[last], end, ordered[2:]))
    if cv2.contourArea(outline) - cv2.contourArea(quad) < max(64, .001 * cv2.contourArea(quad)):
        return None
    scope = cv2.convexHull(np.vstack((quad, trace, np.float32([start, end]) - inward * radius)))[:, 0]
    if np.any((scope < 0) | (scope > np.float32(image.shape[1::-1]) - 1)):
        return None
    if any(cv2.intersectConvexConvex(scope, cv2.convexHull(peer))[0] > 1 for peer in neighbors):
        return None
    return BowedScene(outline, trace[first:last + 1], scope)
