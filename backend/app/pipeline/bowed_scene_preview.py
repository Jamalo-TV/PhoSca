"""Measure a visible image transition where a neighbouring rail confuses a straight fit."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _sample_lab
from app.pipeline.visible_scene_edge import has_enclosing_perimeter


BOWED_SCENE_PREVIEW = "bowed_scene_preview"


@dataclass(frozen=True)
class BowedScene:
    outline: np.ndarray
    trace: np.ndarray
    scope: np.ndarray
    outside_trace: np.ndarray | None = None


@dataclass(frozen=True)
class _OutsideBand:
    first: int
    last: int
    boundary: np.ndarray


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


def _outside_band(
    image: np.ndarray, trace: np.ndarray, stations: np.ndarray, inward: np.ndarray, protected: list[np.ndarray],
    corner_clear: np.ndarray,
) -> _OutsideBand | None:
    """Find one supported outboard band while keeping the entire pale stripe."""
    offsets = np.arange(-3., 49.5, .5, dtype=np.float32)
    points = trace[:, None] - offsets[None, :, None] * inward
    if np.any((points < 0) | (points > np.float32(image.shape[1::-1]) - 1)):
        return None
    band = _sample_lab(image, points.reshape(-1, 2)).reshape(*points.shape[:2], 3)
    exterior = np.median(band[:, 80:96], axis=1)
    columns = np.arange(14, 37)
    pale = np.median(band[:, columns[:, None] - np.arange(3, 7)], axis=2)
    gray = np.median(band[:, columns[:, None] + np.arange(3, 7)], axis=2)
    drop = pale[..., 0] - gray[..., 0]
    local = band[:, columns - 2, 0] - band[:, columns + 2, 0]
    collar = band[:, columns[:, None] + np.arange(7, 62)]
    deviation = np.quantile(np.linalg.norm(collar - gray[:, :, None], axis=3), .90, axis=2)
    drift = np.linalg.norm(gray - exterior[:, None], axis=2)
    change = np.quantile(np.linalg.norm(np.diff(collar, axis=2), axis=3), .90, axis=2)
    valid = ((pale[..., 0] >= 178) & (gray[..., 0] <= 178) & (drop >= 23) & (local >= 16)
             & (deviation <= 17) & (drift <= 10) & (change <= 5))
    score = drop + local - deviation - 2.1 * drift
    peaks = np.where(valid, score, -np.inf).argmax(axis=1)
    rail = offsets[columns[peaks]]
    amount = (trace - stations) @ inward - rail - 2.4
    supported = valid.any(axis=1) & (amount >= 1.25) & corner_clear
    intervals = np.flatnonzero(np.diff(np.r_[False, supported, False])).reshape(-1, 2)
    if len(intervals) == 0:
        return None
    first, stop = max(intervals, key=lambda run: run[1] - run[0])
    last = stop - 1
    if np.linalg.norm(stations[last] - stations[first]) < 100 or np.max(np.abs(np.diff(rail[first:stop]))) > 2:
        return None
    boundary = np.float32(stations[first:stop] + amount[first:stop, None] * inward)
    boundary = cv2.approxPolyDP(boundary, .22, False)[:, 0]
    wedge = np.vstack((stations[first], boundary, stations[last]))
    hull = cv2.convexHull(wedge)
    if any(cv2.intersectConvexConvex(hull, region)[0] > 0 for region in protected):
        return None
    if not _matches_removed_band(image, wedge, stations[first:stop], gray[np.arange(first, stop), peaks[first:stop]]):
        return None
    return _OutsideBand(int(first), int(last), boundary)


def _matches_removed_band(
    image: np.ndarray, wedge: np.ndarray, stations: np.ndarray, gray: np.ndarray,
) -> bool:
    """Check all candidate source centres against the observed exterior."""
    left, top = np.floor(wedge.min(axis=0)).astype(int)
    right, bottom = np.ceil(wedge.max(axis=0)).astype(int) + 1
    mask = _outline_mask(wedge - np.float32([left, top]), (bottom - top, right - left))
    y, x = np.nonzero(mask)
    if len(y) < 180:
        return False
    centres = np.float32(np.column_stack((x + left, y + top)))
    axis = stations[-1] - stations[0]
    fractions = ((centres - stations[0]) @ axis) / (axis @ axis)
    depth = ((stations - stations[0]) @ axis) / (axis @ axis)
    reference = np.column_stack([np.interp(fractions, depth, channel) for channel in gray.T])
    sample_left, sample_top = max(left - 3, 0), max(top - 3, 0)
    lab = cv2.cvtColor(image[sample_top:bottom + 3, sample_left:right + 3], cv2.COLOR_BGR2LAB)
    sample_y, sample_x = y + top - sample_top, x + left - sample_left
    source = lab[sample_y, sample_x].astype(np.float32)
    if np.any(np.linalg.norm(source - reference, axis=1) > 13):
        return False
    # Compare along the band so the nearby bright edge cannot mimic ink.
    # A mark needs similar ground on both sides and connected native evidence.
    tangent = axis / np.linalg.norm(axis)
    flank_points = centres[:, None] + np.float32([-4, -3, 3, 4])[None, :, None] * tangent
    if np.any((flank_points < 0) | (flank_points > np.float32(image.shape[1::-1]) - 1)):
        return False
    flanks = _sample_lab(image, flank_points.reshape(-1, 2)).reshape(-1, 2, 2, 3).mean(axis=2)
    residual = np.min(flanks - source[:, None], axis=1)
    matched = np.linalg.norm(flanks[:, 0] - flanks[:, 1], axis=1) <= 6
    ink = matched & ((residual[:, 0] >= 4) | (residual[:, 2] >= 2))
    if _has_connected_centres(mask.shape, y[ink], x[ink], 4):
        return False

    # A mark may be lighter than the field or show in a different color.
    # Check the 2D neighborhood and discount isolated residuals from grain.
    local = cv2.medianBlur(lab, 7)[sample_y, sample_x]
    squared = (source - local) ** 2
    color = squared[:, 1:].sum(axis=1)
    strong = squared[:, 0] + color >= 64
    chromatic = color >= 9
    return not (_has_connected_centres(mask.shape, y[strong], x[strong], 2)
                or _has_connected_centres(mask.shape, y[chromatic], x[chromatic], 3))


def _has_connected_centres(shape: tuple[int, int], y: np.ndarray, x: np.ndarray, minimum: int) -> bool:
    if len(y) < minimum:
        return False
    marks = np.zeros(shape, dtype=np.uint8)
    marks[y, x] = 1
    _, _, components, _ = cv2.connectedComponentsWithStats(marks, connectivity=8)
    return bool(np.any(components[1:, cv2.CC_STAT_AREA] >= minimum))


def measure_bowed_scene(
    image: np.ndarray, quad: np.ndarray, boundary: dict, neighbors: list[np.ndarray],
    *, protected: list[np.ndarray] | None = None,
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
    corner_clear = np.minimum(fractions, 1 - fractions) * length[index] >= 12
    inside = corner_clear & (additions < -.05)
    measured = np.flatnonzero(inside)
    if len(measured) < 4:
        return None
    first, last = measured[[0, -1]]
    added_edge = cv2.approxPolyDP(extension[first:last + 1], .27, False)[:, 0]
    addition = np.vstack((stations[first], added_edge, stations[last]))
    if cv2.contourArea(addition) < max(64, .001 * cv2.contourArea(quad)):
        return None
    scope = cv2.convexHull(np.vstack((quad, trace, np.float32([start, end]) - inward * radius)))[:, 0]
    if np.any((scope < 0) | (scope > np.float32(image.shape[1::-1]) - 1)):
        return None
    if any(cv2.intersectConvexConvex(scope, cv2.convexHull(peer))[0] > 1 for peer in neighbors):
        return None
    segments = [(first, addition)]
    outside = _outside_band(image, trace, stations, inward, protected or [], corner_clear)
    outside_trace = None
    if outside is not None and (outside.last < first or last < outside.first):
        far = trace[[outside.first, outside.last]] - 49. * inward
        complete_scope = cv2.convexHull(np.vstack((scope, far)))[:, 0]
        if (not any(cv2.intersectConvexConvex(complete_scope, cv2.convexHull(peer))[0] > 1 for peer in neighbors)
                and not has_enclosing_perimeter(image, quad, np.vstack((quad, outside.boundary)), length.min(),
                                                search_margin=.20, maximum_area_ratio=1.86)):
            scope = complete_scope
            clipped = np.vstack((stations[outside.first], outside.boundary, stations[outside.last]))
            segments.append((outside.first, clipped))
            outside_trace = outside.boundary
    outline = np.vstack((start, *(segment for _, segment in sorted(segments)), end, ordered[2:]))
    return BowedScene(outline, trace[first:last + 1], scope, outside_trace)
