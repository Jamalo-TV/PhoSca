"""Mark a supported source interval just outside a manually saved paper edge."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MANUAL_PAPER_EXTENT_SOURCE = "measured_inset_paper_edge"
MANUAL_PAPER_EXTENT_REVISION = 1

_FRACTIONS = np.linspace(.13, .87, 144, dtype=np.float32)
_OFFSETS = np.arange(-7, 7.25, .25, dtype=np.float32)
_INSPECTION_DEPTH = -.5
_INSPECTION_SAMPLE = int(np.searchsorted(_OFFSETS, _INSPECTION_DEPTH))


@dataclass(frozen=True)
class ManualPaperExtent:
    trace: np.ndarray
    evidence_scope: np.ndarray


def _fit_transition(gold: np.ndarray, stations: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray] | None:
    outside = np.median(sliding_window_view(gold[:, :-8], 8, axis=1), axis=2)
    inside = np.median(sliding_window_view(gold[:, 8:], 8, axis=1), axis=2)
    before, after = gold[:, 7:-8], gold[:, 8:-7]
    transition = ((outside >= threshold + 2) & (inside <= threshold - 2)
                  & (before >= threshold) & (after < threshold))
    unique = transition.sum(axis=1) == 1
    if unique.mean() < .75:
        return None
    col = transition.argmax(axis=1)[unique]
    left, right = before[unique, col], after[unique, col]
    depths = _OFFSETS[col + 7] + .25 * (left - threshold) / (left - right)
    points = np.float32(np.column_stack((stations[unique], depths)))
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_HUBER, 0, .001, .001).ravel()
    if abs(vx) < .99:
        return None
    line = y + (stations - x) * vy / vx
    residual = np.abs(line[unique] - depths)
    if np.quantile(residual, .90) > 1:
        return None
    unique[unique] = residual <= .85
    return line, unique


def _sample_gold(gold: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    cols = np.rint(4 * (offsets - _OFFSETS[0])).astype(int)
    return gold[np.arange(len(gold)), np.clip(cols, 0, len(_OFFSETS) - 1)]


def _side_interval(lab: np.ndarray, start: np.ndarray, end: np.ndarray) -> ManualPaperExtent | None:
    edge = end - start
    length = np.linalg.norm(edge)
    stations = _FRACTIONS * length
    inward = np.float32([-edge[1], edge[0]]) / length
    centers = start + _FRACTIONS[:, None] * edge
    rays = centers[:, None] + _OFFSETS[None, :, None] * inward
    if np.any(rays < 0) or np.any(rays > np.float32(lab.shape[1::-1]) - 1):
        return None
    bands = cv2.remap(lab, rays[..., 0].copy(), rays[..., 1].copy(), cv2.INTER_LINEAR)
    gold = bands[..., 2] - bands[..., 1]
    fits = [_fit_transition(gold, stations, level) for level in (4, 6, 8)]
    if any(fit is None for fit in fits):
        return None
    lines, hits = zip(*fits, strict=True)
    interior = np.max(lines, axis=0) + .35
    exterior = np.min(lines, axis=0) - 2.5
    if exterior.min() < _OFFSETS[0] or interior.min() < -4.0:
        return None
    photo_side = _sample_gold(gold, interior / 2)
    backing_side = _sample_gold(gold, exterior)
    observed = (np.logical_and.reduce(hits) & (interior <= -.7) & (np.ptp(lines, axis=0) <= 1.4)
                & (photo_side <= 5) & (backing_side >= 8) & (backing_side - photo_side >= 6))
    chroma = bands[..., 1:]
    backing = np.median(chroma[:, _OFFSETS <= -4.5], axis=1)
    paper = np.median(chroma[:, (_OFFSETS >= 3.5) & (_OFFSETS <= 6.5)], axis=1)
    direction = paper - backing
    contrast = np.sum(direction * direction, axis=1)
    sample = chroma[:, _INSPECTION_SAMPLE]
    mix = ((sample - backing) * direction).sum(axis=1) / np.maximum(contrast, 1)
    corroborated = observed & (contrast >= 64) & (backing[:, 1] - paper[:, 1] >= 4) & (mix >= .50) & (mix <= 1.55)
    sections = np.array_split(np.arange(len(stations)), 8)
    eligible = np.flatnonzero([observed[part].mean() >= .85 and corroborated[part].mean() >= .65 for part in sections])
    if eligible.size == 0 or np.any(np.diff(eligible) != 1):
        return None
    first, last = sections[eligible[0]][0], sections[eligible[-1]][-1]
    # Mark the sampled band for inspection without extrapolating an outer paper edge.
    scope_ends = centers[[0, -1]]
    return ManualPaperExtent(
        centers[[first, last]] + _INSPECTION_DEPTH * inward,
        np.float32([scope_ends[0] - 7 * inward, scope_ends[1] - 7 * inward,
                    scope_ends[1] + 7 * inward, scope_ends[0] + 7 * inward]),
    )


def measure_manual_paper_extent(
    page: np.ndarray, quad: np.ndarray, blockers: list[np.ndarray],
) -> ManualPaperExtent | None:
    """Compare one adequately supported interval with the saved geometry."""
    if np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() < 150:
        return None
    origin = np.maximum(np.floor(quad.min(axis=0)).astype(int) - 10, 0)
    end = np.minimum(np.ceil(quad.max(axis=0)).astype(int) + 11, page.shape[1::-1])
    if (end - origin).prod() > 1_200_000:
        return None
    left, top = origin
    right, bottom = end
    lab = cv2.cvtColor(page[top:bottom, left:right], cv2.COLOR_BGR2LAB).astype(np.float32)
    local = np.float32(quad - origin)
    if cv2.contourArea(local, oriented=True) < 0:
        local = local[::-1]
    candidates = []
    hulls = [cv2.convexHull(blocker) for blocker in blockers]
    for start, stop in zip(local, np.roll(local, -1, axis=0), strict=True):
        measured = _side_interval(lab, start, stop)
        if measured is None:
            continue
        scope = measured.evidence_scope + np.float32(origin)
        if any(cv2.intersectConvexConvex(scope, peer)[0] > 1 for peer in hulls):
            continue
        candidates.append(ManualPaperExtent(measured.trace + np.float32(origin), scope))
    return candidates[0] if len(candidates) == 1 else None
