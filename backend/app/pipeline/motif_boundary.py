"""Measure a narrow paper-to-motif transition in the original scan."""

import cv2
import numpy as np


def _trace_side(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, paper_color: list[float] | tuple[int, ...],
) -> tuple[np.ndarray, float, float] | None:
    tangent = end - start
    length = np.linalg.norm(tangent)
    if length < 60:
        return None
    inward = np.float32([-tangent[1], tangent[0]]) / length
    span = np.linspace(.025, .975, 128, dtype=np.float32)
    offsets = np.arange(-8., 8.1, .25, dtype=np.float32)
    points = start + span[:, None, None] * tangent + offsets[None, :, None] * inward
    if np.any((points < 0) | (points > np.array(image.shape[1::-1]) - 1)):
        return None
    low = np.floor(points.min(axis=(0, 1))).astype(int)
    high = np.ceil(points.max(axis=(0, 1))).astype(int) + 1
    strip = image[low[1]:high[1], low[0]:high[0]].astype(np.float32)
    samples = cv2.remap(strip, np.float32(points[..., 0] - low[0]), np.float32(points[..., 1] - low[1]), cv2.INTER_LINEAR)
    levels = cv2.cvtColor(samples, cv2.COLOR_BGR2GRAY)
    paper = np.median(levels[:, :7], axis=1)
    scene = np.median(levels[:, -7:], axis=1)
    contrast = scene - paper
    matches_paper = np.abs(paper - np.dot(paper_color, (.114, .587, .299))) < 20

    progress = (levels - paper[:, None]) / np.where(contrast != 0, contrast, 1)[:, None]
    earlier, later = progress[:, :-1], progress[:, 1:]
    depth = offsets[:-1] + .25 * np.clip((.5 - earlier) / np.maximum(later - earlier, .001), 0, 1)
    cells = np.arange(len(offsets) - 1)
    # The crossing must enter the picture and stay there, including when the
    # first sharp line encountered is a decorative stroke in the paper rim.
    crossings = (earlier < .5) & (later >= .5) & (np.abs(depth) <= 6.5)
    before = progress[:, np.maximum(cells - 3, 0)]
    after = progress[:, np.minimum(cells + 5, len(offsets) - 1)]
    crossings &= (before <= .38) & (after >= .60)
    crossings &= ((np.abs(contrast) >= 10) & matches_paper)[:, None]
    observed = crossings.any(axis=1)
    if observed.mean() < .65:
        return None
    nearest = np.argmin(np.where(crossings, np.abs(depth), np.inf), axis=1)
    # The line fits the first owned pixel center, half a pixel past the optical transition.
    locations = depth[np.arange(len(span)), nearest] + .5
    knots = []
    for group in np.array_split(np.arange(len(span)), 12):
        visible = group[observed[group]]
        if len(visible) >= .60 * len(group):
            knots.append((np.median(span[visible]), np.median(locations[visible])))
    knots = np.asarray(knots)
    if len(knots) < 10 or knots[0, 0] > .13 or knots[-1, 0] < .87:
        return None
    positions, depths = knots.T
    error = np.abs(locations[observed] - np.interp(span[observed], positions, depths))
    if np.percentile(error, 80) > .8:
        return None
    detrended = depths - np.polyval(np.polyfit(positions, depths, 1), positions)
    polyline = start + positions[:, None] * tangent + depths[:, None] * inward
    return np.float32(polyline), float(np.ptp(detrended) / 2), float(np.max(np.abs(depths)))


def trace_motif_boundary(image: np.ndarray, quad: np.ndarray, edges: dict) -> tuple[list[np.ndarray], bool, bool]:
    """Return supported native sides, curvature evidence and disagreement with the chord."""
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return [], False, False
    traces = []
    curved = False
    disagrees = False
    for side, start, end in zip(("top", "right", "bottom", "left"), quad, np.roll(quad, -1, axis=0), strict=True):
        edge = edges.get(side) or {}
        paper = edge.get("rim_bgr")
        if not edge.get("is_physical_border") or not paper:
            continue
        evidence = _trace_side(image, start, end, paper)
        if evidence is None:
            continue
        polyline, deviation, distance = evidence
        traces.append(polyline)
        curved |= deviation > 1.05
        disagrees |= distance > 1.30
    return traces, curved, disagrees
