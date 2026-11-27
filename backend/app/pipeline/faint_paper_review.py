"""Measure open pale transitions beside a crop whose inner image edges are stronger."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line


ADJACENT_FAINT_PAPER_STEPS = "adjacent_faint_paper_steps"
FAINT_PAPER_REVIEW_REQUIRES_CONFIRMATION = "faint_paper_review_requires_confirmation"
FAINT_PAPER_WITNESS_MARGIN = 8.
FAINT_PAPER_REVIEW_REVISION = 1


@dataclass(frozen=True)
class _OpticalRim:
    line: np.ndarray
    span: np.ndarray
    polarity: int


def _lab_strip(image: np.ndarray, coords: np.ndarray) -> np.ndarray | None:
    low = np.floor(coords.min(axis=(0, 1))).astype(int)
    high = np.ceil(coords.max(axis=(0, 1))).astype(int) + 1
    if np.any(low < 0) or np.any(high > image.shape[1::-1]):
        return None
    patch = image[low[1]:high[1], low[0]:high[0]].astype(np.float32) / 255
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    x, y = np.moveaxis(np.float32(coords - low), -1, 0)
    return cv2.remap(lab, x.copy(), y.copy(), cv2.INTER_LINEAR)


def _strong_scene_edge(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    line, evidence = _side_line(image, start, end, 5)
    if (line is None or evidence["ambiguous"] or evidence["support"] < .88
            or (evidence.get("corner_support") or 0) < .72):
        return None
    points = np.float32(_line_endpoints(start, end, line))
    bounds = _edge_band(image, *points, [-7, -5, -3, 3, 5, 7], fractions=np.linspace(.1, .9, 64, dtype=np.float32))
    if bounds is None:
        return None
    outside, inside = (np.median(part, axis=1) for part in (bounds[:, :3], bounds[:, 3:]))
    pale = (outside[:, 0] >= 170) & (np.linalg.norm(outside[:, 1:] - 128, axis=1) <= 22)
    ink = outside[:, 0] - inside[:, 0] >= 21
    return points if np.mean(pale & ink) >= .65 else None


def _is_repeated_ridge(image: np.ndarray, edge: np.ndarray, distances: np.ndarray, polarity: int) -> bool:
    start, end = edge
    tangent = end - start
    outward = np.float32([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
    stations = np.linspace(.09, .91, 96, dtype=np.float32)
    offsets = np.arange(-32., 38.1, .5, dtype=np.float32)
    positions = start + stations[:, None] * tangent
    lab = _lab_strip(image, positions[:, None] + offsets[:, None] * outward)
    if lab is None:
        return False
    samples = lab[..., 0]
    indices = np.arange(8, len(offsets) - 8)
    center = polarity * np.median(samples[:, indices[:, None] + [-1, 0, 1]], axis=2)
    inside, outside = (polarity * np.median(samples[:, indices[:, None] + sign * np.arange(4, 8)], axis=2)
                       for sign in (-1, 1))
    height = np.minimum(center - inside, center - outside)
    from_ink = distances[0] + stations * (distances[1] - distances[0])
    offsets = offsets[indices]
    stable = ((height >= 1.25) & (offsets + from_ink[:, None] >= 4.5)).mean(axis=0) >= .70
    stable &= np.median(height, axis=0) >= 1.2
    inner, outer = offsets[stable & (offsets <= -5)], offsets[stable & (offsets >= 5)]
    return bool(np.any(np.abs(inner[:, None] + outer) <= 1.))


def _optical_rim(image: np.ndarray, edge: np.ndarray, shortest: float) -> _OpticalRim | None:
    start, end = edge
    tangent = end - start
    length = np.linalg.norm(tangent)
    outward = np.float32([tangent[1], -tangent[0]]) / length
    stations = np.linspace(.075, .925, 124, dtype=np.float32)
    minimum, maximum = max(8, round(.028 * shortest)), min(68, round(.135 * shortest))
    offsets = np.arange(minimum - 8., maximum + 8.1, .5, dtype=np.float32)
    anchors = start + stations[:, None] * tangent
    lab = _lab_strip(image, anchors[:, None, :] + offsets[:, None] * outward)
    if lab is None:
        return None

    choices = np.arange(minimum, maximum + .1, .5, dtype=np.float32)
    ids = np.rint((choices - offsets[0]) * 2).astype(int)
    levels = cv2.medianBlur(lab[..., 0].copy(), 3)
    core = np.median(levels[:, ids[:, None] + [-1, 0, 1]], axis=2)
    inner, outer = (np.median(levels[:, ids[:, None] + np.arange(*limits)], axis=2)
                    for limits in ((-9, -5), (6, 10)))
    broad_inner, broad_outer = (np.median(levels[:, ids[:, None] + np.arange(*limits)], axis=2)
                                for limits in ((-15, -10), (10, 15)))
    neutral = np.linalg.norm(lab[:, ids, 1:], axis=2) <= 17
    pale = neutral & (np.minimum(np.minimum(inner, outer), core) >= 68)

    first, last = np.meshgrid(choices, choices)
    parallel = np.abs(first - last) <= min(.035 * length, .045 * shortest)
    starts, ends = first[parallel], last[parallel]
    distance = starts[:, None] + (ends - starts)[:, None] * stations
    addresses = np.rint((distance - minimum) * 2).astype(int)
    rows = np.arange(len(stations))[None, :]
    candidates = []
    for polarity in (-1, 1):
        near = np.minimum(polarity * (core - inner), polarity * (core - outer))[rows, addresses]
        broad = np.minimum(polarity * (core - broad_inner), polarity * (core - broad_outer))[rows, addresses]
        observed = (near >= 1.15) & (broad >= 1.) & pale[rows, addresses]
        shares = np.stack([part.mean(axis=1) for part in np.array_split(observed, 8, axis=1)])
        stable = (observed.mean(axis=1) >= .88) & (shares.min(axis=0) >= .68)
        contrast = .65 * near + .35 * broad
        rank = .65 * np.quantile(contrast, .25, axis=1) + .35 * np.median(contrast, axis=1)
        for index in np.flatnonzero(stable & (rank >= 1.65)):
            candidates.append((rank[index], int(index), polarity, observed[index]))
    if not candidates:
        return None
    score, best, polarity, supported = max(candidates, key=lambda choice: choice[0])
    if any(value >= .86 * score and np.max(np.abs(distance[index] - distance[best])) > 3.5
           for value, index, _, _ in candidates):
        return None
    valid = np.flatnonzero(supported)
    longest = max(np.split(valid, np.flatnonzero(np.diff(valid) > 4) + 1), key=len)
    if len(longest) < .65 * len(stations):
        return None
    indices = longest[[0, -1]]
    distances = np.float32([starts[best], ends[best]])
    endpoints = edge + distances[:, None] * outward
    if _is_repeated_ridge(image, endpoints, distances, polarity):
        return None
    trace = anchors[indices] + distance[best, indices, None] * outward
    return _OpticalRim(endpoints, trace, polarity)


def adjacent_faint_paper_steps(image: np.ndarray, quad: np.ndarray) -> list[np.ndarray]:
    """Return two measured exterior spans for review, without extrapolating a paper outline."""
    shortest = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
    if not 145 <= shortest <= 660:
        return []
    scenes = [_strong_scene_edge(image, start, end) for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True)]
    adjacent = [index for index, scene in enumerate(scenes) if scene is not None and scenes[index - 1] is not None]
    if len(adjacent) != 1:
        return []
    corner, = adjacent
    first = _optical_rim(image, scenes[corner - 1], shortest)
    second = _optical_rim(image, scenes[corner], shortest)
    if first is None or second is None or first.polarity == second.polarity:
        return []
    first_line, second_line = (np.cross(*np.column_stack((rim.line, [1, 1]))) for rim in (first, second))
    intersection = np.cross(first_line, second_line)
    if abs(intersection[2]) <= 1e-6:
        return []
    meeting_point = intersection[:2] / intersection[2]
    if np.linalg.norm(meeting_point - quad[corner]) > .16 * shortest:
        return []
    return [first.span, second.span]
