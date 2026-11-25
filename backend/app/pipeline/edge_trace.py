"""Measure open source-pixel traces near three points chosen in review."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_vertices_to_pixels, pixel_vertices_to_normalized
from app.pipeline.print_edges import _sample_lab


def _darkward_measurements(light: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    drops = light[:, :-4] - light[:, 4:]
    offsets = offsets[2:-2]
    columns = np.flatnonzero(np.abs(offsets) <= 4)
    peaks = (drops[:, columns] >= drops[:, columns - 1]) & (drops[:, columns] > drops[:, columns + 1])
    outside = np.median(light[:, columns[:, None] + [-8, -5, -3]], axis=2)
    inside = np.median(light[:, columns[:, None] + [9, 11, 13]], axis=2)
    supported = peaks & (outside >= 171) & (outside - inside >= 15) & (drops[:, columns] >= 18)
    scores = np.where(supported, drops[:, columns], 0)
    best = scores.argmax(axis=1)
    rows = np.arange(len(light))
    strength = scores[rows, best]
    index = columns[best]
    selected = offsets[index]
    competing = supported & (scores >= np.maximum(16, .65 * strength[:, None]))
    competing &= np.abs(offsets[columns] - selected[:, None]) >= 1.5

    # A thin printed stroke can have a dark step and then immediately return
    # to the same pale material. It supplies no sustained exterior transition.
    interior = index[:, None] + np.arange(6, 18)
    returned = (-drops[rows[:, None], interior] >= .55 * strength[:, None]) & (
        light[rows[:, None], interior + 4] >= outside[rows, best, None] - 9
    )
    valid = (strength >= 18) & (np.abs(selected) <= 2.5) & ~competing.any(axis=1) & ~returned.any(axis=1)
    return selected, valid


def measure_edge_trace(image: np.ndarray, anchors: list[tuple[float, float]]) -> list[list[dict[str, float]]]:
    """Return observed sections of a selected contrast edge, with unsupported gaps open."""
    width, height = image.shape[1::-1]
    vertices = np.asarray(anchors, dtype=np.float32)
    if vertices.shape != (3, 2) or not np.isfinite(vertices).all() or np.any((vertices < 0) | (vertices > 1)):
        raise ValueError("edge_trace_invalid_points")
    start, middle, end = normalized_vertices_to_pixels(vertices, width, height)
    delta = end - start
    length = np.linalg.norm(delta)
    if length < 80:
        raise ValueError("edge_trace_points_too_close")
    tangent = delta / length
    normal = np.float32([-tangent[1], tangent[0]])
    fraction = (middle - start) @ tangent / length
    depth = (middle - start) @ normal
    if not .25 <= fraction <= .75 or abs(depth) > min(14., .065 * length):
        raise ValueError("edge_trace_invalid_middle")

    # Use the marked points only to set the search corridor. They are never
    # observations themselves, including where they lie over a source gap.
    fractions = np.linspace(.035, .965, 112, dtype=np.float32)
    offsets = np.arange(-15., 15.5, .5, dtype=np.float32)
    guide = depth * fractions * (1 - fractions) / (fraction * (1 - fraction))
    stations = start + fractions[:, None] * delta + guide[:, None] * normal
    positions = stations[:, None] + offsets[None, :, None] * normal
    if np.any((positions < 0) | (positions > np.array([width - 1, height - 1]))):
        return []
    light = _sample_lab(image, positions.reshape(-1, 2)).reshape(len(fractions), len(offsets), 3)[:, :, 0]
    near, direct = _darkward_measurements(light, offsets)
    far, reverse = _darkward_measurements(light[:, ::-1], offsets)
    ambiguous = direct & reverse
    direct &= ~ambiguous
    reverse &= ~ambiguous
    measured, supported = (near, direct) if direct.sum() >= reverse.sum() else (-far, reverse)
    if supported.mean() < .60 or sum(part.mean() >= .60 for part in np.array_split(supported, 4)) < 2:
        return []

    hits = np.flatnonzero(supported)
    depth = guide + measured
    breaks = (np.diff(hits) != 1) | (np.abs(np.diff(depth[hits])) > 1.2)
    trace = stations + measured[:, None] * normal
    segments = []
    for run in np.split(hits, np.flatnonzero(breaks) + 1):
        if len(run) < 8:
            continue
        observed = cv2.approxPolyDP(np.float32(trace[run]), .22, False)[:, 0]
        segments.append([{"x": float(x), "y": float(y)} for x, y in pixel_vertices_to_normalized(observed, width, height)])
    return segments
