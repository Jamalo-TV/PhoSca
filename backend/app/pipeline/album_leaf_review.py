"""Measure a book-like gutter beside a leaf-shaped proposal for count review."""

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic


ALBUM_LEAF_COUNT_REVIEW = "ALBUM_LEAF_COUNT_REVIEW"
ALBUM_LEAF_REVIEW_TRACES = "album_leaf_review_traces"
ALBUM_LEAF_REVIEW_CHILD = "album_leaf_review_child"
ALBUM_LEAF_REVIEW_REVISION = 1
_REFERENCE_SIDE = 1200


def _sample(lab: np.ndarray, points: np.ndarray) -> np.ndarray:
    x, y = np.moveaxis(np.float32(points), -1, 0)
    return cv2.remap(lab, x.copy(), y.copy(), cv2.INTER_LINEAR).astype(np.float32)


def _inside(points: np.ndarray, size: np.ndarray) -> np.ndarray:
    return ((points >= 0) & (points <= size)).all(axis=-1)


def _track_rail(
    stations: np.ndarray, offsets: np.ndarray, strength: np.ndarray, eligible: np.ndarray,
) -> np.ndarray | None:
    row = np.arange(len(stations))
    selected = np.argmax(np.where(eligible, strength, -100.), axis=1)
    values = offsets[selected]
    supported = eligible[row, selected] & (strength[row, selected] >= 48)
    if supported.mean() < .45:
        return None
    fit = np.polyfit(stations[supported], values[supported], 1)
    for _ in range(4):
        stable = supported & (abs(np.polyval(fit, stations) - values) <= 5)
        if stable.mean() < .45:
            return None
        fit = np.polyfit(stations[stable], values[stable], 1)
    return fit if stable.mean() >= .77 else None


def _cap_material(lab, size, position, near, far, endpoint, sign):
    """Measure paired paper ends and the visible dark apron beneath them."""
    steps = np.arange(-100., 101., 4., dtype=np.float32)
    stations = endpoint + sign * steps / _REFERENCE_SIDE
    left, right = np.polyval(near, stations), np.polyval(far, stations)
    across = np.linspace(14, 54, 12, dtype=np.float32)
    shoulders = np.column_stack((left[:, None] - across, right[:, None] + across))
    shoulder_points = position(stations[:, None], shoulders)
    edges = _sample(lab, shoulder_points).reshape(len(stations), 2, -1, 3)
    traces = np.median(edges, axis=2)[..., 0]
    real_samples = _inside(shoulder_points, size).reshape(len(stations), 2, -1).all(axis=2)
    rise = traces[2:] - traces[:-2]
    eligible = ((steps[1:-1, None] >= -57) & (steps[1:-1, None] <= 52)
                & (traces[2:] >= 115) & (traces[:-2] <= 102) & real_samples[2:] & real_samples[:-2])
    best = np.argmax(np.where(eligible, rise, -100.), axis=0)
    if not eligible[best, [0, 1]].all() or min(rise[best, [0, 1]]) < 68:
        return None
    if abs(steps[best[0] + 1] - steps[best[1] + 1]) > 26:
        return None
    center = int(np.floor(np.median(steps[best + 1]) / 4)) + 25
    if center < 6 or center + 6 >= len(stations):
        return None
    offsets = np.column_stack((left - 65, left - 24, (left + right) / 2, right + 24, right + 65))
    apron_points = position(stations[:, None], offsets)
    if (not _inside(apron_points[center - 5:center - 1, [0, 1, 3, 4]], size).all()
            or not _inside(apron_points[center + 7:center + 21, 2], size).all()):
        return None
    samples = _sample(lab, apron_points)
    apron = np.median(samples[center - 5:center - 1, [0, 1, 3, 4]], axis=(0, 1))
    if apron[0] > 89:
        return None
    binding = np.median(samples[center + 7:center + 21, 2], axis=0)
    return float(np.linalg.norm(binding[1:] - apron[1:]))


def _gutter_at_side(lab: np.ndarray, a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    edge = b - a
    length = float(np.linalg.norm(edge))
    if length < 320:
        return []
    normal = np.float32([edge[1], -edge[0]]) / _REFERENCE_SIDE

    def position(station, offset):
        return a + np.asarray(station)[..., None] * edge + np.asarray(offset)[..., None] * normal

    stations = np.linspace(.09, .91, 128, dtype=np.float32)
    offsets = np.arange(-36., 161., 2., dtype=np.float32)
    size = np.asarray(lab.shape[1::-1]) - 1
    positions = position(stations[:, None], offsets[None])
    valid = _inside(positions, size).all(axis=1)
    if valid.mean() < .88:
        return []
    lightness = _sample(lab, positions)[..., 0]
    before = sum(np.roll(lightness, shift, axis=1) for shift in (3, 4, 5)) / 3
    after = sum(np.roll(lightness, -shift, axis=1) for shift in (3, 4, 5)) / 3
    near_band = (offsets >= -15) & (offsets <= 80) & (before >= 124) & (after <= 99) & valid[:, None]
    near = _track_rail(stations, offsets, before - after - .035 * np.maximum(offsets, 0), near_band)
    if near is None:
        return []
    distance = offsets - np.polyval(near, stations)[:, None]
    far_band = (distance >= 20) & (distance <= 102) & (offsets <= 150) & (after >= 122) & (before <= 103) & valid[:, None]
    far = _track_rail(stations, offsets, after - before - .024 * np.maximum(distance, 0), far_band)
    if far is None:
        return []

    left, right = np.polyval(near, stations), np.polyval(far, stations)
    material = position(stations[:, None], np.column_stack((left - 14, (left + right) / 2, right + 14)))
    first, middle, last = _sample(lab, material).transpose(1, 0, 2)
    supported = ((right - left >= 22) & (right - left <= 96) & (first[:, 0] >= 128) & (last[:, 0] >= 128)
                 & (middle[:, 0] <= 104) & (first[:, 0] - middle[:, 0] >= 52) & (last[:, 0] - middle[:, 0] >= 52)
                 & (np.linalg.norm(first - last, axis=1) <= 43) & _inside(material, size).all(axis=1) & valid)
    if supported.mean() < .82 or min(part.mean() for part in np.array_split(supported, 4)) < .64:
        return []
    top = _cap_material(lab, size, position, near, far, 0, 1)
    bottom = _cap_material(lab, size, position, near, far, 1, -1)
    if top is None or bottom is None or max(top, bottom) < 6.5:
        return []
    bounds = stations[[0, -1]]
    return [position(bounds, np.polyval(line, bounds)) for line in (near, far)]


def album_leaf_review_traces(page: np.ndarray, outline: np.ndarray) -> list[np.ndarray]:
    """Return two native gutter rails only when exactly one side supports the cue."""
    quad = order_quad_cyclic(outline)
    lab = cv2.cvtColor(page, cv2.COLOR_BGR2LAB)
    candidates = [_gutter_at_side(lab, start, end) for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True)]
    observed = [traces for traces in candidates if traces]
    return observed[0] if len(observed) == 1 else []
