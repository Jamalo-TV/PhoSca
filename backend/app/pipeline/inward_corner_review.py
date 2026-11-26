"""Measure the two paper edges beyond a severely inset detection corner."""

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _edge_band


INWARD_CORNER_REVIEW_REVISION = 1
INWARD_CORNER_REVIEW_SOURCE = "possible_inward_print_corner"


def _line(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    return np.cross(np.r_[first, 1.], np.r_[last, 1.])


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(point[2]) <= 1e-6:
        return None
    return np.float32(point[:2] / point[2])


def _paper_line(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    """Require a continuous, signed album-to-print step in a bounded native strip."""
    direction = end - start
    length = np.linalg.norm(direction)
    radius = int(np.clip(round(.025 * length), 10, 36))
    fractions = np.linspace(.04, .96, 256, dtype=np.float32)
    offsets = np.arange(-radius - 8, radius + 8.25, .25, dtype=np.float32)
    band = _edge_band(image, start, end, offsets, fractions=fractions)
    if band is None:
        return None

    candidates = np.flatnonzero(np.abs(offsets) <= radius)
    light = band[..., 0]
    outside = light[:, candidates[:, None] - [24, 16, 10]]
    inside = light[:, candidates[:, None] + [24, 16, 10]]
    contrast = np.median(outside, axis=2) - np.median(inside, axis=2)
    step = light[:, candidates - 5] - light[:, candidates + 5]
    scores = np.where((outside.min(axis=2) >= 173) & (contrast >= 20), step, 0)
    location = scores.argmax(axis=1)
    supported = scores.max(axis=1) >= 11
    if supported.mean() < .86:
        return None
    inward = np.float32([-direction[1], direction[0]]) / length
    points = start + fractions[:, None] * direction + offsets[candidates[location], None] * inward
    for _ in range(2):
        dx, dy, x, y = cv2.fitLine(points[supported], cv2.DIST_HUBER, 0, .01, .01).ravel()
        normal = np.float32([-dy, dx])
        residual = np.abs((points - [x, y]) @ normal)
        supported &= residual <= 1.9
        if supported.mean() < .86:
            return None
    if min(part.mean() for part in np.split(supported, 4)) < .65 or np.quantile(residual[supported], .9) > 1.2:
        return None
    return np.r_[normal, -normal @ [x, y]]


def _corner_arm(image: np.ndarray, start: np.ndarray, end: np.ndarray, *, at_start: bool) -> bool:
    length = np.linalg.norm(end - start)
    stations = np.float32([4, 6, 8, 10, 13, 16, 20, 26, 32, 40]) / length
    if stations[-1] >= .5:
        return False
    offsets = [-10, -8, -6, -4, 4, 6, 8, 10]
    fractions = stations if at_start else 1 - stations
    band = _edge_band(image, start, end, offsets, fractions=fractions)
    if band is None:
        return False
    outside, inside = (np.median(half, axis=1) for half in np.split(band, 2, axis=1))
    supported = (outside[:, 0] >= 173) & (outside[:, 0] - inside[:, 0] >= 35)
    if not supported[:2].all() or np.count_nonzero(supported) < 9:
        return False

    # The step must end at the corner and return to the same backing.
    beyond = np.float32([7, 10, 13, 16, 20, 26]) / length
    beyond = -beyond if at_start else 1 + beyond
    background = _edge_band(image, start, end, offsets, fractions=beyond)
    if background is None:
        return False
    stock = np.median(outside[supported], axis=0)
    sides = np.stack([np.median(half, axis=1) for half in np.split(background, 2, axis=1)])
    blank = (np.linalg.norm(sides - stock, axis=2) <= 14).all(axis=0)
    return bool(blank[:2].all() and np.count_nonzero(blank) >= 5)


def _outer_line(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, reversed_winding: bool,
) -> np.ndarray | None:
    return _paper_line(image, end, start) if reversed_winding else _paper_line(image, start, end)


def suggest_inward_corner_outline(image: np.ndarray, quad: np.ndarray, evidence: dict) -> np.ndarray | None:
    """Check a single diagonal truncation and keep the observed opposite arms."""
    if min(image.shape[:2]) < 480:
        return None
    proposals = []
    names = ("top", "right", "bottom", "left")
    for reversed_winding in (False, True):
        cyclic = quad[[0, 3, 2, 1]] if reversed_winding else quad
        sides = names[::-1] if reversed_winding else names
        for index in range(4):
            near, upper, inset, lower = np.roll(cyclic, -index, axis=0)
            shortened = inset - upper
            opposite = lower - near
            length, expected = np.linalg.norm(shortened), np.linalg.norm(opposite)
            if length < 140 or not .38 <= length / expected <= .72:
                continue
            if (shortened @ opposite) / (length * expected) < .97:
                continue
            old_sides = [evidence.get(sides[(index + side) % 4], {}) for side in range(4)]
            if (old_sides[0].get("support", 0) < .85 or old_sides[2].get("support", 1) > .55
                    or any(old_sides[side].get("ambiguous") or old_sides[side].get("support", 0) < .88
                           or (old_sides[side].get("corner_support") or 0) < .80 for side in (1, 3))):
                continue
            second_direction = upper - near
            if abs(second_direction @ opposite) > .19 * np.linalg.norm(second_direction) * expected:
                continue
            corner = _intersection(_line(upper, inset), _line(lower, lower + second_direction))
            if corner is None or not 1.32 <= (corner - upper) @ shortened / length**2 <= 2.6:
                continue

            first = _outer_line(image, upper, corner, reversed_winding)
            second = _outer_line(image, corner, lower, reversed_winding)
            if first is None or second is None:
                continue
            top = _line(near, upper)
            left = _line(lower, near)
            joints = (_intersection(top, first), _intersection(first, second), _intersection(second, left))
            if any(point is None for point in joints):
                continue
            candidate = np.float32([near, *joints])
            if (not cv2.isContourConvex(candidate) or np.any(candidate < 6)
                    or np.any(candidate > np.array(image.shape[1::-1]) - 7)):
                continue
            old_area = cv2.contourArea(quad)
            area = cv2.contourArea(candidate)
            if not 1.15 * old_area <= area <= 1.50 * old_area:
                continue
            if any(cv2.pointPolygonTest(candidate, tuple(point), True) < -2 for point in quad):
                continue
            if reversed_winding:
                arms = ((candidate[2], candidate[1], True), (candidate[3], candidate[2], False))
            else:
                arms = ((candidate[1], candidate[2], False), (candidate[2], candidate[3], True))
            if all(_corner_arm(image, start, end, at_start=at_start) for start, end, at_start in arms):
                proposals.append(order_quad_cyclic(candidate))
    return proposals[0] if len(proposals) == 1 else None
