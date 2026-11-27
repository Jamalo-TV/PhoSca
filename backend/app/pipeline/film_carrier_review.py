"""Locate a sleeve divider when the detected strip groups omit part of the carrier."""

import cv2
import numpy as np

from app.pipeline.contact_print_review import measure_chroma_outer_outline
from app.pipeline.multipanel_prints import (
    _UNIT_QUAD, _native_perforations_supported, regular_perforation_grid, transparent_film_rim,
)


def _divider_between_claims(image: np.ndarray, outline: np.ndarray, children: list[np.ndarray]) -> np.ndarray | None:
    lengths = np.linalg.norm(outline - np.roll(outline, -1, axis=0), axis=1)
    if lengths[::2].mean() > lengths[1::2].mean():
        outline = np.roll(outline, -1, axis=0)
    to_unit = cv2.getPerspectiveTransform(outline, _UNIT_QUAD)
    upper, lower = sorted((cv2.perspectiveTransform(child[None], to_unit)[0] for child in children),
                          key=lambda points: points[:, 1].mean())
    facing = np.float32([upper[:, 1].max(), lower[:, 1].min()])
    left = max(upper[:, 0].min(), lower[:, 0].min())
    right = min(upper[:, 0].max(), lower[:, 0].max())
    center = float(facing.mean())
    if (not .26 <= center <= .74 or not -.012 <= facing[1] - facing[0] <= .027
            or right - left < .76 or abs(upper[:, 0].min() - lower[:, 0].min()) > .05
            or abs(upper[:, 0].max() - lower[:, 0].max()) > .05):
        return None

    width, height = 800, 1040
    start, end = 120, 680
    to_grid = np.diag([width - 1., height - 1., 1.]) @ to_unit
    from_grid = np.linalg.inv(to_grid)
    view = cv2.warpPerspective(image, to_grid, (width, height))
    light = cv2.cvtColor(view[:, start:end], cv2.COLOR_BGR2LAB)[..., 0]
    grid = regular_perforation_grid(light)
    if grid is None:
        return None
    rails, pitches, polarities = grid
    if np.mean(polarities <= -8) < .80 or np.median(polarities) > -18 or max(polarities) > 10:
        return None

    bright = np.percentile(light, 15, axis=1) >= 160
    bands = np.flatnonzero(np.diff(np.r_[False, bright, False])).reshape(-1, 2)
    widths = np.diff(bands, axis=1)[:, 0]
    midpoints = bands.mean(axis=1)
    between_strips = ((midpoints[:, None] > rails[1:-1:2]) & (midpoints[:, None] < rails[2::2])).any(axis=1)
    plausible = (widths >= 4) & (widths <= 24) & between_strips
    predicted = center * (height - 1)
    radius = 7 + abs(facing[1] - facing[0]) * (height - 1) / 2
    candidates = np.flatnonzero(plausible & (abs(midpoints - predicted) <= radius))
    if len(candidates) != 1:
        return None
    candidate, = candidates
    other_widths = widths[plausible & (abs(midpoints - midpoints[candidate]) >= 70)]
    if len(other_widths) < 2 or not .60 <= widths[candidate] / np.median(other_widths) <= 1.50:
        return None
    if not all(_native_perforations_supported(image, from_grid, start, end, row, int(pitch))
               for row, pitch in zip(rails, pitches, strict=True)):
        return None
    y = midpoints[candidate]
    stripe = np.float32([[[max(.10, left + .03) * (width - 1), y], [min(.90, right - .03) * (width - 1), y]]])
    return cv2.perspectiveTransform(stripe, from_grid)[0]


def refitted_film_carrier(
    image: np.ndarray, seed: np.ndarray, children: list[np.ndarray], peers: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Use the refit for measurement and abstain when another claim overlaps it."""
    peer_hulls = [cv2.convexHull(peer) for peer in peers]
    # A disputed raw carrier may use the four-side fit but cannot retry a missing side.
    clear_raw_carrier = not any(cv2.intersectConvexConvex(seed, peer)[0] > 1 for peer in peer_hulls)
    outline = measure_chroma_outer_outline(image, seed, seed, remeasure_one_side=clear_raw_carrier)
    if outline is None:
        return None
    for child in children:
        hull = cv2.convexHull(child)
        area, _ = cv2.intersectConvexConvex(outline, hull)
        if area < .98 * cv2.contourArea(hull):
            return None
    if any(cv2.intersectConvexConvex(outline, peer)[0] > 1 for peer in peer_hulls):
        return None
    seam = _divider_between_claims(image, outline, children)
    if seam is None:
        return None
    rim = transparent_film_rim(image, outline, seam, peers, check_narrow_divider=True)
    return None if rim is None else (outline, seam, rim)
