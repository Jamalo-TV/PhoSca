"""Trace supported corner arcs on a manually marked print over warm album stock."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _sample_lab

ROUNDED_CORNER_REVIEW = "PRINT_ROUNDED_CORNER_REVIEW"
ROUNDED_CORNER_SOURCE = "measured_print_corner_arcs"
ROUNDED_CORNER_WEDGES = "reviewed_corner_wedges"
ROUNDED_CORNER_REVISION = 1

_ANGLES = np.deg2rad(np.linspace(12, 78, 12))
_RAYS = np.float32(np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)]))
_DEPTHS = np.arange(0, 24.5, .5, dtype=np.float32)


@dataclass(frozen=True)
class RoundedCornerReview:
    traces: tuple[np.ndarray, ...]
    wedges: tuple[np.ndarray, ...]
    outline: np.ndarray


def _signed_arc(band: np.ndarray, backing: np.ndarray, *, signed: bool) -> np.ndarray | None:
    distances = (band[..., 1] - band[..., 2] + .52 * (backing[1] - backing[0]) + 8 if signed
                 else np.linalg.norm(band[..., 1:] - backing, axis=2))
    before, after = distances[:, :-6], distances[:, 6:]
    gradient = np.linalg.norm(band[:, 6:] - band[:, :-6], axis=2)
    prefix = np.cumsum(distances <= 9, axis=1)[:, :-6] / np.arange(1, distances.shape[1] - 5)
    supported = (before <= 8) & (after >= 9) & (after - before >= 4) & (gradient >= 18) & (prefix >= .82)
    first = supported.argmax(axis=1)
    positions = np.arange(supported.shape[1])[None, :]
    nearby = supported & (positions <= first[:, None] + 4)
    choices = np.argmax(np.where(nearby, gradient, 0), axis=1)
    points = _RAYS * _DEPTHS[choices + 3, None]
    indices = np.flatnonzero(supported.any(axis=1))
    for run in sorted(np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1), key=len, reverse=True):
        if len(run) < 7 or not (run[0] <= 4 and run[-1] >= 7):
            continue
        arc = points[run]
        if np.linalg.norm(np.diff(arc, axis=0), axis=1).max() > 3.0:
            continue
        if (np.diff(arc[:, 0]) > .85).any() or (np.diff(arc[:, 1]) < -.85).any():
            continue
        direction = arc[-1] - arc[0]
        length = np.linalg.norm(direction)
        if length < 4:
            continue
        normal = np.float32([-direction[1], direction[0]]) / length
        curvature = (arc - arc[0]) @ normal
        if np.abs(curvature).max() < .6 or abs(np.median(curvature[1:-1])) < .22:
            continue
        return arc
    return None


def _measure_corner(
    page: np.ndarray, transform: np.ndarray, origin: np.ndarray, signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    def source(points):
        return cv2.perspectiveTransform(np.float32(origin + points * signs).reshape(1, -1, 2), transform)[0]

    def sample(points):
        positions = source(points)
        if np.any(positions < 0) or np.any(positions > np.float32(page.shape[1::-1]) - 1):
            return None
        return _sample_lab(page, positions).reshape(*points.shape[:-1], 3)

    exterior = np.float32([[x, y] for x in range(-10, -4) for y in range(4, 23)])
    samples = sample(np.stack([exterior, exterior[:, ::-1]]))
    if samples is None:
        return None
    palettes = np.median(samples[..., 1:], axis=1)
    backing = palettes.mean(axis=0)
    if (backing[1] < 134 or np.linalg.norm(palettes[0] - palettes[1]) > 6
            or np.min(np.mean(np.linalg.norm(samples[..., 1:] - backing, axis=2) <= 8, axis=1)) < .81):
        return None
    band = sample(_RAYS[:, None, :] * _DEPTHS[None, :, None])
    if band is None:
        return None
    arc = _signed_arc(band, backing, signed=False)
    if arc is None:
        arc = _signed_arc(band, backing, signed=True)
    if arc is None:
        return None
    # Keep the uncertain transition with the paper. Validate the proposed
    # transparent wedge independently of the angle samples along the arc.
    depth = np.linalg.norm(arc, axis=1, keepdims=True)
    if depth.min() <= 2.05:
        return None
    inside = arc * (1 - 2.05 / depth)
    edge = np.vstack([[inside[0, 0], 0], inside, [0, inside[-1, 1]]]).astype(np.float32)
    wedge = np.vstack([edge, [0, 0]]).astype(np.float32)
    grid = np.float32([(x, y) for x in np.arange(.25, wedge[:, 0].max(), .5)
                       for y in np.arange(.25, wedge[:, 1].max(), .5)
                       if cv2.pointPolygonTest(wedge, (x, y), False) >= 0])
    if len(grid) < 16:
        return None
    colors = sample(grid)
    if colors is None or np.mean(np.linalg.norm(colors[:, 1:] - backing, axis=1) <= 8) < .995:
        return None
    # Extend the measured brightness phase from the exposed side towards the corner.
    # A similarly coloured patch printed on the photograph is insufficient evidence.
    columns = np.float32([-16, -12, -7])
    rows = np.arange(0, 22, .5, dtype=np.float32)
    slopes = np.linspace(-.18, .18, 13, dtype=np.float32)
    anchors = np.broadcast_to(columns, (len(slopes), len(rows), len(columns)))
    phase = rows[None, :, None] + slopes[:, None, None] * (anchors + 7)
    stripes = sample(np.stack([anchors, phase], axis=-1))
    if stripes is None:
        return None
    light = stripes[..., 0]
    residuals = np.median(np.ptp(light, axis=2), axis=1)
    best = int(residuals.argmin())
    if residuals[best] > 15 or np.ptp(np.percentile(np.median(light[best], axis=1), [10, 90])) < 21:
        return None
    prediction = grid[:, 1, None] + slopes[best] * (columns - grid[:, 0, None])
    behind = sample(np.stack([np.broadcast_to(columns, prediction.shape), prediction], axis=-1))
    if behind is None or np.mean(np.abs(np.median(behind[..., 0], axis=1) - colors[:, 0]) <= 11) < .94:
        return None
    return source(arc), source(edge), source(wedge)


def measure_rounded_corners(
    page: np.ndarray, quad: np.ndarray, neighbor_polygons: list[np.ndarray],
) -> RoundedCornerReview | None:
    """Measure only supported arcs, and leave unmeasured corner pixels visible."""
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if not 150 <= lengths.min() <= 650:
        return None
    width, height = np.maximum(lengths[:2], lengths[2:])
    frame = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    transform = cv2.getPerspectiveTransform(frame, np.float32(quad))
    if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1.e-8:
        return None
    hulls = [cv2.convexHull(peer) for peer in neighbor_polygons]
    polygon = []
    traces = []
    wedges = []
    for corner, origin, signs in zip(quad, frame, np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]]), strict=True):
        candidate = _measure_corner(page, transform, origin, signs)
        if candidate is None:
            polygon.append(corner[None])
            continue
        arc, edge, wedge = candidate
        scope = cv2.convexHull(np.vstack([arc, wedge]))
        if any(cv2.intersectConvexConvex(scope, peer)[0] > 1 for peer in hulls):
            polygon.append(corner[None])
            continue
        traces.append(arc)
        wedges.append(wedge)
        polygon.append(edge[::-1] if signs.prod() > 0 else edge)
    if len(traces) < 2:
        return None
    return RoundedCornerReview(tuple(traces), tuple(wedges), np.concatenate(polygon))
