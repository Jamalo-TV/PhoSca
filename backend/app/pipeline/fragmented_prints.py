"""Check a complete print outline that competes with fragments of its image."""

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.multipanel_prints import _edge_has_step
from app.pipeline.print_edges import (
    _backing_color, _edge_band, _is_pale_stock, _pale_backing_transition, _side_line, refine_print_boundary,
)


FRAGMENTED_PRINT_REVISION = 1
FRAGMENTED_PRINT_REVIEW = "FRAGMENTED_PRINT_REVIEW"
_UNIT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def _image_continues_across(image: np.ndarray, strip: np.ndarray, width: int) -> bool:
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, 32], [0, 32]])
    gray = cv2.cvtColor(
        cv2.warpPerspective(image, cv2.getPerspectiveTransform(strip, destination), (width, 33)), cv2.COLOR_BGR2GRAY,
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), .7), 16, 34, L2gradient=True)
    margin = max(12, round(.04 * width))
    edges[:, :margin] = edges[:, -margin:] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
    locations = {}
    # Three separated scene features must cross the proposed division, and
    # at least two must remain continuous farther from the division.
    for label in np.intersect1d(labels[10], labels[22]):
        if label == 0:
            continue
        crossing = np.flatnonzero(labels[16] == label)
        if len(crossing) and stats[label, cv2.CC_STAT_WIDTH] <= .35 * width:
            locations[label] = float(np.median(crossing))
    extended = set(np.intersect1d(labels[8], labels[24])) & locations.keys()
    return len(extended) >= 2 and len(locations) >= 3 and np.ptp(list(locations.values())) >= .40 * width


def _no_internal_split(image: np.ndarray, quad: np.ndarray, fragments: list[np.ndarray]) -> bool:
    to_unit = cv2.getPerspectiveTransform(quad, _UNIT)
    projected = [cv2.perspectiveTransform(fragment[None], to_unit)[0] for fragment in fragments]
    centers = [points.mean(axis=0) for points in projected]
    separation = np.abs(centers[0] - centers[1])
    axis = int(separation.argmax())
    if separation[axis] < .20 or separation[axis] < 2 * separation[1 - axis]:
        return False
    first, second = sorted(projected, key=lambda points: points[:, axis].mean())
    begin, end = sorted((float(first[:, axis].max()), float(second[:, axis].min())))
    center = (begin + end) * .5
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    dimensions = (lengths[:2] + lengths[2:]) * .5
    depth = 16 / dimensions[axis]
    if not .15 <= center <= .85 or center - depth < 0 or center + depth > 1 or end - begin > .18:
        return False
    if any(_edge_has_step(image, quad[i], quad[(i + 1) % 4]) for i in (axis, axis + 2)):
        return False

    from_unit = cv2.getPerspectiveTransform(_UNIT, quad)
    line_points = np.float32([[0, center], [1, center]])
    strip = np.float32([[0, center - depth], [1, center - depth], [1, center + depth], [0, center + depth]])
    if axis == 0:
        line_points, strip = line_points[:, ::-1].copy(), strip[:, ::-1].copy()
    start, stop = cv2.perspectiveTransform(line_points[None], from_unit)[0]
    radius = int(np.clip(round((end - begin) * dimensions[axis] / 2 + 12), 9, 48))
    line, support = _side_line(image, start, stop, radius)
    if (line is not None or support["ambiguous"]) and support["support"] >= .74:
        return False
    strip = cv2.perspectiveTransform(strip[None], from_unit)[0]
    width = int(np.clip(round(dimensions[1 - axis]), 64, 1024))
    # A one-pixel change in rectification can break a thin gradient track.
    return sum(_image_continues_across(image, strip, width + offset) for offset in (-2, 0, 2)) >= 2


def measure_fragmented_print(
    image: np.ndarray, outline: np.ndarray, children: list[np.ndarray],
) -> np.ndarray | None:
    outer = order_quad_cyclic(outline)
    length = float(np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1).min())
    backing = _backing_color(image)
    if length < 90 or not _is_pale_stock(backing):
        return None
    radius = int(np.clip(round(.023 * length), 4, 14))
    quad, sides, reasons = refine_print_boundary(image, outer, search_radius=radius)
    if quad is None or reasons or any(side["support"] < .94 or side["corner_support"] < .75 for side in sides.values()):
        return None
    perimeter = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        band = _edge_band(image, start, end, [-8, -5, -3, 3, 5, 8])
        if band is None:
            return None
        transition, _ = _pale_backing_transition(band, backing)
        perimeter.append(float(np.mean(transition)))
    if min(perimeter) < .58 or np.mean(perimeter) < .84:
        return None

    # A fragment with its own supported perimeter could be a layered print.
    fragments = [order_quad_cyclic(child) for child in children]
    for fragment in fragments:
        rival, rival_sides, _ = refine_print_boundary(image, fragment, search_radius=4)
        if rival is not None and all(side["support"] >= .72 and (side["corner_support"] or 0) >= .45
                                     for side in rival_sides.values()):
            return None
    return quad if _no_internal_split(image, quad, fragments) else None
