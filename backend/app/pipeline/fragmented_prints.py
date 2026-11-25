"""Check a complete print outline that competes with fragments of its image."""

from itertools import combinations

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.multipanel_prints import _edge_has_step
from app.pipeline.print_edges import (
    _backing_color, _edge_band, _is_pale_stock, _pale_backing_transition, _side_line, refine_print_boundary,
)


FRAGMENTED_PRINT_REVISION = 2
FRAGMENTED_PRINT_REVIEW = "FRAGMENTED_PRINT_REVIEW"
FRAGMENT_OUTLINE_REVIEW = "FRAGMENT_OUTLINE_UNCERTAIN"
EXTENDED_PRINT_OUTLINE_REVIEW = "PRINT_EXTENDED_OUTLINE_REVIEW"
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


def _no_internal_split(
    image: np.ndarray, quad: np.ndarray, fragments: list[np.ndarray], *, uncertain_side: int | None = None,
) -> bool:
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
    if any(_edge_has_step(image, quad[i], quad[(i + 1) % 4]) for i in (axis, axis + 2) if i != uncertain_side):
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
    width = int(np.clip(round(dimensions[1 - axis]), 64, 1024))
    if uncertain_side is not None:
        # Require scene tracks around the separator, which need not be aligned
        # with the fragments' estimated bounding edges. This only guides review.
        shift = np.zeros(2, np.float32)
        shift[axis] = .25 * depth
        return all(_image_continues_across(image, cv2.perspectiveTransform((strip + offset)[None], from_unit)[0], width)
                   for offset in (-shift, shift))
    strip = cv2.perspectiveTransform(strip[None], from_unit)[0]
    # A one-pixel change in rectification can break a thin gradient track.
    return sum(_image_continues_across(image, strip, width + offset) for offset in (-2, 0, 2)) >= 2


def _perimeter_support(image: np.ndarray, quad: np.ndarray, backing: np.ndarray) -> list[float]:
    support = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        band = _edge_band(image, start, end, [-8, -5, -3, 3, 5, 8])
        if band is None:
            return []
        transition, _ = _pale_backing_transition(band, backing)
        support.append(float(np.mean(transition)))
    return support


def _has_own_perimeter(image: np.ndarray, fragment: np.ndarray) -> bool:
    rival, sides, _ = refine_print_boundary(image, fragment, search_radius=4)
    return rival is not None and all(side["support"] >= .72 and (side["corner_support"] or 0) >= .45
                                     for side in sides.values())


def _has_oblique_inner_edge(image: np.ndarray, rectified_lab: np.ndarray, to_source: np.ndarray) -> bool:
    height, width = rectified_lab.shape[:2]
    rows = np.rint(np.linspace(.08 * height, .92 * height, 96)).astype(int)
    columns = np.arange(round(.25 * width), round(.96 * width))
    bands = rectified_lab[rows]
    transitions = np.square(bands[:, 4:] - bands[:, :-4]).sum(axis=2) >= 13**2
    # Step over plausible slants with a one-pixel tolerance on each side.
    support = cv2.dilate(transitions.astype(np.uint8), np.ones((1, 3), np.uint8))
    row_indices = np.arange(len(rows))[:, None]
    row_fractions = (rows - (rows[0] + rows[-1]) * .5) / (rows[-1] - rows[0])
    limit = round(.24 * (rows[-1] - rows[0]))
    tested = 0
    for drift in range(-limit, limit + 1, 4):
        shifted = columns + np.rint(drift * row_fractions[:, None]).astype(int)
        inside = (shifted >= round(.15 * width)) & (shifted < round(.97 * width) - 4)
        votes = (inside & support[row_indices, np.clip(shifted, 0, width - 5)].astype(bool)).mean(axis=0)
        column = int(votes.argmax())
        if votes[column] < .72:
            continue
        tested += 1
        if tested > 12:
            return True
        # Confirm each rectified line on the native raster before treating
        # the colour step as evidence of a different print or backing.
        endpoints = np.float32([[[shifted[0, column] + 2, rows[0]], [shifted[-1, column] + 2, rows[-1]]]])
        start, end = cv2.perspectiveTransform(endpoints, to_source)[0]
        line, evidence = _side_line(image, start, end, 7)
        if evidence["support"] >= .74 and (line is not None or evidence["ambiguous"]):
            return True
    return False


def _has_broad_inner_edge(image: np.ndarray, quad: np.ndarray, *, oblique_side: int | None = None) -> bool:
    if oblique_side is not None:
        # Place the proposed extension on the right in the rectified patch.
        quad = np.roll(quad, 1 - oblique_side, axis=0)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    width, height = np.clip(np.rint((lengths[:2] + lengths[2:]) * .5), 64, 1024).astype(int)
    target = _UNIT * np.float32([width - 1, height - 1])
    rectified = cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (width, height))
    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
    for bands in (lab, lab.swapaxes(0, 1)):
        along, across = bands.shape[:2]
        core = bands[along // 12:along - along // 12, across // 12:across - across // 12]
        for span in (1, 3):
            transition = np.square(core[span:] - core[:-span]).sum(axis=2) >= 13 ** 2
            if np.any(transition.mean(axis=1) >= .75):
                return True
    return oblique_side is not None and _has_oblique_inner_edge(
        image, lab, cv2.getPerspectiveTransform(target, quad),
    )


def measure_extended_print_outline(image: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> np.ndarray | None:
    """Check whether one false side of a nearly contained fragment cuts through the scene."""
    length = float(np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1).min())
    if length < 90:
        return None
    radius = int(np.clip(round(.022 * length), 4, 18))
    fragment, inner_evidence, _ = refine_print_boundary(image, inner, search_radius=radius)
    weak = [
        index for index, side in enumerate(inner_evidence.values())
        if side.get("support", 0) < .50 and not side.get("ambiguous")
    ]
    if fragment is None or len(weak) != 1:
        return None
    measured, evidence, reasons = refine_print_boundary(image, outer, search_radius=radius)
    if measured is None or reasons or any(
        side["support"] < .90 or (side["corner_support"] or 0) < .82 for side in evidence.values()
    ):
        return None

    divider = weak[0]
    for index, side in enumerate(inner_evidence.values()):
        if index == divider:
            continue
        if side["support"] < .90 or "edge_width_px" not in side:
            return None
        end = (index + 1) % 4
        vector = measured[end] - measured[index]
        normal = np.float32([-vector[1], vector[0]]) / np.linalg.norm(vector)
        distances = (fragment[[index, end]] - measured[index]) @ normal
        if np.max(np.abs(distances)) > 2:
            return None

    start, end = inner[divider], inner[(divider + 1) % 4]
    vector = end - start
    length = float(np.linalg.norm(vector))
    normal = np.float32([-vector[1], vector[0]]) * (16 / length)
    first, last = start + .15 * vector, end - .15 * vector
    strip = np.float32([first - normal, last - normal, last + normal, first + normal])
    if any(cv2.pointPolygonTest(measured, tuple(point), True) < 2 for point in strip):
        return None
    width = int(np.clip(round(.70 * length), 64, 1024))
    continuous = sum(_image_continues_across(image, strip, width + offset) for offset in (-2, 0, 2)) >= 2
    return measured if continuous and not _has_broad_inner_edge(image, measured, oblique_side=divider) else None


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
    perimeter = _perimeter_support(image, quad, backing)
    if not perimeter or min(perimeter) < .58 or np.mean(perimeter) < .84:
        return None

    # A fragment with its own supported perimeter could be a layered print.
    fragments = [order_quad_cyclic(child) for child in children]
    if any(_has_own_perimeter(image, fragment) for fragment in fragments):
        return None
    return quad if _no_internal_split(image, quad, fragments) else None


def suggest_fragmented_outline(
    image: np.ndarray, outline: np.ndarray, fragments: list[np.ndarray], pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, int] | None:
    """Keep one weak side only as a manual editing hypothesis."""
    outer = order_quad_cyclic(outline)
    length = float(np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1).min())
    backing = _backing_color(image)
    if length < 90 or not _is_pale_stock(backing):
        return None
    radius = int(np.clip(round(.023 * length), 4, 14))
    quad, evidence, reasons = refine_print_boundary(image, outer, search_radius=radius)
    if quad is None or reasons != ["PRINT_BOUNDARY_UNSUPPORTED"]:
        return None
    sides = list(evidence.values())
    weak = [index for index, side in enumerate(sides) if "edge_width_px" not in side]
    if len(weak) != 1:
        return None
    uncertain = weak[0]
    if any((side["corner_support"] or 0) < .75 or side["support"] < (.65 if index == uncertain else .94)
           for index, side in enumerate(sides)):
        return None
    perimeter = _perimeter_support(image, quad, backing)
    if not perimeter or any(value < (.55 if index == uncertain else .90) for index, value in enumerate(perimeter)):
        return None
    fragments = [order_quad_cyclic(fragment) for fragment in fragments]
    if any(_has_own_perimeter(image, fragment) for fragment in fragments):
        return None
    connections = {pair for pair in pairs if _no_internal_split(
        image, quad, [fragments[pair[0]], fragments[pair[1]]], uncertain_side=uncertain,
    )}
    if not connections:
        return None
    # A child without its own connecting scene evidence must substantially
    # duplicate another fragment. A separate print beyond the tested divider
    # may have only a partial contour and would pass the perimeter veto alone.
    transform = cv2.getPerspectiveTransform(quad, _UNIT)
    projected = [cv2.perspectiveTransform(fragment[None], transform)[0] for fragment in fragments]
    for first, second in combinations(range(len(projected)), 2):
        a, b = projected[first], projected[second]
        if np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) > .25:
            continue
        shared = normalized_polygon_overlap(a, b)
        if min(shared.first_containment, shared.second_containment) >= .40:
            connections.add((first, second))
    connected = {0}
    for _ in range(len(fragments) - 1):
        connected |= {neighbor for edge in connections if connected.intersection(edge) for neighbor in edge}
    # Search the interior independently of proposal positions: a misplaced
    # fragment can bridge an actual paper seam in the overlap graph.
    return (quad, uncertain) if len(connected) == len(fragments) and not _has_broad_inner_edge(image, quad) else None
