"""Check a complete print outline that competes with fragments of its image."""

from itertools import combinations

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.multipanel_prints import _edge_has_step
from app.pipeline.print_edges import (
    _backing_color, _edge_band, _is_pale_stock, _line_endpoints, _pale_backing_transition, _side_line, refine_print_boundary,
)


FRAGMENTED_PRINT_REVISION = 5
SOURCE_FRAGMENT_PROPOSAL = "possible_source_fragmented_print"
SOURCE_FRAGMENT_GROUP_REVIEW = "SOURCE_FRAGMENT_GROUP_REVIEW"
FRAGMENTED_PRINT_REVIEW = "FRAGMENTED_PRINT_REVIEW"
FRAGMENT_OUTLINE_REVIEW = "FRAGMENT_OUTLINE_UNCERTAIN"
EXTENDED_PRINT_OUTLINE_REVIEW = "PRINT_EXTENDED_OUTLINE_REVIEW"
COMPACT_SCENE_CROSSING_TRACE = "compact_scene_crossing_trace"
COMPACT_SCENE_CROSSING_REVISION = 1
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


def _crossing_locations(strip: np.ndarray, low: int, high: int) -> list[float]:
    _, labels = cv2.connectedComponents(cv2.Canny(strip, low, high))
    before, after = labels[:3], labels[-3:]
    middle = labels[len(labels) // 2 - 1:len(labels) // 2 + 2]
    locations = []
    for component in np.intersect1d(before, after):
        if not component:
            continue
        central = np.nonzero(middle == component)[1]
        if not len(central):
            continue
        start, stop = (float(np.median(np.nonzero(side == component)[1])) for side in (before, after))
        position = float(np.median(central))
        if abs(position - (start + stop) / 2) <= 4.5 and abs(start - stop) <= 21:
            locations.append(position)
    return sorted(locations)


def compact_scene_crossing_trace(
    image: np.ndarray, fragment: np.ndarray, parent: np.ndarray, other_claims: list[np.ndarray],
) -> np.ndarray | None:
    """Locate an unsupported small-scale cut with visible scene continuities.

    This observes pixels on both sides of the cut. Stacked or printed copies
    can show the same evidence, so it is a review trace only.
    """
    edges = np.roll(fragment, -1, axis=0) - fragment
    lengths = np.linalg.norm(edges, axis=1)
    if not 80 <= lengths.min() <= 250 or lengths.max() > 650:
        return None
    evidence = [_side_line(image, start, start + edge, int(np.clip(round(.027 * length), 5, 10)))
                for start, edge, length in zip(fragment, edges, lengths, strict=True)]
    if any(side["ambiguous"] for _, side in evidence):
        return None
    strong = [index for index, (line, side) in enumerate(evidence)
              if line is not None and side["support"] >= .89 and (side.get("corner_support") or 0) >= .75]
    if len(strong) != 2 or strong[1] - strong[0] == 2:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    spans = []
    for (line, side), start, edge, length in zip(evidence, fragment, edges, lengths, strict=True):
        if line is not None or side["support"] > .49:
            continue
        tangent = edge / length
        normal = np.float32([-tangent[1], tangent[0]])
        stations = start + np.arange(0, length, dtype=np.float32)[:, None] * tangent
        clear = np.array([cv2.pointPolygonTest(parent, tuple(point), True) >= 13 for point in stations])
        for claim in other_claims:
            clear &= [cv2.pointPolygonTest(claim, tuple(point), True) < -12 for point in stations]
        intervals = np.flatnonzero(np.diff(np.r_[False, clear, False])).reshape(-1, 2)
        for first, last in intervals:
            if last - first < 85:
                continue
            coordinates = stations[None, first:last] + np.arange(-8, 9, dtype=np.float32)[:, None, None] * normal
            if not ((coordinates >= 0) & (coordinates <= [width - 1, height - 1])).all():
                continue
            strip = cv2.remap(gray, coordinates[..., 0].copy(), coordinates[..., 1].copy(), cv2.INTER_LINEAR)
            moderate = _crossing_locations(strip, 24, 55)
            strong_contrast = _crossing_locations(strip, 35, 80)
            repeated = []
            for position in strong_contrast:
                if any(abs(position - candidate) < 3 for candidate in moderate) and (
                    not repeated or position - repeated[-1] >= 8
                ):
                    repeated.append(position)
            if len(repeated) >= 4 and np.ptp(repeated) >= .36 * (last - first):
                spans.append(stations[[first + round(repeated[0]), first + round(repeated[-1])]])
    return spans[0] if len(spans) == 1 else None


def nested_scene_fragment_continuities(
    image: np.ndarray, outer: np.ndarray, fragment: np.ndarray,
) -> list[np.ndarray]:
    """Trace two adjacent cuts through scene detail in a contained proposal."""
    return _nested_scene_evidence(image, outer, fragment, require_shared_edges=False)[0]


def _shared_fragment_rail(
    image: np.ndarray, outer: np.ndarray, first: np.ndarray, last: np.ndarray, line: np.ndarray,
) -> np.ndarray | None:
    start, stop = np.float32(_line_endpoints(first, last, line))
    direction = stop - start
    inward = np.float32([-direction[1], direction[0]]) / float(np.linalg.norm(direction))
    offsets = [-7, -4, -2, 2, 4, 7]
    inside = _edge_band(image, start, stop, offsets, fractions=np.linspace(.10, .90, 100, dtype=np.float32))
    if inside is None:
        return None
    backing = np.median(inside[:, :3], axis=(0, 1))
    delta = np.median(inside[:, 3:], axis=1) - np.median(inside[:, :3], axis=1)
    signed = np.median(delta, axis=0)
    strength = np.linalg.norm(signed)
    if strength < 30 or np.mean(delta @ signed >= 18 * strength) < .88:
        return None
    for begin, end in ((-.48, -.10), (1.10, 1.48)):
        stations = np.linspace(begin, end, 64, dtype=np.float32)
        path = start + stations[:, None] * direction
        if min(cv2.pointPolygonTest(outer, tuple(p + 12 * inward), True) for p in path[::4]) < -3:
            continue
        extension = _edge_band(image, start, stop, offsets, fractions=stations)
        if extension is None or np.linalg.norm(np.median(extension[:, :3], axis=(0, 1)) - backing) > 25:
            continue
        change = np.median(extension[:, 3:], axis=1) - np.median(extension[:, :3], axis=1)
        valid = change @ signed >= 18 * strength
        if valid.mean() >= .86 and min(part.mean() for part in np.array_split(valid, 4)) >= .72:
            return path[[0, -1]]
    return None


def _contour_has_material_step(image: np.ndarray, first: np.ndarray, last: np.ndarray) -> bool:
    """Find a consistent narrow step after subtracting a broad illumination ramp."""
    offsets = np.arange(-16., 16.5, .5)
    band = _edge_band(image, first, last, offsets, fractions=np.linspace(.10, .90, 192, dtype=np.float32))
    if band is None:
        return True
    for i in range(12, len(offsets) - 12):
        near = np.median(band[:, i + 2:i + 6], axis=1) - np.median(band[:, i - 5:i - 1], axis=1)
        far = np.median(band[:, i + 8:i + 12], axis=1) - np.median(band[:, i - 11:i - 7], axis=1)
        step = near - .4375 * far
        direction = np.median(step, axis=0)
        strength = np.linalg.norm(direction)
        if strength < 1.25:
            continue
        support = step @ direction >= strength
        if support.mean() >= .53 and min(part.mean() for part in np.array_split(support, 4)) >= .35:
            return True
    return False


def nested_scene_fragment_with_shared_edges(
    image: np.ndarray, outer: np.ndarray, fragment: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Trace scene continuities and the larger print's rails past a small proposal."""
    return _nested_scene_evidence(image, outer, fragment, require_shared_edges=True)


def _nested_scene_evidence(
    image: np.ndarray, outer: np.ndarray, fragment: np.ndarray, *, require_shared_edges: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    directions = np.roll(fragment, -1, axis=0) - fragment
    lengths = np.linalg.norm(directions, axis=1)
    if lengths.min() < 320:
        return [], []
    radius = int(np.clip(round(.014 * lengths.min()), 12, 24))
    strong, traces, shared = [], [], []
    material_steps = 0
    for index, (start, direction) in enumerate(zip(fragment, directions, strict=True)):
        stop = start + direction
        line, side = _side_line(image, start, stop, radius)
        if side["ambiguous"]:
            return [], []
        if line is not None and side["support"] >= .85 and (side["corner_support"] or 0) >= .70:
            strong.append(index)
            if require_shared_edges:
                extension = _shared_fragment_rail(image, outer, start, stop, line)
                if extension is None:
                    return [], []
                shared.append(extension)
            continue
        if line is not None or side["support"] > .30:
            return [], []
        first, last = start + .10 * direction, stop - .10 * direction
        depth = np.float32([-direction[1], direction[0]]) * .015
        strip = np.float32([first - depth, last - depth, last + depth, first + depth])
        if any(cv2.pointPolygonTest(outer, tuple(point), True) < 4 for point in strip):
            return [], []
        if sum(_image_continues_across(image, strip, width) for width in (600, 768, 1000)) < 2:
            return [], []
        if require_shared_edges:
            material_steps += _contour_has_material_step(image, start, stop)
        traces.append(np.float32([first, last]))
    if len(strong) != 2 or (strong[1] - strong[0]) == 2 or material_steps >= 2:
        return [], []
    return traces, shared


def suggest_dark_mounted_outline(
    image: np.ndarray, outline: np.ndarray, fragments: list[np.ndarray],
) -> np.ndarray | None:
    """Find three physical edges and scene continuities for count review."""
    if _backing_color(image)[0] > 70:
        return None
    seed = order_quad_cyclic(cv2.boxPoints(cv2.minAreaRect(outline)))
    shortest = float(np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min())
    if shortest < 300:
        return None
    radius = min(130, round(.054 * shortest))
    quad, evidence, reasons = refine_print_boundary(image, seed, search_radius=radius)
    if quad is None or "PRINT_BOUNDARY_UNSUPPORTED" not in reasons:
        return None
    missing = [index for index, side in enumerate(evidence.values()) if "edge_width_px" not in side]
    if len(missing) != 1:
        return None
    weak, = missing
    for index, side in enumerate(evidence.values()):
        if index == weak:
            continue
        if side["support"] < .68:
            return None
        band = _edge_band(image, quad[index], quad[(index + 1) % 4], [-24, -12, -8, 8, 12, 24])
        if band is None:
            return None
        exterior = np.median(band[:, :3, 0], axis=1)
        print_side = np.median(band[:, 3:, 0], axis=1)
        if np.mean((exterior <= 90) & (print_side >= exterior + 18)) < .76:
            return None

    if _has_broad_inner_edge(image, quad):
        return None

    for fragment in fragments:
        crossing = 0
        for start, end in zip(fragment, np.roll(fragment, -1, axis=0), strict=True):
            tangent = end - start
            length = float(np.linalg.norm(tangent))
            if length < 100:
                continue
            normal = np.float32([-tangent[1], tangent[0]]) * (16 / length)
            first, last = start + .12 * tangent, end - .12 * tangent
            strip = np.float32([first - normal, last - normal, last + normal, first + normal])
            if any(cv2.pointPolygonTest(quad, tuple(point), True) < 2 for point in strip):
                continue
            line, side = _side_line(image, start, end, 25)
            if line is not None or side["ambiguous"]:
                continue
            width = round(min(length * .76, 1024))
            crossing += sum(_image_continues_across(image, strip, width + offset) for offset in (-2, 0, 2)) >= 2
        if crossing < 2:
            return None
    return quad


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
