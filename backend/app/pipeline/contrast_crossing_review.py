"""Find continuous scene detail across unsupported contrast proposals."""

from collections.abc import Iterable, Iterator
from itertools import combinations

import cv2
import numpy as np

from app.pipeline.album_backing import _backing_reference, _supported_backing_transition
from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic, quad_from_polygon
from app.pipeline.print_edges import _line_endpoints, _side_line, refine_print_boundary


CONTRAST_CROSSING_TRACES = "contrast_crossing_traces"


def contrast_review_windows(quad: np.ndarray, width: int, height: int) -> Iterator[tuple[int, tuple[int, int, int, int]]]:
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    shortest = lengths.min()
    size = np.minimum([width, height], round(1.78 * shortest))
    for index, (start, side, length) in enumerate(zip(quad, sides, lengths, strict=True)):
        inward = np.float32([-side[1], side[0]]) / length
        center = start + .5 * side
        windows = set()
        for shift in (0., -.18):
            origin = np.rint(center + shift * shortest * inward - .5 * size).astype(int)
            x, y = np.clip(origin, 0, np.array([width, height]) - size)
            window = (int(x), int(y), int(x + size[0]), int(y + size[1]))
            if window not in windows:
                windows.add(window)
                yield index, window


def _bracketed_span(image: np.ndarray, quad: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    direction = end - start
    length = np.linalg.norm(direction)
    tangent = direction / length
    cosines = np.abs(sides @ tangent / lengths)
    choices = [i for i in range(2) if max(cosines[i], cosines[i + 2]) <= .48]
    if len(choices) != 1:
        return None
    search_line = np.cross(np.r_[start, 1.], np.r_[end, 1.])
    first, = choices
    brackets = []
    for index in (first, first + 2):
        a, side = quad[index], sides[index]
        b = a + side
        fits = []
        for radius in (9, 14, 20, 26, 32):
            line, evidence = _side_line(image, a + .066 * side, b - .066 * side, radius)
            if line is not None and not evidence["ambiguous"] and (evidence["corner_support"] or 0) >= .50:
                fits.append((line, evidence["support"]))
        best = max((support for _, support in fits), default=0.)
        if best < .86:
            reference = _backing_reference(image, quad, lambda _: False)
            if reference is None:
                return None
            line = _supported_backing_transition(image, a, b, reference[0], min(44, round(.16 * lengths.min())))
            if line is None:
                return None
            fits, best = [(line, .90)], .90
        intersections = []
        for line, support in fits:
            if support < best - .065:
                continue
            endpoints = np.float32(_line_endpoints(a, b, line))
            crossing = np.cross(search_line, line)
            if np.linalg.norm(endpoints - [a, b], axis=1).max() > 42 or abs(crossing[2]) < 1.e-6:
                return None
            point = crossing[:2] / crossing[2]
            if not .15 <= (point - a) @ side / lengths[index] ** 2 <= .85:
                return None
            intersections.append(float((point - start) @ tangent))
        if np.ptp(intersections) > 19:
            return None
        brackets.append(intersections)

    near, far = sorted(brackets, key=np.mean)
    lo, hi = max(8., max(near) + 16), min(length - 8., min(far) - 16)
    if hi - lo < 115:
        return None
    stations = start + np.linspace(lo, hi, 180, dtype=np.float32)[:, None] * tangent
    inside = np.array([cv2.pointPolygonTest(quad, tuple(point), True) >= 13 for point in stations])
    return stations[inside][[0, -1]] if inside.mean() >= .82 else None


def _continuous_detail(gray: np.ndarray, span: np.ndarray) -> bool:
    start, end = span
    direction = end - start
    length = np.linalg.norm(direction)
    normal = np.float32([-direction[1], direction[0]]) / length
    stations = np.linspace(0., 1., int(length) + 1, dtype=np.float32)[:, None] * direction
    coords = start + stations[None] + np.arange(-10, 11, dtype=np.float32)[:, None, None] * normal
    if np.any((coords < 0) | (coords > np.float32(gray.shape[1::-1]) - 1)):
        return False
    bands = cv2.remap(gray, coords[..., 0].copy(), coords[..., 1].copy(), cv2.INTER_LINEAR)
    edges = cv2.Canny(cv2.GaussianBlur(bands, (3, 3), .52), 18, 41)
    _, labels = cv2.connectedComponents(edges, connectivity=8)
    across = np.intersect1d(labels[:8], labels[13:])
    carried = across[across != 0]
    if len(carried) < 5:
        return False
    present = np.isin(labels[8:13], carried).any(axis=0)
    return sum(part.any() for part in np.array_split(present, 8)) >= 6 and np.ptp(np.flatnonzero(present)) >= .60 * length


def contrast_crossing_traces(
    image: np.ndarray, quad: np.ndarray, observations: Iterable[tuple[int, np.ndarray, float]],
) -> list[np.ndarray]:
    """Require independent witnesses on two different sides of an unsupported outline."""
    shortest = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min()
    if shortest < 320 or np.any(np.minimum(quad, np.float32(image.shape[1::-1]) - 1 - quad) < 12):
        return []
    _, boundary, reasons = refine_print_boundary(image, quad)
    if reasons != ["PRINT_BOUNDARY_UNSUPPORTED"] or len(boundary) != 4 or any(
        side["support"] > .42 or side["ambiguous"] or (side.get("outward_search") or {}).get("print_rim_supported")
        for side in boundary.values()
    ):
        return []

    witnesses = []
    size = np.float32(image.shape[1::-1]) - 1
    for side, polygon, confidence in observations:
        hull = cv2.convexHull(polygon)[:, 0]
        area = cv2.contourArea(hull)
        if area <= 0 or not .14 <= cv2.intersectConvexConvex(quad, hull)[0] / area <= .79:
            continue
        seed, _ = quad_from_polygon(polygon)
        if seed is None or np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min() < 160:
            continue
        span = _bracketed_span(image, order_quad_cyclic(seed), quad[side], quad[(side + 1) % 4])
        if span is None:
            continue
        _, evidence = _side_line(image, *span, 8)
        if evidence["support"] <= .42 and not evidence["ambiguous"]:
            witnesses.append((side, polygon / size, confidence, span))
    if len(witnesses) < 2:
        return []
    unique = []
    for side, polygon, _, span in sorted(witnesses, key=lambda item: (-item[2], -np.linalg.norm(item[3][1] - item[3][0]))):
        if not any(side == earlier or normalized_polygon_overlap(polygon, shape, raster_size=512).iou >= .10
                   for earlier, shape, _ in unique):
            unique.append((side, polygon, span))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    supported = [(side, span) for side, _, span in unique if _continuous_detail(gray, span)]
    pairs = [[span, other] for (side, span), (neighbor, other) in combinations(supported, 2) if (side - neighbor) % 2 != 0]
    return pairs[0] if len(pairs) == 1 else []
