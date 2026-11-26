"""Measure a possible common perimeter for two suspect classical detections."""

from collections.abc import Callable, Iterable
from itertools import product

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.inward_corner_review import _intersection, _line
from app.pipeline.print_edges import _edge_band, _quad_from_lines, _side_line, refine_print_boundary


SOURCE_FRAGMENT_REVIEW_REVISION = 1
_SIDES = ("top", "right", "bottom", "left")
_CONFLICTS = ("warp_quad", "fragmented_print", "fragment_outline_suggestion", "crossing_print_suggestion",
              "paper_grid_review", "paper_join_review_traces", "film_grid_review_seam", "capture_edge_clipped",
              "capture_split_review_traces", "capture_edge_review_traces", "scalloped_print",
              "adjacent_print_suggestion", "nested_print_suggestion", "manual_added")


def _containment(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    shared = cv2.intersectConvexConvex(first, second)[0]
    return shared / cv2.contourArea(first), shared / cv2.contourArea(second)


def source_fragment_role(mask: dict, confidence: float, quad: np.ndarray, width: int, height: int) -> str | None:
    if mask.get("detector_source") != "edge_contour" or any(mask.get(key) for key in _CONFLICTS):
        return None
    clearance = np.minimum(quad, np.array([width - 1, height - 1]) - quad).min()
    scores = mask.get("scores") or {}
    if (mask.get("source") == "classical_border_quad" and .52 <= scores.get("rectangularity", 0) <= .78
            and confidence >= .72 and scores.get("exterior_background_sides", 0) >= 2 and clearance > 14):
        return "partial"
    if (mask.get("source") == "classical_min_area_rect" and clearance <= 1.5 and confidence <= .68
            and scores.get("exterior_background_sides", 4) <= 2
            and (mask.get("boundary_evidence") or {}).get("score", 1.) <= .60):
        return "frame_crossing"
    return None


def _partial_print_rails(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    fitted, evidence, reasons = refine_print_boundary(image, quad)
    if fitted is None or reasons != ["PRINT_BOUNDARY_UNSUPPORTED"] or len(evidence) != 4:
        return None
    weak = [index for index, name in enumerate(_SIDES) if evidence[name]["support"] < .40]
    if len(weak) != 1 or any(
        side["ambiguous"] or (index != weak[0] and (side["support"] < .92 or (side["corner_support"] or 0) < .84))
        for index, side in enumerate(evidence[name] for name in _SIDES)
    ):
        return None
    rails = np.roll(fitted, 1 - weak[0], axis=0)
    if np.linalg.norm(rails[0] - rails[3]) < 240 or np.linalg.norm(rails[2] - rails[3]) < 100:
        return None
    return rails


def _crossing_claim_lacks_visible_edges(image: np.ndarray, quad: np.ndarray) -> bool:
    height, width = image.shape[:2]
    # A clipped print can still have its own measured edges entirely in frame.
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        span = start + np.linspace(.04, .96, 64, dtype=np.float32)[:, None] * (end - start)
        inside = ((span > 14) & (span < [width - 15, height - 15])).all(axis=1)
        points = span[inside]
        if len(points) < 44 or np.linalg.norm(points[-1] - points[0]) < 150:
            return False
        line, evidence = _side_line(image, points[0], points[-1], 6)
        if line is not None or evidence["ambiguous"] or evidence["support"] > .48:
            return False
    return True


def _fit_signed_side(image: np.ndarray, start: np.ndarray, end: np.ndarray, *, weak: bool) -> np.ndarray | None:
    direction = end - start
    fractions = np.linspace(.065, .935, 256, dtype=np.float32)
    offsets = np.arange(-32, 32.25, .25, dtype=np.float32)
    band = _edge_band(image, start, end, offsets, fractions=fractions)
    if band is None:
        return None
    light = band[..., 0]
    centers = np.flatnonzero(np.abs(offsets) <= 23)
    outside = light[:, centers[:, None] - [10, 16, 24]]
    inside = light[:, centers[:, None] + [10, 16, 24]]
    contrast = np.median(outside, axis=2) - np.median(inside, axis=2)
    step = light[:, centers - 5] - light[:, centers + 5]
    scores = np.where((outside.min(axis=2) >= 173) & (contrast >= 18), step, 0.)
    keep = scores.max(axis=1) >= 10
    inward = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
    points = start + fractions[:, None] * direction + offsets[centers[scores.argmax(axis=1)], None] * inward
    for _ in range(2):
        if keep.sum() < (181 if weak else 231):
            return None
        dx, dy, x, y = cv2.fitLine(points[keep], cv2.DIST_HUBER, 0, .01, .01).ravel()
        normal = np.float32([-dy, dx])
        residual = np.abs((points - [x, y]) @ normal)
        keep &= residual <= 2.2
    if keep.sum() < (181 if weak else 231) or np.quantile(residual[keep], .90) > 1.7:
        return None
    quarters = np.split(keep, 4)
    if min(section.sum() for section in (quarters[1:] if weak else quarters)) < (45 if weak else 47):
        return None
    return np.r_[normal, -normal @ [x, y]]


def _supported_corners(image: np.ndarray, quad: np.ndarray) -> bool:
    for side, (start, end) in enumerate(zip(quad, np.roll(quad, -1, axis=0), strict=True)):
        length = np.linalg.norm(end - start)
        if length < 100:
            return False
        stations = np.float32([4, 6, 9, 13, 18, 25, 35, 45]) / length
        tails = np.float32([7, 10, 14, 19, 25, 32]) / length
        for at_start in (True, False):
            # Both arms at each newly inferred corner must end in the image.
            beyond = side == 1 or (side == 0 and not at_start) or (side == 2 and at_start)
            fractions = stations if at_start else 1 - stations
            if beyond:
                fractions = np.r_[fractions, -tails if at_start else 1 + tails]
            band = _edge_band(image, start, end, [-10, -8, -6, -4, 4, 6, 8, 10], fractions=fractions)
            if band is None:
                return False
            outside, inside = (np.median(part[..., 0], axis=1) for part in np.split(band, 2, axis=1))
            step = outside - inside
            supported = (outside[:8] >= 170) & (step[:8] >= 35)
            if supported.sum() < 7 or not supported[1]:
                return False
            if beyond and (np.count_nonzero(step[8:] <= 21) < 5 or step[8] > 21):
                return False
    return True


def _measure_perimeter(image: np.ndarray, rails: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    top, top_rival, bottom_rival, bottom = rails
    side = top - bottom
    span = np.linalg.norm(side)
    if max(np.linalg.norm(seed[0] - top), np.linalg.norm(seed[3] - bottom)) > .07 * span:
        return None
    tangent = bottom_rival - bottom
    normal = np.float32([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    if abs((seed[2] - bottom) @ normal) > .07 * span:
        return None
    corner = _intersection(_line(top, top_rival), _line(seed[2], seed[2] + side))
    if corner is None:
        return None
    draft = np.float32([top, corner, seed[2], bottom])
    if not cv2.isContourConvex(draft):
        return None
    lines = [_fit_signed_side(image, start, end, weak=index == 1)
             for index, (start, end) in enumerate(zip(draft, np.roll(draft, -1, axis=0), strict=True))]
    if any(line is None for line in lines):
        return None
    quad = _quad_from_lines(np.asarray(lines))
    if quad is None or not cv2.isContourConvex(quad) or cv2.contourArea(quad, oriented=True) <= 0:
        return None
    return quad if _supported_corners(image, quad) else None


def suggest_source_fragment_group(
    image: np.ndarray, quads: list[np.ndarray | None], partials: list[int], crossings: list[int],
    coarse_quads: Callable[[np.ndarray], Iterable[np.ndarray]],
) -> tuple[int, int, np.ndarray, int] | None:
    """Offer a unique outline while retaining both active claims for count review."""
    agreements = []
    height, width = image.shape[:2]
    for first, second in product(partials, crossings):
        partial, crossing = quads[first], quads[second]
        relation = _containment(partial, crossing)
        if min(relation) < .025 or max(relation) > .18:
            continue
        rails = _partial_print_rails(image, partial)
        if rails is None or not _crossing_claim_lacks_visible_edges(image, crossing):
            continue
        minimum = np.minimum(partial.min(axis=0), crossing.min(axis=0))
        maximum = np.maximum(partial.max(axis=0), crossing.max(axis=0))
        padding = .10 * (maximum - minimum).min()
        left, top = np.maximum(0, np.floor(minimum - padding)).astype(int)
        right, bottom = np.minimum([width, height], np.ceil(maximum + padding + 1)).astype(int)
        if not 150_000 <= (right - left) * (bottom - top) <= 2_250_000:
            continue
        fits = []
        for local in coarse_quads(image[top:bottom, left:right]):
            seed = local + np.float32([left, top])
            cover = _containment(partial, seed)
            if cover[0] < .77 or not .30 <= cover[1] <= .68 or _containment(crossing, seed)[0] < .70:
                continue
            seed = min((np.roll(seed, index, axis=0) for index in range(4)),
                       key=lambda points: np.linalg.norm(points[[0, 3]] - rails[[0, 3]], axis=1).sum())
            measured = _measure_perimeter(image, rails, seed)
            if measured is None:
                continue
            crossing_coverage, crossing_share = _containment(crossing, measured)
            if (_containment(partial, measured)[0] < .98 or not .86 <= crossing_coverage <= .96
                    or not .30 <= crossing_share <= .60
                    or any(cv2.pointPolygonTest(measured, tuple(point), True) < -3 for point in partial)):
                continue
            if any(other is None or max(_containment(other, measured)) > .045
                   for index, other in enumerate(quads) if index not in (first, second)):
                continue
            ordered = order_quad_cyclic(measured)
            weak = int(np.argmin(np.linalg.norm(ordered - measured[1], axis=1)))
            fits.append((first, second, ordered, weak))
        if fits and all(np.linalg.norm(fit[2] - fits[0][2], axis=1).max() < 1.3 for fit in fits):
            agreements.append(fits[0])
    return agreements[0] if len(agreements) == 1 else None
