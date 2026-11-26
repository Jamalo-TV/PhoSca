"""Check whether an unmatched fallback outline has its own visible perimeter."""

import cv2
import numpy as np

from app.pipeline.edge_prints import _has_independent_content
from app.pipeline.geometry import (
    normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, pixel_vertices_to_normalized,
    quad_from_polygon,
)
from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line, refine_print_boundary


FUSION_OWNERSHIP_REVISION = 3
NESTED_FALLBACK_CONTOUR = "nested_fallback_contour"
NESTED_FALLBACK_CONTOUR_REVIEW = "NESTED_FALLBACK_CONTOUR_REVIEW"


def _covered_area(quad: np.ndarray, owners: list[np.ndarray]) -> float:
    low = quad.min(axis=0)
    extent = quad.max(axis=0) - low
    scale = min(1.0, 512.0 / float(extent.max()))
    shape = tuple((np.ceil(extent * scale).astype(int) + 1)[::-1])
    outline = np.zeros(shape, np.uint8)
    claimed = np.zeros_like(outline)
    cv2.fillPoly(outline, [np.rint((quad - low) * scale).astype(np.int32)], 1)
    for owner in owners:
        cv2.fillPoly(claimed, [np.rint((owner - low) * scale).astype(np.int32)], 1)
    return float(np.count_nonzero(outline & claimed) / max(1, np.count_nonzero(outline)))


def _continues_across_paper(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, exposed: np.ndarray, fractions: np.ndarray,
) -> bool:
    if np.count_nonzero(exposed) < .35 * len(fractions):
        return False
    band = _edge_band(image, start, end, [-14, -7, -3, 3, 7, 14], fractions=fractions[exposed])
    if band is None:
        return False
    outside, inside = np.median(band[:, :3], axis=1), np.median(band[:, 3:], axis=1)
    neutral = np.maximum(np.linalg.norm(inside[:, 1:] - 128, axis=1), np.linalg.norm(outside[:, 1:] - 128, axis=1))
    step = inside - outside
    if np.linalg.norm(np.median(step, axis=0)) > 3.5:
        return False
    quiet = (np.linalg.norm(step, axis=1) <= 6) & (neutral <= 12)
    return bool(np.mean(quiet & (np.minimum(inside[:, 0], outside[:, 0]) >= 145)) >= .75)


def unsupported_fallback_ownership(
    image: np.ndarray, proposal: np.ndarray, model_outlines: list[np.ndarray],
) -> str | None:
    """Identify conflicting source evidence for count review, preserving the proposed outline."""
    height, width = image.shape[:2]
    if len(proposal) != 4 or not cv2.isContourConvex(proposal):
        return None
    # An area comparison only nominates the outlines for direct pixel checks.
    relevant = [outline for outline in model_outlines
                if normalized_polygon_overlap(proposal, outline).first_containment >= .10]
    if not relevant:
        return None
    quad = order_quad_cyclic(normalized_vertices_to_pixels(proposal, width, height))
    shortest = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
    if shortest < 100:
        return None
    owners = [normalized_vertices_to_pixels(outline, width, height) for outline in relevant]
    if _covered_area(quad, owners) < .70:
        return None
    if len(owners) == 1:
        gap_to_frame = np.minimum(quad, [width - 1, height - 1] - quad)
        if not np.any(np.all(gap_to_frame <= 2, axis=1)):
            return None

    radius = int(np.clip(round(.032 * shortest), 7, 36))
    fractions = np.linspace(.08, .92, 64, dtype=np.float32)
    edges = list(zip(quad, np.roll(quad, -1, axis=0), strict=True))
    distances = []
    for start, end in edges:
        stations = start + fractions[:, None] * (end - start)
        distances.append(np.array([[cv2.pointPolygonTest(owner, tuple(station), True) for station in stations]
                                   for owner in owners]))

    supported = []
    quiet_sides = {
        index for index, ((start, end), side_distances) in enumerate(zip(edges, distances, strict=True))
        if _continues_across_paper(image, start, end, side_distances.max(axis=0) < -14, fractions)
    }
    if not quiet_sides:
        return None
    for index, (start, end) in enumerate(edges):
        line, evidence = _side_line(image, start, end, radius)
        if evidence["ambiguous"]:
            return None
        if line is not None:
            supported.append(index)
            if len(supported) >= 2:
                return None
    if not quiet_sides.difference(supported):
        return None

    if len(owners) == 1:
        clearance = float(np.clip(.02 * shortest, 9, 20))
        if len(supported) == 1 and np.mean(np.abs(distances[supported[0]][0]) <= clearance) >= .8:
            return "shared_confirmed_edge"
        return None

    # The unclaimed portion of a partly covered rear print can still show
    # its own image. Keep it as a candidate wherever that evidence survives.
    claimed = np.zeros((height, width), np.uint8)
    for owner in owners:
        cv2.fillPoly(claimed, [np.rint(owner).astype(np.int32)], 1)
    claimed = cv2.dilate(claimed, np.ones((7, 7), np.uint8))
    if not _has_independent_content(image, claimed, quad):
        return "crossing_modeled_prints"
    return None


def _borrowed_side(stations: np.ndarray, parent: np.ndarray) -> int | None:
    edges = np.roll(parent, -1, axis=0) - parent
    lengths = np.linalg.norm(edges, axis=1)
    units = edges / lengths[:, None]
    normals = np.column_stack((-units[:, 1], units[:, 0]))
    offsets = stations[:, None] - parent[None, :]
    along = np.sum(offsets * units, axis=2)
    across = np.abs(np.sum(offsets * normals, axis=2))
    coverage = np.mean((across <= 5) & (along >= -5) & (along <= lengths + 5), axis=0)
    matching = np.flatnonzero(coverage >= .85)
    return int(matching[0]) if len(matching) == 1 else None


def _weak_interior_span(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, fractions: np.ndarray,
) -> bool:
    band = _edge_band(image, start, end, list(range(-12, 13)), fractions=fractions)
    if band is None:
        return False
    shades = np.quantile(band[:, 12, 0], [.10, .90])
    if shades[1] - shades[0] < 10:
        return False
    gradient = np.linalg.norm(np.diff(band, axis=1), axis=2)
    if np.mean(gradient[:, 10:14].max(axis=1) > 5) > .40:
        return False
    # Check for a coherent thin seam slightly off the proposal or at a small slope.
    rows = np.arange(len(fractions))[:, None]
    lane = np.arange(5, 19)[None, :]
    for drift in (0, -8, -4, 4, 8):
        columns = lane + np.rint(drift * (fractions[:, None] - .5)).astype(int)
        peaks = np.maximum(gradient[rows, columns], gradient[rows, columns + 1])
        if np.mean(peaks >= 5, axis=0).max() >= .45:
            return False
    return True


def _distributed_stock_rail(votes: np.ndarray) -> bool:
    coverage = cv2.dilate(votes, np.ones((3, 1), np.uint8))
    for row in np.flatnonzero(coverage.mean(axis=1) >= .87):
        if min(part.mean() for part in np.array_split(coverage[row], 4)) >= .76:
            return True
    # The gap can be oblique in the parent plane, independent of the two seeds.
    columns = np.arange(votes.shape[1])
    lines = cv2.HoughLinesP(votes, 1, np.pi / 720, threshold=72, minLineLength=100, maxLineGap=12)
    for x1, y1, x2, y2 in lines[:, 0] if lines is not None else ():
        if abs(x2 - x1) < .65 * len(columns) or abs(y2 - y1) > .35 * len(votes):
            continue
        rows = np.rint(y1 + (columns - x1) * (y2 - y1) / (x2 - x1)).astype(int)
        if rows.min() < 0 or rows.max() >= len(votes):
            continue
        support = coverage[rows, columns]
        if support.mean() >= .87 and min(part.mean() for part in np.array_split(support, 4)) >= .76:
            return True
    return _bowed_stock_rail(votes, coverage)


def _bowed_stock_rail(votes: np.ndarray, coverage: np.ndarray) -> bool:
    # Closing nominates a connected path. Only the original votes validate it.
    connected = cv2.morphologyEx(coverage, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    columns = np.arange(votes.shape[1])
    axis = np.linspace(-1, 1, len(columns))
    original = cv2.dilate(votes, np.ones((5, 1), np.uint8))
    candidates = np.flatnonzero((stats[1:, cv2.CC_STAT_WIDTH] >= .82 * len(columns))
                                & (stats[1:, cv2.CC_STAT_HEIGHT] <= .40 * len(votes))) + 1
    for label in candidates:
        x, y, width, height, _ = stats[label]
        section = labels[y:y + height, x:x + width] == label
        centers = np.array([y + np.median(np.flatnonzero(column)) if column.any() else -1 for column in section.T])
        live = centers >= 0
        if np.count_nonzero(live) < .82 * len(columns):
            continue
        fit = np.polynomial.Polynomial.fit(axis[x:x + width][live], centers[live], 3)
        center = fit(axis)
        bend = np.max(np.abs(center - np.linspace(center[0], center[-1], len(center))))
        residual = np.quantile(np.abs(center[x:x + width][live] - centers[live]), .90)
        rows = np.rint(center).astype(int)
        if bend > 28 or residual > 2 or rows.min() < 0 or rows.max() >= len(votes):
            continue
        support = original[rows, columns]
        if support.mean() >= .87 and min(part.mean() for part in np.array_split(support, 4)) >= .76:
            return True
    return False


def _contains_stock_gap(image: np.ndarray, parent: np.ndarray) -> bool:
    lengths = np.linalg.norm(np.roll(parent, -1, axis=0) - parent, axis=1)
    for side in (0, 1):
        depth = round(float(np.mean(lengths[[1 - side, 3 - side]])))
        if not 280 <= depth <= 3500:
            return True
        # Sample across the full print with native resolution along the gap's normal.
        target = np.float32([[0, 0], [223, 0], [223, depth - 1], [0, depth - 1]])
        transform = cv2.getPerspectiveTransform(np.roll(parent, -side, axis=0), target)
        view = cv2.warpPerspective(image, transform, (224, depth))
        lab = cv2.cvtColor(view[:, 8:-8], cv2.COLOR_BGR2LAB).astype(np.float32)
        first, last = round(.17 * depth), round(.83 * depth)
        lightness = lab[:, :, 0]
        chroma = np.linalg.norm(lab[:, :, 1:] - 128, axis=2)
        strip = lab[first:last]
        center = strip[:, :, 0]
        center_chroma = chroma[first:last]
        neutral = np.max(np.abs(lab[first:last, :, 1:] - 128), axis=2) <= 28
        votes = np.zeros_like(center, dtype=np.uint8)
        for offset in (3, 7, 14):
            before = lightness[first - offset:last - offset]
            after = lightness[first + offset:last + offset]
            light = (center >= 150) & (center - np.maximum(before, after) > 15)
            dark = (center <= 100) & (np.minimum(before, after) - center > 15)
            neighboring_chroma = np.minimum(chroma[first - offset:last - offset], chroma[first + offset:last + offset])
            desaturated = (center_chroma <= 10) & (neighboring_chroma - center_chroma > 12)
            # JPEG can attenuate narrow color transitions. Require both shoulders
            # to differ in the same Lab direction before accepting a weaker dip.
            left = lab[first - offset:last - offset] - strip
            right = lab[first + offset:last + offset] - strip
            left_step = np.linalg.norm(left, axis=2)
            right_step = np.linalg.norm(right, axis=2)
            coherent = (np.minimum(left_step, right_step) >= 6) & (
                (left * right).sum(axis=2) >= .75 * left_step * right_step
            )
            color_rail = (center_chroma <= 18) & (neighboring_chroma - center_chroma >= 4) & coherent
            votes |= neutral & (light | dark | desaturated | color_rail)
        if _distributed_stock_rail(votes):
            return True
    return False


def nested_fallback_contour_spans(
    image: np.ndarray, parent_polygon: np.ndarray, child_polygons: list[np.ndarray],
) -> list[np.ndarray]:
    """Measure why two plausible fallback outlines may be parts of the same print."""
    height, width = image.shape[:2]
    seed, method = quad_from_polygon(normalized_vertices_to_pixels(parent_polygon, width, height))
    if method == "minimum_area_rect" or np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min() < 320:
        return []
    parent, _, reasons = refine_print_boundary(image, seed)
    if parent is None or reasons or normalized_polygon_overlap(
        parent_polygon, pixel_vertices_to_normalized(parent, width, height),
    ).iou < .90:
        return []

    fractions = np.linspace(.10, .90, 160, dtype=np.float32)
    spans = []
    for polygon in child_polygons:
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            return []
        quad = order_quad_cyclic(normalized_vertices_to_pixels(polygon, width, height))
        shortest = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
        if shortest < 120:
            return []
        radius = int(np.clip(round(.032 * shortest), 7, 36))
        borrowed = set()
        interior = []
        for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
            line, evidence = _side_line(image, start, end, radius)
            if evidence["ambiguous"]:
                return []
            if line is not None:
                first, last = _line_endpoints(start, end, line)
                shared = _borrowed_side(first + fractions[:, None] * (last - first), parent)
                if shared is not None and evidence["support"] >= .86:
                    borrowed.add(shared)
                    continue
            stations = start + fractions[:, None] * (end - start)
            inside = np.mean([cv2.pointPolygonTest(parent, tuple(point), True) > 18 for point in stations]) >= .95
            if evidence["support"] >= .60 or (inside and (evidence["support"] > .45
                                                         or not _weak_interior_span(image, start, end, fractions))):
                return []
            if inside:
                interior.append(stations[[0, -1]])
        if len(borrowed) != 2 or len(interior) != 1:
            return []
        spans.append(interior[0])
    return [] if _contains_stock_gap(image, parent) else spans
