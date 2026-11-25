"""Check whether an unmatched fallback outline has its own visible perimeter."""

import cv2
import numpy as np

from app.pipeline.edge_prints import _has_independent_content
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic
from app.pipeline.print_edges import _edge_band, _side_line


FUSION_OWNERSHIP_REVISION = 1


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
