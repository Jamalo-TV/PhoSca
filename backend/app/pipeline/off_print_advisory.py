"""Find protected-region suggestions that a reviewer can examine at a paper edge."""

import cv2
import numpy as np

from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line
from app.pipeline.protected_regions import is_advisory_page_face, protected_region_key, region_box_in_pixels


OFF_PRINT_ADVISORY_REVISION = 1


def _outline(box: tuple[float, float, float, float], padding: float = 0.) -> np.ndarray:
    left, top, right, bottom = box
    return np.float32([[left - padding, top - padding], [right + padding, top - padding],
                       [right + padding, bottom + padding], [left - padding, bottom + padding]])


def _locally_supported_paper_edge(page: np.ndarray, paper: np.ndarray, core: np.ndarray) -> bool:
    sides = np.roll(paper, -1, axis=0) - paper
    lengths = np.linalg.norm(sides, axis=1)
    tangents = sides / lengths[:, None]
    inward = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    depths = ((core.mean(axis=0) - paper) * inward).sum(axis=1)
    side = int(np.argmin(depths))
    if depths[side] > -12 or (np.delete(depths, side) < 0).any():
        return False
    positions = (core - paper[side]) @ tangents[side]
    first, last = float(positions.min()), float(positions.max())
    if first < 5 or last > lengths[side] - 5:
        return False
    start, end = max(3., first - 28), min(lengths[side] - 3, last + 28)
    start, end = paper[side] + np.float32([start, end])[:, None] * tangents[side]
    line, evidence = _side_line(page, start, end, 11)
    if line is None or evidence["support"] < .90 or evidence["ambiguous"]:
        return False
    fit = np.float32(_line_endpoints(start, end, line))
    if np.max(np.abs((fit - [start, end]) @ inward[side])) > 2.:
        return False
    band = _edge_band(page, *fit, [-13, -9, -6, -4, -2, 2, 4, 6, 9, 13])
    if band is None:
        return False
    stock = np.median(band[:, 5:], axis=1)
    background = np.median(band[:, :5], axis=1)
    return (np.median(stock[:, 0] - background[:, 0]) >= 20
            and np.mean((stock[:, 0] >= 173) & (np.linalg.norm(stock[:, 1:] - 128, axis=1) < 15)) >= .90)


def off_print_advisory_regions(
    page: np.ndarray, paper: np.ndarray, regions: list[dict], violations: list[dict], neighbors: list[np.ndarray],
) -> list[dict]:
    """Offer original-pixel review only for an unclaimed, very small intrusion onto a measured edge."""
    if not 1 <= len(violations) <= 4 or not cv2.isContourConvex(paper):
        return []
    height, width = page.shape[:2]
    if cv2.contourArea(paper, oriented=True) < 0:
        paper = paper[::-1].copy()
    scale = np.array([width - 1, height - 1])
    blockers = [cv2.convexHull(neighbor) for neighbor in neighbors]
    for region in regions:
        if not is_advisory_page_face(region) and (box := region_box_in_pixels(region, width, height)):
            blockers.append(_outline(box))

    marks = []
    for violation in violations:
        matches = [region for region in regions if protected_region_key(region) == violation]
        if len(matches) != 1 or not is_advisory_page_face(matches[0]):
            return []
        region = matches[0]
        box = region_box_in_pixels(region, width, height)
        if box is None or not np.isfinite(box).all():
            return []
        left, top, right, bottom = box
        padding = float(region.get("margin_ratio", .08)) * max(right - left, bottom - top)
        if not np.isfinite(padding) or padding < 0:
            return []
        core, padded = _outline(box), _outline(box, padding)
        if (np.any(padded < 0) or np.any(padded > scale)
                or cv2.intersectConvexConvex(core, paper)[0] > .20 * cv2.contourArea(core)
                or cv2.intersectConvexConvex(padded, paper)[0] > .28 * cv2.contourArea(padded)
                or any(cv2.intersectConvexConvex(padded, other)[0] > 1. for other in blockers)
                or not _locally_supported_paper_edge(page, paper, core)):
            return []
        marks.append({
            "region": violation,
            "bounds": {"x1": left / (width - 1), "y1": top / (height - 1),
                       "x2": right / (width - 1), "y2": bottom / (height - 1)},
            "claim_outline": [{"x": float(x), "y": float(y)} for x, y in padded / scale],
        })
    return marks
