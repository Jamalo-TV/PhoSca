"""Find protected-region suggestions that a reviewer can examine at a paper edge."""

import cv2
import numpy as np

from app.pipeline.merged_print_review import _ribbed_lane, _stock_side
from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line
from app.pipeline.protected_regions import is_advisory_page_face, protected_region_key, region_box_in_pixels


OFF_PRINT_ADVISORY_REVISION = 2
MANUAL_MARGIN_CONTEXT = "outside_manual_print"


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


def _separating_rib(page: np.ndarray, edge: np.ndarray, outward: np.ndarray, core: np.ndarray) -> np.ndarray | None:
    # Look for a second paper transition across the exposed backing, independently of a neighbor mask.
    distance = (core - edge[0]) @ outward
    if not 9 <= distance.min() <= 26:
        return None
    band = _edge_band(page, *edge, [-9, -7, -5, 4, 6, 9])
    if band is None:
        return None
    backing, print_colors = np.median(band[:, :3], axis=1), np.median(band[:, 3:], axis=1)
    palette = np.median(backing[:, 1:], axis=0)
    if (palette[1] < 134 or np.mean(np.linalg.norm(backing[:, 1:] - palette, axis=1) <= 8) < .90
            or np.mean(np.linalg.norm(backing - print_colors, axis=1) >= 18) < .72):
        return None
    facing = _stock_side(page, *(edge[::-1] + distance.min() * outward), palette, 10)
    if facing is None:
        return None
    gap = (facing[::-1] - edge) @ outward
    if gap.min() < 8 or gap.max() > min(30, 2 * gap.min()):
        return None
    step = facing[1] - facing[0]
    neighbor_inward = np.float32([-step[1], step[0]]) / np.linalg.norm(step)
    if ((core - facing[0]) @ neighbor_inward).min() < -2:
        return None
    lane = (edge + facing[::-1]) / 2
    center_band = _edge_band(page, *lane, [-3, 0, 3])
    if center_band is None or np.mean(np.linalg.norm(center_band[..., 1:] - palette, axis=2) <= 8) < .90:
        return None
    for offset in (0, 2, -2, 4, -4):
        rib = _ribbed_lane(page, lane + offset * outward, palette)
        if (rib is not None and ((rib - edge[0]) @ outward).min() >= 3
                and ((rib - facing[0]) @ neighbor_inward).max() <= -3):
            return rib
    return None


def manual_margin_advisory(
    page: np.ndarray, paper: np.ndarray, regions: list[dict], violations: list[dict], neighbors: list[np.ndarray],
) -> list[dict]:
    """Offer a separately reviewed margin conflict for a manually drawn print."""
    if len(violations) != 1 or not cv2.isContourConvex(paper):
        return []
    violation, = violations
    matches = [region for region in regions if protected_region_key(region) == violation]
    if len(matches) != 1 or not is_advisory_page_face(matches[0]):
        return []
    height, width = page.shape[:2]
    box = region_box_in_pixels(matches[0], width, height)
    if box is None or not np.isfinite(box).all():
        return []
    left, top, right, bottom = box
    size = max(right - left, bottom - top)
    margin = float(matches[0].get("margin_ratio", .08)) * size
    if not np.isfinite(margin) or not 0 < margin <= .18 * size:
        return []
    core, padded = _outline(box), _outline(box, margin)
    scale = np.float32([width - 1, height - 1])
    overlap = cv2.intersectConvexConvex(padded, paper)[0]
    if (np.any(padded < 0) or np.any(padded > scale)
            or not 0 < overlap <= .06 * cv2.contourArea(padded)):
        return []
    for region in regions:
        if region is matches[0] or not (other := region_box_in_pixels(region, width, height)):
            continue
        pad = float(region.get("margin_ratio", .08)) * max(other[2] - other[0], other[3] - other[1])
        if not np.isfinite([*other, pad]).all() or pad < 0:
            return []
        outline = _outline(other, pad)
        area = cv2.intersectConvexConvex(outline, paper)[0]
        if 0 < area < cv2.contourArea(outline) - 1.e-3:
            return []
    if cv2.contourArea(paper, oriented=True) < 0:
        paper = paper[::-1].copy()
    edges = np.roll(paper, -1, axis=0) - paper
    lengths = np.linalg.norm(edges, axis=1)
    tangents = edges / lengths[:, None]
    inward = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    depths = np.einsum("ijk,ik->ij", core[None] - paper[:, None], inward)
    outward = np.flatnonzero(depths.max(axis=1) <= -max(6., .10 * size))
    if len(outward) != 1 or np.any(np.delete(depths, outward, axis=0) < 0):
        return []
    side, = outward
    positions = (core - paper[side]) @ tangents[side]
    limits = [positions.min() - 18, positions.max() + 18]
    if limits[0] < 5 or limits[1] > lengths[side] - 5:
        return []
    start, end = paper[side] + np.float32(limits)[:, None] * tangents[side]
    line, evidence = _side_line(page, start, end, 11)
    if line is None or evidence["ambiguous"] or min(evidence["support"], evidence["start_support"], evidence["end_support"]) < .90:
        return []
    fit = np.float32(_line_endpoints(start, end, line))
    offset = np.max(np.abs((fit - [start, end]) @ inward[side]))
    if offset > 5.5:
        return []
    direction = (fit[1] - fit[0]) / np.linalg.norm(fit[1] - fit[0])
    rib = _separating_rib(page, fit, np.float32([direction[1], -direction[0]]), core)
    if rib is None:
        return []
    corridor = cv2.convexHull(np.vstack([fit, rib]))
    if len(corridor) != 4 or any(cv2.intersectConvexConvex(corridor, cv2.convexHull(neighbor))[0] > 1 for neighbor in neighbors):
        return []
    return [{
        "region": violation,
        "bounds": {"x1": left / (width - 1), "y1": top / (height - 1),
                   "x2": right / (width - 1), "y2": bottom / (height - 1)},
        "claim_outline": [{"x": float(x), "y": float(y)} for x, y in corridor[:, 0] / scale],
        "padded_outline": [{"x": float(x), "y": float(y)} for x, y in padded / scale],
        "paper_edge": [{"x": float(x), "y": float(y)} for x, y in fit / scale],
        "requires_edge_alignment": bool(offset > 2),
    }]


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
