"""Find source-supported markings a proposed motif would remove from light stock."""
from __future__ import annotations

from functools import partial

import cv2
import numpy as np

from app.pipeline.border_removal import CURVED_IMAGE_BOUNDARY, border_chord_quad, find_physical_border
from app.pipeline.faint_margin import faint_margin_region
from app.pipeline.margin_stroke_shapes import isolated_margin_strokes


MOTIF_MARGIN_REVIEW_REVISION = 7
PAPER_STOCK_MOTIF_SOURCE = "paper_stock_motif"
HELD_PRINT_MOTIF_MARGIN_SOURCE = "held_print_motif_margin"


def _mark_clusters(lightness: np.ndarray, owned: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    height, width = lightness.shape
    kernel_size = int(np.clip(height // 3 * 2 + 1, 5, 31))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size + 2, kernel_size))
    # Fill pixels outside the measured strip before estimating its paper tone.
    # The motif seam and the album must not contribute to the local contrast.
    paper = np.where(owned, lightness, np.median(lightness[owned])).astype(np.uint8)
    background = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel)
    dark = owned & (background >= 172) & (background.astype(np.int16) - lightness >= 38)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    x, _, widths, heights, areas = stats[1:].T
    keep = ((x > 0) & (x + widths < width) & (heights >= 3) & (heights < height)
            & (widths <= 6 * heights) & (areas >= 3) & (areas <= 0.18 * width * height)
            & (areas >= 0.13 * widths * heights))
    ids = np.flatnonzero(keep) + 1
    clusters = []
    if 3 <= len(ids) <= 300:
        rows = centers[ids]
        block_heights = stats[ids, cv2.CC_STAT_HEIGHT]
        for (cx, cy), stroke_height in zip(rows, block_heights, strict=True):
            block = ids[(np.abs(rows[:, 1] - cy) <= 0.7 * stroke_height)
                        & (np.abs(rows[:, 0] - cx) <= 9 * stroke_height)
                        & (block_heights >= 0.35 * stroke_height) & (block_heights <= 2.5 * stroke_height)]
            if len(block) < 3:
                continue
            _, _, bw, bh, area = stats[block].T
            # A run of uniformly shaped spots or hairline scratches provides little
            # evidence of a printed mark. Look for a wider stroke or joined letters.
            if np.any(bw > 1.5 * bh) and area.sum() > 2.6 * stroke_height ** 2:
                clusters.append(block)
    if clusters:
        return background, labels, sorted(clusters, key=len, reverse=True)
    return background, labels, isolated_margin_strokes(labels, stats, owned)


def _to_source(points: np.ndarray, origin: np.ndarray, along: np.ndarray, down: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    plane = origin + points[:, :1] * along + points[:, 1:] * down
    return cv2.perspectiveTransform(plane[None].astype(np.float32), inverse)[0]


def _add_region(regions: list[np.ndarray], region: np.ndarray) -> None:
    area = cv2.contourArea(region)
    if area > 0 and not any(cv2.intersectConvexConvex(region, previous)[0]
                            > 0.5 * min(area, cv2.contourArea(previous)) for previous in regions):
        regions.append(region)


def _unexplained_margin_mark(lab: np.ndarray, owned: np.ndarray) -> np.ndarray | None:
    """Flag a compact deviation between similar paper flanks for preview review."""
    values = lab.astype(np.int16)
    marked = np.zeros(owned.shape, bool)
    for dx, dy in ((2, 0), (0, 2), (2, 2), (2, -2), (3, 0), (0, 3), (3, 2), (3, -2)):
        before = np.roll(values, (dy, dx), axis=(0, 1))
        after = np.roll(values, (-dy, -dx), axis=(0, 1))
        valid = owned & np.roll(owned, (dy, dx), axis=(0, 1)) & np.roll(owned, (-dy, -dx), axis=(0, 1))
        if dy:
            valid[:abs(dy)] = valid[-abs(dy):] = False
        if dx:
            valid[:, :dx] = valid[:, -dx:] = False
        difference = before - after
        change = (before + after) * .5 - values
        marked |= (valid & (np.minimum(before[..., 0], after[..., 0]) >= 185)
                   & (np.abs(difference[..., 0]) <= 8) & (np.linalg.norm(difference[..., 1:], axis=2) <= 6)
                   & (change[..., 0] >= 6) & (np.linalg.norm(change, axis=2) >= 20))
    if np.count_nonzero(marked) < 20:
        return None
    _, _, stats, _ = cv2.connectedComponentsWithStats(marked.astype(np.uint8))
    x, y, width, height, area = max(stats[1:], key=lambda component: component[cv2.CC_STAT_AREA])
    if area < 12:
        return None
    return np.float32([[x - .5, y - .5], [x + width - .5, y - .5],
                       [x + width - .5, y + height - .5], [x - .5, y + height - .5]])


def _rule_in_margin(lightness: np.ndarray, owned: np.ndarray) -> np.ndarray | None:
    """Find a thin continuous mark between two similar paper shoulders."""
    values = lightness.astype(np.int16)
    before, line, after = values[:-4], values[2:-2], values[4:]
    paper = np.minimum(before, after)
    dark = (owned[:-4] & owned[2:-2] & owned[4:] & (paper >= 185)
            & (abs(before - after) <= 7) & (paper - line >= 4))
    if not dark.any():
        return None
    _, _, components, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    for x, y, width, height, area in components[1:]:
        if width >= max(36, .15 * lightness.shape[1]) and height <= 4 and area >= .75 * width:
            # Row zero in the contrast map refers to row two of the strip.
            y += 2
            return np.float32([[x - .5, y - .5], [x + width - .5, y - .5],
                               [x + width - .5, y + height - .5], [x - .5, y + height - .5]])
    return None


def motif_margin_marks(
    page: np.ndarray, full: np.ndarray, motif: np.ndarray, inverse: np.ndarray,
    *, preview_guard: bool = False, inspect_rules: bool = False,
) -> list[np.ndarray]:
    """Return source coordinates for markings in the proposed removed paper strips."""
    if motif.shape != (4, 2) or not np.isfinite(motif).all():
        return []
    height, width = full.shape[:2]
    axes = (
        ((0, 0), (1, 0), (0, 1)),
        ((width - 1, 0), (0, 1), (-1, 0)),
        ((width - 1, height - 1), (-1, 0), (0, -1)),
        ((0, height - 1), (0, -1), (1, 0)),
    )
    result = []
    faint_strips = []
    preview_strips = []
    for edge, axes_for_side in enumerate(axes):
        origin, along, down = np.asarray(axes_for_side)
        plane = np.rot90(full, edge)
        segment = motif[[edge, (edge + 1) % 4]] - origin
        start, end = segment @ along
        depth = segment @ down
        if end - start < 20 or max(depth) < 7:
            continue
        # Include the paper corners in both adjoining strip orientations.
        # A date near a corner must survive a quarter turn of the same source.
        collar = max(1, min(3, round(min(height, width) * 0.0025)))
        x0, x1 = collar, plane.shape[1] - collar
        y0 = collar
        y1 = min(plane.shape[0], int(np.ceil(max(depth))) - 1)
        if x1 - x0 < 20 or y1 - y0 < 5:
            continue
        seam = np.interp(np.arange(x0, x1), (start, end), depth)
        owned = np.arange(y0, y1)[:, None] <= seam - 1.5
        if np.count_nonzero(owned) < 100:
            continue
        lab = cv2.cvtColor(plane[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0]
        paper_lightness = np.median(lightness[owned])
        if paper_lightness < 172:
            continue
        project = partial(_to_source, origin=origin + x0 * along + y0 * down, along=along, down=down, inverse=inverse)
        if inspect_rules and (rule := _rule_in_margin(lightness, owned)) is not None:
            _add_region(result, project(rule))
        background, labels, clusters = _mark_clusters(lightness, owned)
        strong_box = None
        checked = set()
        for block in clusters:
            key = tuple(block)
            if key in checked:
                continue
            checked.add(key)
            rows, cols = np.nonzero(np.isin(labels, block))
            stride = max(1, len(rows) // 240)
            native = project(np.column_stack((cols[::stride], rows[::stride])))
            nx, ny = np.rint(native).astype(int).T
            valid = (nx >= 0) & (nx < page.shape[1]) & (ny >= 0) & (ny < page.shape[0])
            native_bgr = page[ny[valid], nx[valid]].reshape(-1, 1, 3)
            if len(native_bgr) < 10:
                continue
            native_lightness = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2LAB)[:, 0, 0]
            stock = background[rows[::stride][valid], cols[::stride][valid]].astype(np.int16)
            if np.count_nonzero(stock - native_lightness >= 32) < 0.7 * len(native):
                continue
            left, right = cols.min(), cols.max()
            top, bottom = rows.min(), rows.max()
            if bottom > top and right > left:
                box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
                _add_region(result, project(box))
                strong_box = (left, top, right, bottom)
                break
        if paper_lightness >= 191 and end - start >= 30 and max(depth) >= 8:
            faint_strips.append((lab, owned, project, background, strong_box))
        if preview_guard:
            preview_strips.append((lab, owned, project))
    if faint_strips:
        already_marked = bool(result)
        forward = np.linalg.inv(inverse)
        for lab, owned, project, background, strong_box in faint_strips:
            if already_marked:
                if strong_box is not None:
                    left, top, right, bottom = strong_box
                    # Keep a halo clear so the same mark cannot seed a faint group.
                    owned = owned.copy()
                    owned[max(0, top - 4):bottom + 5, max(0, left - 4):right + 5] = False
                if np.count_nonzero(owned & (background.astype(np.int16) - lab[:, :, 0] >= 9)) < 15:
                    continue
            region = faint_margin_region(page, lab, owned, project, forward, (width, height), motif,
                                         extend_dashed_note=already_marked)
            if region is not None:
                _add_region(result, region)
    if not result:
        for lab, owned, project in preview_strips:
            box = _unexplained_margin_mark(lab, owned)
            if box is not None:
                _add_region(result, project(box))
    return result


def held_print_motif_margin_marks(
    page: np.ndarray, full: np.ndarray, matrix: np.ndarray, paper: np.ndarray, visible: np.ndarray,
    neighbors: list[np.ndarray],
) -> list[np.ndarray]:
    """Locate corroborated marks for review before a protected paper crop can be approved."""
    height, width = full.shape[:2]
    if min(width, height) < 900 or not cv2.isContourConvex(visible):
        return []
    # Use the observed border for locating the discard strip. It supplies no
    # crop geometry and cannot override the held mode's protection decision.
    plan = find_physical_border(full)
    if CURVED_IMAGE_BOUNDARY not in plan.reason_codes or not all(edge.is_physical_border for edge in plan.edges.values()):
        return []
    inverse = np.linalg.inv(matrix)
    motif = border_chord_quad(plan.edges, width, height)
    candidates = motif_margin_marks(page, full, motif, inverse)
    if not candidates:
        return []

    resized_size = np.maximum(1, np.rint(np.array([width, height]) * 800 / max(width, height))).astype(int)
    scale = resized_size / np.array([width, height])
    to_small = np.eye(3)
    to_small[[0, 1], [0, 1]] = scale
    to_small[:2, 2] = (scale - 1) / 2
    reduced = cv2.resize(full, tuple(resized_size), interpolation=cv2.INTER_AREA)
    reduced_motif = cv2.perspectiveTransform(motif[None], to_small)[0]
    repeats = motif_margin_marks(page, reduced, reduced_motif, inverse @ np.linalg.inv(to_small))
    inset = max(4., .012 * np.linalg.norm(np.roll(paper, -1, axis=0) - paper, axis=1).min())
    peers = [cv2.convexHull(neighbor) for neighbor in neighbors]
    confirmed = []
    for candidate in candidates:
        if (any(cv2.pointPolygonTest(outline, tuple(point), True) < inset
                for outline in (paper, visible) for point in candidate)
                or any(cv2.intersectConvexConvex(candidate, peer)[0] > 1. for peer in peers)):
            continue
        for repeat in repeats:
            intersection, _ = cv2.intersectConvexConvex(candidate, repeat)
            union = cv2.contourArea(candidate) + cv2.contourArea(repeat) - intersection
            if union > 0 and intersection >= .80 * union:
                confirmed.append(candidate)
                break
    return confirmed
