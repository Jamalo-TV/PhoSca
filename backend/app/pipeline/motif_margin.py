"""Find source-supported markings a proposed motif would remove from light stock."""
from __future__ import annotations

from functools import partial

import cv2
import numpy as np

from app.pipeline.faint_margin import faint_margin_region
from app.pipeline.margin_stroke_shapes import isolated_margin_strokes


MOTIF_MARGIN_REVIEW_REVISION = 4


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


def motif_margin_marks(
    page: np.ndarray, full: np.ndarray, motif: np.ndarray, inverse: np.ndarray,
) -> list[np.ndarray]:
    """Return at most one marked paper region per side, in native page pixels."""
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
        if paper_lightness >= 191 and end - start >= 30 and max(depth) >= 8:
            faint_strips.append((lab, owned, project))
        background, labels, clusters = _mark_clusters(lightness, owned)
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
                break
    if not result and faint_strips:
        forward = np.linalg.inv(inverse)
        for lab, owned, project in faint_strips:
            region = faint_margin_region(page, lab, owned, project, forward, (width, height), motif)
            if region is not None:
                _add_region(result, region)
    return result
