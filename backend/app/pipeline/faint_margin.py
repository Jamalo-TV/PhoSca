"""Corroborate faint writing using paper on both sides of its original strokes."""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

_FLANK_STEPS = np.int32([(2, 0), (0, 2), (2, 2), (2, -2), (3, 0), (0, 3),
                         (3, 2), (3, -2), (0, 4), (4, 0), (4, 2), (4, -2)])
_FLANK_OFFSETS = np.vstack(([0, 0], _FLANK_STEPS, -_FLANK_STEPS))


def _glyph_groups(mask: np.ndarray, blue_ink: bool) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    _, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    height, width = mask.shape
    x, _, widths, heights, areas = stats[1:].T
    plausible = ((x > 0) & (x + widths < width) & (heights >= 3) & (heights < height)
                 & (widths >= 2) & (widths <= 3 * heights) & (widths >= 0.18 * heights)
                 & (areas >= 4) & (areas <= 2.7 * heights ** 2))
    ids = np.flatnonzero(plausible) + 1
    if not 3 <= len(ids) <= 180:
        return labels, []
    structured = np.zeros(len(stats), dtype=bool)
    for label in ids:
        left, top, glyph_width, glyph_height, area = stats[label]
        if area >= 0.73 * glyph_width * glyph_height:
            continue
        glyph = labels[top:top + glyph_height, left:left + glyph_width] == label
        indents = sum(np.count_nonzero(np.diff(glyph, axis=axis).sum(axis=axis) >= 2) for axis in (0, 1))
        structured[label] = indents >= max(2, 0.45 * min(glyph_width, glyph_height))

    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    bottoms = stats[:, cv2.CC_STAT_TOP] + heights
    areas = stats[:, cv2.CC_STAT_AREA]
    groups = set()
    for anchor in ids:
        scale = heights[anchor]
        dx, dy = (centers[ids] - centers[anchor]).T
        nearby = ((np.abs(dx) <= 4.7 * scale) & (np.abs(dy) <= scale)
                  & (heights[ids] >= 0.65 * scale) & (heights[ids] <= 1.48 * scale))
        for slope in (-0.18, -0.09, 0, 0.09, 0.18):
            line = ids[nearby & (np.abs(dy - slope * dx) <= 0.43 * scale)]
            line = line[np.argsort(centers[line, 0])]
            gaps = np.diff(centers[line, 0])
            for row in np.split(line, np.flatnonzero(gaps > 1.9 * scale) + 1):
                if len(row) < 3:
                    continue
                has_wide_glyph = np.any((widths[row] > 1.4 * heights[row]) & structured[row])
                if len(row) + has_wide_glyph < 4 or structured[row].sum() < (2 if blue_ink else 3):
                    continue
                span = centers[row, 0]
                baseline = bottoms[row] - slope * span
                low, high = np.quantile(baseline, (0.12, 0.88))
                if (np.median(np.diff(span)) >= 0.35 * scale and np.ptp(span) >= 1.8 * scale
                        and high - low <= 0.53 * scale and areas[row].sum() >= 1.6 * scale ** 2):
                    groups.add(tuple(row))
    return labels, sorted(groups, key=lambda group: (-len(group), group))


def _source_strokes_supported(
    page: np.ndarray, forward: np.ndarray, full_size: tuple[int, int], motif: np.ndarray,
    native: np.ndarray, contrast: np.ndarray, paper_lightness: np.ndarray, blue_ink: bool,
) -> bool:
    rounded = np.rint(native).astype(np.int32)
    height, width = page.shape[:2]
    x, y = rounded.T
    valid = (x >= 5) & (x < width - 5) & (y >= 5) & (y < height - 5)
    if np.count_nonzero(valid) < 12:
        return False
    points = rounded[valid, None, :] + _FLANK_OFFSETS
    pixels = cv2.cvtColor(page[points[..., 1], points[..., 0]], cv2.COLOR_BGR2LAB).astype(np.float32)
    projected = cv2.perspectiveTransform(points.reshape(1, -1, 2).astype(np.float32), forward).reshape(points.shape)
    # Both flanks must fall on the measured paper that would be discarded.
    # Bright samples on the album or inside the motif cannot confirm a stroke.
    on_stock = ((projected >= 1) & (projected <= np.asarray(full_size) - 2)).all(axis=-1)
    edges = np.roll(motif, -1, axis=0) - motif
    displacement = projected[..., None, :] - motif
    cross = edges[:, 0] * displacement[..., 1] - edges[:, 1] * displacement[..., 0]
    inside_motif = (cross >= 0).all(axis=-1) | (cross <= 0).all(axis=-1)
    on_stock &= ~inside_motif
    count = len(_FLANK_STEPS)
    ink = pixels[:, :1]
    first, last = pixels[:, 1:1 + count], pixels[:, 1 + count:]
    left_light, right_light = first[..., 0], last[..., 0]
    similar_paper = ((np.linalg.norm(first[..., 1:] - last[..., 1:], axis=2) <= 5)
                     & (np.abs(left_light - right_light) <= 10)
                     & (0.5 * (left_light + right_light) >= paper_lightness[valid, None] - 10))
    drop = np.minimum(left_light, right_light) - ink[..., 0]
    minimum_drop = np.minimum(contrast[valid, None] * 0.55, 17)
    if blue_ink:
        color_shift = np.minimum(first[..., 2], last[..., 2]) - ink[..., 2]
        supported = (color_shift >= 6) & (drop >= minimum_drop * 0.64)
    else:
        paper_color = 0.5 * (first[..., 1:] + last[..., 1:])
        neutral = np.linalg.norm(paper_color - ink[..., 1:], axis=2) <= 4
        supported = neutral & (drop >= minimum_drop)
    eligible = on_stock[:, :1] & on_stock[:, 1:1 + count] & on_stock[:, 1 + count:]
    return (eligible & similar_paper & supported).any(axis=1).mean() >= (0.35 if blue_ink else 0.46)


def faint_margin_region(
    page: np.ndarray, lab: np.ndarray, owned: np.ndarray, project_to_source: Callable[[np.ndarray], np.ndarray],
    forward: np.ndarray, full_size: tuple[int, int], motif: np.ndarray,
) -> np.ndarray | None:
    if lab.shape[0] < 6 or np.count_nonzero(owned) < 150:
        return None
    paper = np.median(lab[owned], axis=0)
    if paper[0] < 191:
        return None
    filled = np.where(owned[..., None], lab, paper).astype(np.uint8)
    broad = int(np.clip(lab.shape[0] // 3 * 2 + 1, 7, 31))
    narrow = int(np.clip((lab.shape[0] // 3) | 1, 5, 11))
    lightness = lab[:, :, 0].astype(np.float32)
    paper_lightness = cv2.morphologyEx(filled[:, :, 0], cv2.MORPH_CLOSE, np.ones((broad, broad + 2), np.uint8))
    local_lightness = cv2.morphologyEx(filled[:, :, 0], cv2.MORPH_CLOSE, np.ones((narrow, narrow), np.uint8))
    contrast = paper_lightness - lightness
    narrow_stroke = local_lightness - lightness >= np.maximum(9, 0.58 * contrast)
    red_shift = lab[:, :, 1].astype(np.float32) - cv2.medianBlur(filled[:, :, 1], broad)
    blue_shift = lab[:, :, 2].astype(np.float32) - cv2.medianBlur(filled[:, :, 2], broad)
    on_pale_paper = owned & narrow_stroke & (paper_lightness >= 196)
    masks = (
        on_pale_paper & (contrast >= 16) & (np.hypot(red_shift, blue_shift) <= 3),
        on_pale_paper & (contrast >= 13) & (blue_shift <= -6) & (np.abs(red_shift) <= 0.85 * np.abs(blue_shift) + 2),
    )
    for blue_ink, mask in zip((False, True), masks, strict=True):
        labels, groups = _glyph_groups(mask, blue_ink)
        for group in groups:
            rows, columns = np.nonzero(np.isin(labels, group))
            stride = max(1, len(rows) // 230)
            source_rows, source_columns = rows[::stride], columns[::stride]
            native = project_to_source(np.column_stack((source_columns, source_rows)))
            if _source_strokes_supported(page, forward, full_size, motif, native,
                                         contrast[source_rows, source_columns],
                                         paper_lightness[source_rows, source_columns], blue_ink):
                left, right = columns.min(), columns.max()
                top, bottom = rows.min(), rows.max()
                return project_to_source(np.array([[left, top], [right, top], [right, bottom], [left, bottom]]))
    return None
