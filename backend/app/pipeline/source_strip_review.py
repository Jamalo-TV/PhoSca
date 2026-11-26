"""Locate a separate visible picture strip next to a held crop for source review."""

import cv2
import numpy as np


SOURCE_STRIP_REVIEW_REVISION = 1
_SIDES = ("top", "right", "bottom", "left")


def _paper_to_picture_edge(lightness: np.ndarray, columns: np.ndarray, dark_top: np.ndarray, unit: int) -> np.ndarray | None:
    sample = columns[::4]
    depths = np.arange(round(.055 * unit), round(.177 * unit))
    slopes = np.linspace(-.07, .07, 21)
    relative = sample - sample.mean()
    rows = np.rint(np.polyval(dark_top, sample) - depths[None, :, None]
                   + slopes[:, None, None] * relative).astype(int)
    if rows.min() < 3 or rows.max() + 3 >= len(lightness):
        return None
    support = np.mean(lightness[rows - 3, sample] - lightness[rows + 3, sample] >= 20, axis=2)
    slope_index, depth_index = np.unravel_index(np.argmax(support), support.shape)
    if support[slope_index, depth_index] < .69:
        return None
    slope = slopes[slope_index]
    return dark_top + [slope, -depths[depth_index] - slope * sample.mean()]


def _supported_strip(
    lab: np.ndarray, component: np.ndarray, x: int, width: int, unit: int,
) -> np.ndarray | None:
    lightness = lab[..., 0]
    columns = np.arange(x + round(.14 * width), x + round(.86 * width))
    body = component[:, columns]
    if not body.any(axis=0).all():
        return None
    top = body.argmax(axis=0)
    bottom = len(lightness) - 1 - body[::-1].argmax(axis=0)
    dark_top = np.polyfit(columns, top, 1)
    dark_bottom = np.polyfit(columns, bottom, 1)
    top_rows = np.polyval(dark_top, columns)
    bottom_rows = np.polyval(dark_bottom, columns)
    if (np.percentile(abs(top_rows - top), 90) > .025 * unit
            or np.percentile(abs(bottom_rows - bottom), 90) > .018 * unit
            or abs(dark_top[0] - dark_bottom[0]) > .085):
        return None
    picture_edge = _paper_to_picture_edge(lightness, columns, dark_top, unit)
    if picture_edge is None:
        return None

    picture_rows = np.polyval(picture_edge, columns)
    above = np.rint(picture_rows - .045 * unit).astype(int)
    below = np.rint((picture_rows + top_rows) / 2).astype(int)
    after = np.rint(bottom_rows + .025 * unit).astype(int)
    if above.min() < 0 or after.max() >= len(lightness):
        return None
    paper = lab[above, columns]
    middle = np.median(lightness[below, columns])
    dark = np.median(lightness[np.rint((top_rows + bottom_rows) / 2).astype(int), columns])
    pale = np.median(paper[:, 0])
    following = np.median(lightness[after, columns])
    if (pale < 123 or pale - middle < 24 or middle - dark < 44 or following - dark < 70
            or abs(following - pale) > 27 or np.mean(paper[:, 0] >= 120) < .75):
        return None

    lateral = np.arange(-round(.032 * unit), round(.032 * unit) + 1)
    left, right = x + lateral, x + width - 1 + lateral
    if left.min() < 3 or right.max() + 3 >= lightness.shape[1]:
        return None
    rows = np.rint(np.linspace(picture_rows.mean() + .022 * unit, top_rows.mean() - .022 * unit, 20)).astype(int)[:, None]
    left_step = lightness[rows, left - 3] - lightness[rows, left + 3]
    right_step = lightness[rows, right + 3] - lightness[rows, right - 3]
    if min(np.mean(left_step.max(axis=1) >= 22), np.mean(right_step.max(axis=1) >= 22)) < .70:
        return None
    gap_rows = np.rint(np.linspace(.30, .72, 8) * picture_rows.mean()).astype(int)
    gap = lab[gap_rows[:, None], columns].reshape(-1, 3)
    flank = np.concatenate((lab[gap_rows, left.min()], lab[gap_rows, right.max()]))
    if np.linalg.norm(np.median(gap, axis=0) - np.median(flank, axis=0)) > 15:
        return None
    high_pass = abs(lightness - cv2.GaussianBlur(lightness, (0, 0), 1.2))
    samples = below[None, :] + np.arange(-round(.024 * unit), round(.024 * unit) + 1)[:, None]
    if samples.min() < 0 or samples.max() >= len(lightness) or np.percentile(high_pass[samples, columns], 75) < 1.7:
        return None

    # The marker is inside the visible textured band. The dark backing, the
    # gap and the pixels of the earlier crop are left outside the marker.
    first = x + max(2, round(.025 * unit))
    last = x + width - 1 - max(2, round(.025 * unit))
    starts = np.polyval(picture_edge, [first, last]) + .012 * unit
    ends = np.polyval(dark_top, [first, last]) - .008 * unit
    if np.min(ends - starts) < .04 * unit:
        return None
    return np.float32([[first, starts[0]], [last, starts[1]], [last, ends[1]], [first, ends[0]]])


def _strips_at_edge(page: np.ndarray, start: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
    length = float(np.linalg.norm(end - start))
    if length < 150:
        return []
    unit = min(500, round(length))
    tangent = (end - start) / length
    outward = np.float32([-tangent[1], tangent[0]])
    offset = round(.20 * unit)
    columns = np.arange(-offset, round(1.26 * unit), dtype=np.float32) * length / unit
    rows = np.arange(round(.53 * unit), dtype=np.float32) * length / unit
    points = start + columns[None, :, None] * tangent + rows[:, None, None] * outward
    if np.any((points < 0) | (points > np.array(page.shape[1::-1]) - 1)):
        return []
    patch = cv2.remap(page, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper = lab[round(.06 * unit):round(.13 * unit), offset + round(.15 * unit):offset + round(.85 * unit), 0]
    median = np.median(paper)
    if median < 123:
        return []
    dark = np.uint8(lab[..., 0] <= min(32, .25 * median))
    dark[:round(.20 * unit)] = 0
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    findings = []
    for index, (x, y, width, height, area) in enumerate(stats[1:], 1):
        if not (.82 * unit < width < 1.22 * unit and .065 * unit < height < .20 * unit
                and .23 * unit < y < .39 * unit and area > .68 * width * height
                and abs(x + width / 2 - offset - .5 * unit) < .15 * unit):
            continue
        outline = _supported_strip(lab, labels == index, x, width, unit)
        if outline is not None:
            scaled = (outline - np.float32([offset, 0])) * length / unit
            findings.append(start + scaled[:, :1] * tangent + scaled[:, 1:] * outward)
    return findings


def source_strip_outline(
    page: np.ndarray, outline: np.ndarray, boundary: dict, neighbors: list[np.ndarray],
) -> np.ndarray | None:
    """Return one visible, unclaimed strip without deciding which print owns it."""
    candidates = []
    for index, side in enumerate(_SIDES):
        evidence = boundary.get(side) or {}
        if (evidence.get("support", 0.) < .90 or evidence.get("ambiguous")
                or (evidence.get("corner_support") or 0.) < .85):
            continue
        # Reverse the side so the left-hand normal points out of this crop.
        candidates.extend(_strips_at_edge(page, outline[(index + 1) % 4], outline[index]))
    if len(candidates) != 1:
        return None
    region, = candidates
    if any(cv2.intersectConvexConvex(region, cv2.convexHull(neighbor))[0] > 1. for neighbor in neighbors):
        return None
    return region
