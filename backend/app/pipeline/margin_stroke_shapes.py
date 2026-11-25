"""Recognize narrow stroke patterns in a discarded paper strip."""

import cv2
import numpy as np


def _interior_holes(component: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = component.shape[0]
    background = 1 - np.pad(component, 1)
    _, _, stats, centers = cv2.connectedComponentsWithStats(background, connectivity=4)
    # Component 1 is the padded exterior, even when the ink touches its box.
    _, _, widths, heights, areas = stats[2:].T
    keep = ((areas >= max(2, .014 * height ** 2)) & (areas <= .32 * height ** 2)
            & (widths >= 2) & (heights >= 2))
    return centers[2:][keep] - 1, areas[keep]


def _has_regular_strokes(component: np.ndarray) -> bool:
    distance = cv2.distanceTransform(np.pad(component, 2), cv2.DIST_L2, 5)
    peak = distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 1e-4
    radii = distance[peak & (distance > 0)]
    if len(radii) < 3:
        return False
    low, typical, high = np.percentile(radii, [10, 50, 90])
    return bool(.8 <= typical <= .23 * min(component.shape) and low >= .55 * typical and high <= 2.25 * typical)


def _has_aligned_stem(component: np.ndarray, centers: np.ndarray) -> bool:
    height, width = component.shape
    if len(centers) < 2 or width > 1.20 * height or height > 2.3 * width:
        return False
    if np.ptp(centers[:, 1]) < .28 * height or np.ptp(centers[:, 0]) > .24 * width:
        return False
    middle = slice(int(.08 * height), int(.94 * height))
    for facing in (component, component[:, ::-1]):
        side = facing[middle, :max(1, round(.20 * width))]
        seen = side.any(axis=1)
        if seen.mean() < .94 or min(section.mean() for section in np.array_split(seen, 5)) < .80:
            continue
        if side.mean(axis=0).max() >= .78:
            low, high = np.percentile(np.argmax(facing[middle], axis=1)[seen], [10, 90])
            if high - low <= max(.9, .047 * width):
                return True
    return False


def _has_apex_and_legs(component: np.ndarray, center: np.ndarray, area: int) -> bool:
    height, width = component.shape
    x, y = center
    if not (.48 <= width / height <= 1.20 and .34 <= x / width <= .66 and .33 <= y / height <= .70
            and .06 <= area / (height * width) <= .24):
        return False
    top_width = component[round(.11 * height):round(.28 * height)].sum(axis=1).mean()
    gaps = []
    for row in component[round(.74 * height):round(.95 * height)]:
        ink = np.flatnonzero(row)
        gaps.append(len(ink) >= 2 and ink[0] <= .26 * width and ink[-1] >= .74 * width
                    and np.max(np.diff(ink)) >= .24 * width)
    return bool(np.mean(gaps) >= .66 and top_width <= .40 * width)


def _has_spaced_counters(component: np.ndarray, centers: np.ndarray) -> bool:
    height, width = component.shape
    if len(centers) < 3 or width < 1.45 * height:
        return False
    columns = centers[:, 0]
    return bool(np.ptp(columns) >= .52 * height and len(np.unique(columns // max(1, .24 * height))) >= 3)


def isolated_margin_strokes(labels: np.ndarray, stats: np.ndarray, owned: np.ndarray) -> list[np.ndarray]:
    """Find connected handwriting or enclosed glyphs with a visible local collar."""
    if len(stats) > 301:
        return []
    height, width = labels.shape
    clusters = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], start=1):
        if min(w, h) < 6 or max(w, h) > 8 * min(w, h) or not .09 <= area / (w * h) <= .68:
            continue
        if x < 1 or y < 1 or x + w >= width or y + h >= height:
            continue
        collar = np.s_[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
        local = labels[collar]
        if not owned[collar].all() or np.mean((local != 0) & (local != index)) >= .12:
            continue
        component = np.uint8(labels[y:y + h, x:x + w] == index)
        if not _has_regular_strokes(component):
            continue
        for turn in range(4):
            rotated = np.rot90(component, turn)
            centers, areas = _interior_holes(rotated)
            if (_has_spaced_counters(rotated, centers)
                    or _has_aligned_stem(rotated, centers)
                    or (len(centers) == 1 and _has_apex_and_legs(rotated, centers[0], areas[0]))):
                clusters.append(np.array([index]))
                break
    return clusters
