"""Measure two possible paper outlines within a merged detector proposal."""

from itertools import product

import cv2
import numpy as np

from app.pipeline.multipanel_prints import recover_shared_stock_outline
from app.pipeline.print_edges import _edge_band

MERGED_PRINT_PAIR_REVIEW = "MERGED_PRINT_PAIR_REVIEW"
MERGED_PRINT_PAIR = "merged_print_pair"


def _facing_sides(first: np.ndarray, second: np.ndarray) -> tuple[int, int] | None:
    vectors = [np.roll(quad, -1, axis=0) - quad for quad in (first, second)]
    steps = [delta / np.linalg.norm(delta, axis=1)[:, None] for delta in vectors]
    centers = [quad + delta / 2 for quad, delta in zip((first, second), vectors)]
    matches = []
    for i, j in product(range(4), repeat=2):
        if steps[0][i] @ steps[1][j] > -.95:
            continue
        gap = centers[1][j] - centers[0][i]
        inward = np.float32([-steps[0][i][1], steps[0][i][0]])
        if not -45 <= gap @ inward <= 5 or abs(gap @ steps[0][i]) > .25 * np.linalg.norm(vectors[0][i]):
            continue
        matches.append((i, j))
    return matches[0] if len(matches) == 1 else None


def _lane_center(pair: list[np.ndarray], facing: tuple[int, int]) -> np.ndarray:
    a, b = [quad[[i, (i + 1) % 4]] for quad, i in zip(pair, facing)]
    return (a + b[::-1]) / 2


def _stock_palette(image: np.ndarray, lane: np.ndarray) -> np.ndarray | None:
    a, b = lane
    step = 16 / np.linalg.norm(b - a)
    samples = _edge_band(image, a, b, range(-7, 8), fractions=np.float32([-step, 1 + step]))
    if samples is None:
        return None
    colors = np.median(samples[..., 1:], axis=1)
    if np.linalg.norm(colors[0] - colors[1]) > 8 or colors[:, 1].min() < 135:
        return None
    return colors.mean(axis=0)


def _stock_side(image: np.ndarray, a: np.ndarray, b: np.ndarray, backing: np.ndarray, limit: int) -> np.ndarray | None:
    length = np.linalg.norm(b - a)
    offsets = np.arange(-limit - 9, limit + 10, dtype=np.float32)
    fractions = np.linspace(.10, .90, 96, dtype=np.float32)
    lab = _edge_band(image, a, b, offsets, fractions=fractions)
    if lab is None:
        return None
    width = 2 * limit + 1
    centers = np.arange(width) + 9
    outside = np.median(lab[:, centers[:, None] + [-9, -7, -5]], axis=2)
    inside = np.median(lab[:, centers[:, None] + [4, 6, 9]], axis=2)
    stock = ((outside[..., 0] >= 145) & (np.linalg.norm(outside[..., 1:] - backing, axis=2) <= 10)
             & (outside[..., 2] - inside[..., 2] >= 5))
    signed = np.clip((lab[:, centers - 2, 2] - lab[:, centers + 2, 2]) / 10, 0, 1)
    gradient = np.clip(np.linalg.norm(lab[:, centers - 1] - lab[:, centers + 1], axis=2) / 24, 0, 1)
    shifts, slopes = (grid.ravel() for grid in np.meshgrid(np.arange(-limit, limit + 1), np.linspace(-.128, .128, 65)))
    displacement = shifts[:, None] + slopes[:, None] * (fractions - .5) * length
    fits = np.abs(displacement).max(axis=1) <= limit
    shifts, slopes, displacement = shifts[fits], slopes[fits], displacement[fits]
    rank = np.rint(displacement).astype(np.intp) + limit
    rows = np.arange(len(fractions))[None, :]
    support = stock[rows, rank]
    coverage = np.min([part.mean(axis=1) for part in np.array_split(support, 4, axis=1)], axis=0)
    total = support.mean(axis=1)
    scores = (.52 * total + .22 * signed[rows, rank].mean(axis=1)
              + .15 * (gradient[rows, rank] * support).mean(axis=1) + .11 * coverage
              - .025 * np.abs(displacement).mean(axis=1) / limit)
    best = int(np.argmax(scores))
    if total[best] < .80 or coverage[best] < .55:
        return None
    direction = (b - a) / length
    normal = np.float32([-direction[1], direction[0]])
    return np.float32(np.stack((a, b)) + (shifts[best] + slopes[best] * np.array([-.5, .5]) * length)[:, None] * normal)


def _fit_print(image: np.ndarray, seed: np.ndarray, stock: np.ndarray) -> np.ndarray | None:
    rails = []
    for a, b in zip(seed, np.roll(seed, -1, axis=0), strict=True):
        side = _stock_side(image, a, b, stock, 22)
        if side is None:
            side = _stock_side(image, a, b, stock, 44)
        if side is None:
            return None
        rails.append(side)
    lines = np.cross(np.c_[np.asarray(rails)[:, 0], np.ones(4)], np.c_[np.asarray(rails)[:, 1], np.ones(4)])
    corners = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(np.abs(corners[:, 2]) < 1.e-4):
        return None
    quad = np.float32(corners[:, :2] / corners[:, 2:])
    if not cv2.isContourConvex(quad) or np.max(np.linalg.norm(seed - quad, axis=1)) > 52:
        return None
    if np.any((quad < 0) | (quad > np.float32(image.shape[1::-1]) - 1)):
        return None
    return quad


def _ribbed_lane(image: np.ndarray, lane: np.ndarray, stock: np.ndarray) -> np.ndarray | None:
    """Match a bright rib across the lane against independently observed ends."""
    start, stop = lane
    length = np.linalg.norm(stop - start)
    fractions = np.r_[np.linspace(-29, -8, 24), np.linspace(18, length - 18, 112),
                      np.linspace(length + 9, length + 28, 24)].astype(np.float32) / length
    band = _edge_band(image, start, stop, range(-11, 12), fractions=fractions)
    if band is None:
        return None
    light = band[..., 0]
    positions = np.arange(6, 17)  # offsets -5 to +5 around the fitted lane
    score = np.minimum(
        light[:, positions] - light[:, positions - 4],
        light[:, positions] - light[:, positions + 4],
    )
    ends = [int(positions[np.argmax(np.median(block, axis=0))]) for block in (score[:24], score[-24:])]
    if abs(ends[0] - ends[1]) > 8:
        return None
    offsets = np.rint(np.interp(fractions, [fractions[:24].mean(), fractions[-24:].mean()], ends)).astype(np.intp)
    rows = np.arange(len(fractions))
    evidence = (score[rows, offsets - 6] >= 12) & (np.linalg.norm(band[rows, offsets, 1:] - stock, axis=1) <= 7)
    if any(block.mean() < threshold for block, threshold in (
        (evidence[:24], .80), (evidence[24:-24], .88), (evidence[-24:], .80),
    )):
        return None
    tangent = (stop - start) / length
    normal = np.float32([-tangent[1], tangent[0]])
    first = np.flatnonzero(evidence[:24])[0]
    last = len(fractions) - 24 + np.flatnonzero(evidence[-24:])[-1]
    endpoints = [first, last]
    return np.float32(start + fractions[endpoints, None] * (stop - start)
                      + (offsets[endpoints, None] - 11) * normal)


def measured_merged_print_pair(image: np.ndarray, seeds: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray] | None:
    facing = _facing_sides(*seeds)
    if facing is None:
        return None
    if min(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() for quad in seeds) < 200:
        return None
    palette = _stock_palette(image, _lane_center(seeds, facing))
    if palette is None:
        return None
    fits = [_fit_print(image, seed, palette) for seed in seeds]
    if any(quad is None for quad in fits) or cv2.intersectConvexConvex(*fits)[0] > 0:
        return None
    lane = _ribbed_lane(image, _lane_center(fits, facing), palette)
    if lane is None or recover_shared_stock_outline(image, fits) is not None:
        return None
    return fits, lane
