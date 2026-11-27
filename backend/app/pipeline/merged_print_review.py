"""Measure two possible paper outlines within a merged detector proposal."""

from itertools import product

import cv2
import numpy as np

from app.pipeline.album_backing import measure_backing_transitions, ribbed_lane
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
    lane = ribbed_lane(image, _lane_center(fits, facing), palette)
    if lane is None or recover_shared_stock_outline(image, fits) is not None:
        return None
    return fits, lane


def _ribs_across_parent(image: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    left, right = parent[3] - parent[0], parent[2] - parent[1]
    middle = (parent[:2] + parent[[3, 2]]) / 2
    tangent = middle[1] - middle[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.float32([-tangent[1], tangent[0]])
    span = (middle - middle[0]) @ tangent
    ribs = []
    for depth in np.linspace(.43, .57, 48, dtype=np.float32):
        a, b = parent[0] + depth * left, parent[1] + depth * right
        chord = b - a
        for start, stop in product((.035, .07, .105), repeat=2):
            lane = np.float32([a + start * chord, b - stop * chord])
            palette = _stock_palette(image, lane)
            if palette is None:
                continue
            rib = ribbed_lane(image, lane, palette)
            if rib is not None:
                projected = (rib - middle[0]) @ tangent
                line = (rib - middle[0]) @ normal
                offsets = line[0] + (span - projected[0]) * (line[1] - line[0]) / (projected[1] - projected[0])
                ribs.append((rib, palette, offsets))
    if len(ribs) < 2:
        return None
    representative = np.median([offsets for _, _, offsets in ribs], axis=0)
    if any(np.max(np.abs(offsets - representative)) > 1.75 for _, _, offsets in ribs):
        return None
    rib, palette, _ = max(ribs, key=lambda match: np.linalg.norm(match[0][1] - match[0][0]))
    if any(np.linalg.norm(palette - other) > 5 for _, other, _ in ribs):
        return None
    return rib, palette


def measured_merged_print_with_one_tile(
    image: np.ndarray, parent: np.ndarray, child: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Measure both print outlines when only one separate model mask survives."""
    lengths = np.linalg.norm(np.roll(parent, -1, axis=0) - parent, axis=1)
    short, long = .5 * (lengths[:2] + lengths[2:])
    if short > long:
        parent, short, long = np.roll(parent, 1, axis=0), long, short
    if short < 200 or not .42 <= short / long <= .70:
        return None
    axis = parent[2:].mean(axis=0) - parent[:2].mean(axis=0)
    position = float((child.mean(axis=0) - parent[:2].mean(axis=0)) @ axis / (axis @ axis))
    if position < .5:
        parent, position = np.roll(parent, 2, axis=0), 1 - position
    if not .65 <= position <= .87:
        return None
    found = _ribs_across_parent(image, parent)
    if found is None:
        return None
    rib, palette = found
    tangent = rib[1] - rib[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.float32([-tangent[1], tangent[0]])
    cross = (parent - rib[0]) @ normal
    fraction_left = -cross[0] / (cross[3] - cross[0])
    fraction_right = -cross[1] / (cross[2] - cross[1])
    if not .40 <= fraction_left <= .60 or not .40 <= fraction_right <= .60:
        return None
    left = parent[0] + fraction_left * (parent[3] - parent[0])
    right = parent[1] + fraction_right * (parent[2] - parent[1])
    other = np.vstack((parent[:2], np.float32([right, left]) - 4 * normal))
    fits = [measure_backing_transitions(image, seed, palette) for seed in (other, child)]
    if any(fitted is None for fitted in fits) or cv2.intersectConvexConvex(*fits)[0] > 0:
        return None
    # Both measured paper edges must leave the observed rib in the gap.
    facing = _facing_sides(*fits)
    if facing is None:
        return None
    length = np.linalg.norm(rib[1] - rib[0])
    clearance = []
    span = []
    for quad, side, sign in zip(fits, facing, (-1, 1), strict=True):
        edge = quad[[side, (side + 1) % 4]] - rib[0]
        distances = sign * (edge @ normal)
        positions = edge @ tangent
        if (np.any((distances < 2.5) | (distances > 16))
                or positions.min() < 6 or positions.max() > length - 6):
            return None
        clearance.append(float(distances.min()))
        span.append((positions.min(), positions.max()))
    if sum(clearance) < 7.5:
        return None
    start, stop = max(edge[0] for edge in span), min(edge[1] for edge in span)
    if stop - start < .65 * length:
        return None
    fractions = np.linspace(start + 10, stop - 10, 64, dtype=np.float32) / length
    shoulders = _edge_band(image, *rib, [-2, 0, 2], fractions=fractions)
    # A flat gold line can imitate the four-pixel contrast used to locate the rib.
    # Verify that the ridge still falls away on both sides inside the open gap.
    if shoulders is None or np.any((shoulders[:, 1:2, 0] - shoulders[:, [0, 2], 0] >= 4).mean(axis=0) < .70):
        return None
    if recover_shared_stock_outline(image, fits) is not None:
        return None
    return fits, rib
