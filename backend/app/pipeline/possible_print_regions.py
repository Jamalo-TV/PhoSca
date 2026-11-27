"""Compare two source-observed regions within one unresolved print proposal."""

from collections.abc import Callable

import cv2
import numpy as np

from app.pipeline.album_backing import _repeating_profile
from app.pipeline.geometry import (
    normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, quad_from_polygon,
)
from app.pipeline.merged_print_review import _facing_sides, _lane_center
from app.pipeline.multipanel_prints import recover_shared_stock_outline
from app.pipeline.print_edges import _edge_band, _quad_from_lines
from app.pipeline.tile_print_review import tiled_outline_conflicts


POSSIBLE_PRINT_REGIONS = "possible_print_regions"
POSSIBLE_PRINT_REGIONS_REVIEW = "POSSIBLE_PRINT_REGIONS_REVIEW"
POSSIBLE_PRINT_REGIONS_REVISION = 1


def _parent_quad(parent: np.ndarray, width: int, height: int) -> np.ndarray | None:
    try:
        quad, _ = quad_from_polygon(normalized_vertices_to_pixels(parent, width, height))
    except (cv2.error, ValueError):
        return None
    if quad is None:
        return None
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    across, along = (sides[:2] + sides[2:]) / 2
    if across > along:
        quad, across, along = np.roll(quad, -1, axis=0), along, across
    area = cv2.contourArea(quad)
    if (across < 230 or not .49 <= across / along <= .65
            or not .065 <= area / (width * height) <= .14
            or not cv2.isContourConvex(quad)):
        return None
    return quad


def possible_print_region_window(parent: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Find a single bounded inference window for an elongated parent."""
    height, width = image_shape
    if (quad := _parent_quad(parent, width, height)) is None:
        return None
    axis = quad[2:].mean(axis=0) - quad[:2].mean(axis=0)
    if min(abs(axis)) > .30 * max(abs(axis)):
        return None
    factors = (2.4, 1.55) if abs(axis[1]) > abs(axis[0]) else (1.55, 2.4)
    span = np.ptp(quad, axis=0) * np.float32(factors)
    size = np.minimum(np.rint(span).astype(int), [width, height])
    if size[0] * size[1] > .5 * width * height:
        return None
    left, top = np.clip(np.rint(quad.mean(axis=0) - span / 2).astype(int), 0, [width, height] - size)
    return int(left), int(top), int(left + size[0]), int(top + size[1])


def _candidate_seeds(
    parent: np.ndarray, quad: np.ndarray, candidates: list[tuple[np.ndarray, float, float]], width: int, height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    axis_start = quad[:2].mean(axis=0)
    axis = quad[2:].mean(axis=0) - axis_start
    admitted = []
    for polygon, confidence, boundary in candidates:
        if confidence < .82 or boundary < .80:
            continue
        overlap = normalized_polygon_overlap(polygon, parent, raster_size=512)
        if overlap.first_containment < .90 or not .35 <= overlap.second_containment <= .55:
            continue
        seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
        if seed is None or not cv2.isContourConvex(seed) or np.min(np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1)) < 180:
            continue
        position = float((seed.mean(axis=0) - axis_start) @ axis / (axis @ axis))
        admitted.append((position, order_quad_cyclic(seed), polygon))
    if len(admitted) != 2:
        return None
    (near, first, first_polygon), (far, second, second_polygon) = sorted(admitted, key=lambda item: item[0])
    if (not .15 <= near <= .35 or not .65 <= far <= .85
            or normalized_polygon_overlap(first_polygon, second_polygon, raster_size=512).iou > .04):
        return None
    return first, second


def _backing_palette(
    image: np.ndarray, parent: np.ndarray, claimed: Callable[[np.ndarray], bool],
) -> np.ndarray | None:
    """Observe warm repetitive backing away from both of the tested motifs."""
    fractions = np.linspace(.10, .90, 240, dtype=np.float32)
    observations = []
    for start, stop in zip(parent, np.roll(parent, -1, axis=0), strict=True):
        delta = stop - start
        inward = np.float32([-delta[1], delta[0]]) / np.linalg.norm(delta)
        first, last = start + fractions[[0, -1], None] * delta
        for depth in (8, 14, 21, 28):
            band = _edge_band(image, start, stop, range(-depth - 3, -depth + 4), fractions=fractions)
            if band is None:
                continue
            valid = (band[..., 0] >= 150) & (band[..., 2] >= 135) & (band[..., 1] >= 119) & (band[..., 1] <= 143)
            if valid.mean() < .75:
                continue
            palette = np.median(band[valid], axis=0)[1:]
            same = (np.linalg.norm(band[..., 1:] - palette, axis=2) <= 8).mean(axis=1) >= .80
            profile, repetition = _repeating_profile(np.median(band[..., 0], axis=1))
            if same.mean() < .90 or min(part.mean() for part in np.array_split(same, 8)) < .70:
                continue
            if repetition < .72 or profile.std() < 9:
                continue
            near, far = (np.float32([-depth + 3, -depth - 3])[:, None] * inward)
            if not claimed(np.float32([first + far, last + far, last + near, first + near])):
                observations.append((.75 * same.mean() + .25 * repetition, palette))
    if not observations:
        return None
    _, palette = max(observations, key=lambda item: item[0])
    return palette


_FRACTIONS = np.linspace(.11, .89, 152, dtype=np.float32)


def _measure_side(
    image: np.ndarray, start: np.ndarray, stop: np.ndarray, palette: np.ndarray, *, facing: bool,
) -> np.ndarray | None:
    axis = stop - start
    length = np.linalg.norm(axis)
    radius = 26 if facing else min(82, round(length * .32))
    band = _edge_band(image, start, stop, range(-radius - 10, radius + 11), fractions=_FRACTIONS)
    if band is None:
        return None
    indices = np.arange(2 * radius + 1) + 10
    outside_offsets = np.array([-3, -2, -1] if facing else [-7, -5, -3])
    inside_offsets = np.array([2, 3, 4] if facing else [3, 5, 8])
    outside = np.median(band[:, indices[:, None] + outside_offsets], axis=2)
    inside = np.median(band[:, indices[:, None] + inside_offsets], axis=2)
    palette_distance = np.linalg.norm(outside[..., 1:] - palette, axis=2)
    chroma_step = np.linalg.norm(outside[..., 1:] - inside[..., 1:], axis=2)
    light_drop = outside[..., 0] - inside[..., 0]
    edge = np.linalg.norm(band[:, indices - 1] - band[:, indices + 1], axis=2)
    supported = (palette_distance <= 9) & (outside[..., 0] >= 137)
    supported &= ((chroma_step >= 7) | (light_drop >= 25)) & (edge >= 9)
    strength = (.45 * supported + .25 * np.clip(chroma_step / 40, 0, 1)
                + .11 * np.clip(light_drop / 78, 0, 1) + .19 * np.clip(edge / 55, 0, 1))

    shifts, slopes = np.meshgrid(np.arange(-radius, radius + 1), np.linspace(-.32, .32, 129))
    endpoints = shifts.ravel()[:, None] + slopes.ravel()[:, None] * length * np.float32([-.5, .5])
    at = endpoints[:, :1] + np.diff(endpoints, axis=1) * _FRACTIONS
    inside_search = np.max(np.abs(at), axis=1) <= radius
    at = at[inside_search]
    columns = np.rint(at).astype(np.intp) + radius
    rows = np.arange(len(_FRACTIONS))
    votes = supported[rows, columns]
    weakest = np.min([quarter.mean(axis=1) for quarter in np.array_split(votes, 4, axis=1)], axis=0)
    rank = strength[rows, columns].mean(axis=1) + .035 * weakest - .035 * (np.abs(at).mean(axis=1) / radius)
    best = int(np.argmax(rank))
    if facing:
        if votes[best].mean() < .88 or weakest[best] < .78:
            return None
    elif votes[best].mean() < .56:
        return None
    inward = np.float32([-axis[1], axis[0]]) / length
    a, b = np.float32([start, stop]) + endpoints[inside_search][best, :, None] * inward
    return np.cross(np.r_[a, 1.], np.r_[b, 1.])


def _backing_between_regions(
    image: np.ndarray, first: np.ndarray, second: np.ndarray, facing: tuple[int, int], palette: np.ndarray,
    claimed: Callable[[np.ndarray], bool],
) -> bool:
    """Measure a narrow intervening band without assigning physical ownership."""
    a, b = (quad[[side, (side + 1) % 4]] for quad, side in zip((first, second), facing, strict=True))
    axis = a[1] - a[0]
    normal = np.float32([-axis[1], axis[0]]) / np.linalg.norm(axis)
    gap = (a - b[::-1]) @ normal
    if gap.min() < 3.5 or gap.max() > 6.5 or gap.max() - gap.min() > 3:
        return False
    if claimed(np.float32([a[0], a[1], b[0], b[1]])):
        return False
    start, stop = _lane_center([first, second], facing)[::-1]
    length = np.linalg.norm(stop - start)
    stations = np.r_[np.linspace(-13, -5, 12), np.linspace(17, length - 20, 140),
                     np.linspace(length + 12, length + 32, 24)].astype(np.float32)
    offsets = np.arange(-2.0, 2.01, .25, dtype=np.float32)
    band = _edge_band(image, start, stop, offsets, fractions=stations / length)
    if band is None:
        return False
    same = (band[..., 0] >= 166) & (np.linalg.norm(band[..., 1:] - palette, axis=2) <= 8)
    central = same[12:-24]
    distributed = np.min([part.mean(axis=0) for part in np.array_split(central, 4)], axis=0)
    return bool(np.any((central.mean(axis=0) >= .83) & (distributed >= .70)
                       & (same[:12].mean(axis=0) >= .85) & (same[-24:].mean(axis=0) >= .85)))


def measure_possible_print_regions(
    image: np.ndarray, parent: np.ndarray, candidates: list[tuple[np.ndarray, float, float]], claims: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return two native comparison outlines, preserving the original parent claim."""
    height, width = image.shape[:2]
    if (quad := _parent_quad(parent, width, height)) is None:
        return None
    seeds = _candidate_seeds(parent, quad, candidates, width, height)
    if seeds is None or (facing := _facing_sides(*seeds)) is None:
        return None
    scale = np.float32([width - 1, height - 1])

    def claimed(points: np.ndarray) -> bool:
        outline = [{"x": float(x), "y": float(y)} for x, y in points / scale]
        return tiled_outline_conflicts(outline, claims, width, height)

    palette = _backing_palette(image, quad, claimed)
    if palette is None:
        return None
    split_sides = [_measure_side(image, seed[side], seed[(side + 1) % 4], palette, facing=True)
                   for seed, side in zip(seeds, facing, strict=True)]
    if any(line is None for line in split_sides):
        return None
    fitted = []
    for seed, side, split in zip(seeds, facing, split_sides, strict=True):
        lines = [split if index == side else _measure_side(image, start, stop, palette, facing=False)
                 for index, (start, stop) in enumerate(zip(seed, np.roll(seed, -1, axis=0), strict=True))]
        if any(line is None for line in lines):
            return None
        outline = _quad_from_lines(np.stack(lines))
        if (outline is None or not cv2.isContourConvex(outline) or cv2.contourArea(outline, oriented=True) <= 0
                or np.any(np.minimum(outline, scale - outline) < 4) or claimed(outline)):
            return None
        fitted.append(outline)
    first, second = fitted
    if cv2.intersectConvexConvex(first, second)[0] > 0 or not _backing_between_regions(image, first, second, facing, palette, claimed):
        return None
    coverage = [normalized_polygon_overlap(fit / scale, parent, raster_size=1024) for fit in fitted]
    if min(item.first_containment for item in coverage) < .96 or not .78 <= sum(item.second_containment for item in coverage) <= .92:
        return None
    if recover_shared_stock_outline(image, fitted) is not None:
        return None
    return order_quad_cyclic(first), order_quad_cyclic(second)
