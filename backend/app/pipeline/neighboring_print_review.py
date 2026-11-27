"""Verify a possible print at the next position in a measured row or column."""

from itertools import combinations

import cv2
import numpy as np

from app.pipeline.album_backing import _repeating_profile
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, quad_from_polygon
from app.pipeline.print_edges import _edge_band, _line_endpoints, _quad_from_lines, _side_line, refine_print_boundary
from app.pipeline.tile_print_review import TILED_PRINT_PROPOSAL, tiled_outline_conflicts
from app.pipeline.visible_scene_edge import has_enclosing_perimeter


def _vertices(points):
    return np.float32([(point["x"], point["y"]) for point in points])


def _normalized(points):
    return [{"x": float(x), "y": float(y)} for x, y in points]


def next_print_position(
    pending: list[dict], claims: list[dict], width: int, height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find one vacant position supported by comparable held outlines."""
    scale = np.float32([width - 1, height - 1])
    outlines = [row["polygon"] for row in pending
                if row.get("reason") == TILED_PRINT_PROPOSAL and len(row.get("polygon") or []) == 4]
    if not 2 <= len(outlines) <= 4:
        return None
    measured = [order_quad_cyclic(_vertices(points) * scale) for points in outlines]
    positions = []
    for first, second in combinations(measured, 2):
        pitch = second.mean(axis=0) - first.mean(axis=0)
        distance = np.linalg.norm(pitch)
        if distance < 180:
            continue
        axes = [np.diff(quad[:3], axis=0) for quad in (first, second)]
        lengths = [np.linalg.norm(axis, axis=1) for axis in axes]
        alignments = [np.abs(axis @ (pitch / distance)) / size for axis, size in zip(axes, lengths, strict=True)]
        matching_sides = [size[np.argmax(alignment)] for size, alignment in zip(lengths, alignments, strict=True)]
        if (min(alignment.max() for alignment in alignments) < .96
                or max(matching_sides) > 1.25 * min(matching_sides)
                or not .93 <= distance / np.mean(matching_sides) <= 1.35):
            continue
        for quad, direction in ((first, -pitch), (second, pitch)):
            next_quad = quad + direction
            if np.any(np.minimum(next_quad, scale - next_quad) < 4):
                continue
            if not tiled_outline_conflicts(_normalized(next_quad / scale), claims, width, height):
                positions.append((next_quad, direction))
    return positions[0] if len(positions) == 1 else None


_FRACTIONS = np.linspace(.08, .92, 160, dtype=np.float32)


def _leading_edge(image: np.ndarray, side: np.ndarray) -> np.ndarray | None:
    offsets = np.arange(-9, 9.1, .25, dtype=np.float32)
    band = _edge_band(image, *side, offsets, fractions=_FRACTIONS)
    if band is None:
        return None
    light = band[..., 0]
    columns = np.flatnonzero((-5 <= offsets) & (offsets <= 3))
    step = light[:, columns - 4] - light[:, columns + 4]
    wide = light[:, columns - 12] - light[:, columns + 12]
    warmer = band[:, columns - 4, 2] >= band[:, columns + 4, 2]
    choices = (step >= 12) & (wide >= 24) & warmer
    supported = choices.any(axis=1)
    if supported.mean() < .95:
        return None
    peaks = np.argmax(np.where(choices, step + .20 * wide, -1), axis=1)
    depths = offsets[columns[peaks]]
    direction = side[1] - side[0]
    length = np.linalg.norm(direction)
    stations = _FRACTIONS * length
    vx, vy, x, y = cv2.fitLine(
        np.column_stack((stations[supported], depths[supported])), cv2.DIST_HUBER, 0, .01, .01,
    ).ravel()
    if abs(vx) < .1:
        return None
    predicted = y + (vy / vx) * (stations - x)
    agreement = supported & (np.abs(predicted - depths) <= 1.1)
    if (agreement.mean() < .78 or min(part.mean() for part in np.array_split(agreement, 8)) < .50
            or np.max(np.abs(predicted[[0, -1]])) > 2):
        return None
    return _edge_line(side, predicted[[0, -1]])


def _edge_line(side, displacement):
    dx, dy = side[1] - side[0]
    normal = np.float32([-dy, dx]) / np.hypot(dx, dy)
    a, b = side + displacement[:, None] * normal
    return np.cross(np.r_[a, 1.], np.r_[b, 1.])


def _trailing_edge(image: np.ndarray, side: np.ndarray) -> np.ndarray | None:
    offsets = np.arange(-13, 13.5, .5, dtype=np.float32)
    band = _edge_band(image, *side, offsets, fractions=_FRACTIONS)
    if band is None:
        return None
    light = band[..., 0]
    columns = np.flatnonzero(np.abs(offsets) <= 8)
    step = light[:, columns - 2] - light[:, columns + 2]
    wide = light[:, columns - 5] - light[:, columns + 5]
    shoulders = np.maximum(light[:, columns - 6] - light[:, columns - 3], light[:, columns + 3] - light[:, columns + 6])
    choices = (step >= 12) & (wide >= 24) & (step >= shoulders + 3)
    shifts, slopes = np.meshgrid(offsets[columns], np.arange(-.045, .0451, .0045))
    length = np.linalg.norm(side[1] - side[0])
    end_offsets = shifts.ravel()[:, None] + slopes.ravel()[:, None] * length * np.float32([-.5, .5])
    at = end_offsets[:, :1] + np.diff(end_offsets, axis=1) * _FRACTIONS
    bounded = np.abs(at).max(axis=1) <= 8
    at, end_offsets = at[bounded], end_offsets[bounded]
    columns = np.rint((at + 8) * 2).astype(np.intp)
    stations = np.arange(len(_FRACTIONS))[None, :]
    supports = choices[stations, columns]
    coverage = supports.mean(axis=1)
    weakest = np.min([part.mean(axis=1) for part in np.array_split(supports, 8, axis=1)], axis=0)
    quality = .59 * coverage + .21 * weakest + .20 * np.minimum(step[stations, columns] / 50, 1.).mean(axis=1)
    eligible = (coverage >= .90) & (weakest >= .60) & (np.median(wide[stations, columns], axis=1) >= 24)
    if not eligible.any():
        return None
    index = np.argmax(np.where(eligible, quality, -1.))
    rivals = eligible & (quality >= quality[index] - .03) & (np.abs(at - at[index]).max(axis=1) > 3)
    if rivals.any() or np.max(np.abs(end_offsets[index])) > 6:
        return None
    return _edge_line(side, end_offsets[index])


def _lateral_edge(image: np.ndarray, side: np.ndarray) -> np.ndarray | None:
    measured = []
    for radius in (4, 7):
        line, evidence = _side_line(image, *side, radius)
        if line is None or evidence["ambiguous"] or evidence["support"] < .90 or (evidence["corner_support"] or 0) < .60:
            return None
        measured.append(line)
    ends = np.float32([_line_endpoints(*side, line) for line in measured])
    return measured[0] if np.linalg.norm(ends[0] - ends[1], axis=1).max() <= 1 else None


def _ribs_end_at_edge(image: np.ndarray, start: np.ndarray, stop: np.ndarray) -> bool:
    fractions = np.linspace(.08, .92, round(.84 * np.linalg.norm(stop - start)), dtype=np.float32)
    band = _edge_band(image, start, stop, [-14, -12, -10, -8, -6, -5, -4, -3, -2, 1, 2, 3], fractions=fractions)
    if band is None:
        return False
    profiles = [_repeating_profile(np.median(block, axis=1)) for block in np.split(band[..., 0], [4, 6, 9], axis=1)]
    if any(profile.std() < 8 or repeat < .80 for profile, repeat in profiles[:3]) or profiles[3][0].std() == 0:
        return False
    _, middle, near, inside = np.corrcoef([profile for profile, _ in profiles])[0]
    return min(middle, near) >= .85 and abs(inside) < .25


def _native_outline(image: np.ndarray, polygon: np.ndarray, direction: np.ndarray, claims: list[dict]) -> np.ndarray | None:
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
    quad, evidence, reasons = refine_print_boundary(image, order_quad_cyclic(seed))
    if quad is None or len(evidence) != 4 or set(reasons) - {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_CORNERS_UNCERTAIN"}:
        return None
    # Identify the side facing away from the two witnesses, including when
    # the row and captured page are rotated.
    outer = (quad + np.roll(quad, -1, axis=0)) / 2 - quad.mean(axis=0)
    leading = np.argmax((outer @ direction) / np.linalg.norm(outer, axis=1))
    ordered = np.roll(quad, -leading, axis=0)
    methods = (_leading_edge, _lateral_edge, _trailing_edge, _lateral_edge)
    lines = [method(image, np.float32([a, b])) for a, b, method in
             zip(ordered, np.roll(ordered, -1, axis=0), methods, strict=True)]
    if any(line is None for line in lines):
        return None
    fit = _quad_from_lines(np.stack(lines))
    if (fit is None or not cv2.isContourConvex(fit) or cv2.contourArea(fit, oriented=True) <= 0
            or np.any(np.minimum(fit, scale - fit) < 4)
            or normalized_polygon_overlap(fit / scale, polygon, raster_size=1024).iou < .85
            or tiled_outline_conflicts(_normalized(fit / scale), claims, width, height)):
        return None
    flank_supported = False
    for side, (start, stop) in enumerate(zip(fit, np.roll(fit, -1, axis=0), strict=True)):
        delta = stop - start
        length = np.linalg.norm(delta)
        normal = np.float32([-delta[1], delta[0]]) / length
        a, b = start + np.float32([.08, .92])[:, None] * delta
        witness = np.float32([a - 8 * normal, b - 8 * normal, b + 6 * normal, a + 6 * normal])
        if tiled_outline_conflicts(_normalized(witness / scale), claims, width, height):
            return None
        band = _edge_band(image, start, stop, [-8, -6, -4, -2, 0, 2, 4, 6], fractions=_FRACTIONS)
        if band is None or (np.linalg.norm(band[:, 3] - band[:, 5], axis=1) >= 12).mean() < .90:
            return None
        outside = np.median(band[:, :3, 0], axis=1)
        inside = np.median(band[:, 5:, 0], axis=1)
        if (outside - inside >= 12).mean() < .70:
            return None
        if side % 2:
            collar = np.float32([a - 14 * normal, b - 14 * normal, b, a])
            if not tiled_outline_conflicts(_normalized(collar / scale), claims, width, height):
                flank_supported |= _ribs_end_at_edge(image, start, stop)
    if not flank_supported:
        return None
    return order_quad_cyclic(fit)


def neighboring_print_outline(
    image: np.ndarray, masks: list[np.ndarray], direction: np.ndarray, claims: list[dict],
) -> np.ndarray | None:
    """Corroborate a held outline in two local views using original pixels."""
    if len(masks) != 2 or normalized_polygon_overlap(*masks, raster_size=1024).iou < .90:
        return None
    first, second = (_native_outline(image, polygon, direction, claims) for polygon in masks)
    if first is None or second is None:
        return None
    drift = min(np.linalg.norm(first - np.roll(second, turn, axis=0), axis=1).max() for turn in range(4))
    if drift > 1.5:
        return None
    short = float(np.linalg.norm(np.roll(second, -1, axis=0) - second, axis=1).min())
    if has_enclosing_perimeter(image, second, second, short, search_margin=1.65, maximum_area_ratio=12):
        return None
    return second
