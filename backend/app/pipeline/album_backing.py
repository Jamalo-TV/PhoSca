"""Original-pixel evidence from a warm, ribbed album backing."""

from collections.abc import Callable

import cv2
import numpy as np

from app.pipeline.print_edges import _edge_band, _quad_from_lines


def ribbed_lane(image: np.ndarray, lane: np.ndarray, stock: np.ndarray) -> np.ndarray | None:
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


def _repeating_profile(profile: np.ndarray) -> tuple[np.ndarray, float]:
    centered = profile - cv2.GaussianBlur(profile[None, :], (0, 0), 4).ravel()
    correlation = []
    for lag in range(4, 13):
        first, second = centered[:-lag], centered[lag:]
        norm = np.linalg.norm(first) * np.linalg.norm(second)
        correlation.append(float(first @ second / norm) if norm > 0 else 0.)
    return centered, max(correlation)


def _backing_reference(
    image: np.ndarray, seed: np.ndarray, claimed: Callable[[np.ndarray], bool],
) -> tuple[np.ndarray, int] | None:
    references = []
    for index, (start, end) in enumerate(zip(seed, np.roll(seed, -1, axis=0), strict=True)):
        direction = end - start
        length = np.linalg.norm(direction)
        inward = np.float32([-direction[1], direction[0]]) / length
        fractions = np.linspace(.15, .85, round(.70 * length), dtype=np.float32)
        for depth in (.090, .072, .054, .036):
            offset = -round(float(np.clip(depth * length, 7, 30)))
            band = _edge_band(image, start, end, range(offset - 2, offset + 3), fractions=fractions)
            if band is None:
                continue
            light = band[..., 0]
            warm = (light >= 145) & (band[..., 2] >= 135)
            if not warm.any():
                continue
            palette = np.median(band[warm][:, 1:], axis=0)
            if not 123 <= palette[0] <= 139:
                continue
            matches = warm & (np.linalg.norm(band[..., 1:] - palette, axis=2) <= 7)
            stations = matches.mean(axis=1) >= .8
            coverage = stations.mean()
            weakest = min(block.mean() for block in np.array_split(stations, 8))
            if coverage < .82 or weakest < .68:
                continue
            profile, repeat = _repeating_profile(np.median(light, axis=1))
            strength = float(profile.std())
            if strength < 12 or repeat < .52:
                continue
            edge_profiles = [_repeating_profile(light[:, edge])[0] for edge in (0, -1)]
            phase = min(float(np.corrcoef(profile, edge)[0, 1]) if edge.std() > 0 else 0. for edge in edge_profiles)
            if phase < .72:
                continue
            first, last = start + fractions[[0, -1], None] * direction
            outer, inner = (offset + np.float32([-2, 2]))[:, None] * inward
            if claimed(np.float32([first + outer, last + outer, last + inner, first + inner])):
                continue
            score = .38 * coverage + .18 * weakest + .18 * repeat + .14 * phase + .12 * min(strength / 22, 1.)
            references.append((score, palette, index))
    if not references:
        return None
    _, best, index = max(references, key=lambda option: option[0])
    if any(np.linalg.norm(best - palette) > 5 for _, palette, _ in references):
        return None
    return best, index


def _supported_backing_transition(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, palette: np.ndarray, limit: int,
) -> np.ndarray | None:
    direction = end - start
    length = np.linalg.norm(direction)
    offsets = np.arange(-limit - 9, limit + 10, dtype=np.float32)
    fractions = np.linspace(.10, .90, 128, dtype=np.float32)
    band = _edge_band(image, start, end, offsets, fractions=fractions)
    if band is None:
        return None
    centers = np.arange(2 * limit + 1) + 9
    outside = np.median(band[:, centers[:, None] + [-7, -5, -3]], axis=2)
    inside = np.median(band[:, centers[:, None] + [2, 4, 6]], axis=2)
    palette_distance = np.linalg.norm(outside[..., 1:] - palette, axis=2)
    color_change = np.linalg.norm(outside[..., 1:] - inside[..., 1:], axis=2)
    light_drop = outside[..., 0] - inside[..., 0]
    gradient = np.linalg.norm(band[:, centers - 1] - band[:, centers + 1], axis=2)
    evidence = (palette_distance <= 8) & (outside[..., 0] >= 145) & ((color_change >= 7) | (light_drop >= 24))

    shifts, slopes = (grid.ravel() for grid in np.meshgrid(np.arange(-limit, limit + 1), np.linspace(-.23, .23, 93)))
    displacement = shifts[:, None] + slopes[:, None] * (fractions - .5) * length
    bounded = np.abs(displacement).max(axis=1) <= limit
    shifts, slopes, displacement = shifts[bounded], slopes[bounded], displacement[bounded]
    columns = np.rint(displacement).astype(np.intp) + limit
    rows = np.arange(len(fractions))[None, :]
    support = evidence[rows, columns]
    total = support.mean(axis=1)
    portions = np.min([part.mean(axis=1) for part in np.array_split(support, 8, axis=1)], axis=0)
    light = np.clip(light_drop[rows, columns] / 90, 0, 1).mean(axis=1)
    edge = np.clip(gradient[rows, columns] / 70, 0, 1).mean(axis=1)
    color = np.clip(color_change[rows, columns] / 25, 0, 1).mean(axis=1)
    stock = np.minimum(palette_distance[rows, columns] / 8, 2).mean(axis=1)
    scores = .50 * total + .18 * portions + .13 * light + .10 * edge + .09 * color - .035 * stock
    best = int(np.argmax(scores))
    if total[best] < .90 or portions[best] < .80:
        return None
    inward = np.float32([-direction[1], direction[0]]) / length
    endpoints = np.float32([start, end]) + (shifts[best] + slopes[best] * np.float32([-.5, .5]) * length)[:, None] * inward
    return np.cross(np.r_[endpoints[0], 1.], np.r_[endpoints[1], 1.])


def measure_backing_transitions(image: np.ndarray, seed: np.ndarray, palette: np.ndarray) -> np.ndarray | None:
    """Fit four native transitions with evidence of backing on the outside."""
    height, width = image.shape[:2]
    shortest = float(np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min())
    if shortest < 180:
        return None
    limit = min(44, round(.16 * shortest))
    lines = []
    for start, end in zip(seed, np.roll(seed, -1, axis=0), strict=True):
        line = _supported_backing_transition(image, start, end, palette, limit)
        if line is None:
            return None
        lines.append(line)
    fitted = _quad_from_lines(np.stack(lines))
    if (fitted is None or not cv2.isContourConvex(fitted) or cv2.contourArea(fitted, oriented=True) <= 0
            or np.max(np.linalg.norm(fitted - seed, axis=1)) > min(52, shortest * .20)
            or np.any(np.minimum(fitted, np.float32([width - 1, height - 1]) - fitted) < 4)):
        return None
    return fitted


def measured_ribbed_backing_outline(
    image: np.ndarray, seed: np.ndarray, claimed: Callable[[np.ndarray], bool],
) -> np.ndarray | None:
    """Return a count-review contour with independent backing and rib evidence."""
    if np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min() < 180:
        return None
    reference = _backing_reference(image, seed, claimed)
    if reference is None:
        return None
    palette, reference_side = reference
    fitted = measure_backing_transitions(image, seed, palette)
    if fitted is None or claimed(fitted):
        return None
    for index, (start, end) in enumerate(zip(fitted, np.roll(fitted, -1, axis=0), strict=True)):
        if (index - reference_side) % 2 == 0:
            continue
        direction = end - start
        inward = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
        for depth in range(4, 14, 2):
            rib = ribbed_lane(image, np.float32([start, end]) - depth * inward, palette)
            if rib is None:
                continue
            vector = rib[1] - rib[0]
            band = 5 * np.float32([-vector[1], vector[0]]) / np.linalg.norm(vector)
            if not claimed(np.vstack((rib - band, rib[::-1] + band))):
                return fitted
    return None
