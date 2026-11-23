"""Look for a faint outer print boundary beyond a stronger interior edge."""

from collections.abc import Iterator

import cv2
import numpy as np


_RECTANGLE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
_SIDE_NAMES = ("top", "right", "bottom", "left")


def _project(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), transform)[0].reshape(points.shape)


def _sample(source: np.ndarray, transform: np.ndarray, across: np.ndarray, down: np.ndarray) -> np.ndarray | None:
    points = _project(np.stack(np.broadcast_arrays(across, down), axis=-1), transform)
    if np.any((points < 0) | (points > np.array(source.shape[1::-1]) - 1)):
        return None
    return cv2.remap(source, points[..., 0].reshape(-1, 1), points[..., 1].reshape(-1, 1),
                     cv2.INTER_LINEAR).reshape(points.shape[:-1])


def _side_candidates(source: np.ndarray, quad: np.ndarray, side: int) -> Iterator[tuple[np.ndarray, float, float, float]]:
    # In the print plane, the candidate and both adjoining sides continue
    # to meet at right angles as the candidate moves farther out.
    rotated = np.roll(quad, 2 - side, axis=0)
    sides = np.linalg.norm(np.roll(rotated, -1, axis=0) - rotated, axis=1)
    width, height = sides[::2].mean(), sides[1::2].mean()
    transform = cv2.getPerspectiveTransform(_RECTANGLE * [np.float32(width), np.float32(height)], rotated)
    offsets = np.arange(2.5, 8.01, .25, dtype=np.float32)
    positions = np.linspace(.12 * width, .88 * width, 57, dtype=np.float32)
    luminance = _sample(source, transform, positions[:, None, None],
                        height + offsets[:, None] + np.float32([-.95, -.35, .95, 1.55]))
    if luminance is None:
        return
    outer = luminance[..., 2:]
    differences = luminance[..., :2].mean(axis=2) - outer.mean(axis=2)
    steps = np.median(differences, axis=0)
    signs = np.sign(steps)
    strength = np.abs(steps)
    quarters = np.stack([block.mean(axis=0) for block in np.array_split(differences * signs, 4)])
    plausible = (strength >= .47) & (strength <= 5.) & (quarters.min(axis=0) >= .50 * strength)
    side_offsets = np.float32([-.90, -.55, .55, .90])
    for index in np.flatnonzero(plausible):
        offset = offsets[index]
        strip = np.linspace(max(1.45, .35 * offset), offset - .72, 9, dtype=np.float32)
        if np.ptp(strip) < 1.6:
            continue
        levels = outer[:, index]
        if np.percentile(np.abs(levels - np.median(levels)), 80) > 3.:
            continue
        down = height + np.r_[strip, offset + np.float32([1.15, 1.65, 2.15])]
        flanks = _sample(source, transform, np.stack((side_offsets, width - side_offsets))[..., None], down)
        if flanks is None:
            continue
        # Both adjacent print sides must continue to the faint edge and terminate there.
        jumps = flanks[:, 2:].mean(axis=1) - flanks[:, :2].mean(axis=1)
        sustained = np.median(jumps[:, :9] * signs[index], axis=1)
        beyond = np.median(np.abs(jumps[:, 9:]), axis=1).max()
        if sustained.min() < max(.30, .47 * strength[index]) or beyond > .55:
            continue
        candidate = _project(_RECTANGLE * [np.float32(width), np.float32(height + offset)], transform)
        score = float(strength[index] + sustained.min() - .20 * beyond)
        yield np.roll(candidate, side - 2, axis=0), score, float(offset), float(steps[index])


def _unresolved_exterior_step(source: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float] | None:
    edge = end - start
    length = float(np.linalg.norm(edge))
    outwards = np.array([edge[1], -edge[0]], dtype=np.float32) / length
    middle = np.linspace(.08, .92, 96, dtype=np.float32)
    arms = np.r_[np.linspace(-11., -3., 12), np.linspace(length + 3., length + 11., 12)] / length
    along = np.r_[middle, arms].astype(np.float32)
    depths = np.arange(.75, 10.01, .25, dtype=np.float32)
    points = start + along[:, None, None] * edge + depths[None, :, None] * outwards
    if np.any((points < 0) | (points > np.array(source.shape[1::-1]) - 1)):
        return None
    light = cv2.remap(source, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR)
    center, before_arm, after_arm = np.split(light, [len(middle), len(middle) + 12])
    options = []
    for column in range(7, len(depths) - 8):
        core = center[:, column + 4] - center[:, column - 4]
        step = float(np.median(core))
        if not 1.15 <= abs(step) <= 5.0 or np.mean(np.sign(step) * core >= .25) < .72:
            continue
        before = np.median(center[:, column - 2] - center[:, max(column - 8, 0)])
        after = np.median(center[:, column + 8] - center[:, column + 2])
        # A local step is steeper than the adjacent lighting drift and does not
        # continue beyond the print across the album page.
        if max(abs(before), abs(after)) > .62 * abs(step):
            continue
        across = [np.median(arm[:, column + 4] - arm[:, column - 4]) for arm in (before_arm, after_arm)]
        if max(abs(x) for x in across) > .62 * abs(step):
            continue
        options.append((float(depths[column]), -step))
    return max(options, key=lambda option: abs(option[1]), default=None)


def find_outer_stock_candidate(
    image: np.ndarray, quad: np.ndarray, neighbors: list[np.ndarray],
) -> tuple[np.ndarray, str, dict] | None:
    """Find a faint outer boundary, leaving the quad alone when its flanks are uncertain."""
    # Preserve fractional light levels and convert only the local image area.
    origin = np.maximum(0, np.floor(quad.min(axis=0)).astype(int) - 24)
    high = np.minimum(image.shape[1::-1], np.ceil(quad.max(axis=0)).astype(int) + 25)
    patch = image[origin[1]:high[1], origin[0]:high[0]]
    luminance = cv2.cvtColor(np.float32(patch) / 255., cv2.COLOR_BGR2LAB)[..., 0] * 2.55
    low, high = quad.min(axis=0) - 16, quad.max(axis=0) + 16
    nearby = [cv2.convexHull(poly) for poly in neighbors if np.all(poly.min(axis=0) < high)
              and np.all(poly.max(axis=0) > low)]
    overlap = [cv2.intersectConvexConvex(quad, poly)[0] for poly in nearby]

    candidates = []
    for index, name in enumerate(_SIDE_NAMES):
        for extended, score, depth, step in _side_candidates(luminance, quad - np.float32(origin), index):
            candidate = extended + np.float32(origin)
            if (not cv2.isContourConvex(candidate) or cv2.contourArea(candidate, oriented=True) <= 0
                    or np.any((candidate < 0) | (candidate > np.array(image.shape[1::-1]) - 1))
                    or any(cv2.intersectConvexConvex(candidate, poly)[0] > already + 2
                           for poly, already in zip(nearby, overlap, strict=True))):
                continue
            candidates.append((score, index, name, depth, step, candidate))
    if candidates:
        score, index, name, depth, step, candidate = max(candidates, key=lambda item: item[0])
        competing = any(other_score >= .91 * score and (index != other_index or abs(depth - other_depth) > 1.)
                        for other_score, other_index, _, other_depth, _, _ in candidates)
        if not competing:
            return candidate, name, {"depth_px": depth, "luminance_step": step, "flanks_confirmed": True}

    local_quad = quad - np.float32(origin)
    for name, start, end in zip(_SIDE_NAMES, local_quad, np.roll(local_quad, -1, axis=0), strict=True):
        edge = _unresolved_exterior_step(luminance, start, end)
        if edge is not None:
            depth, step = edge
            tangent = end - start
            outward = np.float32([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
            points = np.float32([start, end]) + origin.astype(np.float32)
            strip = np.vstack((points + (depth + 1) * outward, points[::-1] + (depth - 1) * outward))
            if any(cv2.intersectConvexConvex(strip, poly)[0] > 2 for poly in nearby):
                continue
            return quad, name, {"depth_px": depth, "luminance_step": step, "flanks_confirmed": False}
    return None
