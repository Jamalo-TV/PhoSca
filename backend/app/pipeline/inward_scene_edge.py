"""Measure an open scene edge beside a competing album-backing transition."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _line_endpoints, _side_line
from app.pipeline.visible_scene_edge import has_enclosing_perimeter


INWARD_SCENE_EDGE_SCOPE = "inward_scene_edge_scope"
_PIXEL_FOOTPRINT = np.float32([[-2, -2], [-2, 2], [2, 2], [2, -2]])


@dataclass(frozen=True)
class InwardSceneEdge:
    trace: np.ndarray
    scope: np.ndarray
    possible_cut: np.ndarray


def _material_step(image: np.ndarray, ends: np.ndarray) -> tuple[float, bool] | None:
    band = _edge_band(image, *ends, [-15, -12, -9, -6, -3, -2, 1, 2, 4, 6, 9],
                      fractions=np.linspace(.08, .92, 160, dtype=np.float32))
    if band is None:
        return None
    backing = np.median(band[:, :4], axis=1)
    outside = np.median(band[:, 4:6], axis=1)
    inside = np.median(band[:, 6:10], axis=1)
    step = outside[:, 0] - inside[:, 0]
    supported = ((step >= 18) & (np.linalg.norm(outside[:, 1:] - 128, axis=1) <= 15)
                 & (np.linalg.norm(outside - backing, axis=1) <= 21))
    matches_backing = supported.mean() >= .92 and min(part.mean() for part in np.array_split(supported, 8)) >= .80
    return float(np.median(step)), bool(matches_backing)


def _candidate_edges(image, side, inward, scale):
    candidates = []
    for shift in (24, 36, 48):
        for radius in (14, 20):
            line, facts = _side_line(image, *(side + shift * scale * inward), round(radius * scale))
            if line is None or facts["support"] < .89 or (facts["corner_support"] or 0) < .60:
                continue
            ends = np.float32(_line_endpoints(*side, line))
            depths = (ends - side) @ inward
            if depths.min() < 4.5 * scale or depths.max() > 63 * scale:
                continue
            if any(np.linalg.norm(ends - old, axis=1).max() <= 1.5 for old, _ in candidates):
                continue
            material = _material_step(image, ends)
            if material is not None:
                candidates.append((ends, material))
    return candidates


def _line(*points):
    return np.cross(*np.c_[points, np.ones(2)])


def _corners_with_edge(quad: np.ndarray, ends: np.ndarray) -> np.ndarray | None:
    line = _line(*ends)
    corners = np.cross(line, np.float32([_line(*quad[1:3]), _line(quad[3], quad[0])]))
    if np.any(np.abs(corners[:, 2]) < 1e-6):
        return None
    return np.float32(corners[:, :2] / corners[:, 2:])


def _step_magnitudes(image, side, offsets, fractions):
    band = _edge_band(image, *side, offsets, fractions=fractions)
    return None if band is None else np.linalg.norm(np.median(band[:, :3], axis=1) - np.median(band[:, 3:], axis=1), axis=1)


def _band_corners(side, fractions, depths):
    edge = side[1] - side[0]
    inward = np.float32([-edge[1], edge[0]]) / np.linalg.norm(edge)
    ends = side[0] + np.float32(fractions)[:, None] * edge
    return (ends[:, None] + np.float32(depths)[None, :, None] * inward).reshape(-1, 2)


def _terminal_scope(image, quad, inner, corner, index):
    toward = quad[1 if index == 0 else 0] - corner
    toward /= np.linalg.norm(toward)
    axis = np.float32([corner, corner + 78 * toward])
    fits, scopes = [], []
    for near, far in ((6, 56), (12, 76)):
        window = corner + np.float32([[near], [far]]) * toward
        if index == 0:
            window = window[::-1]
        line, evidence = _side_line(image, *window, 6)
        if line is None or evidence["support"] < .90 or (evidence["corner_support"] or 0) < .78:
            return None
        fits.append(np.float32(_line_endpoints(*axis, line)))
        scopes.append(_band_corners(window, (.035, .965), (-9, 9)))
    if np.linalg.norm(fits[0] - fits[1], axis=1).max() > 1 or np.linalg.norm(np.asarray(fits) - axis, axis=2).max() > 6:
        return None
    fitted = np.mean(np.asarray(fits), axis=0)
    joined = np.cross(_line(*fitted), _line(*inner))
    if abs(joined[2]) < 1e-6:
        return None
    vertex = np.float32(joined[:2] / joined[2])
    arm = np.float32([vertex, vertex + fitted[1] - fitted[0]])
    arm_length = np.linalg.norm(arm[1] - arm[0])
    fractions = np.float32([*range(-20, -5), *range(-5, -1), *range(5, 12), *range(12, 24)]) / arm_length
    if index == 0:
        arm = arm[::-1]
        fractions = 1 - fractions
    steps = _step_magnitudes(image, arm, [-3, -2, -1, 1, 2, 3], fractions)
    if steps is None:
        return None
    before, near, inside, far = np.split(steps, (15, 19, 26))
    if np.quantile(before, .9) > 17 or np.quantile(near, .9) > 19 or min(np.quantile(inside, .1), np.quantile(far, .1)) < 32:
        return None
    along = inner[1 - index] - inner[index]
    along /= np.linalg.norm(along)
    cap = np.float32([vertex, vertex + along * 60])
    outside = np.float32(range(-60, -12)) / 60
    outer_steps = _step_magnitudes(image, cap, [-4, -3, -2, 1, 2, 3, 4], outside)
    if outer_steps is None or (outer_steps >= 9).mean() > .15:
        return None
    scopes.extend((_band_corners(arm, (fractions.min(), fractions.max()), (-4, 4)),
                   _band_corners(cap, (outside[0], outside[-1]), (-5, 5))))
    return np.vstack(scopes)


def measure_inward_scene_edge(
    image: np.ndarray, quad: np.ndarray, boundary: dict, peers: list[np.ndarray],
) -> InwardSceneEdge | None:
    """Locate a source span while leaving the intervening band's ownership open."""
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    short = float(lengths.min())
    if short < 430:
        return None
    scale = min(short / 727., 1.65)
    supports = [boundary.get(name, {}).get("support", 0.) for name in _PRINT_SIDES]
    matches = []
    for index, name in enumerate(_PRINT_SIDES):
        facts = boundary.get(name)
        if (not facts or facts["support"] > .45 or facts["ambiguous"] or supports[(index + 2) % 4] < .88
                or max(supports[(index + 1) % 4], supports[(index - 1) % 4]) < .90):
            continue
        ordered = np.roll(quad, 2 - index, axis=0)
        side = ordered[2:]
        edge = side[1] - side[0]
        inward = np.float32([-edge[1], edge[0]]) / lengths[index]
        candidates = _candidate_edges(image, side, inward, scale)
        for inner, (step, _) in candidates:
            if step < 42:
                continue
            rivals = [line for line, (_, proof) in candidates if proof
                      and np.all(((inner - line) @ inward >= 4.5 * scale) & ((inner - line) @ inward <= 22 * scale))]
            if len(rivals) != 1:
                continue
            corners = _corners_with_edge(ordered, inner)
            if corners is None:
                continue
            trace = inner[0] + np.float32([[.08], [.92]]) * (inner[1] - inner[0])
            context = np.vstack((trace, trace - .025 * short * inward, trace + .06 * short * inward))
            body = _band_corners(side, (.033, .967), (-16, max(70 * scale + 4, 63 * scale + 11)))
            for end_index, corner in enumerate(corners):
                terminal = _terminal_scope(image, ordered, inner, corner, end_index)
                if terminal is None:
                    continue
                sampled = np.vstack((body, terminal))
                scope = cv2.convexHull((sampled[:, None] + _PIXEL_FOOTPRINT).reshape(-1, 2))[:, 0]
                if (np.any((scope < 0) | (scope > np.float32(image.shape[1::-1]) - 1))
                        or any(cv2.intersectConvexConvex(scope, peer)[0] > 1 for peer in peers)
                        or has_enclosing_perimeter(image, quad, context, short)):
                    continue
                matches.append(InwardSceneEdge(trace, scope, cv2.convexHull(np.vstack((corners, side)))[:, 0]))
                break
    return matches[0] if len(matches) == 1 else None
