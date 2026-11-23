"""Measure a narrow paper-to-motif transition in the original scan."""

import cv2
import numpy as np


def _add_bracketed_measurements(
    levels: np.ndarray, progress: np.ndarray, crossings: np.ndarray, depth: np.ndarray,
    paper: np.ndarray, contrast: np.ndarray, matches_paper: np.ndarray,
    observed: np.ndarray, locations: np.ndarray, groups: list[np.ndarray],
) -> None:
    # Strong measurements set the search corridor. Weak measurements still
    # need a unique optical crossing and flat collars on both sides of it.
    strong = [group[observed[group]] for group in groups]
    supported_groups = [i for i, (group, hits) in enumerate(zip(groups, strong, strict=True))
                        if len(hits) >= .60 * len(group)]
    gaps = [(strong[left][-5:], strong[right][:5])
            for left, right in zip(supported_groups, supported_groups[1:]) if right - left > 1]
    if not gaps:
        return
    # Prefer the true ends of a long raw gap when a group boundary splits
    # the sustained strong measurements that lead into it.
    hits = np.flatnonzero(observed)
    raw_gaps = np.flatnonzero(np.diff(hits) >= 7)
    narrow = [(hits[i - 4:i + 1], hits[i + 1:i + 6]) for i in raw_gaps if 4 <= i < len(hits) - 5]
    gaps = narrow + gaps
    collar_noise = np.maximum(
        np.mean(np.abs(levels[:, :7] - paper[:, None]), axis=1),
        np.mean(np.abs(levels[:, -7:] - (paper + contrast)[:, None]), axis=1),
    )
    cells = np.arange(crossings.shape[1])
    outside, inside = progress[:, np.maximum(cells - 10, 0)], progress[:, np.minimum(cells + 11, len(cells))]
    weak = crossings & (outside <= .25) & (outside >= -.25) & (inside >= .75) & (inside <= 1.25)
    weak &= (matches_paper & (np.abs(contrast) >= np.maximum(5, 4 * collar_noise)))[:, None]
    unique = weak.sum(axis=1) == 1
    candidate = depth[np.arange(len(depth)), weak.argmax(axis=1)] + .5

    for before, after in gaps:
        left, right = before[-1], after[0]
        # The span includes the sampled ends of the neighboring strong groups.
        if right - left > .38 * len(observed) or left - before[0] > 8 or after[-1] - right > 8:
            continue
        rows = np.arange(left + 1, right)
        bounds = (np.median(before), np.median(after))
        guide = np.interp(rows, bounds, [np.median(locations[before]), np.median(locations[after])])
        paper_guide = np.interp(rows, bounds, [np.median(paper[before]), np.median(paper[after])])
        supported = rows[unique[rows] & (np.abs(candidate[rows] - guide) <= 1.5)
                         & (np.abs(paper[rows] - paper_guide) <= 6)]
        if len(supported) < .75 * len(rows) or np.max(np.diff(np.r_[left, supported, right])) > 6:
            continue
        bright = np.mean(contrast[supported] > 0)
        if .10 < bright < .90:
            continue
        added = supported[~observed[supported]]
        observed[added] = True
        locations[added] = candidate[added]


def _has_competing_step(samples: np.ndarray, offsets: np.ndarray, seam: np.ndarray, strong: np.ndarray) -> bool:
    """Find a second, spatially coherent color transition next to the selected seam."""
    # Compare flat collars on either side of each possible crossing.
    # Keep raw colors so a nearly isoluminant step can also veto the crop.
    windows = np.lib.stride_tricks.sliding_window_view(samples, 13, axis=1)
    before, after = windows[..., :4], windows[..., -4:]
    delta = np.mean(after, axis=-1) - np.mean(before, axis=-1)
    change = np.max(np.abs(delta), axis=2)
    collar_variation = np.maximum(np.ptp(before, axis=-1).max(axis=2), np.ptp(after, axis=-1).max(axis=2))
    noise = np.median(np.max(np.abs(np.diff(samples[:, :7], axis=1)), axis=2), axis=1)
    distance = offsets[6:-6] - seam[:, None]
    localized = collar_variation <= np.maximum(1.25, .22 * change)
    minimum_strength = np.maximum(4.5, 4 * noise)
    rows = np.arange(len(seam))
    for direction in (-1, 1):
        separated = (direction * distance >= 2.5) & (direction * distance <= 6)
        candidates = separated & localized
        peaks = np.argmax(np.where(candidates, change, 0), axis=1)
        supported = strong & candidates[rows, peaks] & (change[rows, peaks] >= minimum_strength)
        positions, steps = distance[rows, peaks], delta[rows, peaks]
        magnitude = np.linalg.norm(steps, axis=1)
        same_color_change = np.sum(steps[1:] * steps[:-1], axis=1) >= .6 * magnitude[1:] * magnitude[:-1]
        aligned = supported[:-1] & supported[1:] & (np.abs(np.diff(positions)) <= .65) & same_color_change
        if np.any(aligned[:-2] & aligned[1:-1] & aligned[2:]):
            return True
    return False


def _trace_side(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, paper_color: list[float] | tuple[int, ...],
) -> tuple[np.ndarray, float, float, bool, bool] | None:
    tangent = end - start
    length = np.linalg.norm(tangent)
    if length < 60:
        return None
    inward = np.float32([-tangent[1], tangent[0]]) / length
    span = np.linspace(.025, .975, 128, dtype=np.float32)
    offsets = np.arange(-8., 8.1, .25, dtype=np.float32)
    points = start + span[:, None, None] * tangent + offsets[None, :, None] * inward
    if np.any((points < 0) | (points > np.array(image.shape[1::-1]) - 1)):
        return None
    low = np.floor(points.min(axis=(0, 1))).astype(int)
    high = np.ceil(points.max(axis=(0, 1))).astype(int) + 1
    strip = image[low[1]:high[1], low[0]:high[0]].astype(np.float32)
    samples = cv2.remap(strip, np.float32(points[..., 0] - low[0]), np.float32(points[..., 1] - low[1]), cv2.INTER_LINEAR)
    levels = cv2.cvtColor(samples, cv2.COLOR_BGR2GRAY)
    paper = np.median(levels[:, :7], axis=1)
    scene = np.median(levels[:, -7:], axis=1)
    contrast = scene - paper
    matches_paper = np.abs(paper - np.dot(paper_color, (.114, .587, .299))) < 20

    progress = (levels - paper[:, None]) / np.where(contrast != 0, contrast, 1)[:, None]
    earlier, later = progress[:, :-1], progress[:, 1:]
    depth = offsets[:-1] + .25 * np.clip((.5 - earlier) / np.maximum(later - earlier, .001), 0, 1)
    cells = np.arange(len(offsets) - 1)
    # The crossing must enter the picture and stay there, including when the
    # first sharp line encountered is a decorative stroke in the paper rim.
    crossings = (earlier < .5) & (later >= .5) & (np.abs(depth) <= 6.5)
    before = progress[:, np.maximum(cells - 3, 0)]
    after = progress[:, np.minimum(cells + 5, len(offsets) - 1)]
    crossings &= (before <= .38) & (after >= .60)
    strong = crossings & ((np.abs(contrast) >= 10) & matches_paper)[:, None]
    strong_rows = strong.any(axis=1)
    if strong_rows.mean() < .65:
        return None
    nearest = np.argmin(np.where(strong, np.abs(depth), np.inf), axis=1)
    # The line fits the first owned pixel center, half a pixel past the optical transition.
    optical = depth[np.arange(len(span)), nearest]
    locations = optical + .5
    groups = np.array_split(np.arange(len(span)), 12)
    observed = strong_rows.copy()
    _add_bracketed_measurements(
        levels, progress, crossings, depth, paper, contrast, matches_paper, observed, locations, groups,
    )
    knots = []
    for group in groups:
        visible = group[observed[group]]
        if len(visible) >= .60 * len(group):
            knots.append((np.median(span[visible]), np.median(locations[visible])))
    knots = np.asarray(knots)
    if len(knots) < 10 or knots[0, 0] > .13 or knots[-1, 0] < .87:
        return None
    positions, depths = knots.T
    error = np.abs(locations[observed] - np.interp(span[observed], positions, depths))
    if np.percentile(error, 80) > .8:
        return None
    detrended = depths - np.polyval(np.polyfit(positions, depths, 1), positions)
    polyline = start + positions[:, None] * tangent + depths[:, None] * inward
    # Weak transitions can locate a likely seam while tracking a shadow instead
    # of a material change. Derive footprint support from the strong rows alone.
    continuous = bool(np.max(np.diff(np.flatnonzero(np.r_[True, strong_rows, True]))) <= 3)
    competing = _has_competing_step(samples, offsets, optical, strong_rows)
    return np.float32(polyline), float(np.ptp(detrended) / 2), float(np.max(np.abs(depths))), continuous, competing


def trace_motif_boundary(image: np.ndarray, quad: np.ndarray, edges: dict) -> tuple[list[np.ndarray], bool, bool, bool]:
    """Return native traces, curvature, chord disagreement and four-side continuity."""
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return [], False, False, False
    traces = []
    curved = False
    disagrees = False
    continuous = True
    for side, start, end in zip(("top", "right", "bottom", "left"), quad, np.roll(quad, -1, axis=0), strict=True):
        edge = edges.get(side) or {}
        paper = edge.get("rim_bgr")
        if not edge.get("is_physical_border") or not paper:
            continue
        evidence = _trace_side(image, start, end, paper)
        if evidence is None:
            continue
        polyline, deviation, distance, side_continuous, competing = evidence
        traces.append(polyline)
        curved |= deviation > 1.05
        disagrees |= distance > 1.30 or competing
        continuous &= side_continuous and not competing
    return traces, curved, disagrees, continuous and len(traces) == 4


def _cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _is_simple_outline(points: np.ndarray) -> bool:
    ends = np.roll(points, -1, axis=0)
    lower, upper = np.minimum(points, ends), np.maximum(points, ends)
    overlapping = np.all((lower[:, None] <= upper[None, :] + 1e-5)
                         & (upper[:, None] >= lower[None, :] - 1e-5), axis=2)
    tangents = ends - points
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    start_side = _cross(tangents[:, None], points[None, :] - points[:, None])
    end_side = _cross(tangents[:, None], ends[None, :] - points[:, None])
    straddles = (np.minimum(start_side, end_side) <= 1e-5) & (np.maximum(start_side, end_side) >= -1e-5)
    candidates = np.triu(overlapping & straddles & straddles.T, k=2)
    candidates[0, -1] = False
    return not candidates.any()


def _corner_between(previous: np.ndarray, following: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    left = previous[-1] - previous[-2]
    right = following[1] - following[0]
    left_length, right_length = np.linalg.norm(left), np.linalg.norm(right)
    if min(left_length, right_length) <= 1e-5:
        return None
    left /= left_length
    right /= right_length
    crossing = _cross(left, right)
    if crossing < .25:
        return None
    gap = following[0] - previous[-1]
    forward, backward = _cross(gap, right) / crossing, _cross(gap, left) / crossing
    return previous[-1] + forward * left, float(forward), float(backward)


def close_motif_outline(quad: np.ndarray, traces: list[np.ndarray]) -> np.ndarray | None:
    """Join four clockwise traces of first owned source pixel centers for a review outline."""
    if len(traces) != 4:
        return None
    try:
        quad = np.asarray(quad, dtype=np.float32)
        observed = [np.asarray(trace, dtype=np.float64) for trace in traces]
    except (TypeError, ValueError):
        return None
    if (quad.shape != (4, 2) or not np.isfinite(quad).all()
            or not cv2.isContourConvex(quad) or cv2.contourArea(quad, oriented=True) <= 0):
        return None
    quad = quad.astype(np.float64)
    chords = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(chords, axis=1)
    directions = chords / lengths[:, None]
    for seed, direction, length, trace in zip(quad, directions, lengths, observed, strict=True):
        if trace.ndim != 2 or trace.shape[1] != 2 or not 3 <= len(trace) <= 128 or not np.isfinite(trace).all():
            return None
        along = (trace - seed) @ direction / length
        if (along[0] < -1e-5 or along[-1] > 1 + 1e-5 or along[0] > .10 or along[-1] < .90
                or np.any(np.diff(along) <= 1e-5) or np.max(np.diff(along)) > .18
                or np.max(np.abs(_cross(direction, trace - seed))) > 7.0 + 1e-5):
            return None

    outline = []
    for index, trace in enumerate(observed):
        previous = observed[index - 1]
        join = _corner_between(previous, trace)
        if join is None:
            return None
        corner, forward, backward = join
        if not (-1e-5 <= forward <= .12 * lengths[index - 1] and -.12 * lengths[index] <= backward <= 1e-5):
            return None
        # Match the original optical search range and bound the unobserved bend near the corner.
        if (np.linalg.norm(corner - quad[index]) > 8
                or abs(_cross(directions[index - 1], corner - previous[-1])) > 2
                or abs(_cross(directions[index], corner - trace[0])) > 2):
            return None
        outline.extend((corner, *trace))

    points = np.float32(outline)
    keep = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > 1e-4
    points = points[keep]
    if len(points) < 4 or cv2.contourArea(points, oriented=True) <= 0 or not _is_simple_outline(points):
        return None
    return points


def _corner_arm_supported(
    image: np.ndarray, corner: np.ndarray, tangent: np.ndarray, along: float, paper_color: np.ndarray,
) -> bool:
    tangent = tangent / np.linalg.norm(tangent)
    inward = np.array([-tangent[1], tangent[0]])
    # Sample both sides of the expected optical crossing, half a pixel before the owned center.
    near = np.array([2.25, 3.25, 4.75, 6.25, 8.25]) * along
    distance = np.array([-4, -3, -2, -1.5, -.5, .5, 1.5, 3, 4])
    points = corner + near[:, None, None] * tangent + distance[None, :, None] * inward
    if np.any((points < 0) | (points > np.array(image.shape[1::-1]) - 1)):
        return False
    low = np.floor(points.min(axis=(0, 1))).astype(int)
    high = np.ceil(points.max(axis=(0, 1))).astype(int) + 1
    patch = image[low[1]:high[1], low[0]:high[0]].astype(np.float32)
    samples = cv2.remap(patch, np.float32(points[..., 0] - low[0]), np.float32(points[..., 1] - low[1]), cv2.INTER_LINEAR)
    gray = cv2.cvtColor(samples, cv2.COLOR_BGR2GRAY)
    paper = np.median(gray[:, :3], axis=1)
    motif = np.median(gray[:, -3:], axis=1)
    contrast = motif - paper
    progress = (gray - paper[:, None]) / np.where(contrast != 0, contrast, 1)[:, None]
    supported = ((np.abs(contrast) >= 12) & (np.abs(paper - paper_color @ [.114, .587, .299]) <= 15)
                 & (np.ptp(gray[:, :3], axis=1) <= 8) & (np.ptp(progress[:, -3:], axis=1) <= .35)
                 & (progress[:, 3] <= .38) & (progress[:, 5] >= .62) & (progress[:, 6] >= .72))
    # Proof several pixels from the corner cannot rule out a clipped or protruding corner.
    return bool(supported[:2].all() and np.count_nonzero(supported) >= 4)


def motif_outline_corners_supported(
    image: np.ndarray, outline: np.ndarray, traces: list[np.ndarray], edges: dict,
) -> bool:
    """Confirm the extrapolated motif corners against both source-visible edge transitions."""
    if len(traces) != 4 or len(outline) < 4 or image.ndim != 3 or image.shape[2] != 3:
        return False
    names = ("top", "right", "bottom", "left")
    for index, current in enumerate(traces):
        previous = traces[index - 1]
        if len(previous) < 2 or len(current) < 2:
            return False
        join = _corner_between(previous.astype(np.float64), current.astype(np.float64))
        if join is None:
            return False
        corner = join[0]
        if not np.any(np.linalg.norm(outline - corner, axis=1) <= 1e-3):
            return False
        for side, tangent, along in (
            (names[index - 1], previous[-1] - previous[-2], -1),
            (names[index], current[1] - current[0], 1),
        ):
            edge = edges.get(side) or {}
            rim = np.asarray(edge.get("rim_bgr"), dtype=np.float32)
            if (not edge.get("is_physical_border") or rim.shape != (3,) or not np.isfinite(rim).all()
                    or not _corner_arm_supported(image, corner, tangent, along, rim)):
                return False
    return True
