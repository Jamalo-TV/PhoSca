"""Measure a narrow paper-to-motif transition in the original scan."""

import cv2
import numpy as np


def _trace_side(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, paper_color: list[float] | tuple[int, ...],
) -> tuple[np.ndarray, float, float] | None:
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
    crossings &= ((np.abs(contrast) >= 10) & matches_paper)[:, None]
    observed = crossings.any(axis=1)
    if observed.mean() < .65:
        return None
    nearest = np.argmin(np.where(crossings, np.abs(depth), np.inf), axis=1)
    # The line fits the first owned pixel center, half a pixel past the optical transition.
    locations = depth[np.arange(len(span)), nearest] + .5
    knots = []
    for group in np.array_split(np.arange(len(span)), 12):
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
    return np.float32(polyline), float(np.ptp(detrended) / 2), float(np.max(np.abs(depths)))


def trace_motif_boundary(image: np.ndarray, quad: np.ndarray, edges: dict) -> tuple[list[np.ndarray], bool, bool]:
    """Return supported native sides, curvature evidence and disagreement with the chord."""
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return [], False, False
    traces = []
    curved = False
    disagrees = False
    for side, start, end in zip(("top", "right", "bottom", "left"), quad, np.roll(quad, -1, axis=0), strict=True):
        edge = edges.get(side) or {}
        paper = edge.get("rim_bgr")
        if not edge.get("is_physical_border") or not paper:
            continue
        evidence = _trace_side(image, start, end, paper)
        if evidence is None:
            continue
        polyline, deviation, distance = evidence
        traces.append(polyline)
        curved |= deviation > 1.05
        disagrees |= distance > 1.30
    return traces, curved, disagrees


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
