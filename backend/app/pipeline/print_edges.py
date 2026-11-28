"""Locate the outer edges of a proposed print in the source image."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


PRINT_SHADOW_EDGE_REVIEW = "PRINT_SHADOW_EDGE_REVIEW"
PRINT_FAINT_RIM_REVIEW = "PRINT_FAINT_OUTER_RIM_REVIEW"
PRINT_OUTER_EDGE_CONFLICT_REVIEW = "PRINT_OUTER_EDGE_CONFLICT_REVIEW"
OPPOSING_OUTER_EDGES_CHECK = "opposing_outer_edges_check_required"
PRINT_SHARED_RIM_UNCERTAIN = "PRINT_SHARED_RIM_UNCERTAIN"
SHARED_PRINT_RIM_SOURCE = "shared_print_rim"
PRINT_EDGE_CONFIRMATION = "PRINT_EDGE_CONFIRMATION_REQUIRED"
COMPETING_PRINT_EDGES = "competing_print_edges"
POSSIBLE_BACKED_PAPER_EDGE = "possible_backed_paper_edge"
PRINT_BOUNDARY_REVISION = 17
CLASSICAL_CONTOUR_REFINEMENT_REVISION = 1
_PRINT_SIDES = ("top", "right", "bottom", "left")


def bowed_print_sides(evidence: dict) -> list[int]:
    bows = np.array([(evidence.get(side) or {}).get("measured_bow_px") or 0 for side in _PRINT_SIDES])
    adjacent = np.roll(bows, 1)
    paired = (bows * adjacent > 0) & (np.minimum(np.abs(bows), np.abs(adjacent)) >= .4)
    return np.flatnonzero((np.abs(bows) >= 1) | paired | np.roll(paired, -1)).tolist()


def _measured_side_bow(fitted: np.ndarray, length: float) -> float | None:
    # Median offsets in separated bands suppress individual texture and mount
    # edges. A projective transform preserves straight lines in the source.
    fractions = fitted[:, 0] / length
    bands = []
    for start in np.arange(7) / 7:
        points = fitted[(fractions >= start) & (fractions < start + 1 / 7)]
        if len(points) >= 2:
            bands.append(np.median(points, axis=0))
    if len(bands) < 6:
        return None
    positions, depths = np.asarray(bands).T
    positions = 2 * positions / length - 1
    if positions[0] > -.64 or positions[-1] < .64:
        return None
    curve = np.polyfit(positions, depths, 2)
    if np.max(np.abs(depths - np.polyval(curve, positions))) > .22:
        return None
    return round(float(curve[0]), 4)


def _median_edge_width(lab: np.ndarray, valid: np.ndarray, columns: np.ndarray, inliers: np.ndarray) -> float | None:
    # Measure the spread of a single color transition on either side of the
    # fitted edge. Unrelated color changes and out-of-frame collars give no vote.
    inside_strip = (columns >= 4) & (columns + 5 < lab.shape[1])
    rows = np.flatnonzero(inliers & inside_strip)
    columns = columns[rows]
    profiles = lab[rows[:, None], columns[:, None] + np.arange(-4, 6)]
    transition = profiles[:, -1] - profiles[:, 0]
    contrast = np.linalg.norm(transition, axis=1)
    components = (np.diff(profiles, axis=1) * transition[:, None]).sum(axis=2) / np.maximum(contrast[:, None], 1)
    usable = ((contrast > 8) & (components.min(axis=1) >= -1.5)
              & valid[rows, columns - 4] & valid[rows, columns + 5])
    if np.count_nonzero(usable) < .5 * len(inliers):
        return None
    width = contrast[usable] / np.maximum(components[usable].max(axis=1), 1)
    return round(float(np.median(width)), 4)


def _observed_curve_segments(
    start: np.ndarray, tangent: np.ndarray, inward: np.ndarray, positions: np.ndarray,
    distances: np.ndarray, supported: np.ndarray,
) -> list[list[list[float]]]:
    hits = np.flatnonzero(supported)
    breaks = (np.diff(hits) != 1) | (np.abs(np.diff(distances[hits])) > 1.2)
    segments = []
    for run in np.split(hits, np.flatnonzero(breaks) + 1):
        if len(run) < 4:
            continue
        points = start + positions[run, None] * tangent + (distances[run, None] + .5) * inward
        sampled = cv2.approxPolyDP(np.float32(points), .28, False)[:, 0]
        segments.append(sampled.tolist())
    return segments


def _side_line(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, radius: int, *, outward_only: bool = False,
    rim_color: np.ndarray | None = None, collect_curve_segments: bool = False,
) -> tuple[np.ndarray | None, dict]:
    length = float(np.linalg.norm(end - start))
    along = (end - start) / length
    inward = np.array([-along[1], along[0]], dtype=np.float32)
    positions = np.linspace(0.035, 0.965, min(112, max(24, int(length / 3))), dtype=np.float32) * length
    margin = 4 if rim_color is not None else 2
    offsets = np.arange(-radius - margin, radius + margin + 1, dtype=np.float32)
    coordinates = start + positions[:, None, None] * along + offsets[None, :, None] * inward
    valid = ((coordinates >= 0) & (coordinates <= np.array(image.shape[1::-1]) - 1)).all(axis=2)
    colors = cv2.remap(image, coordinates[:, :, 0].copy(), coordinates[:, :, 1].copy(), cv2.INTER_LINEAR)
    lab = cv2.cvtColor(colors, cv2.COLOR_BGR2LAB).astype(np.float32)
    gradients = np.linalg.norm(np.diff(lab, axis=1), axis=2)
    gradients *= valid[:, 1:] & valid[:, :-1]
    middle = gradients[:, 1:-1]
    peaks = (middle >= gradients[:, :-2]) & (middle > gradients[:, 2:])
    peaks &= middle >= np.maximum(2.0, 2.0 * np.median(gradients, axis=1, keepdims=True))
    peaks &= np.abs(offsets[1:-2] + 0.5) <= radius + 0.5
    if outward_only:
        peaks &= offsets[1:-2] < -3
    if rim_color is not None:
        # Verify which side of a transition belongs to the paper of this print.
        # Blur and a cast-shadow boundary can otherwise outscore the print edge.
        distance = np.linalg.norm(lab - rim_color, axis=2)
        inside = (distance[:, 3:-1] <= 13) & (distance[:, 4:] <= 13)
        outside = (distance[:, 1:-3] > 7) & (distance[:, :-4] > 7)
        peaks[:, :-1] &= inside & outside & valid[:, 4:] & valid[:, :-4]
        peaks[:, -1] = False
    strengths = np.where(peaks, middle, 0)
    # Keep competing peaks separate until a line has been supported across the side.
    columns = np.argsort(strengths, axis=1, kind="stable")[:, -4:] + 1
    rows = np.arange(len(positions))[:, None]
    left, center, right = (gradients[rows, columns + shift] for shift in (-1, 0, 1))
    curvature = left + right - 2 * center
    subpixel = np.divide(0.5 * (left - right), curvature, out=np.zeros_like(center), where=curvature < -1e-3)
    distances = offsets[columns] + 0.5 + np.clip(subpixel, -0.5, 0.5)
    usable = peaks[rows, columns - 1]
    evidence = {"support": 0.0, "ambiguous": False, "corner_support": None,
                "maximum_residual_px": None, "search_radius_px": radius}
    if collect_curve_segments:
        evidence["curve_segments"] = []
    if np.mean(usable.any(axis=1)) < 0.55:
        return None, evidence

    # Deterministic RANSAC hypotheses from separated portions of the same side.
    def anchors(first: float, last: float) -> tuple[np.ndarray, np.ndarray]:
        rows = np.unique(np.linspace(first, last, 8).astype(int))
        row, peak = np.nonzero(usable[rows])
        return positions[rows[row]], distances[rows[row], peak]

    count = len(positions) - 1
    first_pos, first_dist = anchors(0, 0.33 * count)
    last_pos, last_dist = anchors(0.67 * count, count)
    if not len(first_pos) or not len(last_pos):
        return None, evidence
    slopes = ((last_dist[None, :] - first_dist[:, None]) / (last_pos[None, :] - first_pos[:, None])).ravel()
    starts = (first_dist[:, None] - slopes.reshape(len(first_pos), -1) * first_pos[:, None]).ravel()
    ends = starts + length * slopes
    plausible = (np.maximum(np.abs(starts), np.abs(ends)) <= radius + 0.5) & (np.abs(slopes) <= 0.13)
    starts, slopes = starts[plausible], slopes[plausible]
    if not len(starts):
        return None, evidence

    hypotheses = starts[:, None] + slopes[:, None] * positions
    delta = np.where(usable, np.abs(hypotheses[:, :, None] - distances), np.inf)
    inliers = delta.min(axis=2) <= 1.1
    support = inliers.mean(axis=1)
    weights = np.clip(np.log1p(center) / np.log(48.0), 0.0, 1.0)
    strength = np.where(delta <= 1.1, weights, 0).max(axis=2).mean(axis=1)
    # Evidence needs to be distributed over the side, even when its center is distinctive.
    coverage = np.stack([part.mean(axis=1) for part in np.array_split(inliers, 4, axis=1)], axis=1)
    residual_score = (1 - np.minimum(delta.min(axis=2), 1)).mean(axis=1)
    quality = 0.64 * support + 0.28 * strength + 0.08 * residual_score - 0.035 * np.abs(hypotheses).mean(axis=1) / radius
    distributed = (support >= 0.66) & (coverage.min(axis=1) >= 0.28)
    if rim_color is not None:
        # Matching paper can hide the end of a seam where two prints meet.
        # Only one end can be inferred. Its adjacent edge must corroborate it.
        distributed |= ((support >= 0.78) & (coverage[:, 1:3].min(axis=1) >= 0.8)
                        & (coverage[:, [0, 3]].max(axis=1) >= 0.8))
    quality = np.where(distributed, quality, -1.0)
    winner = int(np.argmax(quality))
    evidence["support"] = round(float(support[winner]), 4)
    if quality[winner] < 0:
        return None, evidence
    separation = np.abs(hypotheses - hypotheses[winner]).mean(axis=1)
    competitors = (separation >= 2.4) & (quality >= quality[winner] - 0.045)
    if competitors.any():
        evidence["ambiguous"] = True
        if collect_curve_segments and rim_color is None and not outward_only:
            strong = (support >= .88) & (coverage.min(axis=1) >= .70)
            rivals = np.flatnonzero(competitors & strong)
            if strong[winner] and len(rivals):
                rival = rivals[np.argmax(quality[rivals])]
                traces = []
                for index in (winner, rival):
                    ends = np.flatnonzero(inliers[index])[[0, -1]]
                    points = start + positions[ends, None] * along + hypotheses[index, ends, None] * inward
                    traces.append(points.tolist())
                evidence[COMPETING_PRINT_EDGES] = traces
        return None, evidence

    closest = delta[winner].argmin(axis=1)
    rows = np.arange(len(positions))
    nearest = distances[rows, closest]
    chosen = inliers[winner]
    corner_span = int(np.ceil(0.12 * len(chosen)))
    start_support = round(float(chosen[:corner_span].mean()), 4)
    end_support = round(float(chosen[-corner_span:].mean()), 4)
    evidence.update(corner_support=min(start_support, end_support), start_support=start_support, end_support=end_support)
    fitted = np.column_stack([positions[chosen], nearest[chosen]]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(fitted, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    slope = vy / vx
    origin = y0 - slope * x0
    residuals = np.abs(fitted[:, 1] - origin - slope * fitted[:, 0])
    evidence["maximum_residual_px"] = round(float(residuals.max()), 4)
    if np.percentile(residuals, 90) > 1.05:
        return None, evidence
    evidence["edge_width_px"] = _median_edge_width(lab, valid, columns[rows, closest], chosen)
    if evidence["support"] >= .82:
        evidence["measured_bow_px"] = _measured_side_bow(fitted, length)
    if collect_curve_segments and abs(evidence.get("measured_bow_px") or 0) >= .4:
        evidence["curve_segments"] = _observed_curve_segments(start, along, inward, positions, nearest, chosen)
    # The transition is measured between pixels. Start at the first print pixel center.
    first = start + inward * (origin + 0.5)
    last = end + inward * (origin + length * slope + 0.5)
    line = np.cross([*first, 1.0], [*last, 1.0])
    return line / np.linalg.norm(line[:2]), evidence


@dataclass(frozen=True)
class _RimObservation:
    color: np.ndarray
    depth: float
    shadow_shoulder_support: float


def _shadow_shoulder_support(
    image: np.ndarray, positions: np.ndarray, inward: np.ndarray, outer_depth: np.ndarray, rim_luminance: np.ndarray,
) -> float:
    # A penumbra falls gradually on both sides of the outside gradient peak.
    # Its dark strip ends in a separate rise at the supported inside line.
    offsets = np.column_stack((outer_depth[:, None] + [-12, -10, -4, -3, -2, -1, 0, 1, 2, 3],
                               np.broadcast_to([3, 6], (len(positions), 2))))
    points = positions[:, None] + offsets[..., None] * inward
    valid = ((points >= 0) & (points <= np.array(image.shape[1::-1]) - 1)).all(axis=(1, 2))
    levels = _sample_lab(image, points.reshape(-1, 2))[:, 0].reshape(offsets.shape)
    far = levels[:, :2].mean(axis=1)
    shoulder = levels[:, 2:4].mean(axis=1)
    fall = far - levels[:, 8:10].mean(axis=1)
    largest_step = np.abs(np.diff(levels[:, 2:10], axis=1)).max(axis=1)
    rebound = levels[:, 10:].mean(axis=1) - rim_luminance.min(axis=1)
    support = valid & (far - shoulder >= 9) & (fall >= 20) & (largest_step <= .33 * fall) & (rebound >= 13)
    return float(np.mean(support)) if min(np.mean(part) for part in np.array_split(support, 4)) >= .50 else 0.0


def _observe_print_rim(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, line: np.ndarray, *, minimum_transition: float,
) -> _RimObservation | None:
    """Require material between an expanded line and the original proposal."""
    tangent = end - start
    inward = np.array([-tangent[1], tangent[0]], np.float32) / np.linalg.norm(tangent)
    positions = start + np.linspace(0.16, 0.84, 64, dtype=np.float32)[:, None] * tangent
    projected = -(positions @ line[:2] + line[2]) / float(line[:2] @ inward)
    depth = float(np.percentile(-projected, 15))
    if depth <= 4:
        return None
    offsets = np.column_stack((projected - 4, projected * 0.76, projected * 0.49, projected * 0.24))
    coordinates = positions[:, None, :] + offsets[:, :, None] * inward
    if np.any((coordinates < 0) | (coordinates > np.array(image.shape[1::-1]) - 1)):
        return None
    x, y = (coordinates[:, :, axis].astype(np.float32) for axis in (0, 1))
    strip = cv2.remap(image, x, y, cv2.INTER_LINEAR)
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = lab[:, 1:]
    rim = np.median(inside, axis=(0, 1))
    if rim[0] < 165 or np.linalg.norm(rim[1:] - 128) > 24:
        return None
    uniform = np.linalg.norm(inside - rim, axis=2) <= 13
    transition = np.linalg.norm(lab[:, 0] - inside[:, 0], axis=1)
    if np.mean(uniform) < .78 or np.mean(transition > minimum_transition) < .75:
        return None
    shadow_support = _shadow_shoulder_support(image, positions, inward, projected, inside[:, :, 0])
    return _RimObservation(rim, depth, shadow_support)


def _line_endpoints(start: np.ndarray, end: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = end - start
    inward = np.array([-tangent[1], tangent[0]], np.float32) / np.linalg.norm(tangent)
    offsets = -((np.array([start, end]) @ line[:2] + line[2]) / (line[:2] @ inward))
    first, last = np.array([start, end]) + offsets[:, None] * inward
    return first.astype(np.float32), last.astype(np.float32)


def unresolved_opposing_outer_edges(image: np.ndarray, quad: np.ndarray, boundary: dict) -> list[np.ndarray]:
    """Verify two opposing outside lines whose paper ownership is unresolved."""
    candidates = []
    for index, side in enumerate(_PRINT_SIDES):
        current = boundary.get(side) or {}
        outward = current.get("outward_search") or {}
        repeatable = outward.get("support", 0) >= .92 and (outward.get("corner_support") or 0) >= .75
        disputed = outward.get("ambiguous") is True and outward.get("support", 0) >= .96
        if (current.get("support", 0) >= .92 and (current.get("corner_support") or 0) >= .75
                and (repeatable or disputed) and outward.get("search_radius_px")):
            candidates.append((index, outward))
    if (len(candidates) != 2 or candidates[1][0] - candidates[0][0] != 2
            or sum(item.get("print_rim_supported") is True for _, item in candidates) != 1):
        return []

    traces = []
    for index, outward in candidates:
        start, end = quad[index], quad[(index + 1) % 4]
        line, support = _side_line(image, start, end, outward["search_radius_px"], outward_only=True)
        if (outward.get("ambiguous") is True and outward.get("print_rim_supported") is not True
                and support["ambiguous"] and support["support"] >= .96):
            # The competing lines justify review, but no single line can be shown here.
            continue
        if (line is None or support["ambiguous"] or support["support"] < .92
                or (support.get("corner_support") or 0) < .75):
            return []
        rim = _observe_print_rim(image, start, end, line, minimum_transition=5)
        if (rim is not None and rim.shadow_shoulder_support >= .70
                or (rim is not None) != (outward.get("print_rim_supported") is True)):
            return []
        first, last = _line_endpoints(start, end, line)
        distances = np.linalg.norm(np.float32([first - start, last - end]), axis=1)
        span = np.float32([first + .12 * (last - first), last + .12 * (first - last)])
        if (distances.min() <= 4 or np.ptp(distances) > .025 * np.linalg.norm(end - start)
                or not ((span >= 0) & (span <= np.array(image.shape[1::-1]) - 1)).all()):
            return []
        traces.append(span)
    return traces


def suggest_partial_corner_edge(image: np.ndarray, quad: np.ndarray, boundary: dict) -> np.ndarray | None:
    """Measure one open edge near an overextended corner for the review overlay."""
    sides = [boundary.get(name) or {} for name in _PRINT_SIDES]
    if any(side.get("ambiguous") for side in sides):
        return None
    weak_sides = [index for index, side in enumerate(sides) if side.get("support", 0) < .55]
    if len(weak_sides) != 1:
        return None
    weak_side, = weak_sides
    shortest = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
    if shortest < 40:
        return None
    radius = min(56, max(4, round(shortest * .042)))
    traces = []
    for corner in (weak_side, (weak_side + 1) % 4):
        opposite = (corner + 2) % 4
        fixed_sides = (sides[opposite], sides[(opposite - 1) % 4])
        if any(side.get("support", 0) < .90 or (side.get("corner_support") or 0) < .85 for side in fixed_sides):
            continue
        probe = quad.copy()
        # This parallelogram opens a new search window. Its inferred corner
        # and its unsupported side are never added to the review overlay.
        probe[corner] = quad[(corner - 1) % 4] + quad[(corner + 1) % 4] - quad[opposite]
        change = np.linalg.norm(probe[corner] - quad[corner])
        if not .08 * shortest <= change <= .30 * shortest or not cv2.isContourConvex(probe):
            continue
        if cv2.pointPolygonTest(quad, tuple(map(float, probe[corner])), True) < -1:
            continue
        measured_side = corner if corner != weak_side else (corner - 1) % 4
        start, end = probe[measured_side], probe[(measured_side + 1) % 4]
        line, measured = _side_line(image, start, end, radius)
        minimum_support = max(.90, sides[measured_side].get("support", 0) + .08)
        if line is None or measured["support"] < minimum_support or (measured["corner_support"] or 0) < .78:
            continue
        first, last = _line_endpoints(start, end, line)
        traces.append(np.array([first + .10 * (last - first), last - .10 * (last - first)]))
    return traces[0] if len(traces) == 1 else None


def _pale_outside_strip(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    tangent = end - start
    inward = np.array([-tangent[1], tangent[0]], np.float32) / np.linalg.norm(tangent)
    positions = start + np.linspace(0.15, 0.85, 48, dtype=np.float32)[:, None] * tangent
    coordinates = positions[:, None, :] - np.array([3.0, 5.0], dtype=np.float32)[None, :, None] * inward
    if np.any((coordinates < 0) | (coordinates > np.array(image.shape[1::-1]) - 1)):
        return False
    sampled = cv2.remap(image, coordinates[:, :, 0].copy(), coordinates[:, :, 1].copy(), cv2.INTER_LINEAR)
    levels = cv2.cvtColor(sampled, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    color = np.median(levels, axis=0)
    return bool(color[0] >= 165 and np.linalg.norm(color[1:] - 128) <= 24
                and np.mean(np.linalg.norm(levels - color, axis=1) <= 13) >= 0.75)


def missing_corner_material(image: np.ndarray, quad: np.ndarray) -> bool:
    """Detect clear rounded or chipped corners that a rectangular warp would fill with page."""
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    directions = sides / lengths[:, None]
    depth = float(np.clip(lengths.min() * 0.006, 2., 10.))
    for index, corner in enumerate(quad):
        first, second = depth * directions[index], -depth * directions[index - 1]
        diagonal = first + second
        points = np.array([
            corner - 1.3 * diagonal, corner + 0.75 * diagonal, corner + 3 * diagonal,
            corner + 5 * first + second, corner + first + 5 * second,
        ], dtype=np.float32)
        if np.any((points < 1) | (points > np.array(image.shape[1::-1]) - 2)):
            continue
        offsets = np.array([[-0.7, 0], [0.7, 0], [0, 0], [0, -0.7], [0, 0.7]], dtype=np.float32)
        positions = points[:, None, :] + offsets
        colors = cv2.remap(image, positions[:, :, 0].copy(), positions[:, :, 1].copy(), cv2.INTER_LINEAR)
        exterior, tip, *interior = np.median(cv2.cvtColor(colors, cv2.COLOR_BGR2LAB), axis=1)
        interior = np.array(interior)
        rim = np.median(interior, axis=0)
        if (np.linalg.norm(exterior - tip) <= 9
                and np.linalg.norm(tip - rim) >= 17
                and np.max(np.linalg.norm(interior - rim, axis=1)) <= 12):
            return True
    return False


def _sample_lab(image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coordinates = coordinates.astype(np.float32, copy=False)
    # OpenCV remap limits the length of its destination axes.
    samples = np.concatenate([
        cv2.remap(image, points[:, 0:1], points[:, 1:2], cv2.INTER_LINEAR)
        for points in np.array_split(coordinates, (len(coordinates) + 16_383) // 16_384)
    ])
    return cv2.cvtColor(samples, cv2.COLOR_BGR2LAB)[:, 0].astype(np.float32)


def observed_rim_points(
    image: np.ndarray, quad: np.ndarray, points: np.ndarray, observed_sides: dict[int, tuple[np.ndarray, float]],
) -> np.ndarray:
    """Verify disputed mask pixels against the source stock of a fitted outer edge."""
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    directions = sides / lengths[:, None]
    inward = np.column_stack((-directions[:, 1], directions[:, 0]))
    upper = np.array(image.shape[1::-1]) - 1
    confirmed = np.zeros(len(points), dtype=bool)
    for index, (stock, observed_depth) in observed_sides.items():
        delta = points - quad[index]
        depth, along = delta @ inward[index], delta @ directions[index]
        reach = min(observed_depth, 0.075 * float(lengths.min()), 24.)
        candidates = np.flatnonzero(
            ~confirmed & (depth > 0) & (depth <= reach) & (along >= 0) & (along <= lengths[index])
        )
        if not len(candidates):
            continue
        edge = quad[index] + along[candidates, None] * directions[index]
        end = depth[candidates, None]
        start = np.minimum(.8, end * .25)
        steps = max(4, int(np.ceil(reach / 1.5)) + 1)
        depths = start + (end - start) * np.linspace(0., 1., steps)
        offsets = np.column_stack((depths, np.broadcast_to([-2., -3.5], (len(candidates), 2))))
        probes = edge[:, None, :] + offsets[:, :, None] * inward[index]
        in_frame = ((probes >= 0) & (probes <= upper)).all(axis=(1, 2))
        if not in_frame.any():
            continue
        distances = np.linalg.norm(
            _sample_lab(image, probes[in_frame].reshape(-1, 2)).reshape(-1, steps + 2, 3) - stock, axis=2,
        )
        continuous_stock = (distances[:, :steps] <= 11).all(axis=1)
        exterior_transition = (distances[:, steps:] >= 10).all(axis=1)
        confirmed[candidates[in_frame]] = continuous_stock & exterior_transition
    return confirmed


def _is_pale_stock(color: np.ndarray) -> bool:
    return color[0] >= 185 and np.max(np.abs(color[1:] - 128)) <= 21


def print_corner_coverage(image: np.ndarray, quad: np.ndarray) -> tuple[bool, bool]:
    """Return whether stock is missing at a corner and whether a holder is visible."""
    sides = np.roll(quad, -1, axis=0) - quad
    directions = sides / np.linalg.norm(sides, axis=1)[:, None]
    inward = np.column_stack((-directions[:, 1], directions[:, 0]))
    edge_positions = quad[:, None, :] + np.linspace(.17, .83, 32)[None, :, None] * sides[:, None, :]
    edge_positions += 2.75 * inward[:, None, :]
    edges = _sample_lab(image, edge_positions.reshape(-1, 2)).reshape(4, 32, 3)
    stock = np.median(edges, axis=(0, 1))
    support = (np.linalg.norm(edges - stock, axis=2) <= 13).mean(axis=1)
    if not _is_pale_stock(stock) or np.min(support) < .78:
        return False, False
    height, width = image.shape[:2]
    outside = edge_positions - 5.5 * inward[:, None, :]
    outside_in_frame = ((outside >= 0) & (outside <= [width - 1, height - 1])).all(axis=(1, 2))
    background = np.median(_sample_lab(image, outside.reshape(-1, 2)).reshape(4, 32, 3), axis=1)
    adjoining = np.stack((background, np.roll(background, 1, axis=0)), axis=1)
    known_background = outside_in_frame & np.roll(outside_in_frame, 1)
    previous = -np.roll(directions, 1, axis=0)
    offsets = np.array([[2.5, 2.5], [5.0, 2.5], [2.5, 5.0]])
    tips = quad[:, None] + directions[:, None] * offsets[None, :, :1] + previous[:, None] * offsets[None, :, 1:]
    tips = _sample_lab(image, tips.reshape(-1, 2)).reshape(4, 3, 3)
    material = np.median(tips, axis=1)
    different = np.linalg.norm(material - stock, axis=1) >= 24
    coherent = np.max(np.linalg.norm(tips - material[:, None], axis=2), axis=1) <= 15
    candidates = different & coherent
    stock_covered = False
    if candidates.any():
        # A holder crosses the outside of both adjoining edges.
        exterior = np.stack((quad + 4 * directions - 2 * inward,
                             quad + 4 * previous - 2 * np.roll(inward, 1, axis=0)), axis=1)[candidates]
        in_frame = ((exterior >= 0) & (exterior <= [width - 1, height - 1])).all(axis=(1, 2))
        colors = _sample_lab(image, exterior.reshape(-1, 2)).reshape(-1, 2, 3)
        same_material = (np.linalg.norm(colors - material[candidates, None], axis=2) <= 18).all(axis=1)
        covered = in_frame & same_material
        stock_covered = bool(np.any(covered))
        # A clipped corner can expose backing on both sides of the missing stock.
        distinct_holder = (np.linalg.norm(adjoining[candidates] - material[candidates, None], axis=2) >= 20).all(axis=1)
        if np.any(covered & known_background[candidates] & distinct_holder):
            return True, True
    outlined = _outlined_corner_mount(image, quad, directions, previous, stock, adjoining, known_background)
    return stock_covered or outlined, outlined


def _outlined_corner_mount(
    image: np.ndarray, quad: np.ndarray, directions: np.ndarray, previous: np.ndarray,
    stock: np.ndarray, adjoining: np.ndarray, known_background: np.ndarray,
) -> bool:
    """Trace an outline of different color from both outside arms into a print corner."""
    shortest = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min()
    reach = int(np.clip(round(.14 * shortest), 16, 50))
    offsets = np.arange(-10, reach + 1, dtype=np.float32)
    first, second = np.meshgrid(offsets, offsets)
    first_arm = (first >= 5) & (second <= -2)
    second_arm = (second >= 5) & (first <= -2)
    interior = (first >= 3) & (second >= 3)
    upper = np.array(image.shape[1::-1]) - 1
    for index in np.flatnonzero(known_background):
        points = quad[index] + first[..., None] * directions[index] + second[..., None] * previous[index]
        in_frame = ((points >= 0) & (points <= upper)).all(axis=2)
        colors = _sample_lab(image, points.reshape(-1, 2)).reshape(*points.shape[:2], 3)[..., 1:]
        colored = np.linalg.norm(colors - stock[1:], axis=2) >= 22
        for neighbor in adjoining[index]:
            colored &= np.linalg.norm(colors - neighbor[1:], axis=2) >= 18
        count, labels = cv2.connectedComponents((colored & in_frame).astype(np.uint8))
        crosses = np.ones(count, dtype=bool)
        crosses[0] = False
        for region in (first_arm, second_arm, interior):
            crosses &= np.bincount(labels[region], minlength=count) >= 5
        if crosses.any():
            return True
    return False


def continuous_motif_rim(image: np.ndarray, quad: np.ndarray) -> bool:
    """Check an unbroken narrow band of the same stock outside all four motif sides."""
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    directions = sides / lengths[:, None]
    outward = np.column_stack((directions[:, 1], -directions[:, 0]))
    positions = np.concatenate([
        start + np.linspace(-2.5, length + 2.5, int(np.ceil(length)) + 6)[:, None] * direction + 2.5 * normal
        for start, length, direction, normal in zip(quad, lengths, directions, outward, strict=True)
    ])
    if np.any(positions < 0) or np.any(positions > np.array(image.shape[1::-1]) - 1):
        return False
    colors = _sample_lab(image, positions)
    stock = np.median(colors, axis=0)
    return _is_pale_stock(stock) and bool(np.all(np.linalg.norm(colors - stock, axis=1) <= 16))


def _edge_band(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, offsets: list[float], *, fractions: np.ndarray | None = None,
) -> np.ndarray | None:
    edge = end - start
    inward = np.array([-edge[1], edge[0]], np.float32) / np.linalg.norm(edge)
    if fractions is None:
        fractions = np.linspace(.10, .90, 64, dtype=np.float32)
    middle = start + fractions[:, None] * edge
    points = middle[:, None] + np.float32(offsets)[None, :, None] * inward
    if np.any((points < 0) | (points > np.array(image.shape[1::-1]) - 1)):
        return None
    return _sample_lab(image, points.reshape(-1, 2)).reshape(*points.shape[:-1], 3)


def line_fit_omits_print_stock(image: np.ndarray, contour: np.ndarray, fitted: np.ndarray) -> bool:
    """Keep a source-supported contour when a fitted side moves onto the print."""
    edges = np.roll(contour, -1, axis=0) - contour
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 260:
        return False
    inward = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    movement = fitted - contour
    depths = np.stack(((movement * inward).sum(axis=1), (np.roll(movement, -1, axis=0) * inward).sum(axis=1)))
    narrowed = np.flatnonzero(depths.min(axis=0) >= max(15., .04 * lengths.min()))
    if len(narrowed) != 1:
        return False
    side, = narrowed
    after = (side + 1) % 4
    if np.max(np.abs(np.delete(depths, side, axis=1))) > .025 * lengths.min():
        return False
    depth = float(depths[:, side].min())
    if depths[:, side].max() > .18 * lengths.min():
        return False

    body = np.linspace(.05, .95, 96, dtype=np.float32)
    ends = np.float32([13, 20, 27, 34]) / lengths[side]
    along = np.r_[body, ends, 1 - ends, -ends, 1 + ends]
    offsets = np.linspace(0, depth - 5, min(128, int(np.ceil(depth)) - 4))
    strip = _edge_band(image, contour[side], contour[after],
                       [-.18 * depth, -.10 * depth, -.065 * depth, -.04 * depth, *offsets], fractions=along)
    fitted_exterior = _edge_band(image, fitted[side], fitted[after], [-min(12, .22 * depth), -min(7, .14 * depth)],
                                 fractions=body)
    if strip is None or fitted_exterior is None:
        return False
    sampled_stock = strip[:, 4:]
    settled = int(np.searchsorted(offsets, .16 * depth))
    inside = sampled_stock[:len(body)]
    stock = np.median(inside[:, settled:], axis=(0, 1))
    matching = np.linalg.norm(inside - stock, axis=2) <= 13
    # Find a second boundary after paper begins and before the fitted edge.
    steady = matching[:, :-2] & matching[:, 1:-1] & matching[:, 2:]
    before = np.maximum.accumulate(steady, axis=1)[:, :-5]
    later = np.maximum.accumulate(steady[:, ::-1], axis=1)[:, ::-1][:, 5:]
    gap = ~matching[:, 3:-4] & ~matching[:, 4:-3]
    interrupted = np.any(before & gap & later, axis=1)
    core = matching[:, settled:]
    exterior = np.linalg.norm(strip[:len(body), 2:4] - stock, axis=2) > 13
    if (not _is_pale_stock(stock) or any(part.mean() < .88 for part in np.array_split(core, 4))
            or interrupted.mean() > .15 or np.min(core.mean(axis=0)) < .80 or exterior.mean() < .82
            or np.mean(np.linalg.norm(fitted_exterior - stock, axis=2) <= 13) < .84):
        return False
    step = sampled_stock[:, settled:settled + max(2, len(offsets) // 6)].mean(axis=1) - strip[:, :2].mean(axis=1)
    direction = np.median(step[:len(body)], axis=0)
    strength = float(np.linalg.norm(direction))
    if strength < 16:
        return False
    projected = step @ (direction / strength)
    if np.percentile(projected[:len(body)], 15) < .55 * strength:
        return False
    arms = np.median(projected[len(body):].reshape(4, -1), axis=1)
    if min(arms[:2]) < .48 * strength or np.max(np.abs(arms[2:])) > .44 * strength:
        return False

    # Both original flanks must meet the extended side across the lost strip.
    for flank, reverse in (((side - 1) % 4, True), (after, False)):
        fractions = np.float32([.21, .34, .47, .60]) * depth / lengths[flank]
        band = _edge_band(image, contour[flank], contour[(flank + 1) % 4], [-7, -4, 4, 7],
                          fractions=1 - fractions if reverse else fractions)
        if band is None:
            return False
        matches = np.linalg.norm(band - stock, axis=2) <= 14
        if matches[:, 2:].mean() < .75 or matches[:, :2].mean() > .25:
            return False
    return True


def paper_rim_reaches_neighbor(
    image: np.ndarray, quad: np.ndarray, side: int, neighbors: list[np.ndarray], *, partial_outward: bool = False,
) -> bool:
    """Check whether the outside of a fitted line continues into neighboring stock."""
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    start, end = quad[side], quad[(side + 1) % 4]
    reach = float(np.clip((.10 if partial_outward else .075) * lengths.min(), 6., 24.))
    # An original model polygon can begin on the neighbor's ink rather than its stock.
    # Allow a one-pixel contact with that polygon, still requiring continuous native stock.
    tolerance = 1. if partial_outward else 0.
    edge = sides[side] / lengths[side]
    outward = np.float32([edge[1], -edge[0]])
    extent = np.stack((start, end, start + reach * outward, end + reach * outward))
    low, high = extent.min(axis=0) - tolerance, extent.max(axis=0) + tolerance
    nearby = [polygon for polygon in neighbors
              if np.all(polygon.max(axis=0) >= low) and np.all(polygon.min(axis=0) <= high)]
    if not nearby:
        return False

    fractions = np.linspace(.15, .85, 16, dtype=np.float32)
    offsets = np.arange(3., reach, dtype=np.float32)
    # Leave the fitted line's antialiased transition out of the stock color.
    exterior = _edge_band(image, start, end, (-offsets).tolist(), fractions=fractions)
    interior = _edge_band(image, start, end, [2., 4.], fractions=fractions)
    if exterior is None or interior is None:
        return False
    stock = np.median(exterior[:, :2], axis=(0, 1))
    if not _is_pale_stock(stock) or np.mean(np.linalg.norm(interior - stock, axis=2) > 16) < .75:
        return False

    matching_sides = 0
    for index, (first, last) in enumerate(zip(quad, np.roll(quad, -1, axis=0), strict=True)):
        if index == side:
            continue
        across = _edge_band(image, first, last, [-3., 2., 4.], fractions=fractions)
        if across is None:
            continue
        difference = np.linalg.norm(across - stock, axis=2)
        if np.mean((difference[:, 0] > 11) & (difference[:, 1:] <= 11).all(axis=1)) >= .80:
            matching_sides += 1
    if matching_sides < 2:
        return False

    # A change of stock color or a band of album paper breaks the connection.
    unbroken = np.logical_and.accumulate(np.linalg.norm(exterior - stock, axis=2) <= 11, axis=1)
    positions = start + fractions[:, None] * sides[side]
    probes = positions[:, None, :] + offsets[None, :, None] * outward
    return any(sum(
        any(cv2.pointPolygonTest(polygon, tuple(point), True) > -tolerance for point in row[continuous])
        for row, continuous in zip(probes, unbroken, strict=True)
    ) >= 10 for polygon in nearby)


def _backing_color(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    inset = min(width, height) // 48
    xs = np.linspace(inset, width - 1 - inset, 64, dtype=np.float32)
    ys = np.linspace(inset, height - 1 - inset, 64, dtype=np.float32)
    samples = np.concatenate([
        np.column_stack([xs, np.full_like(xs, y)]) for y in (inset, height - 1 - inset)
    ] + [np.column_stack([np.full_like(ys, x), ys]) for x in (inset, width - 1 - inset)])
    return np.median(_sample_lab(image, samples), axis=0)


def _stock_ring(image: np.ndarray, quad: np.ndarray, backing: np.ndarray) -> bool:
    bands = [_edge_band(image, start, end, [-2, 2]) for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True)]
    if any(band is None for band in bands):
        return False
    outside, inside = np.moveaxis(np.asarray(bands), 2, 0)
    stock = np.median(inside, axis=(0, 1))
    return bool(
        _is_pale_stock(stock) and np.linalg.norm(stock - backing) > 16
        and (np.mean(np.linalg.norm(inside - stock, axis=2) <= 10, axis=1) >= .75).all()
        and (np.mean(np.linalg.norm(outside - backing, axis=2) <= 16, axis=1) >= .75).all()
    )


def _replace_quad_side(
    quad: np.ndarray, index: int, line: np.ndarray, radius: int, shape: tuple[int, ...],
) -> np.ndarray | None:
    homogeneous = np.column_stack((quad, np.ones(4)))
    lines = np.cross(homogeneous, np.roll(homogeneous, -1, axis=0))
    lines[index] = line
    crossings = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(np.abs(crossings[:, 2]) < 1e-6):
        return None
    candidate = np.float32(crossings[:, :2] / crossings[:, 2:3])
    if (np.max(np.linalg.norm(candidate - quad, axis=1)) > 1.8 * radius
            or not cv2.isContourConvex(candidate) or cv2.contourArea(candidate, oriented=True) <= 0
            or np.any((candidate < 0) | (candidate > np.array(shape[1::-1]) - 1))):
        return None
    return candidate


def retry_ambiguous_stock_edge(
    image: np.ndarray, proposal: np.ndarray, quad: np.ndarray, evidence: dict,
) -> tuple[np.ndarray, dict] | None:
    """Resolve one ambiguous edge of a contrast component with a continuous paper rim."""
    sides = ("top", "right", "bottom", "left")
    missing = [i for i, side in enumerate(sides) if evidence[side]["corner_support"] is None]
    if len(missing) != 1 or any(
        side["support"] < .85 or side["corner_support"] < .60 for side in evidence.values()
        if side["corner_support"] is not None
    ):
        return None
    index = missing[0]
    next_index = (index + 1) % 4
    start, end = proposal[[index, next_index]]
    backing = _backing_color(image)
    if not _stock_ring(image, proposal, backing):
        return None

    radius = 3
    line, observed = _side_line(image, start, end, radius)
    if line is None or observed["support"] < .83 or observed["corner_support"] < .55:
        return None
    candidate = _replace_quad_side(quad, index, line, radius, image.shape)
    if candidate is None:
        return None
    exterior = _edge_band(image, candidate[index], candidate[next_index], [-2, -3, -4, -5, -6, -7, -8])
    # A near-background surround can still belong to the same print.
    if (exterior is None or (np.linalg.norm(exterior - backing, axis=2) >= 2).mean(axis=0).max() >= .50
            or not _stock_ring(image, candidate, backing)):
        return None
    observed["matched_stock_ring"] = True
    return candidate, {**evidence, sides[index]: observed}


def suggest_ambiguous_stock_edge(
    image: np.ndarray, proposal: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray],
) -> tuple[np.ndarray, dict] | None:
    """Improve the review warp using the material observed on three fitted sides."""
    names = ("top", "right", "bottom", "left")
    ambiguous = [i for i, side in enumerate(names) if evidence[side]["ambiguous"]]
    if len(ambiguous) != 1:
        return None
    index = ambiguous[0]
    trusted = [i for i in range(4) if i != index]
    if any(evidence[names[i]]["support"] < .9 or (evidence[names[i]]["corner_support"] or 0) < .8
           or (evidence[names[i]]["maximum_residual_px"] or 0) > 1.1 for i in trusted):
        return None

    offsets = [-5, -3, 3, 5]
    bands = [_edge_band(image, quad[i], quad[(i + 1) % 4], offsets) for i in trusted]
    if any(band is None for band in bands):
        return None
    samples = np.stack(bands)
    stock = np.median(samples[:, :, 2:], axis=(0, 1, 2))
    backing = np.median(samples[:, :, :2], axis=(0, 1, 2))
    if not _is_pale_stock(stock) or np.linalg.norm(stock - backing) <= 16:
        return None

    def material_supported(band: np.ndarray) -> np.ndarray:
        outside, inside = band[..., :2, :], band[..., 2:, :]
        return ((np.linalg.norm(outside - backing, axis=-1) <= 16).all(axis=-1)
                & (np.linalg.norm(outside - stock, axis=-1) >= 12).all(axis=-1)
                & (np.linalg.norm(inside - stock, axis=-1) <= 12).all(axis=-1))

    if not (material_supported(samples).mean(axis=1) >= .9).all():
        return None
    radius = 3
    next_index = (index + 1) % 4
    line, observed = _side_line(image, proposal[index], proposal[next_index], radius, rim_color=stock)
    if line is None and evidence[names[index]]["search_radius_px"] > radius:
        # The physical edge can lie farther from the seed than the ink edge.
        radius = evidence[names[index]]["search_radius_px"]
        line, observed = _side_line(image, proposal[index], proposal[next_index], radius, rim_color=stock)
    if line is None or observed["support"] < .9 or observed["corner_support"] < .8:
        return None
    candidate = _replace_quad_side(quad, index, line, radius, image.shape)
    if candidate is None or any(cv2.intersectConvexConvex(candidate, cv2.convexHull(neighbor))[0] > 1 for neighbor in neighbors):
        return None
    band = _edge_band(image, candidate[index], candidate[next_index], offsets)
    if band is None or material_supported(band).mean() < .9:
        return None

    # Both arms of each changed corner must carry the same local transition.
    for side, from_end in (((index - 1) % 4, True), (index, False), (index, True), (next_index, False)):
        start, end = candidate[side], candidate[(side + 1) % 4]
        fractions = np.float32([4, 6, 8, 10, 13, 16, 20]) / np.linalg.norm(end - start)
        if fractions[-1] >= .5:
            return None
        arm = _edge_band(image, start, end, offsets, fractions=1 - fractions if from_end else fractions)
        if arm is None:
            return None
        supported = material_supported(arm)
        if not supported[:2].all() or np.count_nonzero(supported) < 5:
            return None
    return candidate, {**evidence, names[index]: {**evidence[names[index]], "review_stock_fit": observed}}


def _pale_backing_transition(band: np.ndarray, backing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(band - backing, axis=2)
    outside_distance = distance[:, :3]
    exterior = (np.median(outside_distance, axis=1) <= 14) & (outside_distance.max(axis=1) <= 18)
    inside_distance = np.median(distance[:, 3:], axis=1)
    interior = (inside_distance >= 12) & (inside_distance - np.median(outside_distance, axis=1) >= 5)
    return exterior & interior, (distance <= 14).all(axis=1)


def _pale_backing_at_corners(image: np.ndarray, candidate: np.ndarray, index: int, backing: np.ndarray) -> bool:
    for flank, ends_at_corner in (((index - 1) % 4, True), ((index + 1) % 4, False)):
        start, end = candidate[flank], candidate[(flank + 1) % 4]
        distances = np.float32([-10, -7, -4, -2, 4, 7, 10, 14, 18, 22])
        fractions = distances / np.linalg.norm(end - start)
        band = _edge_band(image, start, end, [-7, -5, -3, 2, 4, 6],
                          fractions=1 - fractions if ends_at_corner else fractions)
        if band is None:
            return False
        material, bare = _pale_backing_transition(band, backing)
        # Along each adjacent edge, the print must start at this corner.
        # An internal ink line leaves photographic material past the corner.
        before_corner = band[:4, 3:].mean(axis=1) - band[:4, :3].mean(axis=1)
        continued_flank = np.linalg.norm(np.median(before_corner, axis=0)) > 3
        if (continued_flank or np.count_nonzero(bare[:4]) < 3
                or np.count_nonzero(material[4:]) < 5 or not material[4:6].any()):
            return False
    start, end = candidate[index], candidate[(index + 1) % 4]
    span = np.linalg.norm(end - start)
    extension = np.float32([3, 5, 8, 12, 16, 20]) / span
    continued = _edge_band(image, start, end, [-5, -3, 3, 5], fractions=np.r_[-extension, 1 + extension])
    if continued is None:
        return False
    changes = continued[:, 2:].mean(axis=1) - continued[:, :2].mean(axis=1)
    arms = changes.reshape(2, -1, 3)
    # A cover crossing the print can terminate the two flanks of the
    # visible region, but its own edge continues past both print corners.
    directions = np.median(arms, axis=1)
    magnitude = np.linalg.norm(directions, axis=1)
    projection = np.sum(arms * directions[:, None], axis=2) / np.maximum(magnitude[:, None], 1)
    strong = np.mean(projection > 3, axis=1)
    aligned = directions[0] @ directions[1] > .8 * np.prod(magnitude)
    return not (bool(np.any(strong >= .83)) or bool(np.all(strong >= .60) and aligned))


def _only_pale_backing_removed(
    image: np.ndarray, original: np.ndarray, candidate: np.ndarray, index: int, backing: np.ndarray,
) -> bool:
    start, end = candidate[index], candidate[(index + 1) % 4]
    tangent = end - start
    length = np.linalg.norm(tangent)
    inward = np.float32([-tangent[1], tangent[0]]) / length
    original_side = original[(index + 1) % 4] - original[index]
    normal = np.float32([-original_side[1], original_side[0]]) / np.linalg.norm(original_side)
    if normal @ inward < .95:
        return False
    stations = int(np.clip(np.ceil(length) + 1, 96, 2048))
    centers = start + np.linspace(.025, .975, stations, dtype=np.float32)[:, None] * tangent
    removed = (centers - original[index]) @ normal / (normal @ inward)
    depth = float(removed.max())
    if depth < 3:
        return True
    offsets = np.arange(3, depth + 1, dtype=np.float32)
    points = centers[:, None] - offsets[:, None] * inward
    in_strip = (offsets <= removed[:, None]) & ((points >= 0) & (points <= np.array(image.shape[1::-1]) - 1)).all(axis=2)
    if not in_strip.any():
        return True
    distance = np.linalg.norm(_sample_lab(image, points[in_strip]) - backing, axis=1)
    return bool(distance.max() <= 20 and np.mean(distance <= 14) >= .98)


def strip_resembles_album_backing(
    image: np.ndarray, quad: np.ndarray, bitmap: np.ndarray, origin: np.ndarray,
) -> bool:
    """Describe a source comparison for review without assigning paper ownership."""
    points = cv2.findNonZero(bitmap)
    if points is None:
        return False
    points = points.reshape(-1, 2) + origin
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    maximum_depth = min(28, max(12, .08 * lengths.min()))
    candidates = []
    for index, (start, side, length) in enumerate(zip(quad, sides, lengths, strict=True)):
        tangent = side / length
        inward = np.float32([-tangent[1], tangent[0]])
        offset = points - start
        depth = offset @ inward
        span = offset @ tangent
        if depth.min() >= -maximum_depth and depth.max() < -1 and span.min() >= -1 and span.max() <= length + 1:
            candidates.append(index)
    if len(candidates) != 1:
        return False
    backing = _backing_color(image)
    colors = image[points[:, 1], points[:, 0]].reshape(-1, 1, 3)
    distance = np.linalg.norm(cv2.cvtColor(colors, cv2.COLOR_BGR2LAB).reshape(-1, 3) - backing, axis=1)
    if distance.max() > 13 or np.mean(distance <= 8) < .995:
        return False
    candidate, evidence, reasons = refine_print_boundary(image, quad, search_radius=3)
    return (candidate is not None and not reasons and np.linalg.norm(candidate - quad, axis=1).max() <= .85
            and all(edge["support"] >= .88 and (edge["corner_support"] or 0) >= .82 for edge in evidence.values())
            and _pale_backing_at_corners(image, quad, candidates[0], backing))


def suggest_missing_pale_backing_edge(
    image: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray],
) -> tuple[np.ndarray, dict] | None:
    """Measure one missing side for a review preview when adjacent prints have a pale gutter."""
    names = ("top", "right", "bottom", "left")
    missing = [index for index, name in enumerate(names) if evidence[name]["corner_support"] is None]
    if not neighbors or len(missing) != 1 or any(evidence[name]["ambiguous"] for name in names):
        return None
    index = missing[0]
    opposite = (index + 2) % 4
    previous, following = (index - 1) % 4, (index + 1) % 4
    if (evidence[names[opposite]]["support"] < .90 or evidence[names[opposite]]["corner_support"] < .65
            or evidence[names[previous]]["start_support"] < .65 or evidence[names[following]]["end_support"] < .65):
        return None
    backing = _backing_color(image)
    if not _is_pale_stock(backing):
        return None
    sides = np.roll(quad, -1, axis=0) - quad
    shortest = float(np.linalg.norm(sides, axis=1).min())
    axis = -sides[opposite] / np.linalg.norm(sides[opposite])
    normal = np.float32([-axis[1], axis[0]])
    offset = sides[index] @ normal
    if not max(6., .025 * shortest) <= abs(offset) <= .22 * shortest:
        return None
    radius = int(np.clip(round(.038 * shortest), 5, 18))
    start, end = quad[index], quad[following]
    hulls = [cv2.convexHull(neighbor) for neighbor in neighbors]
    alternatives = []
    for anchor in (0., .25, .5, .75, 1.):
        line, observed = _side_line(image, start + anchor * offset * normal, end - (1 - anchor) * offset * normal, radius)
        if line is None or observed["support"] < .90 or observed["corner_support"] < .75:
            continue
        candidate = _replace_quad_side(quad, index, line, max(radius, abs(offset)), image.shape)
        if (candidate is None or any(cv2.intersectConvexConvex(candidate, hull)[0] > 1. for hull in hulls)
                or not _pale_backing_at_corners(image, candidate, index, backing)):
            continue
        band = _edge_band(image, candidate[index], candidate[following], [-8, -5, -3, 3, 5, 8])
        if band is None:
            continue
        transitions, _ = _pale_backing_transition(band, backing)
        support = float(np.mean(transitions))
        if support >= .83 and _only_pale_backing_removed(image, quad, candidate, index, backing):
            alternatives.append((candidate, observed, support))
    if not alternatives:
        return None
    candidate, observed, support = max(alternatives, key=lambda entry: (entry[2], entry[1]["support"]))
    if any(np.max(np.linalg.norm(candidate - other, axis=1)) > 2. for other, _, _ in alternatives):
        return None
    side = {**observed, "review_pale_backing_fit": {"native_transition_support": round(support, 4)}}
    return candidate, {**evidence, names[index]: side}


def suggest_partial_backed_edge(
    image: np.ndarray, quad: np.ndarray, evidence: dict, neighbors: list[np.ndarray],
) -> list[np.ndarray]:
    """Show interrupted source evidence alongside a disputed pale gutter."""
    sides = [evidence.get(name) or {} for name in _PRINT_SIDES]
    weak = [index for index, side in enumerate(sides) if side.get("corner_support") is None]
    if not neighbors or len(weak) != 1 or any(side.get("ambiguous") for side in sides):
        return []
    index, = weak
    if (not .55 <= sides[index].get("support", 0) < .80
            or any(side.get("support", 0) < .90 or (side.get("corner_support") or 0) < .85
                   for other, side in enumerate(sides) if index != other)):
        return []

    edges = np.roll(quad, -1, axis=0) - quad
    shortest = float(np.linalg.norm(edges, axis=1).min())
    if shortest < 160:
        return []
    backing = _backing_color(image)
    if not _is_pale_stock(backing):
        return []

    opposite = edges[(index + 2) % 4]
    normal = np.float32([opposite[1], -opposite[0]]) / np.linalg.norm(opposite)
    divergence = float(edges[index] @ normal)
    if not max(6., shortest * .019) <= abs(divergence) <= shortest * .08:
        return []
    following = (index + 1) % 4
    start, end = quad[[index, following]] + np.float32([.5, -.5])[:, None] * divergence * normal
    radius = int(np.clip(round(shortest * .036), 6, 16))
    hulls = [cv2.convexHull(neighbor) for neighbor in neighbors]
    if any(cv2.intersectConvexConvex(quad, hull)[0] > 1 for hull in hulls):
        return []
    candidates = []
    for search in (radius - 2, radius, radius + 2):
        line, side = _side_line(image, start, end, search)
        if line is None or side["support"] < .69 or side["corner_support"] < .86:
            return []
        candidate = _replace_quad_side(quad, index, line, max(radius, abs(divergence)), image.shape)
        if candidate is None or any(cv2.intersectConvexConvex(candidate, hull)[0] > 1 for hull in hulls):
            return []
        shift = (candidate[[index, following]] - quad[[index, following]]) @ normal
        if shift.min() < -.75 or shift.max() < 5:
            return []
        candidates.append(candidate)
    candidate = candidates[1]
    if (any(np.linalg.norm(candidate - other, axis=1).max() > .75 for other in candidates)
            or not _only_pale_backing_removed(image, quad, candidate, index, backing)):
        return []

    for flank, at_end in (((index - 1) % 4, True), (following, False)):
        first, last = candidate[flank], candidate[(flank + 1) % 4]
        fraction = np.float32([-9, -7, -5, -3, 3, 5, 7, 9]) / np.linalg.norm(last - first)
        collar = _edge_band(image, first, last, [-8, -6, -4, 4, 6, 8],
                            fractions=1 + fraction if at_end else -fraction)
        if collar is None:
            return []
        outside, inside = (np.linalg.norm(np.median(part, axis=1) - backing, axis=1)
                           for part in (collar[:, :3], collar[:, 3:]))
        step = inside - outside
        if (np.median(step[:4]) < 15 or abs(np.median(step[4:])) > 3
                or np.count_nonzero((inside[4:] <= 12) & (outside[4:] <= 12) & (abs(step[4:]) <= 5)) < 3):
            return []

    return _observed_backed_edge_spans(image, candidate[index], candidate[following], backing)


def _observed_backed_edge_spans(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, backing: np.ndarray,
) -> list[np.ndarray]:
    edge = end - start
    length = float(np.linalg.norm(edge))
    inset = min(.05, 6 / length)
    fractions = np.linspace(inset, 1 - inset, int(np.clip(round(length), 72, 700)), dtype=np.float32)
    samples = _edge_band(image, start, end, list(range(-9, 10)), fractions=fractions)
    if samples is None:
        return []
    exterior = np.median(samples[:, 1:6], axis=1)
    interior = np.median(samples[:, 13:18], axis=1)
    outside = np.linalg.norm(exterior - backing, axis=1)
    difference = np.linalg.norm(interior - backing, axis=1) - outside
    uniform = (outside <= 12) & (np.linalg.norm(samples[:, 1:6] - backing, axis=2).max(axis=1) <= 17)
    transition = interior - exterior
    direction = transition / np.maximum(np.linalg.norm(transition, axis=1, keepdims=True), 1)
    slopes = (np.diff(samples, axis=1) * direction[:, None]).sum(axis=2)[:, 5:13]
    peak = np.argmax(slopes, axis=1) + 5
    measured = uniform & (difference >= 12) & (peak >= 7) & (peak <= 10) & (slopes.max(axis=1) >= 3)
    if measured.mean() < .45 or uniform.mean() < .95 or np.mean(difference >= 5) < .90:
        return []
    # Leave a gap in the overlay wherever the source no longer shows a step.
    intervals = np.flatnonzero(np.diff(np.r_[False, measured, False])).reshape(-1, 2)
    spans = [(fractions[first], fractions[stop - 1]) for first, stop in intervals
             if stop - first >= 3 and fractions[stop - 1] - fractions[first] >= .035]
    prefix = [span for span in spans if span[0] < .20]
    suffix = [span for span in spans if span[1] > .80]
    if not prefix or not suffix:
        return []
    longest = {max(group, key=lambda span: span[1] - span[0]) for group in (prefix, suffix)}
    return [start + np.float32(span)[:, None] * edge for span in sorted(longest)]


def _quad_from_lines(lines: np.ndarray) -> np.ndarray | None:
    intersections = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(np.abs(intersections[:, 2]) < 1e-6):
        return None
    return (intersections[:, :2] / intersections[:, 2:3]).astype(np.float32)


def _weak_rim_transition(image: np.ndarray, quad: np.ndarray) -> float | None:
    """Verify a small material change at four lines with terminating corners."""
    body_samples = 64
    bands = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        near = np.float32([6, 9, 12, 16, 21]) / np.linalg.norm(end - start)
        body = np.linspace(.14, .86, body_samples, dtype=np.float32)
        fractions = np.r_[body, near, 1 - near, -near, 1 + near]
        sampled = _edge_band(image, start, end, list(range(-14, 15)), fractions=fractions)
        if sampled is None:
            return None
        bands.append(sampled)
    profiles = np.asarray(bands)
    body = profiles[:, :body_samples]
    # The zero offset is index 14. The inside and outside averages start
    # five pixels from the candidate so blur at the line cannot dominate.
    stock = profiles[:, :, 19:24].mean(axis=2)
    step = stock - profiles[:, :, 5:10].mean(axis=2)
    direction = np.median(step[:, :body_samples], axis=(0, 1))
    magnitude = float(np.linalg.norm(direction))
    if not 3.4 <= magnitude <= 9.5:
        return None
    direction /= magnitude
    signed = step @ direction
    body_step = signed[:, :body_samples]
    middle = np.median(body_step, axis=1)
    if np.any(np.quantile(body_step, .10, axis=1) < 2.8):
        return None
    colors = np.median(stock[:, :body_samples], axis=1)
    if (np.any(colors[:, 0] < 210) or np.any(np.linalg.norm(colors[:, 1:] - 128, axis=1) > 17)
            or np.any(np.linalg.norm(colors - np.median(colors, axis=0), axis=1) >= 5)):
        return None
    corners = np.median(signed[:, body_samples:].reshape(4, 4, 5), axis=2)
    if (np.any(corners[:, :2] < .58 * middle[:, None])
            or np.any(np.abs(corners[:, 2:]) > np.maximum(1.8, .32 * middle[:, None]))):
        return None
    changes = np.diff(np.median(body, axis=1) @ direction, axis=1)
    central = changes[:, 11:16].sum(axis=1)
    opposite = np.maximum(0, -changes).sum(axis=1)
    return magnitude if np.all(central >= .76 * middle) and np.all(opposite <= .33 * middle) else None


def refine_print_boundary(
    image: np.ndarray, proposal: np.ndarray, *, search_radius: int | None = None,
    rim_color: np.ndarray | None = None, collect_curve_segments: bool = False,
) -> tuple[np.ndarray | None, dict, list[str]]:
    sides = np.linalg.norm(np.roll(proposal, -1, axis=0) - proposal, axis=1)
    radius = search_radius or min(56, max(4, round(float(sides.min()) * 0.042)))
    wider = min(130, round(float(sides.min()) * 0.112))
    maximum_excursion = radius
    fitted = []
    rings = {}
    supported_outward = {}
    conflicting_outward = 0
    rejected_shadow = False
    for index, (start, end) in enumerate(zip(proposal, np.roll(proposal, -1, axis=0))):
        line, result = _side_line(
            image, start, end, radius, rim_color=rim_color, collect_curve_segments=collect_curve_segments,
        )
        if search_radius is None and rim_color is None and not result["ambiguous"] and wider >= radius + 4:
            first, last = _line_endpoints(start, end, line) if line is not None else (start, end)
            if line is None or _pale_outside_strip(image, first, last):
                alternate, retry = _side_line(
                    image, first, last, wider, outward_only=True, collect_curve_segments=collect_curve_segments,
                )
                if line is not None and result["support"] >= .88:
                    conflicting_outward += int(retry["ambiguous"] and retry["support"] >= .88)
                observation = None if alternate is None else _observe_print_rim(
                    image, first, last, alternate, minimum_transition=8 if line is None else 5,
                )
                if observation is not None and observation.shadow_shoulder_support >= .70:
                    retry["shadow_shoulder_support"] = round(observation.shadow_shoulder_support, 4)
                    rejected_shadow = True
                    observation = None
                result["outward_search"] = {**retry, "print_rim_supported": observation is not None}
                if (line is not None and alternate is not None and retry["support"] >= .88
                        and (retry["corner_support"] or 0) >= .78):
                    supported_outward[index] = (alternate, retry)
                if observation is not None:
                    color, depth = observation.color, observation.depth
                    retry.update(observed_rim_lab=color.tolist(), observed_rim_depth_px=depth)
                    rings[index] = alternate, retry, color, line is not None
                    if line is None:
                        result.update(retry)
                        maximum_excursion = wider
                        line = alternate
        fitted.append((line, result))
    reasons = [PRINT_SHADOW_EDGE_REVIEW] if rejected_shadow else []
    weak_step = None
    resolved_outer_rim = False
    if not rejected_shadow and len(rings) < 4 and len(supported_outward) == 4:
        outer = _quad_from_lines(np.array([line for line, _ in supported_outward.values()]))
        if (outer is not None and cv2.isContourConvex(outer) and cv2.contourArea(outer, oriented=True) > 0
                and np.max(np.linalg.norm(outer - proposal, axis=1)) <= 1.8 * (wider + radius)):
            weak_step = _weak_rim_transition(image, outer)
    if weak_step is not None:
        for index, (alternate, retry) in supported_outward.items():
            evidence = {**fitted[index][1], **retry, "weak_rim_transition_lab": round(weak_step, 4)}
            fitted[index] = alternate, evidence
        maximum_excursion = wider + radius
        reasons.append(PRINT_FAINT_RIM_REVIEW)
    elif any(replaces_supported for _, _, _, replaces_supported in rings.values()):
        # A strong near line can bound the image ink. Move past it when a common
        # physical rim continues on more than one side. An incomplete rim is review evidence.
        coherent = max((
            [index for index, (_, _, color, _) in rings.items() if np.linalg.norm(color - seed) <= 14]
            for _, _, seed, _ in rings.values()
        ), key=len)
        if len(coherent) >= 2:
            for index in coherent:
                alternate, retry, _, _ = rings[index]
                result = fitted[index][1]
                result.update(retry)
                fitted[index] = alternate, result
            maximum_excursion = wider + radius
            resolved_outer_rim = len(coherent) == 4
            if len(coherent) < 4:
                observed_stock = np.median([rings[index][2] for index in coherent], axis=0)
                verified = set(coherent)
                for index in range(4):
                    if index in verified:
                        continue
                    line, retry = _side_line(
                        image, proposal[index], proposal[(index + 1) % 4],
                        maximum_excursion, rim_color=observed_stock, collect_curve_segments=collect_curve_segments,
                    )
                    if line is not None:
                        fitted[index] = line, {**fitted[index][1], **retry, "matched_print_rim": True}
                        verified.add(index)
                if len(verified) < 4:
                    reasons.append("PRINT_OUTER_RIM_UNCERTAIN")
                else:
                    resolved_outer_rim = True
    if not reasons and not resolved_outer_rim and conflicting_outward >= 2:
        reasons.append(PRINT_OUTER_EDGE_CONFLICT_REVIEW)
    evidence = {side: result[1] for side, result in zip(_PRINT_SIDES, fitted, strict=True)}
    if any(line is None for line, _ in fitted):
        reason = "PRINT_BOUNDARY_AMBIGUOUS" if any(item["ambiguous"] for item in evidence.values()) else "PRINT_BOUNDARY_UNSUPPORTED"
        reasons.append(reason)
    elif search_radius is None and bowed_print_sides(evidence):
        reasons.append("PRINT_BOUNDARY_CURVED")
    for index, (line, item) in enumerate(fitted):
        if line is None or item["corner_support"] >= 0.44:
            continue
        previous_line, previous = fitted[index - 1]
        next_line, following = fitted[(index + 1) % 4]
        start_verified = item["start_support"] >= 0.44 or (
            previous_line is not None and previous["end_support"] >= 0.8
        )
        end_verified = item["end_support"] >= 0.44 or (
            next_line is not None and following["start_support"] >= 0.8
        )
        if not (item.get("matched_print_rim") and start_verified and end_verified):
            reasons.append("PRINT_CORNERS_UNCERTAIN")
            break
    if all(line is None for line, _ in fitted):
        return None, evidence, reasons
    # Keep unsupported sides of the original proposal in a review preview; the
    # other three sides still benefit from better independent evidence.
    fallback_lines = np.cross(np.column_stack((proposal, np.ones(4))),
                              np.column_stack((np.roll(proposal, -1, axis=0), np.ones(4))))
    lines = np.array([line if line is not None else fallback for (line, _), fallback in zip(fitted, fallback_lines)])
    corners = _quad_from_lines(lines)
    if corners is None:
        return None, evidence, ["PRINT_BOUNDARY_UNSUPPORTED"]
    if (
        np.any(np.linalg.norm(corners - proposal, axis=1) > 1.8 * maximum_excursion)
        or not cv2.isContourConvex(corners)
        or cv2.contourArea(corners, oriented=True) <= 0
        or np.any(corners < 0)
        or np.any(corners > np.array(image.shape[1::-1]) - 1)
    ):
        return None, evidence, ["PRINT_BOUNDARY_UNSUPPORTED"]
    return corners, evidence, reasons
