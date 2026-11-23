"""Locate the outer edges of a proposed print in the source image."""

from __future__ import annotations

import cv2
import numpy as np


def _side_line(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, radius: int, *, outward_only: bool = False,
    rim_color: np.ndarray | None = None,
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
        return None, evidence

    nearest = distances[np.arange(len(positions)), delta[winner].argmin(axis=1)]
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
    # The transition is measured between pixels. Start at the first print pixel center.
    first = start + inward * (origin + 0.5)
    last = end + inward * (origin + length * slope + 0.5)
    line = np.cross([*first, 1.0], [*last, 1.0])
    return line / np.linalg.norm(line[:2]), evidence


def _observe_print_rim(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, line: np.ndarray, *, minimum_transition: float,
) -> tuple[np.ndarray, float] | None:
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
    return (rim, depth) if np.mean(uniform) >= 0.78 and np.mean(transition > minimum_transition) >= 0.75 else None


def _line_endpoints(start: np.ndarray, end: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = end - start
    inward = np.array([-tangent[1], tangent[0]], np.float32) / np.linalg.norm(tangent)
    offsets = -((np.array([start, end]) @ line[:2] + line[2]) / (line[:2] @ inward))
    first, last = np.array([start, end]) + offsets[:, None] * inward
    return first.astype(np.float32), last.astype(np.float32)


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
    previous = -np.roll(directions, 1, axis=0)
    offsets = np.array([[2.5, 2.5], [5.0, 2.5], [2.5, 5.0]])
    tips = quad[:, None] + directions[:, None] * offsets[None, :, :1] + previous[:, None] * offsets[None, :, 1:]
    tips = _sample_lab(image, tips.reshape(-1, 2)).reshape(4, 3, 3)
    material = np.median(tips, axis=1)
    different = np.linalg.norm(material - stock, axis=1) >= 24
    coherent = np.max(np.linalg.norm(tips - material[:, None], axis=2), axis=1) <= 15
    candidates = different & coherent
    if not candidates.any():
        return False, False
    # A corner holder crosses the outside of both adjoining edges. Print detail
    # on the stock and the end of a neighboring seam do not establish a cover.
    exterior = np.stack((quad + 4 * directions - 2 * inward,
                         quad + 4 * previous - 2 * np.roll(inward, 1, axis=0)), axis=1)[candidates]
    height, width = image.shape[:2]
    in_frame = ((exterior >= 0) & (exterior <= [width - 1, height - 1])).all(axis=(1, 2))
    colors = _sample_lab(image, exterior.reshape(-1, 2)).reshape(-1, 2, 3)
    same_material = (np.linalg.norm(colors - material[candidates, None], axis=2) <= 18).all(axis=1)
    covered = in_frame & same_material
    if not covered.any():
        return False, False

    outside = edge_positions - 5.5 * inward[:, None, :]
    outside_in_frame = ((outside >= 0) & (outside <= [width - 1, height - 1])).all(axis=(1, 2))
    background = np.median(_sample_lab(image, outside.reshape(-1, 2)).reshape(4, 32, 3), axis=1)
    adjoining = np.stack((background, np.roll(background, 1, axis=0)), axis=1)[candidates]
    known_background = (outside_in_frame & np.roll(outside_in_frame, 1))[candidates]
    # A shifted seam can expose the album page at a fitted corner. Only a
    # separate material that differs from both adjoining backgrounds is a holder.
    distinct_holder = (np.linalg.norm(adjoining - material[candidates, None], axis=2) >= 20).all(axis=1)
    return True, bool(np.any(covered & known_background & distinct_holder))


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


def refine_print_boundary(
    image: np.ndarray, proposal: np.ndarray, *, search_radius: int | None = None,
) -> tuple[np.ndarray | None, dict, list[str]]:
    sides = np.linalg.norm(np.roll(proposal, -1, axis=0) - proposal, axis=1)
    radius = search_radius or min(56, max(4, round(float(sides.min()) * 0.042)))
    wider = min(130, round(float(sides.min()) * 0.112))
    maximum_excursion = radius
    fitted = []
    rings = {}
    for index, (start, end) in enumerate(zip(proposal, np.roll(proposal, -1, axis=0))):
        line, result = _side_line(image, start, end, radius)
        if search_radius is None and not result["ambiguous"] and wider >= radius + 4:
            first, last = _line_endpoints(start, end, line) if line is not None else (start, end)
            if line is None or _pale_outside_strip(image, first, last):
                alternate, retry = _side_line(image, first, last, wider, outward_only=True)
                observation = None if alternate is None else _observe_print_rim(
                    image, first, last, alternate, minimum_transition=8 if line is None else 5,
                )
                result["outward_search"] = {**retry, "print_rim_supported": observation is not None}
                if observation is not None:
                    color, depth = observation
                    retry.update(observed_rim_lab=color.tolist(), observed_rim_depth_px=depth)
                    rings[index] = alternate, retry, color, line is not None
                    if line is None:
                        result.update(retry)
                        maximum_excursion = wider
                        line = alternate
        fitted.append((line, result))
    reasons = []
    if any(replaces_supported for _, _, _, replaces_supported in rings.values()):
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
            if len(coherent) < 4:
                rim_color = np.median([rings[index][2] for index in coherent], axis=0)
                verified = set(coherent)
                for index in range(4):
                    if index in verified:
                        continue
                    line, retry = _side_line(
                        image, proposal[index], proposal[(index + 1) % 4],
                        maximum_excursion, rim_color=rim_color,
                    )
                    if line is not None:
                        fitted[index] = line, {**fitted[index][1], **retry, "matched_print_rim": True}
                        verified.add(index)
                if len(verified) < 4:
                    reasons.append("PRINT_OUTER_RIM_UNCERTAIN")
    evidence = {side: result[1] for side, result in zip(("top", "right", "bottom", "left"), fitted)}
    if any(line is None for line, _ in fitted):
        reason = "PRINT_BOUNDARY_AMBIGUOUS" if any(item["ambiguous"] for item in evidence.values()) else "PRINT_BOUNDARY_UNSUPPORTED"
        reasons.append(reason)
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
    intersections = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(np.abs(intersections[:, 2]) < 1e-6):
        return None, evidence, ["PRINT_BOUNDARY_UNSUPPORTED"]
    corners = (intersections[:, :2] / intersections[:, 2:3]).astype(np.float32)
    if (
        np.any(np.linalg.norm(corners - proposal, axis=1) > 1.8 * maximum_excursion)
        or not cv2.isContourConvex(corners)
        or cv2.contourArea(corners, oriented=True) <= 0
        or np.any(corners < 0)
        or np.any(corners > np.array(image.shape[1::-1]) - 1)
    ):
        return None, evidence, ["PRINT_BOUNDARY_UNSUPPORTED"]
    return corners, evidence, reasons
