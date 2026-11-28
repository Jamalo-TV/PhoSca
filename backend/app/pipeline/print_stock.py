"""Look for a faint outer print boundary beyond a stronger interior edge."""

from collections.abc import Iterator

import cv2
import numpy as np

from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.print_edges import _edge_band, _line_endpoints, _replace_quad_side, _side_line


_RECTANGLE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
_SIDE_NAMES = ("top", "right", "bottom", "left")
MARKED_PAPER_MARGIN = "possible_marked_paper_margin"


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


def _pale_stock_shoulder(image: np.ndarray, side: np.ndarray, minimum_step: int) -> np.ndarray | None:
    band = _edge_band(image, *side, [-14, -11, -8, -5, -3, 3, 5, 8, 11, 14])
    if band is None:
        return None
    exterior, interior = (np.median(part, axis=1) for part in np.split(band, 2, axis=1))
    supported = ((interior[:, 0] >= 212) & (interior[:, 0] - exterior[:, 0] >= minimum_step)
                 & (np.linalg.norm(interior[:, 1:] - 128, axis=1) <= 28))
    if supported.mean() < .90 or min(part.mean() for part in np.array_split(supported, 4)) < .78:
        return None
    return np.median(interior, axis=0)


def _continuous_pale_extension(image: np.ndarray, side: np.ndarray, depth: float, palette: np.ndarray) -> bool:
    length = float(np.linalg.norm(side[1] - side[0]))
    stations = np.linspace(.025, .975, round(.95 * length), dtype=np.float32)
    band = _edge_band(image, *side, range(5, int(depth) - 4), fractions=stations)
    if band is None:
        return False
    sparse = band[::max(1, len(band) // 160)]
    paper = (sparse[..., 0] >= 213) & (np.linalg.norm(sparse[..., 1:] - palette[1:], axis=2) <= 8)
    if paper.mean(axis=0).min() < .91 or min(part.mean() for part in np.array_split(paper, 8)) < .90:
        return False

    # Look for a coherent transverse gap or second material edge in the new strip.
    count = sparse.shape[1] - 8
    before = np.median([sparse[:, start:start + count] for start in (0, 1, 2)], axis=0)
    after = np.median([sparse[:, start:start + count] for start in (5, 6, 7)], axis=0)
    change = after - before
    typical = np.median(change, axis=0)
    signed = (change * typical[None]).sum(axis=2) / np.maximum(1., np.linalg.norm(typical, axis=1))
    quarters = np.min([part.mean(axis=0) for part in np.array_split(signed >= 3.5, 4)], axis=0)
    if np.any((np.median(np.linalg.norm(change, axis=2), axis=0) >= 3.5) & (quarters >= .50)):
        return False

    # A short notch can escape the width-wide test. Compare each native station
    # with several separated locations at the same depth on the same stock.
    rows = np.arange(len(band))
    shifts = [round(length * offset) for offset in (-.129, -.107, -.090, .090, .107, .129)]
    surrounding = np.median([band[np.clip(rows + shift, 0, len(band) - 1)] for shift in shifts], axis=0)
    delta = cv2.GaussianBlur(surrounding, (5, 7), 0) - cv2.GaussianBlur(band, (5, 7), 0)
    interruption = (np.linalg.norm(delta, axis=2) >= 7) | (delta[..., 0] >= 5)
    components = cv2.connectedComponentsWithStats(cv2.morphologyEx(np.uint8(interruption), cv2.MORPH_OPEN,
                                                                  np.ones((3, 3), dtype=np.uint8)))[2][1:]
    return not any(x <= 2 and w >= 8 and h >= 14 and area >= 80 for x, _, w, h, area in components)


def _pale_stock_corner(image: np.ndarray, side: np.ndarray, *, at_end: bool, extension: float = 0.) -> bool:
    length = float(np.linalg.norm(side[1] - side[0]))
    distances = np.float32([12, 20, 30, 42]) / length
    fractions = np.r_[1 - distances, 1 + distances] if at_end else np.r_[distances, -distances]
    offsets = [-10, -8, -6, -4, 4, 6, 8, 10]
    around = _edge_band(image, *side, offsets, fractions=fractions)
    if around is None:
        return False
    change = around[:, 4:, 0].mean(axis=1) - around[:, :4, 0].mean(axis=1)
    if change[:4].min() < 13 or np.abs(change[4:]).max() > 18:
        return False
    if extension == 0:
        return True
    # Confirm the neighboring edge across the whole added cap, not just beyond
    # the old corner or over the next few samples beside the new corner.
    along = np.linspace(14., extension + 12., round(extension) - 1, dtype=np.float32) / length
    cap = _edge_band(image, *side, offsets, fractions=1 - along if at_end else along)
    if cap is None:
        return False
    outer, inner = (np.median(samples, axis=1) for samples in np.split(cap, 2, axis=1))
    supported = (inner[:, 0] >= 212) & (inner[:, 0] - outer[:, 0] >= 13)
    return supported.mean() >= .92 and min(part.mean() for part in np.array_split(supported, 4)) >= .85


def _claimed_stock_addition(quad: np.ndarray, candidate: np.ndarray, claims: list[np.ndarray]) -> bool:
    for claim in claims:
        amount, intersection = cv2.intersectConvexConvex(candidate, cv2.convexHull(claim))
        if amount > 1. and amount - cv2.intersectConvexConvex(intersection, quad)[0] > 1.:
            return True
    return False


def _expand_stock_side(
    quad: np.ndarray, index: int, line: np.ndarray, radius: int, shape: tuple[int, ...],
) -> np.ndarray | None:
    expanded = _replace_quad_side(quad, index, line, radius, shape)
    if expanded is None:
        return None
    next_index = (index + 1) % 4
    opposite = [(index + 2) % 4, (index + 3) % 4]
    expanded[opposite] = quad[opposite]
    # Float32 corner intersections can move an unchanged half-plane inward.
    # Round the two new corners outward from those neighboring lines.
    for corner, neighbor in ((index, (index - 1) % 4), (next_index, next_index)):
        edge = quad[(neighbor + 1) % 4] - quad[neighbor]
        outward = np.array([edge[1], -edge[0]], dtype=np.float32)
        direction = np.copysign(np.full_like(outward, np.inf), outward)
        direction[outward == 0] = expanded[corner, outward == 0]
        expanded[corner] = np.nextafter(np.nextafter(expanded[corner], direction), direction)
    precise = expanded.astype(np.float64)
    edges = np.roll(precise, -1, axis=0) - precise
    delta = quad[None].astype(np.float64) - precise[:, None]
    containment = edges[:, 0, None] * delta[..., 1] - edges[:, 1, None] * delta[..., 0]
    if np.any(containment < -1e-8) or np.any((expanded < 0) | (expanded > np.array(shape[1::-1]) - 1)):
        return None
    return expanded


def recover_pale_stock_preview(
    image: np.ndarray, quad: np.ndarray, boundary: dict, claims: list[np.ndarray],
) -> tuple[np.ndarray, dict] | None:
    """Restore a partly omitted pale margin in a held working crop."""
    shortest = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
    if shortest < 512:
        return None
    candidates = []
    for index, name in enumerate(_SIDE_NAMES):
        outward = (boundary.get(name) or {}).get("outward_search") or {}
        if not outward.get("print_rim_supported"):
            continue
        next_index = (index + 1) % 4
        original = quad[[index, next_index]]
        reach = outward["search_radius_px"]
        fits = [_side_line(image, *original, radius, outward_only=True) for radius in (reach, max(8, reach - 18))]
        if any(line is None or edge["ambiguous"] or edge["support"] < .92 or edge["corner_support"] < .85
               for line, edge in fits):
            continue
        line, _ = fits[0]
        rail = np.float32(_line_endpoints(*original, line))
        edge = original[1] - original[0]
        length = np.linalg.norm(edge)
        outward_depth = (original - rail) @ (np.float32([-edge[1], edge[0]]) / length)
        if (outward_depth.min() < max(36., .025 * shortest) or outward_depth.max() > .075 * shortest
                or np.ptp(outward_depth) > .008 * length
                or np.max(np.abs(rail @ fits[1][0][:2] + fits[1][0][2])) > .90):
            continue
        palette = _pale_stock_shoulder(image, rail, 44)
        if palette is None or not _continuous_pale_extension(image, rail, outward_depth.min(), palette):
            continue
        expanded = _expand_stock_side(quad, index, line, reach, image.shape)
        if expanded is None or _claimed_stock_addition(quad, expanded, claims):
            continue
        for companion in ((index - 1) % 4, next_index):
            indices = [companion, (companion + 1) % 4]
            support = boundary[_SIDE_NAMES[companion]]
            other, evidence = _side_line(image, *quad[indices], support["search_radius_px"])
            if other is None or evidence["ambiguous"] or evidence["support"] < .93 or evidence["corner_support"] < .80:
                continue
            candidate = _replace_quad_side(expanded, companion, other, support["search_radius_px"], image.shape)
            if candidate is None or np.linalg.norm(candidate[indices] - expanded[indices], axis=1).max() > 12:
                continue
            adjoining = _pale_stock_shoulder(image, candidate[indices], 14)
            if adjoining is None or np.linalg.norm(adjoining - palette) > 9:
                continue
            at_end = companion == next_index
            if not (_pale_stock_corner(image, candidate[[index, next_index]], at_end=at_end)
                    and _pale_stock_corner(image, candidate[indices], at_end=not at_end, extension=float(outward_depth.max()))):
                continue
            # The companion proves the cap exists. Its fitted rail can still
            # cross prior content along the old side, so retain that side.
            candidates.append((expanded, {"side": name, "companion": _SIDE_NAMES[companion],
                                           "depth_px": np.round(outward_depth, 3).tolist()}))
    return candidates[0] if len(candidates) == 1 else None


def _interposed_stock_seam(image: np.ndarray, new: np.ndarray, old: np.ndarray) -> bool:
    """Find a narrow material stripe between pale flanks along straight or oblique paths."""
    start, stop = new
    length = float(np.linalg.norm(stop - start))
    tangent = (stop - start) / length
    normal = np.float32([-tangent[1], tangent[0]])
    stations = np.arange(16, length - 16, 6, dtype=np.float32)
    old_edge = old[1] - old[0]
    fraction = (stations - (old[0] - start) @ tangent) / (old_edge @ tangent)
    reach = (old[0] - start) @ normal + fraction * (old_edge @ normal)
    offsets = np.arange(0, int(reach.max()) + 1, .5, dtype=np.float32)
    lab = _edge_band(image, *new, offsets, fractions=stations / length)
    if lab is None or len(offsets) < 30:
        return True
    mid = np.arange(14, len(offsets) - 14)
    center = np.median(lab[:, mid[:, None] + np.arange(-1, 2)], axis=2)
    stripe = np.zeros(center.shape[:2], bool)
    for near, far in ((3, 7), (6, 12)):
        flank = np.arange(near, far + 1)
        before = np.median(lab[:, mid[:, None] - flank], axis=2)
        after = np.median(lab[:, mid[:, None] + flank], axis=2)
        a, b = center - before, center - after
        first, second = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2)
        stripe |= ((first >= 3) & (second >= 3) & ((a * b).sum(axis=2) >= .55 * first * second)
                   & (np.linalg.norm(before - after, axis=2) <= 8.5)
                   & (np.minimum(before[..., 0], after[..., 0]) >= 203))
    stripe &= offsets[mid][None] <= reach[:, None] - 3
    first, last = np.meshgrid(
        np.arange(2., reach[0] - 2.9, .75, dtype=np.float32),
        np.arange(2., reach[-1] - 2.9, .75, dtype=np.float32), indexing="ij",
    )
    plausible = np.abs(first - last) < max(18., min(72., .041 * length))
    along = (stations - stations[0]) / (stations[-1] - stations[0])
    columns = np.arange(len(stations))[None]
    endpoints = np.column_stack((first[plausible], last[plausible]))
    for start in range(0, len(endpoints), 256):
        starts, stops = endpoints[start:start + 256].T
        paths = starts[:, None] * (1 - along) + stops[:, None] * along
        indices = np.rint((paths - 7.) * 2).astype(np.int32)
        hit = np.zeros(indices.shape, bool)
        for shift in (-1, 0, 1):
            row = indices + shift
            hit |= ((row >= 0) & (row < stripe.shape[1])
                    & stripe[columns, row.clip(0, stripe.shape[1] - 1)])
        hit &= paths <= reach[None] - 3
        counts = np.pad(hit.cumsum(axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        if np.any(counts[:, 26:] - counts[:, :-26] >= 19):
            return True
    return False


def _bright_exterior_cap(image: np.ndarray, rail: np.ndarray, other: np.ndarray, *, at_end: bool) -> bool:
    for side, terminal, samples in (
        (rail, at_end, [-8, -5, -3, 4, 6, 8]),
        (other, not at_end, [-8, -5, -3, 7, 9, 11]),
    ):
        points = np.r_[np.arange(12., 49., 2.), -np.arange(12., 49., 2.)] / np.linalg.norm(side[1] - side[0])
        band = _edge_band(image, *side, samples, fractions=np.float32(1 - points if terminal else points))
        if band is None:
            return False
        outer, inner = (np.median(pixels[..., 0], axis=1) for pixels in np.split(band, 2, axis=1))
        near, past = np.split(outer - inner, 2)
        if np.median(near) < 18 or np.quantile(near, .10) < 12 or abs(np.median(past)) > 8:
            return False
    return True


def _cut_stock_mark(
    image: np.ndarray, quad: np.ndarray, expanded: np.ndarray, rail: np.ndarray, cap: np.ndarray, *,
    at_end: bool, depth: float, palette: np.ndarray,
) -> np.ndarray | None:
    tangent = rail[1] - rail[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.float32([-tangent[1], tangent[0]])
    along = -tangent if at_end else tangent
    corners = cap + np.float32([[-12, -10], [120, -10], [120, depth + 40], [-12, depth + 40]]) @ np.vstack((along, normal))
    low = np.floor(corners.min(axis=0)).astype(int)
    high = np.ceil(corners.max(axis=0)).astype(int) + 1
    if np.any(low < 0) or np.any(high > np.array(image.shape[1::-1])):
        return None
    left, top = low
    right, bottom = high
    lab = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2Lab)
    old = _outline_mask(quad.astype(np.float64) - low, lab.shape[:2]) > 0
    new = _outline_mask(expanded.astype(np.float64) - low, lab.shape[:2]) > 0
    dark = np.uint8(lab[..., 0] < palette[0] - 73)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    proposals = []
    for label, (x, y, width, height, area) in enumerate(stats[1:], 1):
        narrow, long = sorted((width, height))
        if not (area >= 120 and 3 <= narrow <= 34 and 1.65 * narrow <= long <= 118):
            continue
        mark = labels == label
        if not (.13 <= old[mark].mean() <= .85 and new[mark].all()):
            continue
        points = np.column_stack(np.nonzero(mark)[::-1]) + low
        if not 2 <= np.min((points - rail[0]) @ normal) <= 20:
            continue
        binary_mark = np.uint8(mark)
        outer_ring = cv2.dilate(binary_mark, np.ones((21, 21), np.uint8)) != 0
        inner_ring = cv2.dilate(binary_mark, np.ones((7, 7), np.uint8)) != 0
        surrounding = outer_ring & ~inner_ring & new
        collar = [lab[surrounding & part] for part in (old, ~old)]
        if any(len(side) < 60 or np.mean((side[:, 0] >= 199) & (np.linalg.norm(side[:, 1:] - palette[1:], axis=1) <= 10)) < .70
               for side in collar):
            continue
        x1, y1 = np.maximum([left, top], [x + left - 13, y + top - 12])
        x2, y2 = np.minimum([right - 1, bottom - 1], [x + left + width + 12, y + top + height + 14])
        proposals.append(np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
    return proposals[0] if len(proposals) == 1 else None


def marked_paper_margin_comparison(
    image: np.ndarray, quad: np.ndarray, boundary: dict, claims: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Propose a separate held comparison that includes a mark cut by the working outline."""
    earlier = [index for index, name in enumerate(_SIDE_NAMES) if (boundary.get(name) or {}).get("review_pale_stock_extension")]
    if len(earlier) != 1:
        return None
    index = (earlier[0] + 2) % 4
    after = (index + 1) % 4
    outward = (boundary.get(_SIDE_NAMES[index]) or {}).get("outward_search") or {}
    if not outward.get("print_rim_supported"):
        return None
    radius = outward["search_radius_px"]
    old = quad[[index, after]]
    fits = [_side_line(image, *old, reach, outward_only=True) for reach in (radius, max(8, radius - 18))]
    if any(line is None or evidence["ambiguous"] or evidence["support"] < .96 or evidence["corner_support"] < .92
           for line, evidence in fits):
        return None
    line = fits[0][0]
    rail = np.float32(_line_endpoints(*old, line))
    direction = rail[1] - rail[0]
    depth = (old - rail) @ (np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction))
    if not (28 <= depth.min() <= depth.max() <= 105) or np.max(np.abs(rail @ fits[1][0][:2] + fits[1][0][2])) > .75:
        return None
    band = _edge_band(image, *rail, [-7, -5, -3, 3, 5, 7], fractions=np.linspace(.035, .965, 176, dtype=np.float32))
    if band is None:
        return None
    outside, inside = (np.median(samples, axis=1) for samples in np.split(band, 2, axis=1))
    supported = (inside[:, 0] >= 212) & (outside[:, 0] - inside[:, 0] >= 13)
    if supported.mean() < .94 or min(part.mean() for part in np.array_split(supported, 4)) < .86:
        return None
    palette = np.median(inside, axis=0)
    if np.linalg.norm(palette[1:] - 128) > 28 or not _continuous_pale_extension(image, rail, depth.min(), palette):
        return None
    expanded = _expand_stock_side(quad, index, line, radius, image.shape)
    if expanded is None or _claimed_stock_addition(quad, expanded, claims):
        return None
    new = expanded[[index, after]]
    marks = []
    for companion in ((index - 1) % 4, after):
        indices = [companion, (companion + 1) % 4]
        reach = boundary[_SIDE_NAMES[companion]]["search_radius_px"]
        adjacent, evidence = _side_line(image, *quad[indices], reach)
        if adjacent is None or evidence["ambiguous"] or evidence["support"] < .91 or evidence["corner_support"] < .82:
            continue
        cap_fit = _replace_quad_side(expanded, companion, adjacent, reach, image.shape)
        if cap_fit is None or np.linalg.norm(cap_fit[indices] - expanded[indices], axis=1).max() > 12:
            continue
        at_end = companion == after
        if _bright_exterior_cap(image, cap_fit[[index, after]], cap_fit[indices], at_end=at_end):
            cap = cap_fit[after if at_end else index]
            mark = _cut_stock_mark(image, quad, expanded, new, cap, at_end=at_end, depth=float(depth.min()), palette=palette)
            if mark is not None:
                marks.append(mark)
    if len(marks) != 1 or _interposed_stock_seam(image, new, old):
        return None
    added_strip = np.float32([old[0], new[0], new[1], old[1]])
    return expanded, marks[0], added_strip
