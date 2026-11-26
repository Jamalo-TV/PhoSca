"""Find a measured outer outline to compare with an uncertain crop."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _replace_quad_side, _sample_lab


PRINT_OUTLINE_SUGGESTION_REVISION = 2
POSSIBLE_OVEREXTENDED_PRINT_EDGE = "possible_overextended_print_edge"


def _outline_at_distance(
    distance: np.ndarray, offset: np.ndarray, seed: np.ndarray, scale: np.ndarray, threshold: int,
) -> np.ndarray | None:
    mask = cv2.morphologyEx(np.uint8(distance <= threshold), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seed_area = cv2.contourArea(seed)
    outlines = []
    for contour in contours:
        if not 1.02 * seed_area <= cv2.contourArea(contour) <= 1.27 * seed_area:
            continue
        points = contour[:, 0]
        if np.any(points.min(axis=0) < 3) or np.any(points.max(axis=0) > np.array(distance.shape[1::-1]) - 4):
            continue
        hull = cv2.convexHull(contour)
        corners = cv2.approxPolyDP(hull, .012 * cv2.arcLength(hull, True), True)[:, 0]
        if len(corners) != 4:
            continue
        quad = order_quad_cyclic(corners + offset)
        match = normalized_polygon_overlap(seed / scale, quad / scale)
        if .81 <= match.iou <= .96 and match.first_containment >= .985:
            outlines.append(quad)
    return outlines[0] if len(outlines) == 1 else None


def _visible_stock_perimeter(image: np.ndarray, outline: np.ndarray, stock: np.ndarray) -> bool:
    for start, end in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        length = float(np.linalg.norm(end - start))
        count = int(np.clip(np.ceil(length), 64, 1600))
        inset = 6 / length
        fractions = np.linspace(inset, 1 - inset, count, dtype=np.float32)
        band = _edge_band(image, start, end, [-8, -5, -3, 3, 5, 8], fractions=fractions)
        if band is None or np.median(band[:, 3:, 0]) < 180:
            return False
        distance = np.linalg.norm(band[:, :, 1:] - stock, axis=2)
        votes = (distance[:, :3].mean(axis=1) >= 15) & (distance[:, 3:].mean(axis=1) <= 10)
        body = votes[(fractions >= .035) & (fractions <= .965)]
        if body.mean() < .83 or any(part.mean() < .50 for part in np.array_split(body, 4)):
            return False
        # A separating gap across two adjacent prints can disappear in the
        # median and closing operations. Check the interior rim in the source.
        interrupted = (distance[:, 3:] > 16).all(axis=1)
        if (interrupted[:-1] & interrupted[1:]).any():
            return False
    return True


def suggest_outer_print_outline(
    image: np.ndarray, seed: np.ndarray, preview: np.ndarray, neighbors: list[np.ndarray],
) -> np.ndarray | None:
    """Return one plausible larger stock outline for review."""
    extent = float(np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min())
    if extent < 180:
        return None
    scale = np.float32(image.shape[1::-1]) - 1
    pad = .20 * extent
    low = np.maximum(0, np.floor(seed.min(axis=0) - pad).astype(int))
    high = np.minimum(scale, np.ceil(seed.max(axis=0) + pad)).astype(int) + 1
    patch = image[low[1]:high[1], low[0]:high[0]]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = lab[..., 1:]
    initial = cv2.fillConvexPoly(np.zeros(lab.shape[:2], np.uint8), np.rint(seed - low).astype(np.int32), 1)
    pale = (initial != 0) & (lab[..., 0] >= 180) & (np.linalg.norm(chroma - 128, axis=2) <= 16)
    if np.count_nonzero(pale) < .10 * cv2.contourArea(seed):
        return None
    stock = np.median(chroma[pale], axis=0)
    distance = cv2.medianBlur(np.linalg.norm(chroma - stock, axis=2), 5)
    narrow = _outline_at_distance(distance, low, seed, scale, 8)
    outline = _outline_at_distance(distance, low, seed, scale, 12)
    if narrow is None or outline is None or np.max(np.linalg.norm(narrow - outline, axis=1)) > 5:
        return None
    normalized = outline / scale
    if (np.max(np.linalg.norm(outline - preview, axis=1)) < max(7., .018 * extent)
            or normalized_polygon_overlap(preview / scale, normalized).first_containment < .98):
        return None
    for neighbor in neighbors:
        other = neighbor / scale
        match = normalized_polygon_overlap(normalized, other)
        if max(match.first_containment, match.second_containment) <= .02:
            continue
        before = normalized_polygon_overlap(seed / scale, other)
        if not (before.second_containment >= .92 and match.second_containment >= .985
                and match.first_containment <= .68):
            return None
    return outline if _visible_stock_perimeter(image, outline, stock) else None


def _paper_edge_start(image: np.ndarray, start: np.ndarray, end: np.ndarray, reach: int, at_end: bool):
    length = np.linalg.norm(end - start)
    distances = np.arange(-.15 * reach, reach, .5, dtype=np.float32)
    fractions = distances / length
    if at_end:
        fractions = 1 - fractions
    band = _edge_band(image, start, end, [-11, -8, -5, -3, 3, 5, 8, 11], fractions=fractions)
    if band is None:
        return None
    outside, inside = (np.median(part, axis=1) for part in (band[:, :4], band[:, 4:]))
    contrast = np.linalg.norm(outside - inside, axis=1)
    edge = (contrast >= 28) & (outside[:, 0] - inside[:, 0] > 20)
    for index in range(26, len(distances) - 51):
        next_edge = slice(index + 5, index + 51)
        if (edge[next_edge].mean() >= .95 and np.percentile(contrast[index - 26:index - 4], 80) < 11
                and np.mean(contrast[index:index + 6] > 12) >= .67):
            return start + fractions[index] * (end - start), distances[index], outside[next_edge]
    return None


def _continues_into_backing(image: np.ndarray, ends: np.ndarray, normal: np.ndarray) -> bool:
    offsets = np.float32([-11, -8, -5, -3, 3, 5, 8, 11])
    for shift in range(-8, 9):
        points = ends[:, None] + (offsets + shift)[None, :, None] * normal
        if np.any((points < 0) | (points > np.array(image.shape[1::-1]) - 1)):
            return True
        colors = _sample_lab(image, points.reshape(-1, 2)).reshape(len(ends), len(offsets), 3)
        delta = np.median(colors[:, 4:], axis=1) - np.median(colors[:, :4], axis=1)
        for arm in np.split(delta, 2):
            direction = np.median(arm, axis=0)
            strength = np.linalg.norm(direction)
            if strength >= 2.5 and np.mean(arm @ direction >= 1.5 * strength) >= .78:
                return True
    return False


def suggest_overextended_print_outline(
    image: np.ndarray, quad: np.ndarray, evidence: dict,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Measure a plausible replacement for one side of an overextended review crop."""
    sides = [evidence.get(name) or {} for name in _PRINT_SIDES]
    missing = [index for index, side in enumerate(sides) if side.get("corner_support") is None]
    if len(missing) != 1 or any(side.get("ambiguous") for side in sides):
        return None
    index, = missing
    if any(side.get("support", 0) < .84 for other, side in enumerate(sides) if other != index):
        return None
    shortest = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
    if shortest < 280:
        return None
    previous, following = (index - 1) % 4, (index + 1) % 4
    reach = int(.25 * shortest)
    first = _paper_edge_start(image, quad[previous], quad[index], reach, True)
    last = _paper_edge_start(image, quad[following], quad[(index + 2) % 4], reach, False)
    if first is None or last is None:
        return None
    start, start_distance, start_colors = first
    end, end_distance, end_colors = last
    near, far = sorted((start_distance, end_distance))
    initial = quad.copy()
    initial[[index, following]] = [start, end]
    if not (-10 <= near <= .04 * shortest and .10 * shortest <= far <= .24 * shortest
            and cv2.isContourConvex(initial)):
        return None

    backing = np.median(np.concatenate((start_colors, end_colors)), axis=0)
    fractions = np.linspace(.018, .982, 320, dtype=np.float32)
    offsets = np.arange(-16, 17, dtype=np.float32)
    collar = _edge_band(image, start, end, offsets, fractions=fractions)
    if collar is None:
        return None
    columns = np.arange(5, len(offsets) - 7)
    exterior = np.median(collar[:, columns[:, None] - [2, 3, 4, 5]], axis=2)
    interior = collar[:, columns[:, None] + [2, 3, 5, 7], 0].max(axis=2)
    eligible = (np.linalg.norm(exterior - backing, axis=2) <= 18) & (interior <= exterior[..., 0] - 27)
    steps = np.where(eligible, collar[:, columns - 1, 0] - collar[:, columns + 1, 0], 0)
    peak = steps.argmax(axis=1)
    depths = offsets[columns[peak]]
    observed = steps[np.arange(len(peak)), peak] >= 18
    if observed.mean() < .90:
        return None
    curve = np.polyfit(fractions[observed], depths[observed], 3)
    observed &= np.abs(depths - np.polyval(curve, fractions)) < 2.5
    if observed.mean() < .90 or min(part.mean() for part in np.array_split(observed, 4)) < .87:
        return None
    curve = np.polyfit(fractions[observed], depths[observed], 3)
    if np.quantile(np.abs(depths - np.polyval(curve, fractions))[observed], .90) > 1.7:
        return None

    tangent = end - start
    length = np.linalg.norm(tangent)
    normal = np.float32([-tangent[1], tangent[0]]) / length
    extensions = np.r_[np.arange(-28., -5.), length + np.arange(6., 29.)] / length
    projected = start + extensions[:, None] * tangent + np.polyval(curve, extensions)[:, None] * normal
    if _continues_into_backing(image, projected, normal):
        return None

    # Sample the wedge bounded by the prior side and the new hypothesis.
    stations = np.linspace(.10, .90, 24, dtype=np.float32)
    wedge = -((quad[[index, following]] - start) @ normal)
    depth = np.interp(stations, fractions, np.interp(fractions, [0, 1], wedge))
    space = (start + stations[:, None, None] * tangent
             - depth[:, None, None] * np.linspace(.24, .88, 20)[None, :, None] * normal)
    if np.any((space < 0) | (space > np.array(image.shape[1::-1]) - 1)):
        return None
    colors = _sample_lab(image, space.reshape(-1, 2)).reshape(24, 20, 3)
    backing_support = (np.linalg.norm(colors - backing, axis=2) < 15).mean(axis=1)
    old = _edge_band(image, quad[index], quad[following], [-7, -5, -3, 3, 5, 7],
                     fractions=np.linspace(.09, .95, 64, dtype=np.float32))
    if backing_support.mean() < .80 or np.count_nonzero(backing_support >= .55) < 20 or old is None:
        return None
    old_step = np.linalg.norm(np.median(old[:, 3:], axis=1) - np.median(old[:, :3], axis=1), axis=1) >= 11
    if old_step.mean() >= .55 or max(part.mean() for part in np.array_split(old_step, 4)) >= .83:
        return None

    points = np.float32(start + fractions[:, None] * tangent + depths[:, None] * normal)
    vx, vy, x, y = cv2.fitLine(points[observed], cv2.DIST_HUBER, 0, .001, .001).ravel()
    # An integer edge sample locates a transition between source centres.
    x, y = [x, y] - .5 * normal
    candidate = _replace_quad_side(quad, index, np.array([-vy, vx, vy * x - vx * y]), reach, image.shape)
    if candidate is None:
        return None
    intervals = np.flatnonzero(np.diff(np.r_[False, observed, False])).reshape(-1, 2)
    traces = [cv2.approxPolyDP(points[first:stop], .4, False)[:, 0]
              for first, stop in intervals if stop - first >= 5]
    return (candidate, traces) if traces else None
