"""Find review outlines where a classical proposal crosses visible paper edges."""

from collections.abc import Callable, Iterator
from itertools import combinations

import cv2
import numpy as np

from app.pipeline.adjacent_prints import _joined_segments, _near_staggered_neighbor
from app.pipeline.edge_trace import measure_edge_trace
from app.pipeline.edge_prints import _photo_texture
from app.pipeline.geometry import (
    normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, pixel_vertices_to_normalized,
)
from app.pipeline.print_edges import _edge_band, _side_line, refine_print_boundary


CROSSING_PRINT_REVISION = 4
CROSSING_PRINT_REVIEW = "CROSSING_PRINT_OUTLINE_REVIEW"
REVIEWED_CROSSING_OUTLINE = "reviewed_crossing_outline"
CONTESTED_TILE_OUTLINE = "contested_tile"


def tile_crosses_unsupported_frame_claim(image: np.ndarray, tile: np.ndarray, claim: np.ndarray) -> bool:
    """Check for a measured print across an unsupported outline borrowing the capture frame."""
    height, width = image.shape[:2]
    frame = np.column_stack((claim, [width - 1, height - 1] - claim))
    touches = frame <= 1.5
    if (not touches[:, ::2].any() or not touches[:, 1::2].any()
            or np.count_nonzero(touches.any(axis=1)) < 2):
        return False
    _, sides, reasons = refine_print_boundary(image, claim)
    if reasons != ["PRINT_BOUNDARY_UNSUPPORTED"] or len(sides) != 4 or any(
        side["support"] > .40 or side["ambiguous"] for side in sides.values()
    ):
        return False
    fitted, sides, reasons = refine_print_boundary(image, tile, search_radius=4)
    if (fitted is None or reasons or len(sides) != 4
            or any(side["support"] < .90 or side["ambiguous"] for side in sides.values())
            or np.max(np.linalg.norm(order_quad_cyclic(fitted) - tile, axis=1)) > 4):
        return False

    tile_shortest = np.linalg.norm(np.roll(tile, -1, axis=0) - tile, axis=1).min()
    # Inspect the supposed edge with room to sample the same photo on both sides.
    crossings = []
    for start, stop in zip(claim, np.roll(claim, -1, axis=0), strict=True):
        stations = np.linspace(start, stop, 160, dtype=np.float32)
        interior = [station for station in stations if cv2.pointPolygonTest(tile, tuple(station), True) >= 13]
        if len(interior) >= 2 and np.linalg.norm(interior[-1] - interior[0]) >= max(96, .45 * tile_shortest):
            crossings.append((interior[0], interior[-1]))
    if len(crossings) != 1:
        return False
    (start, stop), = crossings
    line, evidence = _side_line(image, start, stop, 8)
    return line is None and not evidence["ambiguous"] and evidence["support"] <= .40


def _pale_leaf_foreground(image: np.ndarray, *, near_leaf_edge: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    """Separate dark regions inside the observed pale page without closing gaps."""
    height, width = image.shape[:2]
    short = min(height, width)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    coarse = lab[::4, ::4]
    smooth = cv2.GaussianBlur(lab, (0, 0), 1.1)[::4, ::4]
    pale = ((coarse[:, :, 0] > 180) & (np.linalg.norm(coarse[:, :, 1:] - 128, axis=2) < 13)
            & (np.linalg.norm(coarse - smooth, axis=2) < 5))
    if pale.mean() < .12:
        return None
    light = np.quantile(coarse[:, :, 0][pale], .64)
    pale &= (coarse[:, :, 0] >= light - 7) & (coarse[:, :, 0] < light + 8)
    ys, xs = np.ogrid[:height, :width]
    x, y = xs.astype(np.float32) / width - .5, ys.astype(np.float32) / height - .5
    sample_y, sample_x = np.nonzero(pale)
    terms = np.column_stack((x[0, 4 * sample_x], y[4 * sample_y, 0], np.ones(len(sample_x), np.float32)))
    fit, _, rank, _ = np.linalg.lstsq(terms, coarse[pale], rcond=None)
    if rank != 3:
        return None
    lab -= fit[2]
    lab -= x[..., None] * fit[0]
    lab -= y[..., None] * fit[1]
    distance_sq = np.einsum("hwi,hwi->hw", lab, lab)
    count, components, stats, _ = cv2.connectedComponentsWithStats(np.uint8(distance_sq <= 13**2))
    major = (np.arange(count) > 0) & (stats[:, cv2.CC_STAT_AREA] > .035 * height * width)
    points = cv2.findNonZero(major[components].astype(np.uint8))
    if points is None:
        return None
    hull = cv2.convexHull(points)
    if not .50 <= cv2.contourArea(hull) / (width * height) <= .97:
        return None
    leaf = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(leaf, hull, 1)
    leaf[[0, -1], :] = leaf[:, [0, -1]] = 0
    ordinary_inset = max(9., .032 * short)
    inset = 2. if near_leaf_edge else ordinary_inset
    distance = cv2.distanceTransform(leaf, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    mask = np.uint8((distance_sq > 28**2) & (distance > inset))
    size = max(3, round(.0036 * short) | 1)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    # An inner strip is needed on all four sides. Cutting a region at the
    # hull inset would invent a paper edge.
    return opened, distance > ordinary_inset + 2


def _outside_pale_paper(image: np.ndarray, start: np.ndarray, stop: np.ndarray, scale: float) -> bool:
    offsets = np.float32([-13, -19, -27, -36, -48, 4, 8, 13, 19, 27]) * scale
    samples = _edge_band(image, start, stop, offsets, fractions=np.linspace(.08, .92, 72, dtype=np.float32))
    if samples is None:
        return False
    paper, scene = np.split(samples.astype(np.float32), 2, axis=1)
    stock = np.median(paper, axis=1)
    support = ((stock[:, 0] > 171) & (np.linalg.norm(stock[:, 1:] - 128, axis=1) < 17)
               & (np.linalg.norm(paper - stock[:, None], axis=2).max(axis=1) < 11)
               & (np.linalg.norm(scene - stock[:, None], axis=2).mean(axis=1) > 9))
    return support.mean() >= .83 and min(part.mean() for part in np.array_split(support, 4)) >= .65


def _shadow_on_pale_paper(image: np.ndarray, start: np.ndarray, stop: np.ndarray, scale: float) -> bool:
    offsets = np.float32([-35, -26, -20, -14, -11, -8, -5, -3, 4, 7]) * scale
    samples = _edge_band(image, start, stop, offsets, fractions=np.linspace(.11, .89, 112, dtype=np.float32))
    if samples is None:
        return False
    light = samples[:, :, 0]
    far, shoulder, near, scene = (strip.mean(axis=1) for strip in np.split(light, [3, 6, 8], axis=1))
    support = (far >= 180) & (far - shoulder >= 3.2) & (shoulder - near >= 3) & (near - scene >= 11)
    return support.mean() >= .75 and min(part.mean() for part in np.array_split(support, 4)) >= .65


def crossing_print_outlines(
    image: np.ndarray, quad_from_contour: Callable[[np.ndarray], tuple[np.ndarray, str]],
) -> list[np.ndarray]:
    height, width = image.shape[:2]
    if min(height, width) < 480:
        return []
    ratio = min(1., 2048 / max(height, width))
    dimensions = (round(width * ratio), round(height * ratio))
    reduced = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA) if ratio < 1 else image
    foreground = _pale_leaf_foreground(reduced)
    if foreground is None:
        return []
    mask, interior = foreground
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = dimensions[0] * dimensions[1]
    short = min(dimensions)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not .06 * frame_area < area < .26 * frame_area:
            continue
        xs, ys = contour[:, 0].T
        if not interior[ys, xs].all():
            continue
        seed, kind = quad_from_contour(contour)
        if kind != "classical_border_quad" or area < .72 * cv2.contourArea(seed):
            continue
        lengths = np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1)
        if min(lengths) < .17 * short or not .50 < lengths[::2].sum() / lengths[1::2].sum() < 2:
            continue
        candidates.append(order_quad_cyclic(seed * (np.float32([width - 1, height - 1]) / (np.array(dimensions) - 1))))
    if len(candidates) > 4:
        return []
    scale = np.clip(min(width, height) / 1920, .5, 1.5)
    outlines = []
    for seed in candidates:
        fitted, sides, reasons = refine_print_boundary(image, seed, search_radius=round(18 * scale))
        if (fitted is None or reasons or any(
            side["support"] < .86 or (side["corner_support"] or 0) < .65 for side in sides.values()
        )):
            continue
        backed = [_outside_pale_paper(image, start, stop, scale)
                  for start, stop in zip(fitted, np.roll(fitted, -1, axis=0), strict=True)]
        if not any(backed[i] and backed[i + 2] for i in (0, 1)):
            continue
        if not any(_shadow_on_pale_paper(image, start, stop, scale)
                   for start, stop in zip(fitted, np.roll(fitted, -1, axis=0), strict=True)):
            continue
        if _photo_texture(image, fitted):
            # A fit can follow a neighbour a few pixels away. Use the
            # observed component as the editable starting outline.
            outlines.append(seed)
    return outlines


def _source_side_trace(image: np.ndarray, first: np.ndarray, last: np.ndarray) -> list:
    fractions = np.linspace(.02, .98, 112, dtype=np.float32)
    offsets = np.arange(-20., 21.)
    band = _edge_band(image, first, last, offsets, fractions=fractions)
    if band is None:
        return []
    light = band[..., 0]
    columns = np.arange(5, len(offsets) - 6)
    drops = light[:, columns] - light[:, columns + 3]
    scores = np.where(light[:, columns - 4] >= 170, drops, 0)
    peak = scores.argmax(axis=1)
    observed = scores.max(axis=1) >= 16
    if observed.mean() < .75:
        return []
    depths = offsets[columns[peak]] + 1.5
    keep = observed.copy()
    # Bracket the source evidence with the same three-point measurement
    # used in review. The guide itself contributes no displayed pixels.
    for _ in range(3):
        curve = np.polyfit(fractions[keep], depths[keep], 2)
        errors = np.abs(np.polyval(curve, fractions) - depths)
        keep &= errors <= max(1.4, np.percentile(errors[keep], 88))
    if (observed & (errors <= 2.2)).mean() < .75:
        return []
    direction = last - first
    inward = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
    stations = np.float32([0, .5, 1])
    anchors = first + stations[:, None] * direction + np.polyval(curve, stations)[:, None] * inward
    points = pixel_vertices_to_normalized(anchors, *image.shape[1::-1])
    try:
        return measure_edge_trace(image, points)
    except ValueError:
        return []


def _native_line(points: np.ndarray) -> np.ndarray:
    line = np.cross(*np.column_stack((points, np.ones(2))))
    return line / np.linalg.norm(line[:2])


def _at_height(line: np.ndarray, height: float) -> np.ndarray:
    return np.float32([-(line[1] * height + line[2]) / line[0], height])


def _local_side_support(image: np.ndarray, start: np.ndarray, end: np.ndarray, radius: int) -> bool:
    points = start + np.linspace(0, 1, 5, dtype=np.float32)[:, None] * (end - start)
    offsets = []
    for first, last in zip(points[:-1], points[1:], strict=True):
        line, evidence = _side_line(image, first, last, radius)
        if line is None or evidence["support"] < .90 or (evidence["corner_support"] or 0) < .60:
            return False
        offsets.append(np.column_stack((np.float32([first, last]), np.ones(2))) @ line)
    return all(abs(before[1] - after[0]) <= 2 for before, after in zip(offsets[:-1], offsets[1:], strict=True))


def _staggered_near_extension(
    image: np.ndarray, rail_end: np.ndarray, line: np.ndarray, corner: np.ndarray, anchor_bottom: float,
) -> bool:
    # The shared rail can end early in the thumbnail. Validate the remaining
    # span and check that it borders pale stock beyond the known print.
    if rail_end[1] >= anchor_bottom or corner[1] - anchor_bottom < 62:
        return False
    if not _local_side_support(image, corner, _at_height(line, rail_end[1] - 9), 8):
        return False
    band = _edge_band(
        image, _at_height(line, corner[1] - 8), _at_height(line, anchor_bottom + 15),
        [-11, -7, -4, 4, 7, 11], fractions=np.linspace(0, 1, 48, dtype=np.float32),
    )
    if band is None:
        return False
    stock, print_face = (np.median(side, axis=1) for side in np.split(band, 2, axis=1))
    supported = ((stock[:, 0] >= 170) & (np.linalg.norm(stock[:, 1:] - 128, axis=1) < 18)
                 & (stock[:, 0] - print_face[:, 0] >= 25))
    return supported.mean() >= .92 and all(part.mean() >= .78 for part in np.split(supported, 4))


def _staggered_far_endpoint(
    image: np.ndarray, near: np.ndarray, far: np.ndarray, *, upper: bool,
) -> np.ndarray | None:
    # Look for the end of the light-to-dark edge along the crossbar. A joined
    # rail may switch between the paper edge and the edge of the album leaf.
    distance = np.arange(-34, 35, dtype=np.float32)
    direction = (far - near) / (far[0] - near[0])
    stations = distance / (far[0] - near[0])
    if upper:
        band = _edge_band(image, near, far, [-10, -7, -5, 5, 7, 10], fractions=1 + stations)
    else:
        band = _edge_band(image, far, near, [-10, -7, -5, 5, 7, 10], fractions=-stations)
    if band is None:
        return None
    outside, inside = (np.median(side, axis=1) for side in np.split(band, 2, axis=1))
    delta = outside - inside
    print_edge = (outside[:, 0] >= 158) & (delta[:, 0] >= 25)
    backing = np.linalg.norm(delta, axis=1) < 16
    bracket = []
    for index in range(10, len(distance) - 11):
        if (print_edge[index - 10:index - 2].mean() >= .875
                and backing[index + 3:index + 11].mean() >= .875
                and np.median(delta[index - 7:index - 2, 0]) - np.median(delta[index + 3:index + 8, 0]) >= 24):
            bracket.append(distance[index])
    if not bracket or max(bracket) - min(bracket) > 7:
        return None
    return np.float32(far + np.median(bracket) * direction)


def _staggered_search_seeds(
    segments: list[tuple[int, np.ndarray]], anchor: np.ndarray,
    *, extension_image: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    left, top = anchor.min(axis=0)
    right, bottom = anchor.max(axis=0)
    width, height = right - left, bottom - top
    if min(width, height) < 220:
        return
    arms = []
    for axis, ends in segments:
        if axis != 1 or np.linalg.norm(ends[1] - ends[0]) < .48 * height:
            continue
        line = _native_line(ends)
        position = _at_height(line, (top + bottom) / 2)[0]
        if right - .06 * width <= position <= right + 1.40 * width:
            arms.append((position, ends, line))
    if len(arms) > 8:
        return
    for (near, first, left_line), (far, second, right_line) in combinations(sorted(arms, key=lambda arm: arm[0]), 2):
        if abs(near - right) > .12 * width or not .72 * width <= far - near <= 1.28 * width:
            continue
        for axis, crossbar in segments:
            if axis != 0 or np.linalg.norm(crossbar[1] - crossbar[0]) < .72 * (far - near):
                continue
            line = _native_line(crossbar)
            corners = np.cross(line, [left_line, right_line])
            if np.min(np.abs(corners[:, 2])) < .2:
                continue
            bottom_corners = np.float32(corners[:, :2] / corners[:, 2, None])
            if (not .98 <= (bottom_corners[0, 1] - top) / height <= 1.18
                    or np.linalg.norm(bottom_corners - crossbar, axis=1).max() > .12 * width):
                continue
            extension = bottom_corners[:, 1] - [first[1, 1], second[1, 1]]
            if np.any(extension < -.06 * height) or extension[1] > .29 * height:
                continue
            if extension[0] > .14 * height and (
                extension_image is None or extension[0] > .18 * height
                or not _staggered_near_extension(extension_image, first[1], left_line, bottom_corners[0], bottom)
            ):
                continue
            yield left_line, right_line, bottom_corners, top, height


def _staggered_native_quads(
    image: np.ndarray, seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]], anchor: np.ndarray,
    *, verify_far_endpoints: bool = False,
) -> list[np.ndarray]:
    result = []
    for left_line, right_line, (bottom_left, bottom_right), top, height in seeds:
        if not _local_side_support(image, bottom_right, bottom_left, 8):
            continue
        slope = bottom_right[1] - bottom_left[1]
        for y in np.arange(top + .045 * height, top + .205 * height, 16):
            start, end = _at_height(left_line, y), _at_height(right_line, y + slope)
            top_line, evidence = _side_line(image, start, end, 14)
            if top_line is None or evidence["support"] < .88 or (evidence["corner_support"] or 0) < .65:
                continue
            corners = np.cross(top_line, [left_line, right_line])
            if np.min(np.abs(corners[:, 2])) < .2:
                continue
            top_left, top_right = np.float32(corners[:, :2] / corners[:, 2, None])
            far_top, far_bottom = top_right, bottom_right
            if verify_far_endpoints:
                far_top = _staggered_far_endpoint(image, top_left, top_right, upper=True)
                far_bottom = _staggered_far_endpoint(image, bottom_left, bottom_right, upper=False)
                if far_top is None or far_bottom is None or not _local_side_support(image, far_bottom, bottom_left, 8):
                    continue
            quad = np.float32([top_left, far_top, far_bottom, bottom_left])
            if (not cv2.isContourConvex(quad) or np.any(quad < 4)
                    or np.any(quad > np.asarray(image.shape[1::-1]) - 5)):
                continue
            overlap, _ = cv2.intersectConvexConvex(quad, anchor)
            if overlap > .02 * min(cv2.contourArea(quad), cv2.contourArea(anchor)):
                continue
            _, near = _side_line(image, bottom_left, top_left, 10)
            if (near["ambiguous"] or near["support"] < .90 or (near["corner_support"] or 0) < .70
                    or not _local_side_support(image, far_top, far_bottom, 11)
                    or not _photo_texture(image, quad)):
                continue
            if not any(np.max(np.linalg.norm(quad - previous, axis=1)) <= 5 for previous in result):
                result.append(quad)
    return result


def _quarter_turn(points: np.ndarray, turn: int, width: int, height: int) -> np.ndarray:
    x, y = points.T
    positions = ((x, y), (y, width - 1 - x), (width - 1 - x, height - 1 - y), (height - 1 - y, x))
    return np.column_stack(positions[turn]).astype(np.float32)


def staggered_print_source_traces(image: np.ndarray, anchor: np.ndarray, *, working_side: int = 1280) -> list:
    """Observe three free sides of a possible print next to a line-pair detection."""
    height, width = image.shape[:2]
    scale = min(1., working_side / max(height, width))
    working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    segments = _joined_segments(working, min(working.shape[:2]))
    if not segments or len(segments) > 48:
        return []
    searches = []
    for turn in range(4):
        nearby = []
        for rail in segments:
            axis = (rail.axis + turn) % 2
            ends = _quarter_turn(rail.ends / scale, turn, width, height)
            nearby.append((axis, ends[ends[:, axis].argsort()]))
        outline = _quarter_turn(anchor, turn, width, height)
        searches.append((turn, nearby, outline))
    for verify_far_endpoints in (False, True):
        candidates = _staggered_trace_candidates(image, searches, verify_far_endpoints)
        if candidates:
            return candidates[0] if len(candidates) == 1 else []
    return []


def _staggered_trace_candidates(
    image: np.ndarray, searches: list[tuple[int, list[tuple[int, np.ndarray]], np.ndarray]],
    verify_far_endpoints: bool,
) -> list:
    height, width = image.shape[:2]
    candidates = []
    for turn, nearby, outline in searches:
        turned = np.rot90(image, turn)
        seeds = list(_staggered_search_seeds(nearby, outline, extension_image=turned if verify_far_endpoints else None))
        if not 1 <= len(seeds) <= 4:
            continue
        turned = np.ascontiguousarray(turned)
        for quad in _staggered_native_quads(turned, seeds, outline, verify_far_endpoints=verify_far_endpoints):
            # These three sides face the album, while the near side runs
            # against another proposed print. Persist only signed source traces.
            sides = [_source_side_trace(turned, a, b) for a, b in zip(quad[:3], quad[1:], strict=True)]
            if not all(sides):
                continue
            native_height, native_width = turned.shape[:2]
            traces = []
            for trace in [run for side in sides for run in side]:
                points = normalized_vertices_to_pixels([(point["x"], point["y"]) for point in trace],
                                                       native_width, native_height)
                source = _quarter_turn(points, (-turn) % 4, native_width, native_height)
                traces.append([{"x": float(x), "y": float(y)}
                               for x, y in pixel_vertices_to_normalized(source, width, height)])
            candidates.append(traces)
    return candidates


def near_leaf_edge_traces(
    image: np.ndarray, prepared: np.ndarray, crossing: np.ndarray, other_prints: list,
    quad_from_contour: Callable[[np.ndarray], tuple[np.ndarray, str]],
) -> list:
    """Find open source transitions near a potentially missing adjoining print."""
    height, width = image.shape[:2]
    scale = min(1., 2048 / max(height, width))
    dimensions = (round(width * scale), round(height * scale))
    working = cv2.resize(prepared, dimensions, interpolation=cv2.INTER_AREA) if scale < 1 else prepared
    evidence = _pale_leaf_foreground(working, near_leaf_edge=True)
    if evidence is None:
        return []
    mask, interior = evidence
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = dimensions[0] * dimensions[1]
    scale_to_source = np.float32([width - 1, height - 1]) / (np.array(dimensions) - 1)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not .06 * frame_area < area < .26 * frame_area:
            continue
        xs, ys = contour[:, 0].T
        if np.count_nonzero(~interior[ys, xs]) < 4:
            continue
        quad, kind = quad_from_contour(contour)
        if kind != "classical_border_quad" or area < .82 * cv2.contourArea(quad):
            continue
        quad = order_quad_cyclic(quad * scale_to_source)
        if not _near_staggered_neighbor(quad, crossing):
            continue
        outline = pixel_vertices_to_normalized(quad, width, height)
        if any(normalized_polygon_overlap(outline, other).first_containment >= .12 for other in other_prints):
            continue
        candidates.append(quad)
    if len(candidates) != 1 or not _photo_texture(image, candidates[0]):
        return []
    quad = candidates[0]
    sides = [_source_side_trace(image, first, last) for first, last in zip(quad, np.roll(quad, -1, axis=0), strict=True)]
    if sum(bool(side) for side in sides) < 3:
        return []
    return [trace for side in sides for trace in side]


def reviewed_outline_covers_traces(quad: list[dict], traces: list) -> bool:
    if not traces:
        return False
    points = np.float32([(point["x"], point["y"]) for run in traces for point in run])
    if len(points) < 3:
        return False
    extent = cv2.convexHull(points)[:, 0]
    # The hull only retires a stale hint when a reviewer creates that print.
    # It is never persisted, counted, or passed to crop rendering.
    outline = [(point["x"], point["y"]) for point in quad]
    return normalized_polygon_overlap(outline, extent).iou >= .82
