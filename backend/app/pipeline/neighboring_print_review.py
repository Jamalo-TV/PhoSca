"""Check a vacant position beside measured album prints against the source."""

from collections.abc import Callable
from itertools import combinations

import cv2
import numpy as np

from app.pipeline.album_backing import _backing_reference, _repeating_profile
from app.pipeline.contested_scene_edge import _common_spans
from app.pipeline.edge_prints import _photo_texture
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, quad_from_polygon
from app.pipeline.near_capture_review import _Rail, _claimed_contours, _signed_stations, _trace_intervals
from app.pipeline.print_edges import _edge_band, _line_endpoints, _quad_from_lines, _side_line, refine_print_boundary
from app.pipeline.tile_print_review import TILED_PRINT_PROPOSAL, tiled_outline_conflicts
from app.pipeline.visible_scene_edge import has_enclosing_perimeter


NEIGHBORING_PRINT_EDGE_TRACES = "neighboring_print_edge_traces"
NEIGHBORING_SOURCE_GUIDE = "neighboring_source_guide"
FOLLOWUP_PRINT_EDGE_GUIDE = "followup_print_edge_guide"


def _vertices(points):
    return np.float32([(point["x"], point["y"]) for point in points])


def _normalized(points):
    return [{"x": float(x), "y": float(y)} for x, y in points]


def next_print_position(
    pending: list[dict], claims: list[dict], width: int, height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find one vacant position supported by comparable held outlines."""
    scale = np.float32([width - 1, height - 1])
    outlines = [row["polygon"] for row in pending
                if row.get("reason") == TILED_PRINT_PROPOSAL and len(row.get("polygon") or []) == 4]
    if not 2 <= len(outlines) <= 4:
        return None
    measured = [order_quad_cyclic(_vertices(points) * scale) for points in outlines]
    positions = []
    for first, second in combinations(measured, 2):
        pitch = second.mean(axis=0) - first.mean(axis=0)
        distance = np.linalg.norm(pitch)
        if distance < 180:
            continue
        axes = [np.diff(quad[:3], axis=0) for quad in (first, second)]
        lengths = [np.linalg.norm(axis, axis=1) for axis in axes]
        alignments = [np.abs(axis @ (pitch / distance)) / size for axis, size in zip(axes, lengths, strict=True)]
        matching_sides = [size[np.argmax(alignment)] for size, alignment in zip(lengths, alignments, strict=True)]
        if (min(alignment.max() for alignment in alignments) < .96
                or max(matching_sides) > 1.25 * min(matching_sides)
                or not .93 <= distance / np.mean(matching_sides) <= 1.35):
            continue
        for quad, direction in ((first, -pitch), (second, pitch)):
            next_quad = quad + direction
            if np.any(np.minimum(next_quad, scale - next_quad) < 4):
                continue
            if not tiled_outline_conflicts(_normalized(next_quad / scale), claims, width, height):
                positions.append((next_quad, direction))
    return positions[0] if len(positions) == 1 else None


_FRACTIONS = np.linspace(.08, .92, 160, dtype=np.float32)


def _leading_edge(image: np.ndarray, side: np.ndarray) -> np.ndarray | None:
    offsets = np.arange(-9, 9.1, .25, dtype=np.float32)
    band = _edge_band(image, *side, offsets, fractions=_FRACTIONS)
    if band is None:
        return None
    light = band[..., 0]
    columns = np.flatnonzero((-5 <= offsets) & (offsets <= 3))
    step = light[:, columns - 4] - light[:, columns + 4]
    wide = light[:, columns - 12] - light[:, columns + 12]
    warmer = band[:, columns - 4, 2] >= band[:, columns + 4, 2]
    choices = (step >= 12) & (wide >= 24) & warmer
    supported = choices.any(axis=1)
    if supported.mean() < .95:
        return None
    peaks = np.argmax(np.where(choices, step + .20 * wide, -1), axis=1)
    depths = offsets[columns[peaks]]
    direction = side[1] - side[0]
    length = np.linalg.norm(direction)
    stations = _FRACTIONS * length
    vx, vy, x, y = cv2.fitLine(
        np.column_stack((stations[supported], depths[supported])), cv2.DIST_HUBER, 0, .01, .01,
    ).ravel()
    if abs(vx) < .1:
        return None
    predicted = y + (vy / vx) * (stations - x)
    agreement = supported & (np.abs(predicted - depths) <= 1.1)
    if (agreement.mean() < .78 or min(part.mean() for part in np.array_split(agreement, 8)) < .50
            or np.max(np.abs(predicted[[0, -1]])) > 2):
        return None
    return _edge_line(side, predicted[[0, -1]])


def _edge_line(side, displacement):
    dx, dy = side[1] - side[0]
    normal = np.float32([-dy, dx]) / np.hypot(dx, dy)
    a, b = side + displacement[:, None] * normal
    return np.cross(np.r_[a, 1.], np.r_[b, 1.])


def _trailing_edge(image: np.ndarray, side: np.ndarray) -> np.ndarray | None:
    offsets = np.arange(-13, 13.5, .5, dtype=np.float32)
    band = _edge_band(image, *side, offsets, fractions=_FRACTIONS)
    if band is None:
        return None
    light = band[..., 0]
    columns = np.flatnonzero(np.abs(offsets) <= 8)
    step = light[:, columns - 2] - light[:, columns + 2]
    wide = light[:, columns - 5] - light[:, columns + 5]
    shoulders = np.maximum(light[:, columns - 6] - light[:, columns - 3], light[:, columns + 3] - light[:, columns + 6])
    choices = (step >= 12) & (wide >= 24) & (step >= shoulders + 3)
    shifts, slopes = np.meshgrid(offsets[columns], np.arange(-.045, .0451, .0045))
    length = np.linalg.norm(side[1] - side[0])
    end_offsets = shifts.ravel()[:, None] + slopes.ravel()[:, None] * length * np.float32([-.5, .5])
    at = end_offsets[:, :1] + np.diff(end_offsets, axis=1) * _FRACTIONS
    bounded = np.abs(at).max(axis=1) <= 8
    at, end_offsets = at[bounded], end_offsets[bounded]
    columns = np.rint((at + 8) * 2).astype(np.intp)
    stations = np.arange(len(_FRACTIONS))[None, :]
    supports = choices[stations, columns]
    coverage = supports.mean(axis=1)
    weakest = np.min([part.mean(axis=1) for part in np.array_split(supports, 8, axis=1)], axis=0)
    quality = .59 * coverage + .21 * weakest + .20 * np.minimum(step[stations, columns] / 50, 1.).mean(axis=1)
    eligible = (coverage >= .90) & (weakest >= .60) & (np.median(wide[stations, columns], axis=1) >= 24)
    if not eligible.any():
        return None
    index = np.argmax(np.where(eligible, quality, -1.))
    rivals = eligible & (quality >= quality[index] - .03) & (np.abs(at - at[index]).max(axis=1) > 3)
    if rivals.any() or np.max(np.abs(end_offsets[index])) > 6:
        return None
    return _edge_line(side, end_offsets[index])


def _lateral_edge(image: np.ndarray, side: np.ndarray) -> np.ndarray | None:
    measured = []
    for radius in (4, 7):
        line, evidence = _side_line(image, *side, radius)
        if line is None or evidence["ambiguous"] or evidence["support"] < .90 or (evidence["corner_support"] or 0) < .60:
            return None
        measured.append(line)
    ends = np.float32([_line_endpoints(*side, line) for line in measured])
    return measured[0] if np.linalg.norm(ends[0] - ends[1], axis=1).max() <= 1 else None


def _ribs_end_at_edge(image: np.ndarray, start: np.ndarray, stop: np.ndarray) -> bool:
    fractions = np.linspace(.08, .92, round(.84 * np.linalg.norm(stop - start)), dtype=np.float32)
    band = _edge_band(image, start, stop, [-14, -12, -10, -8, -6, -5, -4, -3, -2, 1, 2, 3], fractions=fractions)
    if band is None:
        return False
    profiles = [_repeating_profile(np.median(block, axis=1)) for block in np.split(band[..., 0], [4, 6, 9], axis=1)]
    if any(profile.std() < 8 or repeat < .80 for profile, repeat in profiles[:3]) or profiles[3][0].std() == 0:
        return False
    _, middle, near, inside = np.corrcoef([profile for profile, _ in profiles])[0]
    return min(middle, near) >= .85 and abs(inside) < .25


def _vacant_layout_cell(
    anchors: list[np.ndarray], held: list[np.ndarray], scale: np.ndarray, claimed: Callable[[np.ndarray], bool],
) -> tuple[np.ndarray, np.ndarray] | None:
    cells = []
    for anchor in anchors:
        lengths = np.linalg.norm(np.roll(anchor, -1, axis=0) - anchor, axis=1)
        if lengths.min() < 200 or lengths.max() > 1.20 * lengths.min():
            continue
        axes = np.diff(anchor[:3], axis=0) / lengths[:2, None]
        comparable = [quad for quad in held
                      if np.max(np.abs(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1) / lengths - 1)) <= .25]
        for first, second in combinations(comparable, 2):
            delta = np.stack([first.mean(axis=0), second.mean(axis=0)]) - anchor.mean(axis=0)
            distance = np.linalg.norm(delta, axis=1)
            if distance.min() < .98 * lengths.min():
                continue
            unit = delta / distance[:, None]
            alignment = np.abs(unit @ axes.T)
            match = alignment.argmax(axis=1)
            pitch = distance / lengths[match]
            if (match[0] == match[1] or abs(unit[0] @ unit[1]) > .13
                    or alignment.max(axis=1).min() < .97 or not np.all((.98 <= pitch) & (pitch <= 1.30))):
                continue
            cell = first + second - anchor
            if (cv2.isContourConvex(cell) and cv2.contourArea(cell, oriented=True) > 0
                    and np.min(np.minimum(cell, scale - cell)) >= 4 and not claimed(cell)):
                cells.append((cell, np.stack([first, second])))
    return cells[0] if len(cells) == 1 else None


def _measure_layout_side(
    image: np.ndarray, cell: np.ndarray, side: int, palette: np.ndarray, claimed: Callable[[np.ndarray], bool],
) -> _Rail | None:
    start, stop = cell[[side, (side + 1) % 4]]
    span = start + np.float32([.08, .92])[:, None] * (stop - start)
    measurements = [_side_line(image, *span, radius) for radius in (15, 18)]
    if any(line is None or evidence["ambiguous"] or evidence["support"] < .80 for line, evidence in measurements):
        return None
    (line, evidence), (crosscheck, _) = measurements
    measured = np.float32(_line_endpoints(*span, line))
    if np.abs(measured @ crosscheck[:2] + crosscheck[2]).max() > 1.5:
        return None
    a, b = measured
    direction = b - a
    normal = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
    # Ownership must be clear for the entire sampled strip, including rib and colour witnesses.
    witness = np.float32([a - 14 * normal, b - 14 * normal, b + 9 * normal, a + 9 * normal])
    if np.min(np.minimum(witness, np.float32(image.shape[1::-1]) - 1 - witness)) < 1 or claimed(witness):
        return None
    if not _ribs_end_at_edge(image, a, b):
        return None
    _, signed = _signed_stations(image, measured, palette)
    return _Rail(line, measured, evidence["support"], signed)


def neighboring_layout_traces(
    image: np.ndarray, active: list[np.ndarray], pending: list[dict], claims: list[dict],
) -> list[np.ndarray]:
    """Measure opposite open sides in one unclaimed cell. The layout never supplies a crop."""
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    outlines = [row["polygon"] for row in pending
                if row.get("reason") == TILED_PRINT_PROPOSAL and len(row.get("polygon") or []) == 4]
    if len(outlines) != 3 or not active:
        return []
    held = [order_quad_cyclic(_vertices(outline) * scale) for outline in outlines]
    anchors = [quad_from_polygon(outline * scale)[0] for outline in active if len(outline) >= 4]
    contours = _claimed_contours(claims, width, height, 0)

    def claimed(quad):
        return any(cv2.intersectConvexConvex(quad, contour)[0] > 1 for contour in contours)

    layout = _vacant_layout_cell(anchors, held, scale, claimed)
    if layout is None:
        return []
    cell, peers = layout
    reference = _backing_reference(image, cell, claimed)
    if reference is None:
        return []
    palette, side = reference
    opposite = (side + 2) % 4
    traces = []
    for index in (side, opposite):
        rail = _measure_layout_side(image, cell, index, palette, claimed)
        intervals = [] if rail is None else _trace_intervals(image, rail, palette, contours)
        if not intervals:
            return []
        traces.extend(intervals)

    envelopes = [cell, *(cv2.convexHull(np.vstack((cell, peer)))[:, 0] for peer in peers)]
    shortest = np.linalg.norm(np.roll(cell, -1, axis=0) - cell, axis=1).min()
    if any(has_enclosing_perimeter(image, envelope, envelope, shortest, search_margin=.65, maximum_area_ratio=3.8)
           for envelope in envelopes):
        return []
    return traces


def followup_print_position(
    previous: np.ndarray, upper: np.ndarray, active: list[np.ndarray], held: list[np.ndarray],
) -> np.ndarray | None:
    """Translate an upper outline using a measured row and a matching lower neighbor."""
    vertices = np.roll(previous, -1, axis=0) - previous
    lengths = np.linalg.norm(vertices, axis=1)
    if lengths.min() < 180 or lengths.max() > 1.35 * lengths.min():
        return None
    axes = vertices[:2] / lengths[:2, None]
    center = previous.mean(axis=0)
    upwards = upper.mean(axis=0) - center
    directions = np.abs(axes @ upwards)
    vertical = int(directions.argmax())
    if (directions[vertical] < .97 * np.linalg.norm(upwards)
            or not .88 <= np.linalg.norm(upwards) / lengths[vertical] <= 1.30):
        return None
    row = 1 - vertical
    candidates = []
    for peer in active:
        if np.max(np.abs(np.linalg.norm(np.roll(peer, -1, axis=0) - peer, axis=1) / lengths - 1)) > .25:
            continue
        offset = peer.mean(axis=0) - center
        pitch = np.linalg.norm(offset)
        if not .95 <= pitch / lengths[row] <= 1.35 or abs(offset @ axes[row]) < .97 * pitch:
            continue
        lower_neighbors = [quad for quad in held if np.linalg.norm(
            quad.mean(axis=0) - (peer.mean(axis=0) - upwards),
        ) <= .19 * lengths[vertical]]
        if len(lower_neighbors) == 1:
            candidates.append(upper + offset)
    return candidates[0] if len(candidates) == 1 else None


def _open_rail_witness(start: np.ndarray, stop: np.ndarray, exterior: int) -> np.ndarray:
    delta = stop - start
    normal = np.float32([-delta[1], delta[0]]) / np.linalg.norm(delta)
    return np.float32([start - exterior * normal, stop - exterior * normal,
                      stop + 10 * normal, start + 10 * normal])


def _neighboring_source_spans(
    image: np.ndarray, quads: list[np.ndarray], side: int, palette: np.ndarray, contours: list[np.ndarray],
    *, rib_witness: bool = False,
) -> list[np.ndarray]:
    views = []
    for quad in quads:
        a, b = quad[[side, (side + 1) % 4]]
        a, b = a + np.float32([[.08], [.92]]) * (b - a)
        candidates = []
        for radius in ((14, 18) if rib_witness else (10, 24, 30)):
            line, evidence = _side_line(image, a, b, radius)
            if line is None or evidence["ambiguous"] or evidence["support"] < .82:
                continue
            span = np.float32(_line_endpoints(a, b, line))
            if rib_witness:
                if _ribs_end_at_edge(image, *span):
                    candidates.append((line, [span]))
            else:
                _, signed = _signed_stations(image, span, palette)
                if signed.mean() >= .88 and min(block.mean() for block in np.array_split(signed, 4)) >= .70:
                    candidates.append((line, _trace_intervals(image, _Rail(line, span, evidence["support"], signed),
                                                               palette, contours)))
        views.append(candidates)
    matches = []
    first, second = views
    for line, spans in first:
        for _, other in second:
            shared = _common_spans(spans, other, contours)
            length = sum(np.linalg.norm(end - start) for start, end in shared)
            if length >= 100:
                matches.append((length, line, shared))
    if not matches:
        return []
    _, _, longest = max(matches, key=lambda match: match[0])
    return longest if all(np.max(np.abs(np.vstack(longest) @ line[:2] + line[2])) <= 2 for _, line, _ in matches) else []


def neighboring_source_guide(image: np.ndarray, polygons: list[np.ndarray], claims: list[dict]) -> dict | None:
    """Show observed sides of a vacant cell, with an independent opposite rib witness."""
    if len(polygons) != 2 or normalized_polygon_overlap(*polygons, raster_size=1024).iou < .90:
        return None
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    seeds = [quad_from_polygon(polygon * scale)[0] for polygon in polygons]
    if any(seed is None for seed in seeds):
        return None
    quads = [order_quad_cyclic(seed) for seed in seeds]
    if any(np.min(np.minimum(quad, scale - quad)) < 24 or not _photo_texture(image, quad) for quad in quads):
        return None
    contours = _claimed_contours(claims, width, height, 0)

    def claimed(polygon):
        return any(cv2.intersectConvexConvex(polygon, contour)[0] > 1 for contour in contours)

    scope = cv2.convexHull(np.vstack(quads))[:, 0]
    if claimed(scope):
        return None
    references = [_backing_reference(image, quad, claimed) for quad in quads]
    if any(reference is None for reference in references):
        return None
    (palette, side), (second_palette, second_side) = references
    if side != second_side or np.linalg.norm(palette - second_palette) > 6:
        return None
    opposite = _neighboring_source_spans(image, quads, (side + 2) % 4, palette, contours, rib_witness=True)
    if not opposite:
        return None
    measured = [_neighboring_source_spans(image, quads, index, palette, contours)
                for index in (side, (side - 1) % 4, (side + 1) % 4)]
    if not measured[0] or bool(measured[1]) == bool(measured[2]):
        return None
    spans = [span for pieces in measured for span in pieces]
    # Include the entire sampled shoulders and the position in claim retirement.
    witnesses = [scope, *(_open_rail_witness(*span, 9) for span in spans),
                 *(_open_rail_witness(*span, 15) for span in opposite)]
    if any(claimed(polygon) or np.min(np.minimum(polygon, scale - polygon)) < 1 for polygon in witnesses):
        return None
    shortest = np.linalg.norm(np.roll(quads[0], -1, axis=0) - quads[0], axis=1).min()
    if has_enclosing_perimeter(image, quads[0], scope, shortest, search_margin=.65, maximum_area_ratio=3.8):
        return None
    return {"traces": [_normalized(span / scale) for span in spans],
            "witnesses": [_normalized(polygon / scale) for polygon in witnesses]}


def followup_print_edge_guide(
    image: np.ndarray, projected: np.ndarray, peer: np.ndarray, claims: list[dict],
) -> dict | None:
    """Search a nearby position for three independently supported open source sides."""
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    if (not cv2.isContourConvex(projected) or np.min(np.minimum(projected, scale - projected)) < 24
            or not _photo_texture(image, projected)):
        return None
    rails = []
    for a, b in zip(projected, np.roll(projected, -1, axis=0), strict=True):
        start, stop = a + np.float32([[.08], [.92]]) * (b - a)
        fits = [_side_line(image, start, stop, radius) for radius in (14, 17, 20)]
        if any(line is None or facts["ambiguous"] or facts["support"] < .94 for line, facts in fits):
            continue
        line, _ = fits[1]
        span = np.float32(_line_endpoints(start, stop, line))
        if any(np.abs(span @ rival[:2] + rival[2]).max() > 1.5 for rival, _ in fits):
            continue
        rails.append(span)
    if len(rails) != 3:
        return None
    phases = [bool(_ribs_end_at_edge(image, *span)) for span in rails]
    if phases.count(True) != 1:
        return None
    phase_rail = rails[phases.index(True)]
    phase_witness = _open_rail_witness(*phase_rail, 15)
    phase_scope = _normalized(phase_witness / scale)
    if np.min(np.minimum(phase_witness, scale - phase_witness)) < 1 or tiled_outline_conflicts(
        phase_scope, claims, width, height,
    ):
        return None
    backing = _edge_band(image, *phase_rail, [-14, -12, -10])
    if backing is None or np.median(backing[..., 0]) < 145:
        return None
    palette = np.median(backing[..., 1:].reshape(-1, 2), axis=0)
    if not 123 <= palette[0] <= 139 or palette[1] < 135:
        return None
    contours = _claimed_contours(claims, width, height, 0)
    spans, witnesses = [], [phase_scope]
    for phase, span in zip(phases, rails, strict=True):
        stations, signed = _signed_stations(image, span, palette)
        if signed.mean() < .85:
            return None
        exterior = 15 if phase else 10
        # Clip the sampled strip to claim-free intervals on the measured line.
        delta = span[1] - span[0]
        normal = np.float32([-delta[1], delta[0]]) / np.linalg.norm(delta)
        corridor = _open_rail_witness(*span, exterior)
        low, high = corridor.min(axis=0) - 1, corridor.max(axis=0) + 1
        for contour in contours:
            vertices = contour[:, 0]
            if np.any(vertices.max(axis=0) < low) or np.any(vertices.min(axis=0) > high):
                continue
            for offset in range(-exterior, 11):
                signed &= [cv2.pointPolygonTest(contour, tuple(point), True) < -.5
                           for point in stations + offset * normal]
        indices = np.flatnonzero(signed)
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        observed = []
        for run in runs:
            if len(run) < 14:
                continue
            trace = stations[run[[0, -1]]]
            region = _open_rail_witness(*trace, exterior)
            if np.min(np.minimum(region, scale - region)) < 1:
                continue
            witness = _normalized(region / scale)
            if not tiled_outline_conflicts(witness, claims, width, height):
                observed.append(trace)
                witnesses.append(witness)
        if sum(np.linalg.norm(stop - start) for start, stop in observed) < .15 * np.linalg.norm(delta):
            return None
        spans.extend(observed)
    shortest = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1).min()
    envelope = cv2.convexHull(np.vstack([projected, peer]))[:, 0]
    inside = cv2.convexHull(np.vstack(spans))[:, 0]
    if any(has_enclosing_perimeter(image, shape, context, shortest, search_margin=.65, maximum_area_ratio=3.8)
           for shape, context in ((projected, inside), (envelope, envelope))):
        return None
    return {"traces": [_normalized(span / scale) for span in spans], "witnesses": witnesses}


def _native_outline(image: np.ndarray, polygon: np.ndarray, direction: np.ndarray, claims: list[dict]) -> np.ndarray | None:
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
    quad, evidence, reasons = refine_print_boundary(image, order_quad_cyclic(seed))
    if quad is None or len(evidence) != 4 or set(reasons) - {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_CORNERS_UNCERTAIN"}:
        return None
    # Identify the side facing away from the two witnesses, including when
    # the row and captured page are rotated.
    outer = (quad + np.roll(quad, -1, axis=0)) / 2 - quad.mean(axis=0)
    leading = np.argmax((outer @ direction) / np.linalg.norm(outer, axis=1))
    ordered = np.roll(quad, -leading, axis=0)
    methods = (_leading_edge, _lateral_edge, _trailing_edge, _lateral_edge)
    lines = [method(image, np.float32([a, b])) for a, b, method in
             zip(ordered, np.roll(ordered, -1, axis=0), methods, strict=True)]
    if any(line is None for line in lines):
        return None
    fit = _quad_from_lines(np.stack(lines))
    if (fit is None or not cv2.isContourConvex(fit) or cv2.contourArea(fit, oriented=True) <= 0
            or np.any(np.minimum(fit, scale - fit) < 4)
            or normalized_polygon_overlap(fit / scale, polygon, raster_size=1024).iou < .85
            or tiled_outline_conflicts(_normalized(fit / scale), claims, width, height)):
        return None
    flank_supported = False
    for side, (start, stop) in enumerate(zip(fit, np.roll(fit, -1, axis=0), strict=True)):
        delta = stop - start
        length = np.linalg.norm(delta)
        normal = np.float32([-delta[1], delta[0]]) / length
        a, b = start + np.float32([.08, .92])[:, None] * delta
        witness = np.float32([a - 8 * normal, b - 8 * normal, b + 6 * normal, a + 6 * normal])
        if tiled_outline_conflicts(_normalized(witness / scale), claims, width, height):
            return None
        band = _edge_band(image, start, stop, [-8, -6, -4, -2, 0, 2, 4, 6], fractions=_FRACTIONS)
        if band is None or (np.linalg.norm(band[:, 3] - band[:, 5], axis=1) >= 12).mean() < .90:
            return None
        outside = np.median(band[:, :3, 0], axis=1)
        inside = np.median(band[:, 5:, 0], axis=1)
        if (outside - inside >= 12).mean() < .70:
            return None
        if side % 2:
            collar = np.float32([a - 14 * normal, b - 14 * normal, b, a])
            if not tiled_outline_conflicts(_normalized(collar / scale), claims, width, height):
                flank_supported |= _ribs_end_at_edge(image, start, stop)
    if not flank_supported:
        return None
    return order_quad_cyclic(fit)


def neighboring_print_outline(
    image: np.ndarray, masks: list[np.ndarray], direction: np.ndarray, claims: list[dict],
) -> np.ndarray | None:
    """Corroborate a held outline in two local views using original pixels."""
    if len(masks) != 2 or normalized_polygon_overlap(*masks, raster_size=1024).iou < .90:
        return None
    first, second = (_native_outline(image, polygon, direction, claims) for polygon in masks)
    if first is None or second is None:
        return None
    drift = min(np.linalg.norm(first - np.roll(second, turn, axis=0), axis=1).max() for turn in range(4))
    if drift > 1.5:
        return None
    short = float(np.linalg.norm(np.roll(second, -1, axis=0) - second, axis=1).min())
    if has_enclosing_perimeter(image, second, second, short, search_margin=1.65, maximum_area_ratio=12):
        return None
    return second
