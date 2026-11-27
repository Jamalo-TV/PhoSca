"""Source checks for uncounted print suggestions from enlarged views."""

from collections.abc import Callable

import cv2
import numpy as np

from app.pipeline.album_backing import (
    _backing_reference, _repeating_profile, _supported_backing_transition, measured_ribbed_backing_outline,
)
from app.pipeline.geometry import (
    normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, pixel_vertices_to_normalized,
    quad_from_polygon,
)
from app.pipeline.print_edges import _edge_band, _line_endpoints, _quad_from_lines, refine_print_boundary
from app.pipeline.visible_scene_edge import has_enclosing_perimeter


TILED_PRINT_PROPOSAL = "possible_tiled_print"


def tiled_outline_conflicts(outline: list[dict], claims: list[dict], width: int, height: int) -> bool:
    """Keep the proposed outline clear of every current paper geometry."""
    def hull(vertices):
        normalized = [(point["x"], point["y"]) for point in vertices]
        return cv2.convexHull(normalized_vertices_to_pixels(normalized, width, height))

    target = hull(outline)
    for claim in claims:
        if source := claim.get("source_mask"):
            if tiled_outline_conflicts(outline, [source], width, height):
                return True
        for key in ("polygon", "visible_polygon", "warp_quad"):
            vertices = claim.get(key) or []
            if len(vertices) >= 3:
                try:
                    if cv2.intersectConvexConvex(target, hull(vertices))[0] > 1:
                        return True
                except (TypeError, ValueError, KeyError, cv2.error):
                    return True
    return False


def tile_print_outline(
    image: np.ndarray, polygon: np.ndarray, *, require_clear_sides: bool = False, minimum_side_support: float = .70,
) -> np.ndarray | None:
    """Check the four sides in the unenhanced source before offering a draft."""
    height, width = image.shape[:2]
    seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
    if seed is None:
        return None
    outline, sides, reasons = refine_print_boundary(image, seed)
    if (outline is None or len(sides) != 4
            or min(side["support"] for side in sides.values()) < minimum_side_support
            or sum(side["support"] >= .85 and not side["ambiguous"] for side in sides.values()) < 3):
        return None
    # A curved paper edge can still identify a missing print for count review.
    # Require four clear, strongly measured sides for this case.
    if require_clear_sides or reasons == ["PRINT_BOUNDARY_CURVED"]:
        if set(reasons) - {"PRINT_BOUNDARY_CURVED"}:
            return None
        if not all(side["support"] >= .90 and not side["ambiguous"] for side in sides.values()):
            return None
    elif set(reasons) - {"PRINT_BOUNDARY_AMBIGUOUS"}:
        return None
    if np.any(np.minimum(outline, [width - 1, height - 1] - outline) < 4):
        return None
    return outline


def ribbed_backing_tile_outline(image: np.ndarray, polygon: np.ndarray, claims: list[dict]) -> list[dict] | None:
    height, width = image.shape[:2]
    if tiled_outline_conflicts([{"x": float(x), "y": float(y)} for x, y in polygon], claims, width, height):
        return None
    seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
    if seed is None:
        return None

    def normalized(quad: np.ndarray) -> list[dict]:
        return [{"x": float(x), "y": float(y)} for x, y in pixel_vertices_to_normalized(quad, width, height)]

    def claimed(quad: np.ndarray) -> bool:
        return tiled_outline_conflicts(normalized(quad), claims, width, height)

    fitted = measured_ribbed_backing_outline(image, seed, claimed)
    if fitted is None:
        return None
    vertices = pixel_vertices_to_normalized(fitted, width, height)
    if normalized_polygon_overlap(vertices, polygon, raster_size=512).iou < .85:
        return None
    return normalized(fitted)


def _signed_tile_edge_support(image: np.ndarray, line: np.ndarray, palette: np.ndarray) -> float:
    fractions = np.linspace(.10, .90, 160, dtype=np.float32)
    band = _edge_band(image, *line, [-6, -4, -2, 1, 3, 5], fractions=fractions)
    if band is None:
        return 0.
    outside = np.median(band[:, :3], axis=1)
    inside = np.median(band[:, 3:], axis=1)
    exterior_distance = np.linalg.norm(outside[:, 1:] - palette, axis=1)
    interior_distance = np.linalg.norm(inside[:, 1:] - palette, axis=1)
    signed_change = (outside[:, 0] - inside[:, 0] >= 20) | (interior_distance - exterior_distance >= 5)
    return float(np.mean((exterior_distance <= 8) & signed_change))


def _transverse_tile_backing(image: np.ndarray, line: np.ndarray, palette: np.ndarray) -> bool:
    fractions = np.linspace(.10, .90, 160, dtype=np.float32)
    band = _edge_band(image, *line, [-6, -2, 3], fractions=fractions)
    if band is None:
        return False
    exterior = band[:, :2]
    stock = (np.linalg.norm(exterior[..., 1:] - palette, axis=2) <= 8) & (exterior[..., 0] >= 135)
    if stock.all(axis=1).mean() < .80:
        return False
    far, near, inside = [_repeating_profile(column) for column in band[..., 0].T]
    if min(far[0].std(), near[0].std()) < 12 or min(far[1], near[1]) < .52 or inside[0].std() == 0:
        return False
    correlation = np.corrcoef([far[0], near[0], inside[0]])
    return correlation[0, 1] >= .85 and abs(correlation[0, 2]) < .25


def _fit_unclaimed_tile_views(
    image: np.ndarray, polygons: tuple[np.ndarray, np.ndarray], claimed: Callable[[np.ndarray], bool],
) -> list[tuple[np.ndarray, list[dict]]] | None:
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    fits = []
    for polygon in polygons:
        pixels = normalized_vertices_to_pixels(polygon, width, height)
        if claimed(pixels):
            return None
        seed, _ = quad_from_polygon(pixels)
        if seed is None:
            return None
        outline, sides, reasons = refine_print_boundary(image, order_quad_cyclic(seed))
        if (outline is None or len(sides) != 4 or set(reasons) - {"PRINT_BOUNDARY_AMBIGUOUS"}
                or min(side["support"] for side in sides.values()) < .85
                or np.any(np.minimum(outline, scale - outline) < 4) or claimed(outline)):
            return None
        fits.append((outline, list(sides.values())))
    return fits


def measured_adjacent_tile_outline(
    image: np.ndarray, observations: list[tuple[np.ndarray, float]], claims: list[dict],
) -> np.ndarray | None:
    """Measure a held print outline when two close views disagree on a single edge."""
    if len(observations) != 2:
        return None
    (first, first_score), (second, second_score) = sorted(observations, key=lambda row: row[1], reverse=True)
    if first_score < .80 or second_score < .60 or normalized_polygon_overlap(first, second, raster_size=512).iou < .90:
        return None
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])

    def claimed(polygon: np.ndarray) -> bool:
        return tiled_outline_conflicts([{"x": float(x), "y": float(y)} for x, y in polygon / scale], claims, width, height)

    fits = _fit_unclaimed_tile_views(image, (first, second), claimed)
    if fits is None:
        return None

    primary, primary_sides = fits[0]
    rival, rival_sides = fits[1]
    # Match the physical sides even if the two fits start on different vertices.
    shift = min(range(4), key=lambda offset: np.linalg.norm(primary - np.roll(rival, offset, axis=0)))
    rival = np.roll(rival, shift, axis=0)
    rival_sides = np.roll(rival_sides, shift)
    tangent = np.roll(primary, -1, axis=0) - primary
    lengths = np.linalg.norm(tangent, axis=1)
    shortest = float(lengths.min())
    if shortest < 180:
        return None
    inward = np.column_stack((-tangent[:, 1], tangent[:, 0])) / lengths[:, None]
    displacement = rival - primary
    normal_delta = np.column_stack(((displacement * inward).sum(axis=1),
                                   (np.roll(displacement, -1, axis=0) * inward).sum(axis=1)))
    disputes = np.flatnonzero((normal_delta.max(axis=1) <= -5) & (normal_delta.min(axis=1) >= -20))
    if len(disputes) != 1:
        return None
    side = int(disputes[0])
    stable_sides = [i for i in range(4) if i != side]
    if (not primary_sides[side]["ambiguous"] or np.max(np.abs(normal_delta[stable_sides])) > 4
            or any(evidence[index]["ambiguous"] or evidence[index]["support"] < .90
                   for evidence in (primary_sides, rival_sides) for index in stable_sides)):
        return None

    reference = _backing_reference(image, primary, claimed)
    if reference is None or (reference[1] - side) % 2 == 0:
        return None
    palette = reference[0]
    edge = [side, (side + 1) % 4]
    if (_signed_tile_edge_support(image, primary[edge], palette) < .80
            or _signed_tile_edge_support(image, rival[edge], palette) >= .30):
        return None
    for adjacent in ((side - 1) % 4, (side + 1) % 4):
        line = primary[[adjacent, (adjacent + 1) % 4]]
        start, stop = line[0] + np.float32([.10, .90])[:, None] * tangent[adjacent]
        outward = 6 * inward[adjacent]
        if (claimed(np.float32([start - outward, stop - outward, stop, start]))
                or not _transverse_tile_backing(image, line, palette)):
            return None

    radius = min(20, round(.065 * shortest))
    primary_line = _supported_backing_transition(image, *primary[edge], palette, radius)
    rival_line = _supported_backing_transition(image, *rival[edge], palette, radius)
    if primary_line is None or rival_line is None:
        return None
    primary_ends = np.float32(_line_endpoints(*primary[edge], primary_line))
    rival_ends = np.float32(_line_endpoints(*primary[edge], rival_line))
    if (np.max(np.linalg.norm(primary_ends - rival_ends, axis=1)) > 4
            or np.max(np.linalg.norm(primary_ends - primary[edge], axis=1)) > 8):
        return None
    points = np.column_stack((primary, np.ones(4)))
    lines = np.cross(points, np.roll(points, -1, axis=0))
    lines[side] = primary_line
    outline = _quad_from_lines(lines)
    if (outline is None or not cv2.isContourConvex(outline) or cv2.contourArea(outline, oriented=True) <= 0
            or np.any(np.minimum(outline, scale - outline) < 4) or claimed(outline)
            or any(normalized_polygon_overlap(outline / scale, mask, raster_size=512).iou < .85
                   for mask in (first, second))):
        return None
    # A larger print can reproduce two neighboring views and the ribs between them.
    if has_enclosing_perimeter(image, outline, outline, shortest, search_margin=1.15, maximum_area_ratio=3.1):
        return None
    return outline


def _signed_grid_rim_offsets(image: np.ndarray, line: np.ndarray, palette: np.ndarray) -> np.ndarray | None:
    offsets = np.arange(-9, 10, dtype=np.float32)
    fractions = np.linspace(.08, .92, 160, dtype=np.float32)
    band = _edge_band(image, *line, offsets, fractions=fractions)
    if band is None:
        return None
    centers = np.arange(6, 13)
    light = band[..., 0]
    near = light[:, centers - 1] - light[:, centers + 1]
    wide = light[:, centers - 2] - light[:, centers + 2]
    exterior = np.median(band[:, centers[:, None] - [5, 4, 3]], axis=2)
    backed = np.linalg.norm(exterior[..., 1:] - palette, axis=2) <= 8
    support = (near >= 12) & backed
    weakest_eighth = np.min([part.mean(axis=0) for part in np.array_split(support, 8)], axis=0)
    contrast = np.median(wide, axis=0)
    good = (support.mean(axis=0) >= .90) & (weakest_eighth >= .60) & (contrast >= 24)
    ranked = np.argsort(-contrast[good], kind="stable")
    return offsets[centers[good]][ranked]


def measured_grid_tile_outline(
    image: np.ndarray, observations: list[tuple[np.ndarray, float]], claims: list[dict],
) -> np.ndarray | None:
    """Resolve one uncertain grid-view side against another view and source pixels."""
    if len(observations) != 2 or not all(score >= .80 for _, score in observations):
        return None
    first, second = (polygon for polygon, _ in observations)
    if normalized_polygon_overlap(first, second, raster_size=512).iou < .90:
        return None
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])

    def claimed(polygon: np.ndarray) -> bool:
        return tiled_outline_conflicts([{"x": float(x), "y": float(y)} for x, y in polygon / scale], claims, width, height)

    fits = _fit_unclaimed_tile_views(image, (first, second), claimed)
    if fits is None:
        return None
    compared = [(outline, [index for index, side in enumerate(sides) if side["ambiguous"]]) for outline, sides in fits]
    compared.sort(key=lambda fit: len(fit[1]))
    (base, base_uncertainty), (rival, rival_uncertainty) = compared
    if base_uncertainty or len(rival_uncertainty) != 1:
        return None
    shift = min(range(4), key=lambda offset: np.linalg.norm(base - np.roll(rival, offset, axis=0)))
    rival = np.roll(rival, shift, axis=0)
    side = (rival_uncertainty[0] + shift) % 4
    tangent = np.roll(base, -1, axis=0) - base
    lengths = np.linalg.norm(tangent, axis=1)
    shortest = float(lengths.min())
    if shortest < 180:
        return None
    inward = np.column_stack((-tangent[:, 1], tangent[:, 0])) / lengths[:, None]
    displacement = rival - base
    normal_delta = np.column_stack(((displacement * inward).sum(axis=1),
                                   (np.roll(displacement, -1, axis=0) * inward).sum(axis=1)))
    if np.max(np.abs(normal_delta[np.arange(4) != side])) > 4 or np.max(np.abs(normal_delta[side])) > 15:
        return None

    reference = _backing_reference(image, base, claimed)
    if reference is None or (reference[1] - side) % 2 == 0:
        return None
    palette = reference[0]
    edge = [side, (side + 1) % 4]
    line = base[edge]
    offsets = _signed_grid_rim_offsets(image, line, palette)
    if offsets is None or not len(offsets) or np.ptp(offsets) > 3 or abs(offsets[0]) > 2:
        return None
    chosen = offsets[0]
    alternate = _signed_grid_rim_offsets(image, rival[edge], palette)
    if alternate is None or np.any(np.abs(normal_delta[side] + alternate[:, None] - chosen) > 2.5):
        return None

    candidate = line + chosen * inward[side]
    start, stop = candidate[0] + np.float32([.08, .92])[:, None] * (candidate[1] - candidate[0])
    outward = 7 * inward[side]
    if claimed(np.float32([start - outward, stop - outward, stop, start])):
        return None
    points = np.column_stack((base, np.ones(4)))
    lines = np.cross(points, np.roll(points, -1, axis=0))
    lines[side] = np.cross(np.r_[candidate[0], 1.], np.r_[candidate[1], 1.])
    outline = _quad_from_lines(lines)
    if (outline is None or not cv2.isContourConvex(outline) or cv2.contourArea(outline, oriented=True) <= 0
            or np.any(np.minimum(outline, scale - outline) < 4) or claimed(outline)
            or any(normalized_polygon_overlap(outline / scale, mask, raster_size=512).iou < .85
                   for mask in (first, second))):
        return None
    if has_enclosing_perimeter(image, outline, outline, shortest, search_margin=1.15, maximum_area_ratio=3.1):
        return None
    return outline


def opposing_view_print_outlines(
    image: np.ndarray, first_view: list[tuple[np.ndarray, float]], second_view: list[tuple[np.ndarray, float]],
    *, acceptance: float,
) -> list[np.ndarray]:
    """Measure uniquely agreed proposals whose outlines are already in source coordinates."""
    pairs = [
        (i, j) for i, first in enumerate(first_view) for j, second in enumerate(second_view)
        if normalized_polygon_overlap(first[0], second[0], raster_size=512).iou >= .80
    ]
    height, width = image.shape[:2]
    scale = np.float32([width - 1, height - 1])
    outlines = []
    for i, j in pairs:
        if sum(left == i or right == j for left, right in pairs) != 1:
            continue
        first, first_confidence = first_view[i]
        second, second_confidence = second_view[j]
        if max(first_confidence, second_confidence) < acceptance:
            continue
        shapes = first, second
        fitted = []
        for shape in shapes:
            try:
                outline = tile_print_outline(image, shape, require_clear_sides=True)
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                continue
            if outline is not None and all(
                normalized_polygon_overlap(outline / scale, other, raster_size=512).iou >= .84
                for other in shapes
            ):
                fitted.append(order_quad_cyclic(outline))
        if not fitted:
            continue
        if len(fitted) == 2 and np.max(np.linalg.norm(fitted[0] - fitted[1], axis=1)) > 4:
            continue
        outlines.append(fitted[0])
    return outlines
