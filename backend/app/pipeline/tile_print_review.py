"""Source checks for uncounted print suggestions from enlarged views."""

import cv2
import numpy as np

from app.pipeline.album_backing import measured_ribbed_backing_outline
from app.pipeline.geometry import (
    normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic, pixel_vertices_to_normalized,
    quad_from_polygon,
)
from app.pipeline.print_edges import refine_print_boundary


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
