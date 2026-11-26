"""Source checks for uncounted print suggestions from enlarged views."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_vertices_to_pixels, quad_from_polygon
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


def tile_print_outline(image: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    """Check the four sides in the unenhanced source before offering a draft."""
    height, width = image.shape[:2]
    seed, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
    if seed is None:
        return None
    outline, sides, reasons = refine_print_boundary(image, seed)
    if (outline is None or len(sides) != 4
            or min(side["support"] for side in sides.values()) < .70
            or sum(side["support"] >= .85 and not side["ambiguous"] for side in sides.values()) < 3):
        return None
    # A curved paper edge can still identify a missing print for count review.
    # Require four clear, strongly measured sides for this case.
    if reasons == ["PRINT_BOUNDARY_CURVED"]:
        if not all(side["support"] >= .90 and not side["ambiguous"] for side in sides.values()):
            return None
    elif set(reasons) - {"PRINT_BOUNDARY_AMBIGUOUS"}:
        return None
    if np.any(np.minimum(outline, [width - 1, height - 1] - outline) < 4):
        return None
    return outline
