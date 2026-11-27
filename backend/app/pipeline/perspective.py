from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.adjacent_stock_review import ADJACENT_STOCK_REVIEW_SOURCE, suggest_adjacent_stock_outline
from app.pipeline.album_layer_review import suggest_album_layer_preview
from app.pipeline.album_opening_review import POSSIBLE_ALBUM_OPENING_EDGE, album_opening_traces
from app.pipeline.album_windows import (
    ALBUM_WINDOW_OCCLUDED, ALBUM_WINDOW_SOURCE, ALBUM_WINDOW_STOCK_UNCERTAIN, album_window_leaves,
)
from app.pipeline.border_removal import CURVED_IMAGE_BOUNDARY, border_chord_quad, find_physical_border, wide_paper_motif_proposal
from app.pipeline.bowed_edge_review import (
    POSSIBLE_ALBUM_LAYER_EDGE, POSSIBLE_BOWED_PRINT_EDGE, POSSIBLE_DISPLACED_PRINT_EDGES, POSSIBLE_LOCAL_PRINT_EDGE,
    POSSIBLE_MOUNTED_EDGE_STRIP, suggest_bowed_print_preview, suggest_displaced_print_preview,
)
from app.pipeline.crop_sampling import OWNED_BILINEAR, warp_owned_crop, warp_owned_outline
from app.pipeline.dark_album_review import DARK_BACKING_OUTLINE_REVIEW, DARK_MOUNTED_EDGE_SOURCE
from app.pipeline.decisions import (
    MANUAL_NEIGHBOR_CONTENT_LOSS, MANUAL_NEIGHBOR_OVERLAP, MOTIF_MARGIN_MARKS_REVIEW, MOUNTED_PRINT_OUTLINE_REVIEW,
)
from app.pipeline.edge_prints import (
    CAPTURE_EDGE_HOLD, CAPTURE_EDGE_SOURCE, CAPTURE_SPLIT_REVIEW_SOURCE, capture_edge_requires_review,
)
from app.pipeline.embossed_album_pockets import POSSIBLE_EMBOSSED_ALBUM_PAPER_EDGES, embossed_carrier_paper_traces
from app.pipeline.crossing_prints import CROSSING_PRINT_REVIEW, REVIEWED_CROSSING_OUTLINE
from app.pipeline.fragmented_prints import (
    EXTENDED_PRINT_OUTLINE_REVIEW, FRAGMENT_OUTLINE_REVIEW, FRAGMENTED_PRINT_REVIEW, SOURCE_FRAGMENT_GROUP_REVIEW,
)
from app.pipeline.geometry import (
    normalized_polygon_overlap,
    normalized_vertices_to_pixels,
    order_quad_cyclic,
    pixel_vertices_to_normalized,
    quad_from_polygon,
    rectify_crop_quad,
)
from app.pipeline.image_ops import normalized_box_to_pixels, save_jpeg
from app.pipeline.merged_print_review import MERGED_PRINT_PAIR, MERGED_PRINT_PAIR_REVIEW
from app.pipeline.inward_corner_review import INWARD_CORNER_REVIEW_SOURCE, suggest_inward_corner_outline
from app.pipeline.local_print_edge_review import suggest_local_print_edge_preview
from app.pipeline.motif_boundary import (
    PARTIAL_MOTIF_SEAMS, close_motif_outline, motif_outline_corners_supported, motif_seams_intersect_claim,
    partial_motif_seams, trace_motif_boundary,
)
from app.pipeline.motif_margin import (
    HELD_PRINT_MOTIF_MARGIN_SOURCE, PAPER_STOCK_MOTIF_SOURCE, held_print_motif_margin_marks, motif_margin_marks,
)
from app.pipeline.mounted_edge_preview import suggest_mounted_strip_preview
from app.pipeline.off_print_advisory import MANUAL_MARGIN_CONTEXT, manual_margin_advisory, off_print_advisory_regions
from app.pipeline.multipanel_prints import DISPUTED_GRID_PREVIEW_SOURCE, GRID_OUTLINE_REVIEW_SOURCE
from app.pipeline.manual_paper_boundary import (
    MANUAL_CAPTURE_OUTLINE_REVIEW, MANUAL_PAPER_BOUNDARY_REVIEW, MANUAL_PAPER_BOUNDARY_REVISION,
    check_manual_paper_boundary, shared_rim_cut_segments,
)
from app.pipeline.orientation import correct_photo_orientation, rotate_image
from app.pipeline.overlap_edge_review import suggest_exposed_overlap_edges
from app.pipeline.page_plane import (
    PRINT_ASPECT_REFERENCE_REVIEW,
    MeasuredPrintReference,
    PrintPagePlane,
    fit_calibrated_page_plane,
    fit_measured_page_plane,
    fit_uncalibrated_page_plane,
    measured_page_aspect,
)
from app.pipeline.partial_print_review import PARTIAL_PRINT_EXTENT_REVIEW, PARTIAL_PRINT_EXTENT_TRACES
from app.pipeline.print_edges import (
    COMPETING_PRINT_EDGES,
    POSSIBLE_BACKED_PAPER_EDGE,
    PRINT_EDGE_CONFIRMATION,
    PRINT_FAINT_RIM_REVIEW,
    PRINT_SHADOW_EDGE_REVIEW,
    PRINT_SHARED_RIM_UNCERTAIN,
    SHARED_PRINT_RIM_SOURCE,
    continuous_motif_rim,
    missing_corner_material,
    observed_rim_points,
    paper_rim_reaches_neighbor,
    print_corner_coverage,
    refine_print_boundary,
    retry_ambiguous_stock_edge,
    strip_resembles_album_backing,
    suggest_ambiguous_stock_edge,
    suggest_missing_pale_backing_edge,
    suggest_partial_backed_edge,
    suggest_partial_corner_edge,
)
from app.pipeline.print_outline_suggestion import (
    POSSIBLE_INWARD_PRINT_EDGE, POSSIBLE_OVEREXTENDED_PRINT_EDGE, suggest_inward_print_edge,
    suggest_outer_print_outline, suggest_overextended_print_outline, suggest_slanted_print_edge_comparison,
    verify_print_edge_at_claim, visible_gap_at_unchanged_claim,
)
from app.pipeline.print_stock import find_outer_stock_candidate
from app.pipeline.projective_prints import PROJECTIVE_PRINT_OUTLINE_REVIEW
from app.pipeline.protected_regions import (
    OFF_PRINT_ADVISORY, PAGE_FACE_SUGGESTIONS, ProtectedRegionRemoval, detect_protected_regions, is_advisory_page_face,
    merge_protected_regions, protected_region_key, region_box_in_pixels, removed_protected_regions,
    validate_protected_regions,
)
from app.pipeline.rounded_corner_review import (
    ROUNDED_CORNER_REVIEW, ROUNDED_CORNER_SOURCE, ROUNDED_CORNER_WEDGES, measure_rounded_corners,
)
from app.pipeline.scalloped_prints import SCALLOPED_PRINT_REVIEW, SCALLOPED_PRINT_REVISION
from app.pipeline.segmentation_filter import CONTRAST_COMPONENT_SOURCE, refine_visible_paper_rim
from app.pipeline.source_strip_review import source_strip_outline

_PRINT_BOUNDARY_SOURCE = "physical_print_boundary"
_PAPER_RIM_SOURCE = "visible_paper_rim"
_REFINED_VISIBILITY_SOURCES = {_PRINT_BOUNDARY_SOURCE, _PAPER_RIM_SOURCE}
_LOCALIZED_OCCLUSION_REASONS = {"PRINT_CORNERS_UNCERTAIN", "PRINT_POSSIBLY_OCCLUDED"}
_CROP_PREVIEW_REASONS = {PRINT_EDGE_CONFIRMATION, PRINT_ASPECT_REFERENCE_REVIEW, FRAGMENTED_PRINT_REVIEW,
                         SCALLOPED_PRINT_REVIEW, PRINT_SHADOW_EDGE_REVIEW, PRINT_FAINT_RIM_REVIEW,
                         EXTENDED_PRINT_OUTLINE_REVIEW, PRINT_SHARED_RIM_UNCERTAIN}
_STOCK_REVIEW_CONFLICTS = ("paper_grid_review", "film_grid_review_seam", "paper_join_review_traces",
                           "crossing_print_suggestion", "fragment_outline_suggestion", "capture_edge_clipped",
                           "scalloped_print", "scalloped_ownership_outline")


def _pale_backing_review_fit(boundary: dict) -> bool:
    return any(isinstance(side, dict) and side.get("review_pale_backing_fit") for side in boundary.values())


@dataclass(frozen=True)
class CropResult:
    """Faithful rectified pixels plus any optional, reversible derivative."""

    rectified_untrimmed: np.ndarray
    image: np.ndarray
    metadata: dict
    image_only: np.ndarray | None = None
    image_only_preview: np.ndarray | None = None
    full_print_preview: np.ndarray | None = None


def _explicit_quad(
    mask: dict, key: str, width: int, height: int, *, preserve_width_edge: bool = False,
) -> np.ndarray | None:
    vertices = mask.get(key)
    if vertices is None:
        return None
    try:
        points = np.array([[float(p["x"]), float(p["y"])] for p in vertices], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{key}") from exc
    if points.shape != (4, 2) or not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
        raise ValueError(f"invalid_{key}")
    points = normalized_vertices_to_pixels(points, width, height)
    if not preserve_width_edge:
        points = _order_points(points)
    _validate_quad(points, width, height)
    if preserve_width_edge and cv2.contourArea(points, oriented=True) < 0:
        # Reverse the winding without changing which edge fixes the measured width.
        points = points[[1, 0, 3, 2]]
    return points


def validate_review_quads(mask: dict, width: int, height: int, *, require_cyclic: bool = False) -> None:
    """Validate authoritative reviewer vertices before storing a correction revision."""
    measured = mask.get("print_size_mm") is not None
    outer = _explicit_quad(mask, "warp_quad", width, height, preserve_width_edge=measured or require_cyclic)
    if measured and outer is None:
        raise ValueError("measured_print_requires_warp_quad")
    inner = _explicit_quad(mask, "image_quad", width, height)
    if inner is None:
        return
    if outer is None or any(cv2.pointPolygonTest(outer, tuple(point), True) < -0.02 for point in inner):
        raise ValueError("image_quad_outside_print")


def manual_print_intersects(
    crop_quad: list[dict] | None, marked_quad: list[dict], width: int, height: int,
    *, minimum_overlap_px: float | None = None,
) -> bool:
    """Treat a material intersection with an unverified mark as ambiguous ownership."""
    try:
        crop = _explicit_quad({"warp_quad": crop_quad}, "warp_quad", width, height)
        mark = _explicit_quad({"warp_quad": marked_quad}, "warp_quad", width, height)
    except ValueError:
        return False
    if crop is None or mark is None:
        return False
    if minimum_overlap_px is not None:
        return cv2.intersectConvexConvex(crop, mark)[0] > minimum_overlap_px
    return _material_print_intersection(crop, mark)


def _material_print_intersection(crop: np.ndarray, mark: np.ndarray) -> bool:
    overlap, _ = cv2.intersectConvexConvex(crop, mark)
    return overlap >= max(6.0, 0.002 * cv2.contourArea(crop))


def _points_from_mask(
    mask: dict | None, width: int, height: int, *, preserve_width_edge: bool = False,
) -> np.ndarray | None:
    if not mask:
        return None
    explicit = _explicit_quad(mask, "warp_quad", width, height, preserve_width_edge=preserve_width_edge)
    if explicit is not None:
        return explicit
    polygon = mask.get("polygon")
    if not polygon:
        return None
    points = normalized_vertices_to_pixels(
        [[float(point["x"]), float(point["y"])] for point in polygon],
        width,
        height,
    )
    if len(points) < 4:
        return None
    if len(points) == 4:
        return points
    points, _ = quad_from_polygon(points)
    return points


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order a quadrilateral clockwise from its top-left corner.

    Sum/difference ordering can assign one point to two corners for rotated or
    symmetric quads.  A centroid-based cyclic order is unambiguous once the
    geometry has been validated.
    """
    try:
        return order_quad_cyclic(points)
    except ValueError as exc:
        reason = str(exc)
        if reason == "duplicate_quad_vertices":
            raise ValueError("duplicate_perspective_corners") from exc
        if reason == "non_convex_quad_vertices":
            raise ValueError("non_convex_perspective_quad") from exc
        raise ValueError("invalid_perspective_quad") from exc


def _validate_quad(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("invalid_perspective_quad")
    if len(np.unique(np.round(points, decimals=3), axis=0)) != 4:
        raise ValueError("duplicate_perspective_corners")
    if (
        (points[:, 0] < -1e-4).any()
        or (points[:, 0] > width - 1 + 1e-4).any()
        or (points[:, 1] < -1e-4).any()
        or (points[:, 1] > height - 1 + 1e-4).any()
    ):
        raise ValueError("perspective_corners_outside_page")
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        raise ValueError("non_convex_perspective_quad")
    area = abs(float(cv2.contourArea(contour)))
    min_area = max(16.0, width * height * 0.0001)
    if area < min_area:
        raise ValueError("degenerate_perspective_quad")
    side_lengths = [float(np.linalg.norm(points[(index + 1) % 4] - points[index])) for index in range(4)]
    if min(side_lengths) < 2.0:
        raise ValueError("perspective_quad_side_too_short")
    angles: list[float] = []
    for index in range(4):
        previous = points[(index - 1) % 4] - points[index]
        following = points[(index + 1) % 4] - points[index]
        cosine = float(np.dot(previous, following) / max(np.linalg.norm(previous) * np.linalg.norm(following), 1e-9))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    if min(angles) < 8.0 or max(angles) > 172.0:
        raise ValueError("perspective_quad_angle_unstable")
    return {"area": area, "min_side": min(side_lengths), "max_side": max(side_lengths), "min_angle": min(angles)}


def _projective_jacobian(matrix: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    x, y = float(point[0]), float(point[1])
    numerator_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    numerator_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if abs(float(denominator)) < 1e-9:
        return None
    denominator_sq = denominator * denominator
    return np.array(
        [
            [
                (matrix[0, 0] * denominator - numerator_x * matrix[2, 0]) / denominator_sq,
                (matrix[0, 1] * denominator - numerator_x * matrix[2, 1]) / denominator_sq,
            ],
            [
                (matrix[1, 0] * denominator - numerator_y * matrix[2, 0]) / denominator_sq,
                (matrix[1, 1] * denominator - numerator_y * matrix[2, 1]) / denominator_sq,
            ],
        ],
        dtype=np.float64,
    )


def _assess_homography(
    source: np.ndarray,
    destination: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, float]:
    """Measure local stretch and corner-perturbation sensitivity."""
    samples: list[np.ndarray] = []
    for vertical in (0.15, 0.5, 0.85):
        left = source[0] * (1.0 - vertical) + source[3] * vertical
        right = source[1] * (1.0 - vertical) + source[2] * vertical
        for horizontal in (0.15, 0.5, 0.85):
            samples.append(left * (1.0 - horizontal) + right * horizontal)

    minimum_scales: list[float] = []
    maximum_scales: list[float] = []
    anisotropies: list[float] = []
    for point in samples:
        jacobian = _projective_jacobian(matrix, point)
        if jacobian is None or not np.isfinite(jacobian).all():
            raise ValueError("unstable_perspective_jacobian")
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        minimum = float(np.min(singular_values))
        maximum = float(np.max(singular_values))
        if minimum <= 1e-6:
            raise ValueError("unstable_perspective_jacobian")
        minimum_scales.append(minimum)
        maximum_scales.append(maximum)
        anisotropies.append(maximum / minimum)

    source_side_lengths = [float(np.linalg.norm(source[(index + 1) % 4] - source[index])) for index in range(4)]
    perturbation = max(0.25, min(2.0, min(source_side_lengths) * 0.002))
    center = source.mean(axis=0).reshape(1, 1, 2).astype(np.float32)
    baseline = cv2.perspectiveTransform(center, matrix).reshape(2)
    output_diagonal = max(1.0, float(np.linalg.norm(destination[2] - destination[0])))
    perturbation_drifts: list[float] = []
    for corner_index in range(4):
        for axis in range(2):
            for direction in (-1.0, 1.0):
                perturbed = source.copy()
                perturbed[corner_index, axis] += direction * perturbation
                try:
                    candidate_matrix = cv2.getPerspectiveTransform(perturbed.astype(np.float32), destination)
                    projected = cv2.perspectiveTransform(center, candidate_matrix).reshape(2)
                except cv2.error as exc:
                    raise ValueError("unstable_perspective_perturbation") from exc
                perturbation_drifts.append(float(np.linalg.norm(projected - baseline)) / output_diagonal)

    max_anisotropy = max(anisotropies)
    scale_variation = max(maximum_scales) / max(min(minimum_scales), 1e-9)
    max_perturbation_drift = max(perturbation_drifts, default=0.0)
    if max_anisotropy > 12.0 or scale_variation > 25.0:
        raise ValueError("excessive_perspective_stretch")
    if max_perturbation_drift > 0.025:
        raise ValueError("unstable_perspective_perturbation")
    return {
        "minimum_local_scale": min(minimum_scales),
        "maximum_local_scale": max(maximum_scales),
        "maximum_local_anisotropy": max_anisotropy,
        "local_scale_variation": scale_variation,
        "corner_perturbation_pixels": perturbation,
        "maximum_perturbation_drift_ratio": max_perturbation_drift,
    }


def _input_evidence(mask: dict | None, quad_metrics: dict[str, float]) -> tuple[dict, dict]:
    """Read the segmentation evidence contract, with a legacy-mask fallback."""
    source_mask: dict = mask or {}
    fusion_value = source_mask.get("fusion")
    fusion: dict = fusion_value if isinstance(fusion_value, dict) else {}
    supplied_boundary = source_mask.get("boundary_evidence")
    if not isinstance(supplied_boundary, dict):
        supplied_boundary = fusion.get("boundary_evidence")
    boundary = (
        dict(supplied_boundary)
        if isinstance(supplied_boundary, dict)
        else {
            "score": None,
            "available": False,
        }
    )

    supplied_geometry = source_mask.get("geometry_evidence")
    if isinstance(supplied_geometry, dict):
        geometry = dict(supplied_geometry)
    else:
        # Persisted masks created before the evidence contract can still expose
        # deterministic correction-time validation without inventing boundary
        # evidence that was never measured.
        minimum_side = float(quad_metrics["min_side"])
        maximum_side = max(float(quad_metrics["max_side"]), 1e-9)
        minimum_angle = float(quad_metrics["min_angle"])
        side_balance = float(np.clip(minimum_side / maximum_side, 0.0, 1.0))
        angle_stability = float(np.clip(minimum_angle / 90.0, 0.0, 1.0))
        score = 0.55 + 0.25 * side_balance + 0.20 * angle_stability
        geometry = {
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "valid": True,
            "source": "correction_quad_validation",
            "side_balance": round(side_balance, 4),
            "angle_stability": round(angle_stability, 4),
        }
    return boundary, geometry


def _perspective_confidence(assessment: dict[str, float], normalized_condition_number: float) -> float:
    """Convert independent stability measurements to a bounded risk score."""
    anisotropy = float(assessment["maximum_local_anisotropy"])
    scale_variation = float(assessment["local_scale_variation"])
    perturbation = float(assessment["maximum_perturbation_drift_ratio"])

    def bounded(value: float) -> float:
        return max(0.0, min(1.0, value))

    components = (
        bounded(1.0 - max(0.0, anisotropy - 1.0) / 11.0),
        bounded(1.0 - max(0.0, scale_variation - 1.0) / 24.0),
        bounded(1.0 - perturbation / 0.025),
        bounded(1.0 - float(np.log10(max(normalized_condition_number, 1.0))) / 4.0),
    )
    # A weak stability component must remain visible rather than being hidden
    # by otherwise easy geometry.
    return round(min(sum(components) / len(components), min(components) + 0.14), 4)


def _rectified_protected_regions(
    regions: list[dict],
    *,
    page_width: int,
    page_height: int,
    matrix: np.ndarray,
    target_width: int,
    target_height: int,
) -> list[dict]:
    """Map known page-space regions into the rectified crop coordinate space."""
    transformed: list[dict] = []
    for region in regions:
        box = region.get("box") or region.get("bounding_box") or {}
        try:
            x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1, x2 = x1 * page_width, x2 * page_width
            y1, y2 = y1 * page_height, y2 * page_height
        if x2 <= x1 or y2 <= y1:
            continue
        corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        px1, py1 = np.min(projected, axis=0)
        px2, py2 = np.max(projected, axis=0)
        if px2 <= 0 or py2 <= 0 or px1 >= target_width or py1 >= target_height:
            continue
        mapped = dict(region)
        mapped["box"] = {
            "x1": float(np.clip(px1 / max(target_width, 1), 0.0, 1.0)),
            "y1": float(np.clip(py1 / max(target_height, 1), 0.0, 1.0)),
            "x2": float(np.clip(px2 / max(target_width, 1), 0.0, 1.0)),
            "y2": float(np.clip(py2 / max(target_height, 1), 0.0, 1.0)),
        }
        mapped["coordinate_space"] = "rectified_crop_normalized"
        transformed.append(mapped)
    return transformed


def _analysis_padding(
    destination: np.ndarray,
    rectified_to_page: np.ndarray,
    page_width: int,
    page_height: int,
) -> tuple[int, int, int, int]:
    """Extend the detection on each side only as far as real page pixels permit."""
    window = destination.copy()
    longest_search = min(64, max(3, round(0.035 * float(np.min(destination[2])))))
    steps = np.arange(1, longest_search + 1, dtype=np.float32)[:, None]
    page_max = np.array([page_width - 1, page_height - 1]) + 1e-4
    padding = []
    for axis, side, sign in ((0, [0, 3], -1), (1, [0, 1], -1), (0, [1, 2], 1), (1, [2, 3], 1)):
        pairs = np.repeat(window[side][None, :, :], longest_search, axis=0)
        pairs[:, :, axis] += sign * steps
        projected = cv2.perspectiveTransform(pairs.reshape(-1, 1, 2), rectified_to_page).reshape(-1, 2, 2)
        inside = np.all((projected >= -1e-4) & (projected <= page_max), axis=(1, 2))
        distance = int(np.logical_and.accumulate(inside).sum())
        window[side, axis] += distance * sign
        padding.append(distance)
    return padding[0], padding[1], padding[2], padding[3]


_UNIT_QUAD = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
def _advisory_face_overlap(protection: dict, allow_page_face_overlap: bool) -> bool:
    return allow_page_face_overlap and all(
        violation["source"] in PAGE_FACE_SUGGESTIONS for violation in protection["violations"]
    )


def _orthogonal_aspect(homography: np.ndarray, camera: np.ndarray) -> float | None:
    axes = np.linalg.solve(camera, homography[:, :2])
    norms = np.linalg.norm(axes, axis=0)
    if min(norms) <= 1e-8 or abs(float(np.dot(axes[:, 0], axes[:, 1]) / np.prod(norms))) > 0.06:
        return None
    ratio = float(norms[0] / norms[1])
    return ratio if 0.1 <= ratio <= 10 else None


def _estimated_aspect(points: np.ndarray, center: tuple[float, float], side: int) -> float | None:
    homography = cv2.getPerspectiveTransform(_UNIT_QUAD, points)
    x, y = center
    projective = homography[2, :2]
    if np.min(np.abs(projective)) < 0.018:
        return None
    directions = homography[:2, :2] - np.array([[x], [y]]) * projective
    focal_sq = -float(np.dot(directions[:, 0], directions[:, 1]) / np.prod(projective))
    if not (0.35 * side) ** 2 <= focal_sq <= (4 * side) ** 2:
        return None
    focal = np.sqrt(focal_sq)
    camera = np.array([[focal, 0, x], [0, focal, y], [0, 0, 1]], dtype=float)
    return _orthogonal_aspect(homography, camera)


def page_plane_from_prints(
    image: np.ndarray, camera: np.ndarray | None, masks: list[dict], *,
    measured_references: list[MeasuredPrintReference] | None = None,
) -> PrintPagePlane | None:
    if len(masks) < 6:
        return None
    height, width = image.shape[:2]
    quads = []
    for mask in masks:
        if mask.get("album_window_occluded") or mask.get("source") == ALBUM_WINDOW_SOURCE:
            continue
        try:
            seed = _points_from_mask(mask, width, height)
            if seed is None:
                continue
            seed = _order_points(seed)
            if _validate_quad(seed, width, height)["min_side"] < 60:
                continue
            quad, evidence, reasons = refine_print_boundary(image, seed)
        except (KeyError, TypeError, ValueError, cv2.error):
            continue
        if (quad is None or reasons or np.any((quad < .5) | (quad > np.array([width, height]) - 1.5))
                or any(side["support"] < .80 or (side["corner_support"] or 0) < .65 for side in evidence.values())):
            continue
        if any(cv2.intersectConvexConvex(quad, peer)[0] > 1 for peer in quads):
            continue
        quads.append(quad)
    if camera is not None:
        return fit_calibrated_page_plane(quads, camera, image.shape)
    plane = fit_uncalibrated_page_plane(quads, image.shape)
    return plane if plane is not None else fit_measured_page_plane(image, quads, measured_references or [])


def _crop_dimensions(
    points: np.ndarray, width: int, height: int, *, metric: bool,
    camera_matrix: np.ndarray | None, print_aspect_ratio: float | None,
    page_plane: PrintPagePlane | None = None,
) -> tuple[int, int, dict, list[str]]:
    top, right, bottom, left = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    target_width = round(float(max(top, bottom))) + 1
    target_height = round(float(max(left, right))) + 1
    aspect = None
    method = "projected_side_lengths"
    plane_evidence = None
    reasons = []
    if print_aspect_ratio is not None:
        aspect = float(print_aspect_ratio)
        if not 0.1 <= aspect <= 10:
            raise ValueError("invalid_print_aspect_ratio")
        method = "known_print_aspect"
    elif camera_matrix is not None:
        camera = np.asarray(camera_matrix, dtype=float)
        if camera.shape != (3, 3) or not np.isfinite(camera).all() or abs(np.linalg.det(camera)) < 1e-7:
            raise ValueError("invalid_camera_matrix")
        estimate = page_plane.estimate_aspect(points) if page_plane is not None else None
        if estimate is not None:
            aspect, plane_evidence = estimate
        else:
            homography = cv2.getPerspectiveTransform(_UNIT_QUAD, points)
            aspect = _orthogonal_aspect(homography, camera)
        if aspect is None:
            reasons.append("PRINT_ASPECT_UNCERTAIN")
        else:
            method = "calibrated_camera"
    elif metric:
        directions = np.roll(points, -1, axis=0) - points
        unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        right_angle_error = float(np.max(np.abs(np.sum(unit * np.roll(unit, -1, axis=0), axis=1))))
        opposite_difference = max(abs(float(np.log(top / bottom))), abs(float(np.log(left / right))))
        # Even a barely visible trapezoid can shift the aspect by several percent
        # when a single tilted axis gives no usable focal-length estimate.
        if max(right_angle_error, opposite_difference) > 0.012:
            center = ((width - 1) / 2, (height - 1) / 2)
            side = max(width, height)
            estimates = [_estimated_aspect(points, center, side)]
            if estimates[0] is not None:
                # A focal length reconstructed from two vanishing directions can be ill-conditioned.
                for axis, resolution in enumerate((width, height)):
                    for sign in (-1, 1):
                        shifted = list(center)
                        shifted[axis] += sign * 0.025 * resolution
                        perturbed = points.copy()
                        perturbed[0, axis] += sign * 0.5
                        estimates.append(_estimated_aspect(perturbed, tuple(shifted), side))
            if len(estimates) > 1 and all(item is not None for item in estimates) and max(estimates) / min(estimates) <= 1.025:
                aspect = estimates[0]
                method = "estimated_pinhole_camera"
            else:
                reasons.append("PRINT_ASPECT_UNCERTAIN")
    if aspect is not None:
        # Preserve the largest observed sampling dimension while fixing the shape of the print.
        # Measured geometry fixes the span between the rectified corner pixel centers.
        offset = 1 if method in {"known_print_aspect", "calibrated_camera"} else 0
        scale = max((target_width - offset) / aspect, target_height - offset)
        target_width, target_height = round(scale * aspect) + offset, round(scale) + offset
    details = {"method": method, "estimated_ratio": aspect}
    if plane_evidence is not None:
        details["page_plane"] = plane_evidence
    return target_width, target_height, details, reasons


def _geometry_for_mode(
    points: np.ndarray, matrix: np.ndarray, image: np.ndarray, page_width: int, page_height: int,
    rotation: int, reasons: list[str],
) -> dict:
    height, width = image.shape[:2]
    if rotation == 90:
        turn = np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]])
    elif rotation == 180:
        turn = np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]])
    elif rotation == 270:
        turn = np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]])
    else:
        turn = np.eye(3)
    return {
        "status": "review_needed" if reasons else "ready",
        "reason_codes": list(dict.fromkeys(reasons)),
        "warp_quad": _vertices_on_page(points, page_width, page_height),
        "page_to_output": (turn @ matrix).tolist(),
        "shape": {"height": width, "width": height} if rotation in (90, 270) else {"height": height, "width": width},
    }


def _vertices_on_page(points: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
    return [
        {"x": round(float(x), 8), "y": round(float(y), 8)}
        for x, y in pixel_vertices_to_normalized(points, width, height)
    ]


def _rim_like_sides(image: np.ndarray) -> set[str]:
    """A long, uniform outer strip could be a border even without a visible inner seam."""
    sides = {"top": image, "right": image[:, ::-1].transpose(1, 0, 2),
             "bottom": image[::-1], "left": image.transpose(1, 0, 2)}
    uniform = {}
    for name, side in sides.items():
        span = side.shape[1]
        cols = np.linspace(0.08 * (span - 1), 0.92 * (span - 1), min(96, span)).astype(int)
        # Ignore the first row, which may contain a mixed sample along the print's outline.
        stripe = side[1: min(5, len(side)), cols].astype(np.float32)
        median = np.median(stripe, axis=(0, 1))
        if np.mean(np.max(np.abs(stripe - median), axis=2) <= 12) >= 0.78:
            uniform[name] = median
    # Neutral and repeated strips can be paper. A single saturated strip
    # extending into the scene is ordinary photographed content (e.g. sky).
    return {
        name for name, color in uniform.items()
        if np.ptp(color) <= 25
        or any(name != other and np.max(np.abs(color - candidate)) <= 15 for other, candidate in uniform.items())
    }


def _proposal_polygon(mask: dict, width: int, height: int) -> np.ndarray | None:
    # A repeat run and its neighbors must use the original proposal, regardless
    # of which instances already have a measured outline in the database.
    if mask.get("visible_polygon_source") in _REFINED_VISIBILITY_SOURCES | {GRID_OUTLINE_REVIEW_SOURCE}:
        polygon = mask.get("proposal_visible_polygon") or mask.get("polygon")
    else:
        polygon = mask.get("visible_polygon") or mask.get("polygon")
    if not polygon or len(polygon) < 3:
        return None
    try:
        return normalized_vertices_to_pixels([(point["x"], point["y"]) for point in polygon], width, height)
    except (KeyError, TypeError, ValueError):
        return None


def apply_refined_visibility(mask: dict | None, correction: dict) -> dict | None:
    outline = correction.get("refined_visible_polygon")
    if mask is None or mask.get("visible_polygon_source") == GRID_OUTLINE_REVIEW_SOURCE:
        return mask
    was_refined = mask.get("visible_polygon_source") in _REFINED_VISIBILITY_SOURCES
    if not outline and not was_refined:
        return mask
    prior = mask.get("proposal_visible_polygon") if was_refined else mask.get("visible_polygon")
    updated = dict(mask)
    updated.pop("proposal_visible_polygon", None)
    updated.pop("visible_polygon_source", None)
    updated.pop("visible_polygon", None)
    if outline:
        updated["visible_polygon"] = outline
        updated["visible_polygon_source"] = correction.get("refined_visibility_source", _PRINT_BOUNDARY_SOURCE)
        if prior:
            updated["proposal_visible_polygon"] = prior
    elif prior:
        updated["visible_polygon"] = prior
    return updated


def _adjacent_stock_neighbor(
    seed: np.ndarray, masks: list[dict], width: int, height: int,
) -> np.ndarray | None:
    shortest = np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min()
    if shortest < 100:
        return None
    low, high = seed.min(axis=0), seed.max(axis=0)
    nearby = []
    for mask in masks:
        visible = _proposal_polygon(mask, width, height)
        if visible is None:
            continue
        separation = np.maximum(0, np.maximum(visible.min(axis=0) - high, low - visible.max(axis=0)))
        if np.linalg.norm(separation) <= .08 * shortest:
            nearby.append(mask)
    if len(nearby) != 1:
        return None
    neighbor, = nearby
    if neighbor.get("warp_quad") or any(neighbor.get(name) for name in _STOCK_REVIEW_CONFLICTS):
        return None
    try:
        points = _points_from_mask(neighbor, width, height)
        if points is None:
            return None
        points = _order_points(points)
        _validate_quad(points, width, height)
        return points
    except (KeyError, TypeError, ValueError, cv2.error):
        return None


def _clipped_visible_corner(quad: np.ndarray, visible: np.ndarray | None) -> bool:
    if visible is None or len(visible) <= 4:
        return False
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    depth = float(np.clip(0.052 * lengths.min(), 4, 34))
    directions = sides / lengths[:, None]
    for index, corner in enumerate(quad):
        first = depth * directions[index]
        second = -depth * directions[index - 1]
        inside_corner = cv2.pointPolygonTest(visible, tuple(corner + first + second), True)
        edges_present = all(
            cv2.pointPolygonTest(visible, tuple(point), True) > 1.5
            for point in (corner + 3.2 * first + second, corner + first + 3.2 * second)
        )
        if inside_corner < -2.0 and edges_present:
            return True
    return False


def _occluding_neighbor(
    page_image: np.ndarray, src_points: np.ndarray, own: np.ndarray | None, neighbors: list[np.ndarray],
    *, inferred_sides: tuple[int, ...] = (), observed_sides: dict[int, tuple[np.ndarray, float]] | None = None,
) -> np.ndarray | None:
    if not neighbors:
        return None
    low = src_points.min(axis=0)
    high = src_points.max(axis=0)
    ratio = min(1.0, 380.0 / float(np.max(high - low)))
    shape = tuple((np.ceil((high - low) * ratio).astype(int) + 1)[::-1])

    def raster(points: np.ndarray) -> np.ndarray:
        return cv2.fillPoly(np.zeros(shape, np.uint8), [np.rint((points - low) * ratio).astype(np.int32)], 1)

    footprint = cv2.erode(raster(src_points), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    exposed = footprint if own is None else footprint & (1 - raster(own))
    footprint_area = int(footprint.sum())
    minimum_overlap = max(6, round(footprint_area * 0.002))
    sides = np.roll(src_points, -1, axis=0) - src_points
    reach = float(np.clip(0.028 * np.min(np.linalg.norm(sides, axis=1)), 4, 24))
    inferred_edge_samples = [
        src_points[index] + np.linspace(0.15, 0.85, 16)[:, None] * sides[index] for index in inferred_sides
    ]
    for polygon in neighbors:
        lower, upper = polygon.min(axis=0), polygon.max(axis=0)
        if np.all(upper > low) and np.all(lower < high):
            overlap = raster(polygon) & exposed
            if np.count_nonzero(overlap) >= minimum_overlap:
                if own is None or not observed_sides:
                    return polygon
                rows, columns = np.nonzero(overlap)
                claimed = np.column_stack((columns, rows)) / ratio + low
                confirmed_stock = observed_rim_points(page_image, src_points, claimed, observed_sides)
                if np.count_nonzero(~confirmed_stock) >= minimum_overlap:
                    return polygon
            # A single exterior contour can cover an interior occluder hole.
            # A distinct, substantially smaller enclosed proposal is review evidence.
            if np.all(lower > low) and np.all(upper < high):
                area = cv2.contourArea(polygon) * ratio ** 2
                if minimum_overlap <= area <= 0.80 * footprint_area and all(
                    cv2.pointPolygonTest(src_points, tuple(point), True) >= reach for point in polygon
                ):
                    return polygon
        # Paper color can continue across two touching prints and make the other
        # print's ink edge appear to be this print's physical boundary.
        if any(
            sum(cv2.pointPolygonTest(polygon, tuple(point), True) >= -reach for point in samples) >= 8
            for samples in inferred_edge_samples
        ):
            return polygon
    return None


def _lost_visible_paper(
    page: np.ndarray, current: np.ndarray, reference: dict, neighbors: list[np.ndarray],
) -> tuple[list[list[dict]], int]:
    height, width = page.shape[:2]
    previous = _explicit_quad({"warp_quad": reference["print_warp_quad"]}, "warp_quad", width, height)
    visible = _proposal_polygon({"polygon": (reference.get("segmentation_mask") or {}).get("visible_polygon")},
                                width, height)
    if previous is None or visible is None:
        return [], 0
    if not reference.get("ready_before_edit") and _occluding_neighbor(page, previous, None, neighbors) is None:
        return [], 0
    low = np.maximum(0, np.floor(previous.min(axis=0)).astype(int) - 1)
    high = np.minimum([width, height], np.ceil(previous.max(axis=0)).astype(int) + 2)
    shape = tuple((high - low)[::-1])
    owned = np.zeros(shape, np.uint8)
    excluded = np.zeros(shape, np.uint8)

    def draw(bitmap, polygon, color):
        points = np.rint((polygon - low) * 256).astype(np.int32)
        cv2.fillPoly(bitmap, [points], color, shift=8)

    draw(owned, previous, 1)
    draw(excluded, visible, 1)
    owned &= excluded
    excluded.fill(0)
    draw(excluded, current, 1)
    if not reference.get("ready_before_edit"):
        for neighbor in neighbors:
            draw(excluded, neighbor, 1)
    # A one-pixel band absorbs sampling noise at the competing outlines.
    owned[cv2.dilate(excluded, np.ones((3, 3), np.uint8)) != 0] = 0
    minimum = max(6, round(0.002 * cv2.contourArea(previous)))
    disputed = cv2.countNonZero(owned)
    if disputed < minimum:
        return [], 0
    contours, _ = cv2.findContours(owned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(contour) >= minimum:
            points = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2) + low
            outlines.append(_vertices_on_page(np.vstack((points, points[0])), width, height))
    backing_match = bool(outlines and not reference.get("ready_before_edit")
                         and strip_resembles_album_backing(page, current, owned, low))
    return outlines, disputed if backing_match else 0


def _orient_crop(
    image: np.ndarray, enabled: bool, model_path: Path | None, *, manual_rotation: int | None,
) -> tuple[np.ndarray, dict]:
    if manual_rotation is not None:
        if type(manual_rotation) is not int or manual_rotation not in (0, 90, 180, 270):
            raise ValueError("invalid_orientation_degrees")
        return rotate_image(image, manual_rotation), {
            "rotation_degrees": manual_rotation, "method": "manual", "decision": "reviewer_selected",
        }
    if enabled:
        return correct_photo_orientation(image, model_path=model_path)
    return image, {"rotation_degrees": 0, "method": "disabled", "decision": "preserve"}


def _inner_crop_plan(
    corrected: np.ndarray, matrix: np.ndarray, inner: np.ndarray | None, *, trim_border: bool,
    page_width: int, page_height: int, page_regions: list[dict], protect_regions: bool, allow_wide_search: bool,
) -> tuple[np.ndarray | None, dict, list[str]]:
    height, width = corrected.shape[:2]
    if inner is not None:
        rectified = _order_points(cv2.perspectiveTransform(inner.reshape(1, 4, 2), matrix)[0])
        if np.allclose(rectified, _UNIT_QUAD * [width - 1, height - 1], rtol=0, atol=1e-4):
            return None, {"method": "manual_image_quad", "applied": False}, []
        return rectified, {"method": "manual_image_quad", "applied": True, "confidence": 1.0}, []
    if not trim_border:
        return None, {"method": "disabled", "applied": False}, ["IMAGE_BORDER_CLASSIFICATION_DISABLED"]
    regions = _rectified_protected_regions(
        page_regions, page_width=page_width, page_height=page_height,
        matrix=matrix, target_width=width, target_height=height,
    )
    if protect_regions:
        regions = merge_protected_regions(regions, detect_protected_regions(corrected))
    plan = find_physical_border(corrected, protected_regions=regions)
    if CURVED_IMAGE_BOUNDARY in plan.reason_codes:
        diagnostics = plan.metadata()
        if all(edge.is_physical_border for edge in plan.edges.values()):
            diagnostics["curve_seed_quad"] = border_chord_quad(plan.edges, width, height)
        return None, diagnostics, [CURVED_IMAGE_BOUNDARY]
    near_outline = {
        name for name, edge in plan.edges.items() if edge.is_physical_border and max(edge.insets) <= 2.2
    }
    substantial = sum(edge.is_physical_border and min(edge.insets) >= 4 for edge in plan.edges.values())
    shift = np.zeros(2, dtype=np.float32)
    if near_outline and substantial >= 2 and min(width, height) >= 18:
        # A mixed sample of album paper can look like a one-row rim after warping.
        # Exclude that row during measurement while retaining the original source pixels.
        left, top, right, bottom = (3 if name in near_outline else 0 for name in ("left", "top", "right", "bottom"))
        retry = find_physical_border(corrected[top : height - bottom, left : width - right])
        if CURVED_IMAGE_BOUNDARY in retry.reason_codes:
            return None, retry.metadata(), [CURVED_IMAGE_BOUNDARY]
        if retry.crop_quad is None or any(not retry.edges[name].is_physical_border for name in near_outline):
            return None, {**plan.metadata(), "applied": False}, ["IMAGE_BOUNDARY_UNCERTAIN"]
        if any(reason.endswith("_VETO") for reason in plan.reason_codes):
            return None, {**plan.metadata(), "applied": False}, ["BORDER_PROTECTED_CONTENT_VETO"]
        shift = np.array([left, top], dtype=np.float32)
        candidate = retry.crop_quad + shift
        if regions and not validate_protected_regions(candidate, regions, width=width, height=height)["passed"]:
            return None, {**plan.metadata(), "applied": False}, ["BORDER_PROTECTED_CONTENT_VETO"]
        plan = retry
    diagnostics = plan.metadata()
    border_candidates = {name: edge for name, edge in plan.edges.items() if edge.is_physical_border}
    ambiguous = {reason for reason in plan.reason_codes if reason.endswith("_VETO") or "UNCERTAIN" in reason}
    uniform = _rim_like_sides(corrected)
    if plan.crop_quad is None and not uniform and not ambiguous:
        diagnostics["method"] = "confirmed_borderless"
        return None, diagnostics, []
    # A uniform patch of photographed sky or a wall can create one straight seam.
    rim_colors = [np.array(edge.rim_bgr) for edge in border_candidates.values() if edge.rim_bgr is not None]
    corroborated = any(
        np.max(np.abs(first - second)) <= 25
        for index, first in enumerate(rim_colors) for second in rim_colors[index + 1:]
    )
    if (plan.crop_quad is None or not corroborated or uniform - border_candidates.keys() or ambiguous
            or plan.confidence < 0.68):
        if allow_wide_search and len(uniform) == 4 and "BORDER_PROTECTED_CONTENT_VETO" not in ambiguous:
            wide = wide_paper_motif_proposal(corrected)
            if wide is not None:
                if regions and not validate_protected_regions(
                    wide, regions, width=width, height=height, require_containment=True,
                )["passed"]:
                    return None, diagnostics, ["BORDER_PROTECTED_CONTENT_VETO"]
                return wide, {"method": "wide_paper_rim", "applied": True}, []
        diagnostics["applied"] = False
        return None, diagnostics, ["IMAGE_BOUNDARY_UNCERTAIN", *sorted(ambiguous)]
    quad = plan.crop_quad + shift
    diagnostics["crop_quad"] = quad.tolist()
    return quad, diagnostics, []


def _is_rectangular_in_measured_plane(quad: np.ndarray) -> bool:
    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    opposite_error = np.abs(lengths[:2] - lengths[2:])
    perpendicular_error = np.abs(np.sum(edges * np.roll(edges, -1, axis=0), axis=1)) / lengths
    return max(float(opposite_error.max()), float(perpendicular_error.max())) <= 2.0


def _independent_motif_visible(page: np.ndarray, source: np.ndarray, neighbors: list[np.ndarray]) -> bool:
    if any(cv2.intersectConvexConvex(source, cv2.convexHull(polygon))[0] > 1.0 for polygon in neighbors):
        return False
    return continuous_motif_rim(page, source)


def _motif_inside_scalloped_stock(inner: np.ndarray, outline: np.ndarray, neighbors: list[np.ndarray]) -> bool:
    hulls = [cv2.convexHull(neighbor) for neighbor in neighbors]
    if any(cv2.intersectConvexConvex(inner, hull)[0] > 0 for hull in hulls):
        return False
    for start, end in zip(inner, np.roll(inner, -1, axis=0), strict=True):
        # The contour can be concave between corners. Check the entire side
        # at spacing of at most two native pixels, with a margin beyond that.
        stations = np.linspace(start, end, 1 + int(np.ceil(np.linalg.norm(end - start) / 2)))
        if any(cv2.pointPolygonTest(outline, tuple(point), True) < 2 for point in stations):
            return False
        if any(cv2.pointPolygonTest(hull, tuple(point), True) > -2 for hull in hulls for point in stations):
            return False
    return True


def _render_outline_preview(page: np.ndarray, outline: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    preview, preview_matrix = warp_owned_outline(page, outline, matrix)
    return preview, {
        "source_outline": _vertices_on_page(outline, *page.shape[1::-1]),
        "page_to_output": preview_matrix.tolist(),
        "shape": {"height": preview.shape[0], "width": preview.shape[1]},
    }


def _render_review_quad_preview(
    page: np.ndarray, quad: np.ndarray, rotation: int, neighbors: list[np.ndarray], regions: list[dict],
    camera_matrix: np.ndarray | None, page_plane: PrintPagePlane | None, *, source: str,
    prior_protection: dict | None = None,
) -> tuple[np.ndarray, dict] | None:
    """Straighten the proposed print outline independently of the primary crop."""
    height, width = page.shape[:2]
    try:
        if any(_material_print_intersection(quad, cv2.convexHull(neighbor)) for neighbor in neighbors):
            return None
        protection = validate_protected_regions(quad, regions, width=width, height=height)
        if not protection["passed"]:
            previous = prior_protection or {}
            if (not _advisory_face_overlap(protection, previous.get("review_untrimmed") is True)
                    or any(violation not in (previous.get("violations") or []) for violation in protection["violations"])):
                return None
        out_width, out_height, aspect, reasons = _crop_dimensions(
            quad, width, height, metric=True, camera_matrix=camera_matrix, print_aspect_ratio=None, page_plane=page_plane,
        )
        if min(out_width, out_height) < 2 or out_width * out_height > 2 * width * height:
            return None
        destination = _UNIT_QUAD * np.float32([out_width - 1, out_height - 1])
        matrix = cv2.getPerspectiveTransform(quad, destination)
        _assess_homography(quad, destination, matrix)
        pixels, supported = warp_owned_crop(page, matrix, (out_width, out_height))
        if not supported:
            return None
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return None

    geometry = _geometry_for_mode(quad, matrix, pixels, width, height, rotation, reasons)
    return rotate_image(pixels, rotation), {
        "source": source,
        "source_outline": geometry["warp_quad"],
        "page_to_output": geometry["page_to_output"],
        "shape": geometry["shape"],
        "sampling": OWNED_BILINEAR,
        "aspect_ratio_estimation": aspect,
        "reason_codes": reasons,
        **({"protected_content_review": True} if (prior_protection or {}).get("review_untrimmed") else {}),
    }


def _neighbors_for_edge_comparison(
    page: np.ndarray, previous: np.ndarray, candidate: np.ndarray, neighbor_masks: list[dict],
) -> list[np.ndarray]:
    peers = [(mask, outline) for mask in neighbor_masks
             if (outline := _proposal_polygon(mask, *page.shape[1::-1])) is not None]
    outlines = [outline for _, outline in peers]
    overlapping = [index for index, outline in enumerate(outlines)
                   if _material_print_intersection(candidate, cv2.convexHull(outline))]
    if len(overlapping) != 1:
        return outlines
    index, = overlapping
    mask, outline = peers[index]
    if (mask.get("source") != "yolo_seg_mask" or mask.get("detector_source") != "yolo_onnx"
            or mask.get("warp_quad") or any(mask.get(key) for key in _STOCK_REVIEW_CONFLICTS)):
        return outlines
    area, intersection = cv2.intersectConvexConvex(candidate, cv2.convexHull(outline))
    if area <= .012 * cv2.contourArea(candidate) and visible_gap_at_unchanged_claim(page, previous, candidate, intersection):
        # The observed gap supports this comparison. Keep the peer claim
        # for the stored crop, review decisions and export gates.
        outlines.pop(index)
    return outlines


def _linked_grid_cut(
    neighbor: dict, grid: dict, outline: np.ndarray, points: np.ndarray, claim: np.ndarray, width: int, height: int,
) -> np.ndarray | None:
    link = neighbor.get("paper_grid_review") or {}
    if (neighbor.get("source") != "yolo_seg_mask" or neighbor.get("detector_source") != "yolo_onnx"
            or neighbor.get("warp_quad") is not None or neighbor.get("manual_added")
            or link.get("outline") != grid["outline"] or not link.get("cut")):
        return None
    try:
        vertices = np.float32([(point["x"], point["y"]) for point in link["cut"]])
    except (KeyError, TypeError, ValueError):
        return None
    if vertices.shape != (2, 2) or not np.isfinite(vertices).all() or np.any((vertices < 0) | (vertices > 1)):
        return None
    cut = normalized_vertices_to_pixels(vertices, width, height)
    if (np.linalg.norm(cut[1] - cut[0]) <= 1
            or any(cv2.pointPolygonTest(outline, tuple(point), True) < -0.02 for point in cut)
            or cv2.intersectConvexConvex(outline, claim)[0] < .99 * cv2.contourArea(claim)):
        return None
    if any(abs(cv2.pointPolygonTest(points, tuple(point), True)) > 2 for point in np.linspace(*cut, num=5)):
        return None
    return cut


def review_outline_intersects_claim(
    source_outline: list[dict], claim: dict, width: int, height: int,
) -> bool:
    """Check a current reviewer claim against a stored source outline."""
    try:
        outline = _explicit_quad({"outline": source_outline}, "outline", width, height)
        quad = _explicit_quad(claim, "warp_quad", width, height)
        points = quad if quad is not None else _proposal_polygon(claim, width, height)
        return outline is not None and points is not None and cv2.intersectConvexConvex(outline, cv2.convexHull(points))[0] > 1
    except (ValueError, cv2.error):
        return False


def review_traces_intersect_claim(
    guides: list[list[dict]], claim: dict, width: int, height: int,
) -> bool:
    try:
        traces = [normalized_vertices_to_pixels([(p["x"], p["y"]) for p in guide], width, height) for guide in guides]
        visible = _proposal_polygon(claim, width, height)
        if visible is not None and motif_seams_intersect_claim(traces, visible):
            return True
        paper = _explicit_quad(claim, "warp_quad", width, height)
        return paper is not None and motif_seams_intersect_claim(traces, paper)
    except (KeyError, TypeError, ValueError, cv2.error):
        return False


def _render_disputed_grid_preview(
    page: np.ndarray, outline: np.ndarray, grid: dict, rotation: int, neighbor_masks: list[dict],
    regions: list[dict], camera_matrix: np.ndarray | None, page_plane: PrintPagePlane | None,
) -> tuple[np.ndarray, dict] | None:
    """Show the one-sheet hypothesis while keeping the active source claims intact."""
    height, width = page.shape[:2]
    shortest_side = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1).min()
    cuts = []
    try:
        for neighbor in neighbor_masks:
            explicit = _explicit_quad(neighbor, "warp_quad", width, height)
            points = explicit if explicit is not None else _proposal_polygon(neighbor, width, height)
            if points is None or not np.isfinite(points).all():
                return None
            claim = cv2.convexHull(points)
            if cv2.contourArea(claim) <= 0:
                return None
            cut = _linked_grid_cut(neighbor, grid, outline, points, claim, width, height)
            if cut is not None:
                if any(np.allclose(cut, previous, rtol=0, atol=1) or np.allclose(cut, previous[::-1], rtol=0, atol=1)
                       for previous in cuts):
                    return None
                cuts.append(cut)
                continue
            area, overlap = cv2.intersectConvexConvex(outline, claim)
            if area <= 1:
                continue
            # A coarse contrast contour can reach slightly across a corner of
            # its neighbor. Require the print edge to continue through the intersection.
            if (explicit is not None or neighbor.get("manual_added") or neighbor.get("paper_grid_review")
                    or neighbor.get("source") != "classical_border_quad" or neighbor.get("detector_source") != "contrast_region"
                    or area > max(6., .00035 * cv2.contourArea(outline)) or overlap is None
                    or np.linalg.norm(np.ptp(overlap[:, 0], axis=0)) > .10 * shortest_side
                    or not verify_print_edge_at_claim(page, outline, overlap[:, 0])):
                return None
    except (ValueError, cv2.error):
        return None
    if len(cuts) > 2:
        return None
    preview = _render_review_quad_preview(
        page, outline, rotation, [], regions, camera_matrix, page_plane, source=DISPUTED_GRID_PREVIEW_SOURCE,
    )
    if preview is not None and cuts:
        _, metadata = preview
        matrix = np.asarray(metadata["page_to_output"])
        shape = metadata["shape"]
        metadata["projected_grid_cuts"] = [
            _vertices_on_page(cv2.perspectiveTransform(cut[None], matrix)[0], shape["width"], shape["height"])
            for cut in cuts
        ]
    return preview


def _keeps_protected_preview_content(
    page: np.ndarray, original: np.ndarray, alternative: np.ndarray, regions: list[dict], protection: dict,
) -> bool:
    height, width = page.shape[:2]
    advisory = (protection.get("violations") or []) if protection.get("review_untrimmed") else []
    for region in regions:
        box = region_box_in_pixels(region, width, height)
        if box is None:
            continue
        key = protected_region_key(region)
        if key in advisory and key["source"] in PAGE_FACE_SUGGESTIONS:
            continue
        x1, y1, x2, y2 = box
        pad = float(region.get("margin_ratio", .08)) * max(x2 - x1, y2 - y1)
        padded = np.float32([[x1 - pad, y1 - pad], [x2 + pad, y1 - pad],
                             [x2 + pad, y2 + pad], [x1 - pad, y2 + pad]])
        before = cv2.intersectConvexConvex(original, padded)[0]
        after = cv2.intersectConvexConvex(alternative, padded)[0]
        if before - after > 1:
            return False
    return True


def _regions_after_album_mark_review(
    page: np.ndarray, quad: np.ndarray, regions: list[dict], review: dict, neighbors: list[np.ndarray],
) -> list[dict]:
    width, height = page.shape[1::-1]
    original = _explicit_quad({"warp_quad": review["previous_quad"]}, "warp_quad", width, height)
    if original is None:
        raise ValueError("album_page_mark_review_stale")
    if review.get("source") == OFF_PRINT_ADVISORY:
        corner_offsets = np.linalg.norm(original[:, None, :] - quad[None, :, :], axis=2)
        if corner_offsets.min(axis=1).max() > 1.:
            raise ValueError("album_page_mark_outline_changed")
        protection = validate_protected_regions(quad, regions, width=width, height=height)
        manual_margin = review.get("context") == MANUAL_MARGIN_CONTEXT
        current = (manual_margin_advisory(page, quad, regions, protection["violations"], neighbors)
                   if manual_margin
                   else off_print_advisory_regions(page, quad, regions, protection["violations"], neighbors))
        targets = [region["region"] for region in review.get("regions") or []]
        if (not current or [region["region"] for region in current] != targets
                or review.get("confirmed_indices") != list(range(len(targets)))):
            raise ValueError("album_page_mark_review_stale")
        if manual_margin and any(region["requires_edge_alignment"] for region in current):
            raise ValueError("outside_manual_print_edge_unresolved")
        remaining = [region for region in regions if protected_region_key(region) not in targets]
        if manual_margin:
            remaining.extend({**region, "margin_ratio": 0.} for region in regions if protected_region_key(region) in targets)
        if not _keeps_protected_preview_content(page, original, quad, remaining, {}):
            raise ValueError("protected_region_cut")
        return remaining

    target = review["region"]
    region_indices = [index for index, region in enumerate(regions) if protected_region_key(region) == target]
    if len(region_indices) != 1 or not is_advisory_page_face(regions[region_indices[0]]):
        raise ValueError("album_page_mark_review_stale")
    index, = region_indices
    box = region_box_in_pixels(regions[index], width, height)
    if box is None:
        raise ValueError("album_page_mark_review_stale")
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    if cv2.pointPolygonTest(original, center, True) < 2 or cv2.pointPolygonTest(quad, center, True) > -2:
        raise ValueError("album_page_mark_inside_crop")
    remaining = [region for other, region in enumerate(regions) if other != index]
    if not _keeps_protected_preview_content(page, original, quad, remaining, {}):
        raise ValueError("protected_region_cut")
    return remaining


def _dual_crop_result(
    page: np.ndarray, full_image: np.ndarray, source_quad: np.ndarray, matrix: np.ndarray, metadata: dict,
    inner: np.ndarray | None, full_reasons: list[str], *, trim_border: bool, auto_orient: bool,
    orientation_model_path: Path | None, manual_rotation: int | None, page_regions: list[dict],
    protect_regions: bool, owned_sampling: bool,
    neighbors: list[np.ndarray], corner_mount_detected: bool, print_outline: np.ndarray | None,
    motif_suggestion: np.ndarray | None, print_edge_traces: list,
    paper_stock_margin_regions: list | None = None,
    adjacent_stock_outline: np.ndarray | None = None,
    neighbor_masks: list[dict] | None = None,
) -> CropResult:
    page_height, page_width = page.shape[:2]
    height, width = full_image.shape[:2]
    image_only = None
    image_only_preview = None
    image_mode = {"status": "review_needed", "reason_codes": full_reasons.copy()}
    diagnostics = {"method": "deferred_until_print_boundary_is_verified", "applied": False}
    adjusted, orientation = _orient_crop(
        full_image, auto_orient, orientation_model_path, manual_rotation=manual_rotation,
    )
    rotation = int(orientation.get("rotation_degrees") or 0) % 360
    full_mode = _geometry_for_mode(source_quad, matrix, full_image, page_width, page_height, rotation, full_reasons)
    if print_edge_traces:
        full_mode["suggested_edge_traces"] = print_edge_traces
    if owned_sampling:
        full_mode["sampling"] = OWNED_BILINEAR
    full_print_preview = None
    preview_projection = np.asarray(full_mode["page_to_output"])
    if print_outline is not None:
        try:
            full_print_preview, full_mode["outline_preview"] = _render_outline_preview(page, print_outline, preview_projection)
        except (ValueError, cv2.error):
            pass  # The source page remains available for review.

    independent = corner_mount_detected and set(full_reasons) <= _LOCALIZED_OCCLUSION_REASONS
    scalloped_motif = (print_outline is not None and inner is not None and set(full_reasons) == {SCALLOPED_PRINT_REVIEW}
                       and _motif_inside_scalloped_stock(inner, print_outline, neighbors))
    inboard_review = adjacent_stock_outline is not None and trim_border and inner is None
    stock_motif_review = trim_border and inner is None and full_reasons == ["PRINT_BOUNDARY_AMBIGUOUS"] and (
        inboard_review or any(isinstance(side, dict) and side.get("review_stock_fit")
                              for side in metadata["boundary_refinement"].values())
    )
    needs_closed_rim = independent or stock_motif_review
    preview_only = bool(full_reasons) and set(full_reasons) <= _CROP_PREVIEW_REASONS and not scalloped_motif
    if trim_border and inner is None and paper_stock_margin_regions:
        diagnostics = {"method": "saved_margin_review", "applied": False}
        image_mode["reason_codes"].insert(0, MOTIF_MARGIN_MARKS_REVIEW)
        image_mode["margin_mark_source"] = PAPER_STOCK_MOTIF_SOURCE
        image_mode["margin_mark_regions"] = paper_stock_margin_regions
    elif motif_suggestion is not None:
        diagnostics = {"method": "native_scalloped_print_motif", "applied": False}
        if protect_regions and not validate_protected_regions(motif_suggestion, page_regions, width=page_width,
                                                              height=page_height)["passed"]:
            image_mode["reason_codes"].append("BORDER_PROTECTED_CONTENT_VETO")
        else:
            image_mode["suggested_warp_quad"] = _vertices_on_page(motif_suggestion, page_width, page_height)
            try:
                image_only_preview, image_mode["outline_preview"] = _render_outline_preview(page, motif_suggestion, preview_projection)
            except (ValueError, cv2.error):
                pass  # Preserve the measured editing suggestion if rendering is unresolved.
    elif print_outline is not None and not scalloped_motif:
        image_mode["reason_codes"].insert(0, "IMAGE_BOUNDARY_UNCERTAIN")
    elif not full_reasons or needs_closed_rim or preview_only or scalloped_motif:
        quad, diagnostics, image_reasons = _inner_crop_plan(
            full_image, matrix, inner, trim_border=trim_border, page_width=page_width, page_height=page_height,
            page_regions=page_regions, protect_regions=protect_regions, allow_wide_search=not needs_closed_rim,
        )
        wide_rim = diagnostics["method"] == "wide_paper_rim"
        suggested_quad = None
        seam_traces = []
        margin_regions = None
        outline = None
        needs_native_line_support = False
        all_edges = diagnostics.get("edges", {})
        four_rim_edges = len(all_edges) == 4 and all(edge.get("is_physical_border") for edge in all_edges.values())
        seam_seed = diagnostics.pop("curve_seed_quad", quad)
        if seam_seed is not None and four_rim_edges and inner is None and not independent:
            native_seed = cv2.perspectiveTransform(seam_seed.reshape(1, 4, 2), np.linalg.inv(matrix))[0]
            review_fragments: list[np.ndarray] = []
            traces, curved, disagrees, continuously_observed = trace_motif_boundary(
                page, native_seed, all_edges, review_fragments=review_fragments,
            )
            if curved:
                image_reasons.append(CURVED_IMAGE_BOUNDARY)
            elif disagrees and not image_reasons:
                image_reasons.append("IMAGE_BOUNDARY_UNCERTAIN")
            needs_native_line_support = not continuously_observed
            if curved or disagrees or needs_native_line_support or CURVED_IMAGE_BOUNDARY in image_reasons:
                seam_traces = [_vertices_on_page(trace, page_width, page_height)
                               for trace in (*traces, *review_fragments)]
            if curved or disagrees or CURVED_IMAGE_BOUNDARY in image_reasons:
                if continuously_observed and CURVED_IMAGE_BOUNDARY in image_reasons:
                    outline = close_motif_outline(native_seed, traces)
                quad = None
        if needs_closed_rim and (quad is None or (inner is None and not four_rim_edges)):
            image_reasons.append("IMAGE_VISIBILITY_UNCERTAIN")
            quad = None
        if not image_reasons and quad is None:
            image_only = adjusted
            image_mode = {**full_mode, "method": diagnostics["method"]}
        elif quad is not None:
            try:
                _validate_quad(quad, width, height)
                source = cv2.perspectiveTransform(quad.reshape(1, 4, 2), np.linalg.inv(matrix))[0]
                if needs_closed_rim or wide_rim or (inner is None and four_rim_edges):
                    # The rectified scan corroborates the paper rim. Locate its last
                    # boundary in source pixels, where resampling cannot blur the seam.
                    radius = max(3, min(16, round(max(width, height) / 450))) if wide_rim else 3
                    refined, native_evidence, native_reasons = refine_print_boundary(page, source, search_radius=radius)
                    diagnostics["native_boundary_refinement"] = native_evidence
                    native_verified = refined is not None and not native_reasons
                    # A straight fit can corroborate an incomplete trace, for
                    # example under glare. It must observe all sampled support
                    # before the motif is marked ready.
                    line_supported = native_verified and not (inboard_review and needs_native_line_support) and (preview_only or (
                        len(native_evidence) == 4 and all(
                            edge["support"] == 1 and edge["corner_support"] == 1 for edge in native_evidence.values()
                        )
                    ))
                    if (needs_native_line_support or inboard_review) and not line_supported:
                        image_reasons.append("IMAGE_BOUNDARY_UNCERTAIN")
                    elif (needs_closed_rim or wide_rim) and (
                        not native_verified or len(native_evidence) != 4
                        or any(edge["support"] < .85 or (edge["corner_support"] or 0) < .85
                               for edge in native_evidence.values())
                        or (inner is not None and np.max(np.linalg.norm(source - refined, axis=1)) > 1.25)
                    ):
                        image_reasons.append("IMAGE_VISIBILITY_UNCERTAIN")
                    elif inner is None and native_verified:
                        candidate = cv2.perspectiveTransform(refined.reshape(1, 4, 2), matrix)[0]
                        _validate_quad(candidate, width, height)
                        source, quad = refined, candidate
                _validate_quad(source, page_width, page_height)
                measured = metadata["aspect_ratio_estimation"]["method"] in {
                    "known_print_aspect", "measured_page_reference", "calibrated_camera", "estimated_page_plane",
                    "measured_page_plane",
                }
                if (needs_closed_rim or wide_rim) and not image_reasons and not _independent_motif_visible(page, source, neighbors):
                    image_reasons.append("IMAGE_VISIBILITY_UNCERTAIN")
                if stock_motif_review and any(cv2.pointPolygonTest(source_quad, tuple(point), True) < 3 for point in source):
                    image_reasons.append("IMAGE_VISIBILITY_UNCERTAIN")
                if (measured or independent or wide_rim or inboard_review) and not _is_rectangular_in_measured_plane(quad):
                    image_reasons.append("IMAGE_RECTANGLE_INCONSISTENT")
                if protect_regions:
                    source_protection = validate_protected_regions(source, page_regions, width=page_width, height=page_height)
                    if not source_protection["passed"] or (inner is not None and removed_protected_regions(
                        source_quad, source, page_regions, width=page_width, height=page_height,
                    )):
                        image_reasons.append("BORDER_PROTECTED_CONTENT_VETO")
                if not image_reasons and inner is None and (not full_reasons or needs_closed_rim):
                    margin_regions = motif_margin_marks(
                        page, full_image, quad, np.linalg.inv(matrix),
                        preview_guard=stock_motif_review, inspect_rules=inboard_review,
                    )
                    if inboard_review and not margin_regions:
                        # Inspect the proposed extended paper as a second margin
                        # witness. The motif is still fitted and shown in the old plane.
                        expanded = _render_review_quad_preview(
                            page, adjacent_stock_outline, 0, neighbors, page_regions, None, None,
                            source=ADJACENT_STOCK_REVIEW_SOURCE,
                        )
                        if expanded is None:
                            image_reasons.append("IMAGE_VISIBILITY_UNCERTAIN")
                        else:
                            expanded_image, expanded_geometry = expanded
                            expanded_matrix = np.asarray(expanded_geometry["page_to_output"])
                            expanded_motif = cv2.perspectiveTransform(source[None], expanded_matrix)[0]
                            margin_regions = motif_margin_marks(
                                page, expanded_image, expanded_motif, np.linalg.inv(expanded_matrix),
                                preview_guard=True, inspect_rules=True,
                            )
                if not image_reasons and not margin_regions and wide_rim:
                    rectified_regions = _rectified_protected_regions(
                        page_regions, page_width=page_width, page_height=page_height, matrix=matrix,
                        target_width=width, target_height=height,
                    )
                    if validate_protected_regions(quad, rectified_regions, width=width, height=height,
                                                  require_containment=True)["passed"]:
                        suggested_quad = _vertices_on_page(source, page_width, page_height)
                        image_reasons.append("WIDE_MOTIF_CONFIRMATION_REQUIRED")
                    else:
                        image_reasons.append("BORDER_PROTECTED_CONTENT_VETO")
                if not image_reasons and not margin_regions:
                    top, right, bottom, left = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
                    inner_width = round(float((top + bottom) / 2)) + 1
                    inner_height = round(float((left + right) / 2)) + 1
                    destination = _UNIT_QUAD * np.array([inner_width - 1, inner_height - 1], dtype=np.float32)
                    combined = cv2.getPerspectiveTransform(quad, destination) @ matrix
                    _assess_homography(source, destination, combined)
                    pixels, supported = warp_owned_crop(page, combined, (inner_width, inner_height))
                    if supported:
                        image_mode = _geometry_for_mode(
                            source, combined, pixels, page_width, page_height, rotation,
                            full_reasons if preview_only or stock_motif_review else [],
                        )
                        image_mode["sampling"] = OWNED_BILINEAR
                        if independent or scalloped_motif:
                            image_mode["independent_of_full_print"] = True
                        if scalloped_motif:
                            image_mode["scalloped_motif_ownership_revision"] = SCALLOPED_PRINT_REVISION
                        if stock_motif_review:
                            if not inboard_review:
                                full_mode["suggested_warp_quad"] = full_mode["warp_quad"]
                            image_mode["outline_preview"] = {
                                "source": PAPER_STOCK_MOTIF_SOURCE,
                                "source_outline": image_mode["warp_quad"],
                                "page_to_output": image_mode["page_to_output"],
                                "shape": image_mode["shape"],
                            }
                            image_only_preview = rotate_image(pixels, rotation)
                        else:
                            image_only = rotate_image(pixels, rotation)
                    else:
                        image_reasons.append("IMAGE_BOUNDARY_SAMPLING_UNCERTAIN")
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                image_reasons.append("IMAGE_QUAD_UNSTABLE")
        if (image_reasons and margin_regions is None and seam_seed is not None and inner is None
                and (not full_reasons or needs_closed_rim)):
            margin_regions = motif_margin_marks(
                page, full_image, seam_seed, np.linalg.inv(matrix),
                preview_guard=stock_motif_review, inspect_rules=inboard_review,
            )
        if margin_regions:
            image_reasons.insert(0, MOTIF_MARGIN_MARKS_REVIEW)
        if image_reasons and inboard_review and not margin_regions:
            diagnostics = {"method": "deferred_until_print_boundary_is_verified", "applied": False}
        elif image_reasons:
            diagnostics["applied"] = False
            if stock_motif_review:
                image_reasons.extend(full_reasons)
            image_mode = {"status": "review_needed", "reason_codes": list(dict.fromkeys(image_reasons))}
            if margin_regions:
                image_mode["margin_mark_regions"] = [_vertices_on_page(region, page_width, page_height)
                                                     for region in margin_regions]
                if stock_motif_review:
                    image_mode["margin_mark_source"] = PAPER_STOCK_MOTIF_SOURCE
            if suggested_quad is not None:
                image_mode["suggested_warp_quad"] = suggested_quad
            if seam_traces:
                image_mode["suggested_seam_traces"] = seam_traces
            if outline is not None and not stock_motif_review and not margin_regions and all(
                cv2.pointPolygonTest(source_quad, tuple(point), True) >= -1e-4 for point in outline
            ):
                outline_hull = cv2.convexHull(outline)
                overlaps = any(cv2.intersectConvexConvex(outline_hull, cv2.convexHull(neighbor))[0] > 1.0
                               for neighbor in neighbors)
                protects_content = not protect_regions or validate_protected_regions(
                    outline, page_regions, width=page_width, height=page_height,
                )["passed"]
                if not overlaps and protects_content and motif_outline_corners_supported(page, outline, traces, all_edges):
                    try:
                        image_only_preview, image_mode["outline_preview"] = _render_outline_preview(
                            page, outline, preview_projection,
                        )
                    except (ValueError, cv2.error):
                        pass  # Incomplete preview geometry leaves the measured lines available for review.
    elif trim_border and inner is None and set(full_reasons) == {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_ASPECT_UNCERTAIN"}:
        try:
            guides = [_vertices_on_page(trace, page_width, page_height)
                      for trace in partial_motif_seams(page, full_image, matrix)]
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            guides = []
        if guides and not any(review_traces_intersect_claim(guides, mask, page_width, page_height)
                              for mask in neighbor_masks or []):
            image_mode["suggested_seam_traces"] = guides
            image_mode["suggested_seam_source"] = PARTIAL_MOTIF_SEAMS

    metadata.update(
        crop_modes={"full_print": full_mode, "image_only": image_mode},
        orientation=orientation,
        orientation_confidence=orientation.get("confidence"),
        final_shape=full_mode["shape"],
        untrimmed_shape=full_mode["shape"],
        border_trim={**diagnostics, "applied": bool(diagnostics.get("applied") and image_only is not None)},
        border_trim_confidence=diagnostics.get("confidence") if diagnostics.get("applied") and not inboard_review else None,
    )
    return CropResult(rectified_untrimmed=adjusted, image=adjusted, metadata=metadata,
                      image_only=image_only, image_only_preview=image_only_preview, full_print_preview=full_print_preview)


def crop_and_correct_photo(
    page_image: np.ndarray,
    bounding_box: dict,
    mask: dict | None = None,
    *,
    trim_border: bool = True,
    auto_orient: bool = True,
    orientation_model_path: Path | None = None,
    protected_regions: list[dict] | None = None,
    protect_regions: bool = True,
) -> np.ndarray:
    result = correct_photo_crop(
        page_image,
        bounding_box,
        mask,
        trim_border=trim_border,
        auto_orient=auto_orient,
        orientation_model_path=orientation_model_path,
        protected_regions=protected_regions,
        protect_regions=protect_regions,
    )
    return result.image


def _outer_contour_hypothesis(page: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    from app.pipeline.segmentation import (
        _CLASSICAL_MIN_AREA_RATIO, _find_contrast_candidates, _quadrilateral_from_contour, detect_photos_classical,
    )

    height, width = page.shape[:2]
    x, y, w, h = cv2.boundingRect(seed)
    pad = max(8, round(0.65 * min(w, h)))
    left, top = max(0, x - pad), max(0, y - pad)
    roi = page[top : min(height, y + h + pad), left : min(width, x + w + pad)]
    scale = min(1.0, 840 / max(roi.shape[:2]))
    sample = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else roi
    contours = detect_photos_classical(sample, max_line_refinement_pixels=sample.shape[0] * sample.shape[1])
    seed_area = float(cv2.contourArea(seed))

    def enclosing_quads(candidates) -> list[np.ndarray]:
        enclosing = []
        for points in candidates:
            if points is None:
                continue
            candidate = _order_points(points + np.float32([left, top]))
            candidate_area = float(cv2.contourArea(candidate))
            if 1.10 * seed_area < candidate_area < 1.95 * seed_area:
                overlap, _ = cv2.intersectConvexConvex(seed, candidate)
                if overlap >= 0.97 * seed_area:
                    enclosing.append(candidate)
        return enclosing

    supported = enclosing_quads(_points_from_mask(detection.mask, roi.shape[1], roi.shape[0]) for detection in contours.detections)
    if not supported:
        # Full-page candidate ranking can discard the print as a container
        # when a pale area in the image separates it into strong subregions.
        pixel_scale = np.float32([roi.shape[1] - 1, roi.shape[0] - 1]) / [sample.shape[1] - 1, sample.shape[0] - 1]
        supported = enclosing_quads(
            _quadrilateral_from_contour(candidate.contour)[0] * pixel_scale
            for candidate in _find_contrast_candidates(sample, _CLASSICAL_MIN_AREA_RATIO)
        )
    # A second plausible container is ambiguity, even if one scores higher.
    return supported[0] if len(supported) == 1 else None


@dataclass(frozen=True)
class _PrintInspection:
    quad: np.ndarray
    metrics: dict
    boundary: dict
    reasons: list[str]
    corner_mount: bool
    protection: dict
    occluding_neighbor: np.ndarray | None = None


def _inspect_print(
    page: np.ndarray, quad: np.ndarray, mask: dict, visible: np.ndarray | None,
    neighbors: list[np.ndarray], regions: list[dict], *, dual_mode: bool, manual: bool,
    allow_page_face_overlap: bool, protect_regions: bool, enable_classical_retry: bool,
    manual_neighbors: list[dict] | None = None, collect_curve_segments: bool = False,
) -> _PrintInspection:
    height, width = page.shape[:2]
    metrics = _validate_quad(quad, width, height)
    boundary: dict = {}
    reasons: list[str] = []
    corner_mount = False
    edge_confirmation = False
    manual_neighbor = None
    if dual_mode and not manual and mask.get("warp_quad") is None:
        refined, boundary, reasons = refine_print_boundary(page, quad, collect_curve_segments=collect_curve_segments)
        if refined is not None and mask.get("detector_source") == CONTRAST_COMPONENT_SOURCE:
            if not reasons:
                extension = find_outer_stock_candidate(page, refined, neighbors)
                if extension is not None:
                    refined, side, evidence = extension
                    key = "outer_stock_candidate" if evidence["flanks_confirmed"] else "ambiguous_stock_step"
                    boundary[side][key] = evidence
                    edge_confirmation = True
            elif reasons == ["PRINT_BOUNDARY_AMBIGUOUS"]:
                recovered = retry_ambiguous_stock_edge(page, quad, refined, boundary)
                if recovered is not None:
                    refined, boundary = recovered
                    reasons = []
                    edge_confirmation = True
        if enable_classical_retry and "PRINT_OUTER_RIM_UNCERTAIN" in reasons:
            hypothesis = _outer_contour_hypothesis(page, quad)
            if hypothesis is not None:
                recovered, evidence, remaining = refine_print_boundary(page, hypothesis)
                if recovered is not None and not remaining:
                    refined, reasons = recovered, []
                    boundary = {**evidence, "seed": "local_outer_contour"}
        suggestion = None
        if refined is not None and reasons == ["PRINT_BOUNDARY_AMBIGUOUS"]:
            suggestion = suggest_ambiguous_stock_edge(page, quad, refined, boundary, neighbors)
            if suggestion is not None:
                refined = suggestion[0]
        if (enable_classical_retry and refined is not None and "PRINT_BOUNDARY_UNSUPPORTED" in reasons
                and set(reasons) <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN"}):
            pale_backing = suggest_missing_pale_backing_edge(page, refined, boundary, neighbors)
            if pale_backing is not None:
                refined, boundary = pale_backing
                reasons = []
                edge_confirmation = True
        if refined is not None and "PRINT_OUTER_RIM_UNCERTAIN" not in reasons:
            refined_protection = validate_protected_regions(refined, regions, width=width, height=height)
            if not refined_protection["passed"] and not _advisory_face_overlap(refined_protection, allow_page_face_overlap):
                reasons.append("PRINT_BOUNDARY_PROTECTED_CONTENT_VETO")
            else:
                if suggestion is not None:
                    boundary = suggestion[1]
                quad = refined
                metrics = _validate_quad(quad, width, height)
    if dual_mode:
        manual_outline = mask.get("manual_added") is True or mask.get(REVIEWED_CROSSING_OUTLINE) is True
        outline_review = next(((kind, reason) for kind, reason in (
            ("projective", PROJECTIVE_PRINT_OUTLINE_REVIEW), ("mounted", MOUNTED_PRINT_OUTLINE_REVIEW),
            ("dark_backing", DARK_BACKING_OUTLINE_REVIEW),
        ) if mask.get(f"{kind}_outline_review") is True), None)
        if outline_review is not None:
            kind, reason = outline_review
            confirmed = False
            if mask.get(f"{kind}_outline_confirmed") is True:
                fitted, sides, unresolved = refine_print_boundary(page, quad, search_radius=3)
                confirmed = (fitted is not None and not unresolved and np.linalg.norm(fitted - quad, axis=1).max() <= 1.
                             and all(edge["support"] >= .90 and (edge["corner_support"] or 0) >= .80
                                     for edge in sides.values()))
            if not confirmed:
                reasons.append(reason)
        if not manual and mask.get("paper_edge_confirmation") is True:
            reasons.append(PRINT_EDGE_CONFIRMATION)
        scalloped = not manual and mask.get("scalloped_print") is True and visible is not None
        if scalloped:
            reasons.append(SCALLOPED_PRINT_REVIEW)
        if not manual and mask.get("fragmented_print") is True:
            reasons.append(FRAGMENTED_PRINT_REVIEW)
        if not manual and mask.get("extended_print_outline") is True:
            reasons.append(EXTENDED_PRINT_OUTLINE_REVIEW)
        if not manual and mask.get("fragment_outline_suggestion"):
            reasons.append(FRAGMENT_OUTLINE_REVIEW)
        if not manual and mask.get("source_fragment_group_review") is not None:
            reasons.append(SOURCE_FRAGMENT_GROUP_REVIEW)
        if not manual and mask.get("crossing_print_suggestion"):
            reasons.append(CROSSING_PRINT_REVIEW)
        if not manual and mask.get(PARTIAL_PRINT_EXTENT_TRACES):
            reasons.append(PARTIAL_PRINT_EXTENT_REVIEW)
        if not manual and MERGED_PRINT_PAIR in mask:
            reasons.append(MERGED_PRINT_PAIR_REVIEW)
        if mask.get("album_window_occluded") is True:
            reasons.append(ALBUM_WINDOW_OCCLUDED)
        elif mask.get("source") == ALBUM_WINDOW_SOURCE:
            reasons.append(ALBUM_WINDOW_STOCK_UNCERTAIN)
        elif manual_outline:
            outline = pixel_vertices_to_normalized(quad, width, height)
            for leaf in album_window_leaves(page):
                if (leaf.print_outline is not None
                        and normalized_polygon_overlap(outline, leaf.print_outline).first_containment >= .80):
                    reasons.append(ALBUM_WINDOW_OCCLUDED if leaf.aperture_quad is not None else ALBUM_WINDOW_STOCK_UNCERTAIN)
                    break
        if mask.get("capture_edge_clipped") is True or (not manual and capture_edge_requires_review(mask)) or (
            (not manual or manual_outline) and np.any((quad < 0.5) | (quad > np.array([width, height]) - 1.5))
        ):
            reasons.append(CAPTURE_EDGE_HOLD)
        sides = [boundary.get(side, {}) for side in ("top", "right", "bottom", "left")]
        inferred = tuple(index for index, evidence in enumerate(sides) if evidence.get("matched_print_rim"))
        observed = {
            index: (np.array(evidence["observed_rim_lab"]), evidence["observed_rim_depth_px"])
            for index, evidence in enumerate(sides)
            if not reasons and evidence.get("support", 0) >= 0.9
            and (evidence.get("corner_support") or 0) >= 0.8
            and "observed_rim_lab" in evidence and index not in inferred
        }
        occluding_neighbor = _occluding_neighbor(
            page, quad, None if manual or manual_outline else visible, neighbors,
            inferred_sides=inferred, observed_sides=observed,
        )
        occluded = occluding_neighbor is not None
        if manual and occluded:
            manual_neighbor = occluding_neighbor
        if not occluded and manual_neighbors:
            target = _vertices_on_page(quad, width, height)
            mark = next((mark for mark in manual_neighbors if manual_print_intersects(target, mark, width, height)), None)
            if mark is not None:
                occluded = True
                if manual:
                    manual_neighbor = _explicit_quad({"warp_quad": mark}, "warp_quad", width, height)
        if (not scalloped and _clipped_visible_corner(quad, visible)) or occluded:
            reasons.append("PRINT_POSSIBLY_OCCLUDED")
        if manual_neighbor is not None:
            reasons.append(MANUAL_NEIGHBOR_OVERLAP)
        if not reasons and neighbors:
            # An observed outward strip may still lack an assignable paper edge.
            for index, evidence in enumerate(sides):
                outward = evidence.get("outward_search", {})
                if not outward.get("print_rim_supported") or evidence.get("observed_rim_depth_px") or index in inferred:
                    continue
                support = outward["support"]
                first, last = sorted((outward["start_support"], outward["end_support"]))
                # An outward line that ends at one corner can follow a neighbor's ink.
                one_end_missing = support >= .84 and evidence["support"] >= .90 and first <= .20 and last >= .80
                if (support >= .88 or one_end_missing) and paper_rim_reaches_neighbor(
                    page, quad, index, neighbors, partial_outward=support < .88,
                ):
                    evidence["shared_rim_unresolved"] = True
            if any(evidence.get("shared_rim_unresolved") for evidence in sides):
                reasons.append(PRINT_SHARED_RIM_UNCERTAIN)
        if not reasons and missing_corner_material(page, quad):
            reasons.append("PRINT_CORNERS_UNCERTAIN")
        if not occluded and set(reasons) <= _LOCALIZED_OCCLUSION_REASONS:
            covered, corner_mount = print_corner_coverage(page, quad)
            if covered and not reasons:
                reasons.append("PRINT_POSSIBLY_OCCLUDED")
    protection = (
        validate_protected_regions(quad, regions, width=width, height=height)
        if protect_regions else {"passed": True, "regions_checked": 0, "violations": []}
    )
    if not protection["passed"]:
        margin_marks = (manual_margin_advisory(page, quad, regions, protection["violations"], neighbors)
                        if dual_mode and manual and mask.get("manual_added") is True else [])
        if not margin_marks and not _advisory_face_overlap(protection, allow_page_face_overlap):
            raise ValueError("protected_region_cut")
        if margin_marks:
            protection["manual_margin_marks"] = margin_marks
        protection["review_untrimmed"] = True
        reasons.append("POSSIBLE_CONTENT_CUTOFF")
    if edge_confirmation and not reasons:
        reasons.append(PRINT_EDGE_CONFIRMATION)
    return _PrintInspection(quad, metrics, boundary, reasons, corner_mount, protection, manual_neighbor)


def _source_visible_outline(
    page: np.ndarray, mask: dict, inspection: _PrintInspection, visible: np.ndarray | None,
    neighbors: list[np.ndarray], neighbor_masks: list[dict],
) -> tuple[np.ndarray, str] | None:
    if not inspection.boundary or visible is None:
        return None
    height, width = page.shape[:2]
    if not inspection.reasons or (
        inspection.reasons == [PRINT_EDGE_CONFIRMATION] and _pale_backing_review_fit(inspection.boundary)
    ):
        overlap = normalized_polygon_overlap(
            pixel_vertices_to_normalized(visible, width, height),
            pixel_vertices_to_normalized(inspection.quad, width, height),
        )
        return (inspection.quad, _PRINT_BOUNDARY_SOURCE) if overlap.first_containment >= 0.95 else None
    if not inspection.protection["passed"] or "PRINT_BOUNDARY_PROTECTED_CONTENT_VETO" in inspection.reasons:
        return None
    primary = _proposal_polygon({"polygon": mask.get("polygon")}, width, height)
    if primary is not None and not np.array_equal(primary, visible):
        primary_neighbors = [
            points for neighbor in neighbor_masks
            if (points := _proposal_polygon({"polygon": neighbor.get("polygon")}, width, height)) is not None
        ]
        observed = refine_visible_paper_rim(page, primary, primary_neighbors)
        if observed is not None:
            agreement = normalized_polygon_overlap(
                pixel_vertices_to_normalized(visible, width, height),
                pixel_vertices_to_normalized(observed, width, height),
            )
            if agreement.first_containment >= .90 and agreement.second_containment >= .90:
                return observed, _PAPER_RIM_SOURCE
    rim = refine_visible_paper_rim(page, visible, neighbors)
    return (rim, _PAPER_RIM_SOURCE) if rim is not None else None


def measure_detected_visibility(
    page: np.ndarray, mask: dict, visible: np.ndarray | None, neighbors: list[np.ndarray],
    regions: list[dict], *, geometry_valid: bool, protect_regions: bool, enable_classical_retry: bool,
) -> tuple[np.ndarray, str] | None:
    if visible is None or not geometry_valid or mask.get("warp_quad") or mask.get("source") == ALBUM_WINDOW_SOURCE:
        return None
    try:
        seed = _points_from_mask(mask, *page.shape[1::-1])
        if seed is None:
            return None
        inspection = _inspect_print(
            page, _order_points(seed), mask, visible, neighbors, regions,
            dual_mode=True, manual=False, allow_page_face_overlap=True, protect_regions=protect_regions,
            enable_classical_retry=enable_classical_retry,
        )
        return _source_visible_outline(page, mask, inspection, visible, neighbors, [])
    except (ValueError, cv2.error):
        return None


def correct_photo_crop(
    page_image: np.ndarray,
    bounding_box: dict,
    mask: dict | None = None,
    *,
    trim_border: bool = True,
    auto_orient: bool = True,
    orientation_model_path: Path | None = None,
    protected_regions: list[dict] | None = None,
    protect_regions: bool = True,
    allow_page_face_overlap: bool = False,
    allow_border_expansion: bool = True,
    dual_mode: bool = False,
    enable_classical_retry: bool = False,
    camera_matrix: np.ndarray | None = None,
    page_plane: PrintPagePlane | None = None,
    print_aspect_ratio: float | None = None,
    measured_page_references: list[MeasuredPrintReference] | None = None,
    neighbor_masks: list[dict] | None = None,
    manual_boundary_reference: dict | None = None,
    manual_scalloped_polygon: list[dict] | None = None,
    reviewed_album_mark: dict | None = None,
    previous_manual_crop: dict | None = None,
    manual_neighbor_reference: dict | None = None,
    shared_rim_reference: list[list[dict]] | None = None,
) -> CropResult:
    height, width = page_image.shape[:2]
    mask = mask or {}
    if dual_mode:
        validate_review_quads(mask, width, height)
    measured_edge = print_aspect_ratio is not None and mask.get("warp_quad") is not None
    src_points = _points_from_mask(mask, width, height, preserve_width_edge=measured_edge)
    if src_points is None:
        x1, y1, x2, y2 = normalized_box_to_pixels(bounding_box, width, height)
        # Warp vertices are inclusive centers. The last pixel in a half-open box is max - 1.
        x2 -= 1
        y2 -= 1
        if x2 <= x1 or y2 <= y1:
            raise ValueError("degenerate_perspective_quad")
        src_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if not measured_edge:
        src_points = _order_points(src_points)
    seed_points = src_points
    manual = not allow_border_expansion or str(mask.get("source") or "").startswith("manual")
    manual_rotation = mask.get("orientation_degrees") if manual else None
    regions = [] if not protect_regions else (
        protected_regions if protected_regions is not None else detect_protected_regions(page_image)
    )
    visible = _proposal_polygon(mask, width, height) if dual_mode else None
    neighbor_polygons = [
        points for neighbor in neighbor_masks or []
        if (points := _proposal_polygon(neighbor, width, height)) is not None
    ] if dual_mode else []
    if reviewed_album_mark is not None:
        if not manual or not protect_regions or not dual_mode:
            raise ValueError("album_page_mark_review_stale")
        regions = _regions_after_album_mark_review(page_image, src_points, regions, reviewed_album_mark, neighbor_polygons)
    if manual and regions and previous_manual_crop is not None:
        try:
            previous = _points_from_mask(previous_manual_crop, width, height)
        except (KeyError, TypeError, ValueError):
            previous = None  # A corrupt historical outline cannot establish ownership.
        if previous is not None:
            prior_visibility = previous_manual_crop.get("visible_polygon")
            visible_owner = _proposal_polygon({"polygon": prior_visibility}, width, height) if prior_visibility else None
            removed = removed_protected_regions(
                _order_points(previous), src_points, regions, width=width, height=height, visible=visible_owner,
            )
            if removed:
                raise ProtectedRegionRemoval("full_print", removed)
        inner_edit = _explicit_quad(mask, "image_quad", width, height) if dual_mode else None
        if inner_edit is not None:
            removed = removed_protected_regions(src_points, inner_edit, regions, width=width, height=height)
            if removed:
                raise ProtectedRegionRemoval("image_only", removed)
    inspection = _inspect_print(
        page_image, src_points, mask, visible, neighbor_polygons, regions,
        dual_mode=dual_mode, manual=manual, allow_page_face_overlap=allow_page_face_overlap,
        protect_regions=protect_regions, enable_classical_retry=enable_classical_retry,
        collect_curve_segments=True,
        manual_neighbors=[
            neighbor["warp_quad"] for neighbor in neighbor_masks or []
            if (neighbor.get("manual_added") is True or neighbor.get(REVIEWED_CROSSING_OUTLINE) is True)
            and neighbor.get("warp_quad")
        ],
    )
    src_points, quad_metrics = inspection.quad, inspection.metrics
    if dual_mode and not manual and shared_rim_reference and PRINT_SHARED_RIM_UNCERTAIN not in inspection.reasons:
        inspection.reasons.append(PRINT_SHARED_RIM_UNCERTAIN)
    boundary_refinement, full_reasons = inspection.boundary, inspection.reasons.copy()
    missing_paper = []
    backing_match_pixels = 0
    if dual_mode and manual and manual_neighbor_reference and inspection.occluding_neighbor is None and neighbor_polygons:
        missing_paper, backing_match_pixels = _lost_visible_paper(
            page_image, src_points, manual_neighbor_reference, neighbor_polygons,
        )
        if missing_paper:
            full_reasons.append(MANUAL_NEIGHBOR_CONTENT_LOSS)
    curve_segments = [segment for edge in boundary_refinement.values() if isinstance(edge, dict)
                      for segment in edge.pop("curve_segments", [])]
    competing_segments = []
    for edge in boundary_refinement.values():
        if isinstance(edge, dict):
            segments = edge.pop(COMPETING_PRINT_EDGES, [])
            if not edge.get("review_stock_fit"):
                competing_segments.extend(segments)
    print_edge_traces = []
    if ("PRINT_BOUNDARY_CURVED" in full_reasons
            and set(full_reasons) <= {"PRINT_BOUNDARY_CURVED", EXTENDED_PRINT_OUTLINE_REVIEW}):
        print_edge_traces = [_vertices_on_page(np.asarray(segment), width, height) for segment in curve_segments]
    elif (dual_mode and not manual and enable_classical_retry and "PRINT_BOUNDARY_UNSUPPORTED" in full_reasons
          and set(full_reasons) <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN", "POSSIBLE_CONTENT_CUTOFF"}):
        trace = suggest_partial_corner_edge(page_image, src_points, boundary_refinement)
        if trace is not None:
            print_edge_traces = [_vertices_on_page(trace, width, height)]
    checked_manual_boundary = dual_mode and manual and manual_boundary_reference is not None
    capture_outline = checked_manual_boundary and manual_boundary_reference.get("capture_outline")
    backed_edge = checked_manual_boundary and bool(manual_boundary_reference.get("backed_edge_review"))
    if checked_manual_boundary:
        manual_boundary_hold, print_edge_traces = check_manual_paper_boundary(page_image, src_points, manual_boundary_reference)
        if manual_boundary_hold:
            if backed_edge:
                full_reasons.append(PRINT_EDGE_CONFIRMATION)
            else:
                full_reasons.append(MANUAL_CAPTURE_OUTLINE_REVIEW if capture_outline else MANUAL_PAPER_BOUNDARY_REVIEW)
    manual_scalloped = _proposal_polygon({"polygon": manual_scalloped_polygon}, width, height) if dual_mode and manual else None
    if manual_scalloped is not None:
        full_reasons.append(SCALLOPED_PRINT_REVIEW)
        print_edge_traces.append(_vertices_on_page(np.vstack((manual_scalloped, manual_scalloped[0])), width, height))
    blurred_boundary = not full_reasons and any(
        (edge.get("edge_width_px") or 0) >= 4.75
        for edge in boundary_refinement.values() if isinstance(edge, dict)
    )
    protection, corner_mount_detected = inspection.protection, inspection.corner_mount
    if reviewed_album_mark is not None:
        protection["reviewed_album_mark_id"] = reviewed_album_mark["id"]
    boundary_supports_preview = bool(boundary_refinement and set(full_reasons) <= _CROP_PREVIEW_REASONS and visible is not None)
    target_width, target_height, aspect, aspect_reasons = _crop_dimensions(
        src_points, width, height, metric=dual_mode, camera_matrix=camera_matrix, print_aspect_ratio=print_aspect_ratio,
        page_plane=page_plane if boundary_supports_preview else None,
    )
    if (set(full_reasons) <= _CROP_PREVIEW_REASONS and aspect_reasons == ["PRINT_ASPECT_UNCERTAIN"] and camera_matrix is None
            and measured_page_references):
        inferred, estimates = measured_page_aspect(page_image, src_points, measured_page_references)
        if inferred is None:
            aspect["page_reference_estimates"] = estimates
        else:
            target_width, target_height, aspect, _ = _crop_dimensions(
                src_points, width, height, metric=dual_mode, camera_matrix=None, print_aspect_ratio=inferred,
            )
            aspect.update(method="measured_page_reference", page_reference_estimates=estimates)
            aspect_reasons = [PRINT_ASPECT_REFERENCE_REVIEW]
    if (boundary_supports_preview and camera_matrix is None and print_aspect_ratio is None
            and page_plane is not None and page_plane.independent_fits
            and (not aspect_reasons or aspect_reasons == ["PRINT_ASPECT_UNCERTAIN"])):
        estimate = page_plane.estimate_aspect(src_points)
        output_aspect = (target_width - 1) / (target_height - 1)
        if estimate is not None and (aspect_reasons or abs(output_aspect / estimate[0] - 1) > .01):
            target_width, target_height, aspect, _ = _crop_dimensions(
                src_points, width, height, metric=dual_mode, camera_matrix=None, print_aspect_ratio=estimate[0],
            )
            method = "measured_page_plane" if page_plane.reference_photo_id is not None else "estimated_page_plane"
            aspect.update(method=method, page_plane=estimate[1])
            aspect_reasons = [PRINT_ASPECT_REFERENCE_REVIEW]
    full_reasons.extend(aspect_reasons)
    if target_width * target_height > 2 * width * height:
        raise ValueError("excessive_perspective_output_area")
    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_points, destination)
    # The condition number of a pixel-coordinate homography grows with source
    # resolution and rejected valid 12 MP mobile captures. Evaluate the same
    # transform in normalized coordinates so the gate measures geometry rather
    # than pixel scale.
    normalized_source = src_points / np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    normalized_destination = destination / np.array(
        [max(target_width - 1, 1), max(target_height - 1, 1)], dtype=np.float32
    )
    normalized_matrix = cv2.getPerspectiveTransform(
        normalized_source.astype(np.float32), normalized_destination.astype(np.float32)
    )
    condition_number = float(np.linalg.cond(matrix))
    normalized_condition_number = float(np.linalg.cond(normalized_matrix))
    if not np.isfinite(normalized_condition_number) or normalized_condition_number > 1e4:
        raise ValueError("unstable_perspective_transform")
    homography_assessment = _assess_homography(src_points, destination, matrix)
    perspective_confidence = _perspective_confidence(homography_assessment, normalized_condition_number)
    boundary_evidence, geometry_evidence = _input_evidence(mask, quad_metrics)
    rectified_to_page = np.linalg.inv(matrix)
    left = top = right = bottom = 0
    if not dual_mode and trim_border and allow_border_expansion and protection["passed"] and mask.get("source") in {
        "yolo_box", "yolo_seg_mask", "classical_min_area_rect", "classical_border_quad", "chromatic_mount_boundary"
    }:
        left, top, right, bottom = _analysis_padding(destination, rectified_to_page, width, height)
    page_to_analysis = matrix.copy()
    page_to_analysis[:2] += np.array([[left], [top]]) * matrix[2]
    stock_suggestion = _pale_backing_review_fit(boundary_refinement) or any(
        isinstance(side, dict) and side.get("review_stock_fit") for side in boundary_refinement.values()
    )
    owned_sampling = dual_mode and (
        set(full_reasons) <= _CROP_PREVIEW_REASONS
        or mask.get("capture_edge_clipped") is True
        or (stock_suggestion and not set(full_reasons) & {CAPTURE_EDGE_HOLD, "PRINT_POSSIBLY_OCCLUDED"})
    )
    if owned_sampling:
        analysis, supported = warp_owned_crop(page_image, matrix, (target_width, target_height))
        if not supported:
            full_reasons.append("PRINT_BOUNDARY_SAMPLING_UNCERTAIN")
    else:
        analysis = cv2.warpPerspective(
            page_image, page_to_analysis, (left + target_width + right, top + target_height + bottom),
            borderMode=cv2.BORDER_CONSTANT if dual_mode else cv2.BORDER_REPLICATE,
        )
    if blurred_boundary:
        # Retain the measured warp as a preview and keep its sampling guard.
        full_reasons.append("PRINT_BOUNDARY_BLURRED")
    corrected = analysis[top : top + target_height, left : left + target_width]
    if corrected.size == 0 or min(corrected.shape[:2]) < 2:
        raise ValueError("empty_perspective_output")
    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    reprojection_error = float(np.max(np.linalg.norm(projected - destination, axis=1)))
    if not np.isfinite(reprojection_error) or reprojection_error > 0.05:
        raise ValueError("perspective_reprojection_error")
    source_pixels_per_output_pixel = float(
        np.sqrt(quad_metrics["area"] / max(float(target_width * target_height), 1.0))
    )
    normalized_quad = _vertices_on_page(src_points, width, height)
    metadata = {
        "source": mask.get("source") if mask else "bounding_box",
        "source_quad": normalized_quad,
        "warp_quad": normalized_quad,
        "coordinate_space": "original_page_normalized_vertices_v1",
        "target_width": target_width,
        "target_height": target_height,
        "source_pixels_per_output_pixel": round(source_pixels_per_output_pixel, 4),
        "trim_border": trim_border,
        "auto_orient": auto_orient and manual_rotation is None,
        "evidence_schema": "segmentation_detection_evidence_v1",
        "boundary_evidence": boundary_evidence,
        "geometry_evidence": geometry_evidence,
        "boundary_confidence": boundary_evidence.get("score"),
        "geometry_confidence": geometry_evidence.get("score"),
        "perspective_confidence": perspective_confidence,
        "quad_validation": {
            **quad_metrics,
            "transform_condition_number": condition_number,
            "normalized_transform_condition_number": normalized_condition_number,
            "reprojection_error": reprojection_error,
            **homography_assessment,
        },
        "protected_regions": protection,
    }
    if dual_mode:
        metadata.update(boundary_refinement=boundary_refinement, aspect_ratio_estimation=aspect)
        if checked_manual_boundary:
            metadata["manual_paper_boundary"] = {
                "revision": MANUAL_PAPER_BOUNDARY_REVISION, "requires_review": manual_boundary_hold,
            }
            if capture_outline:
                metadata["manual_paper_boundary"]["capture_outline_verified"] = not manual_boundary_hold
            if backed_edge:
                metadata["manual_paper_boundary"]["backed_edge_verified"] = not manual_boundary_hold
        observed = _source_visible_outline(
            page_image, mask, inspection, visible, neighbor_polygons, neighbor_masks or [],
        )
        if observed is not None:
            outline, source = observed
            metadata["refined_visible_polygon"] = _vertices_on_page(outline, width, height)
            metadata["refined_visibility_source"] = source
        rounded = None
        if (manual and mask.get("manual_added") is True and protect_regions and protection["passed"]
                and set(full_reasons) <= {"PRINT_ASPECT_UNCERTAIN"}):
            protected_boxes = []
            for region in regions:
                if box := region_box_in_pixels(region, width, height):
                    left, top, right, bottom = box
                    padding = float(region.get("margin_ratio", .08)) * max(right - left, bottom - top)
                    protected_boxes.append(np.float32([[left - padding, top - padding], [right + padding, top - padding],
                                                       [right + padding, bottom + padding], [left - padding, bottom + padding]]))
            rounded = measure_rounded_corners(page_image, src_points, [*neighbor_polygons, *protected_boxes])
        corner_bounds = []
        if rounded is not None:
            for wedge, trace in zip(rounded.wedges, rounded.traces, strict=True):
                scope = np.vstack([wedge, trace])
                left, top = scope.min(axis=0)
                right, bottom = scope.max(axis=0)
                corner_bounds.append(_vertices_on_page(np.float32([[left, top], [right, top], [right, bottom], [left, bottom]]),
                                                       width, height))
        if rounded is not None or (manual and mask.get(ROUNDED_CORNER_WEDGES)):
            full_reasons.append(ROUNDED_CORNER_REVIEW)
        scalloped_outline = manual_scalloped if manual else visible
        print_outline = scalloped_outline if SCALLOPED_PRINT_REVIEW in full_reasons and protection["passed"] and not (
            {"PRINT_POSSIBLY_OCCLUDED", "POSSIBLE_CONTENT_CUTOFF"} & set(full_reasons)
        ) else None
        adjacent_stock_outline = None
        if (enable_classical_retry and not manual and full_reasons == ["PRINT_BOUNDARY_AMBIGUOUS"]
                and not stock_suggestion and neighbor_masks and not any(mask.get(name) for name in _STOCK_REVIEW_CONFLICTS)):
            neighbor_seed = _adjacent_stock_neighbor(seed_points, neighbor_masks, width, height)
            if neighbor_seed is not None:
                candidate = suggest_adjacent_stock_outline(
                    page_image, seed_points, src_points, boundary_refinement, neighbor_seed,
                )
                if (candidate is not None
                        and not any(cv2.intersectConvexConvex(candidate, cv2.convexHull(neighbor))[0] > 1
                                    for neighbor in neighbor_polygons)
                        and _keeps_protected_preview_content(page_image, src_points, candidate, regions, protection)
                        and validate_protected_regions(candidate, regions, width=width, height=height)["passed"]):
                    adjacent_stock_outline = candidate
        result = _dual_crop_result(
            page_image, corrected, src_points, matrix, metadata, _explicit_quad(mask, "image_quad", width, height),
            full_reasons, trim_border=trim_border, auto_orient=auto_orient, orientation_model_path=orientation_model_path,
            manual_rotation=manual_rotation, page_regions=regions, protect_regions=protect_regions,
            owned_sampling=owned_sampling,
            neighbors=neighbor_polygons, corner_mount_detected=corner_mount_detected,
            print_outline=print_outline,
            print_edge_traces=print_edge_traces,
            paper_stock_margin_regions=mask.get("paper_stock_margin_regions"),
            motif_suggestion=_explicit_quad(mask, "scalloped_motif_suggestion", width, height)
            if print_outline is not None and trim_border else None,
            adjacent_stock_outline=adjacent_stock_outline,
            neighbor_masks=neighbor_masks,
        )
        modes = result.metadata["crop_modes"]
        if rounded is not None:
            mode = modes["full_print"]
            mode["suggested_edge_source"] = ROUNDED_CORNER_SOURCE
            mode["suggested_edge_traces"] = [_vertices_on_page(trace, width, height) for trace in rounded.traces]
            mode[ROUNDED_CORNER_WEDGES] = corner_bounds
            try:
                preview, evidence = _render_outline_preview(page_image, rounded.outline, np.asarray(mode["page_to_output"]))
                mode["outline_preview"] = {**evidence, "source": ROUNDED_CORNER_SOURCE}
                result = replace(result, full_print_preview=preview)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                pass  # Source traces still show the measured evidence.
        aspect_pending_after_mark_review = (
            (reviewed_album_mark or {}).get("source") == OFF_PRINT_ADVISORY
            and set(full_reasons) - {ROUNDED_CORNER_REVIEW} == {"PRINT_ASPECT_UNCERTAIN"}
        )
        held_by_protection = (not manual and protection.get("review_untrimmed")
                              and set(full_reasons) == {"PRINT_BOUNDARY_AMBIGUOUS", "POSSIBLE_CONTENT_CUTOFF", "PRINT_ASPECT_UNCERTAIN"})
        if (trim_border and visible is not None and (held_by_protection or aspect_pending_after_mark_review)
                and result.metadata["border_trim"]["method"] == "deferred_until_print_boundary_is_verified"):
            try:
                claims = [*neighbor_polygons, *(
                    _explicit_quad(neighbor, "warp_quad", width, height) for neighbor in neighbor_masks or []
                    if neighbor.get("warp_quad") is not None
                )]
                marks = held_print_motif_margin_marks(page_image, corrected, matrix, src_points, visible, claims)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                marks = []
            if marks:
                modes["image_only"]["margin_mark_source"] = HELD_PRINT_MOTIF_MARGIN_SOURCE
                modes["image_only"]["margin_mark_regions"] = [_vertices_on_page(mark, width, height) for mark in marks]
                modes["image_only"]["reason_codes"].insert(0, MOTIF_MARGIN_MARKS_REVIEW)
        if PRINT_SHARED_RIM_UNCERTAIN in full_reasons:
            cuts = shared_rim_reference or shared_rim_cut_segments(_vertices_on_page(src_points, width, height), boundary_refinement)
            if cuts:
                modes["full_print"]["shared_rim_cut_traces"] = cuts
        if inspection.occluding_neighbor is not None:
            contour = _vertices_on_page(inspection.occluding_neighbor, width, height)
            modes["full_print"]["neighbor_overlap_outline"] = [*contour, contour[0]]
        if missing_paper:
            modes["full_print"]["neighbor_content_loss_outlines"] = missing_paper
            modes["full_print"]["neighbor_review_source_size"] = {"width": width, "height": height}
            if backing_match_pixels:
                modes["full_print"]["neighbor_backing_color_match_pixels"] = backing_match_pixels
        if backed_edge and manual_boundary_hold:
            modes["full_print"]["suggested_edge_source"] = POSSIBLE_BACKED_PAPER_EDGE
        split_traces = mask.get("capture_split_review_traces")
        if not manual and (capture_traces := split_traces or mask.get("capture_edge_review_traces")):
            full_mode = modes["full_print"]
            full_mode.setdefault("suggested_edge_traces", []).extend(capture_traces)
            full_mode["suggested_edge_source"] = CAPTURE_SPLIT_REVIEW_SOURCE if split_traces else CAPTURE_EDGE_SOURCE
        if not manual and (scalloped_outline := mask.get("scalloped_ownership_outline")):
            full_mode = modes["full_print"]
            full_mode["suggested_warp_quad"] = mask["scalloped_ownership_group"]
            full_mode.setdefault("suggested_edge_traces", []).append([*scalloped_outline, scalloped_outline[0]])
        if not manual and mask.get("visible_polygon_source") == GRID_OUTLINE_REVIEW_SOURCE:
            modes["full_print"]["suggested_warp_quad"] = mask["visible_polygon"]
            for mode in modes.values():
                mode["status"] = "review_needed"
                if PRINT_EDGE_CONFIRMATION not in mode["reason_codes"]:
                    mode["reason_codes"].append(PRINT_EDGE_CONFIRMATION)
            rotation = int(result.metadata["orientation"].get("rotation_degrees") or 0) % 360
            try:
                outline = _explicit_quad(mask, "visible_polygon", width, height)
            except ValueError:
                outline = None
            preview = None if outline is None else _render_review_quad_preview(
                page_image, outline, rotation, neighbor_polygons, regions, camera_matrix, page_plane,
                source=GRID_OUTLINE_REVIEW_SOURCE,
            )
            if preview is not None:
                pixels, modes["full_print"]["outline_preview"] = preview
                result = replace(result, full_print_preview=pixels)
            return result
        uncertain = {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_BOUNDARY_UNSUPPORTED", "PRINT_OUTER_RIM_UNCERTAIN"}
        paper_grid = mask.get("paper_grid_review") or {}
        grid_outline = not manual and paper_grid.get("outline") and "cut" not in paper_grid
        if grid_outline or (enable_classical_retry and not manual and mask.get("detector_source") == "contrast_region"
                and not mask.get("warp_quad") and not print_edge_traces and uncertain.intersection(full_reasons)
                and CAPTURE_EDGE_HOLD not in full_reasons
                and all(mode["status"] == "review_needed" for mode in modes.values())):
            try:
                alternative = (_explicit_quad(paper_grid, "outline", width, height) if grid_outline
                               else suggest_outer_print_outline(page_image, seed_points, src_points, neighbor_polygons))
                if alternative is not None:
                    _validate_quad(alternative, width, height)
            except (ValueError, cv2.error):
                alternative = None
            if alternative is not None:
                modes["full_print"]["suggested_warp_quad"] = _vertices_on_page(alternative, width, height)
                for mode in modes.values():
                    mode["status"] = "review_needed"
                    if PRINT_EDGE_CONFIRMATION not in mode["reason_codes"]:
                        mode["reason_codes"].append(PRINT_EDGE_CONFIRMATION)
                if grid_outline and _keeps_protected_preview_content(page_image, src_points, alternative, regions, protection):
                    rotation = int(result.metadata["orientation"].get("rotation_degrees") or 0) % 360
                    preview = _render_disputed_grid_preview(
                        page_image, alternative, paper_grid, rotation, neighbor_masks or [], regions, camera_matrix, page_plane,
                    )
                    if preview is not None:
                        pixels, modes["full_print"]["outline_preview"] = preview
                        result = replace(result, full_print_preview=pixels)
        full_mode = modes["full_print"]
        if PROJECTIVE_PRINT_OUTLINE_REVIEW in full_reasons and (mounted_traces := mask.get("mounted_paper_edge_traces")):
            full_mode["suggested_edge_traces"] = mounted_traces
            full_mode["suggested_edge_source"] = DARK_MOUNTED_EDGE_SOURCE
            if enable_classical_retry and not manual and all(mode["status"] == "review_needed" for mode in modes.values()):
                try:
                    traces = [normalized_vertices_to_pixels([(p["x"], p["y"]) for p in trace], width, height)
                              for trace in mounted_traces if len(trace) == 2]
                    strip = suggest_mounted_strip_preview(page_image, src_points, traces, neighbor_polygons, regions)
                    if strip is not None:
                        pixels, preview = _render_outline_preview(
                            page_image, strip.outline, np.asarray(full_mode["page_to_output"]),
                        )
                        full_mode["outline_preview"] = {
                            **preview, "source": POSSIBLE_MOUNTED_EDGE_STRIP, "source_side": strip.side,
                            "source_search_band": _vertices_on_page(strip.search_band, width, height),
                        }
                        result = replace(result, full_print_preview=pixels)
                except (ValueError, cv2.error, np.linalg.LinAlgError):
                    pass
        if (competing_segments and not manual and "PRINT_BOUNDARY_AMBIGUOUS" in full_reasons
                and not full_mode.get("suggested_edge_traces") and not full_mode.get("suggested_warp_quad")
                and not (mask.get("film_grid_review_seam") or mask.get("paper_grid_review"))
                and set(full_reasons) - {PARTIAL_PRINT_EXTENT_REVIEW}
                <= {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_CORNERS_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"}):
            full_mode["suggested_edge_traces"] = [
                _vertices_on_page(np.asarray(segment), width, height) for segment in competing_segments
            ]
            full_mode["suggested_edge_source"] = COMPETING_PRINT_EDGES
            if (enable_classical_retry and mask.get("detector_source") == "yolo_onnx" and not mask.get("warp_quad")
                    and not any(mask.get(key) for key in _STOCK_REVIEW_CONFLICTS)):
                spans = suggest_exposed_overlap_edges(page_image, src_points, boundary_refinement, neighbor_polygons)
                if spans:
                    full_mode["exposed_overlap_edge_traces"] = [_vertices_on_page(span, width, height) for span in spans]
        elif (enable_classical_retry and not manual and "PRINT_BOUNDARY_UNSUPPORTED" in full_reasons
              and set(full_reasons) <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"}
              and all(mode["status"] == "review_needed" for mode in modes.values())
              and not full_mode.get("suggested_edge_traces") and not full_mode.get("suggested_warp_quad")
              and not any(mask.get(key) for key in ("warp_quad", "paper_grid_review", "film_grid_review_seam",
                                                  "crossing_print_suggestion", "fragment_outline_suggestion"))):
            traces = suggest_partial_backed_edge(page_image, src_points, boundary_refinement, neighbor_polygons)
            if traces:
                full_mode["suggested_edge_traces"] = [_vertices_on_page(trace, width, height) for trace in traces]
                full_mode["suggested_edge_source"] = POSSIBLE_BACKED_PAPER_EDGE
        uncertain_edges = set(full_reasons)
        slanted = (mask.get("detector_source") == "yolo_onnx"
                   and not any(mask.get(key) for key in _STOCK_REVIEW_CONFLICTS)
                   and uncertain_edges == {"PRINT_OUTER_RIM_UNCERTAIN", "PRINT_BOUNDARY_UNSUPPORTED",
                                           "PRINT_CORNERS_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"})
        if (enable_classical_retry and not manual and "PRINT_BOUNDARY_UNSUPPORTED" in uncertain_edges
                and (slanted or uncertain_edges <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN",
                                                       "PRINT_ASPECT_UNCERTAIN", "POSSIBLE_CONTENT_CUTOFF"})
                and all(mode["status"] == "review_needed" for mode in modes.values())
                and not any(full_mode.get(key) for key in ("suggested_edge_traces", "suggested_warp_quad", "outline_preview"))
                and not any(mask.get(key) for key in ("warp_quad", "paper_grid_review", "film_grid_review_seam",
                                                      "crossing_print_suggestion", "fragment_outline_suggestion"))):
            comparison = suggest_slanted_print_edge_comparison if slanted else suggest_overextended_print_outline
            alternative = comparison(page_image, src_points, boundary_refinement)
            if alternative is not None:
                outline, traces = alternative
                rotation = int(result.metadata["orientation"].get("rotation_degrees") or 0) % 360
                preview = None
                marks = slanted and motif_margin_marks(
                    page_image, corrected, cv2.perspectiveTransform(outline.reshape(1, 4, 2), matrix)[0],
                    rectified_to_page, preview_guard=True,
                )
                if not marks and _keeps_protected_preview_content(page_image, src_points, outline, regions, protection):
                    comparison_neighbors = _neighbors_for_edge_comparison(
                        page_image, src_points, outline, neighbor_masks or [],
                    ) if slanted else neighbor_polygons
                    preview = _render_review_quad_preview(
                        page_image, outline, rotation, comparison_neighbors, regions, camera_matrix, page_plane,
                        source=POSSIBLE_OVEREXTENDED_PRINT_EDGE, prior_protection=protection,
                    )
                if preview is not None:
                    pixels, full_mode["outline_preview"] = preview
                    if len(comparison_neighbors) < len(neighbor_polygons):
                        full_mode["outline_preview"]["neighbor_edge_in_review"] = True
                    full_mode["suggested_edge_traces"] = [_vertices_on_page(trace, width, height) for trace in traces]
                    full_mode["suggested_edge_source"] = POSSIBLE_OVEREXTENDED_PRINT_EDGE
                    result = replace(result, full_print_preview=pixels)
        if (enable_classical_retry and not manual and mask.get("source") == "classical_border_quad"
                and mask.get("detector_source") == "edge_contour" and not mask.get("warp_quad")
                and "PRINT_BOUNDARY_AMBIGUOUS" in full_reasons
                and set(full_reasons) <= {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_ASPECT_UNCERTAIN"}
                and not any(full_mode.get(key) for key in ("suggested_warp_quad", "outline_preview"))
                and full_mode.get("suggested_edge_source") in {None, COMPETING_PRINT_EDGES}
                and not any(mask.get(name) for name in _STOCK_REVIEW_CONFLICTS)):
            candidate = suggest_inward_corner_outline(page_image, src_points, boundary_refinement)
            if candidate is not None and _keeps_protected_preview_content(page_image, src_points, candidate, regions, protection):
                rotation = int(result.metadata["orientation"].get("rotation_degrees") or 0) % 360
                preview = _render_review_quad_preview(
                    page_image, candidate, rotation, neighbor_polygons, regions, camera_matrix, page_plane,
                    source=INWARD_CORNER_REVIEW_SOURCE,
                )
                if preview is not None:
                    pixels, full_mode["outline_preview"] = preview
                    full_mode["suggested_warp_quad"] = _vertices_on_page(candidate, width, height)
                    full_mode["suggested_outline_source"] = INWARD_CORNER_REVIEW_SOURCE
                    result = replace(result, full_print_preview=pixels)
        if adjacent_stock_outline is not None and not full_mode.get("suggested_warp_quad"):
            full_mode["suggested_warp_quad"] = _vertices_on_page(adjacent_stock_outline, width, height)
            full_mode["suggested_outline_source"] = ADJACENT_STOCK_REVIEW_SOURCE
        if (enable_classical_retry and not manual and mask.get("detector_source") == "yolo_onnx"
                and "PRINT_BOUNDARY_UNSUPPORTED" in full_reasons
                and set(full_reasons) <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"}
                and all(mode["status"] == "review_needed" for mode in modes.values())
                and not any(full_mode.get(key) for key in ("suggested_edge_traces", "suggested_warp_quad", "outline_preview"))
                and not any(mask.get(key) for key in ("warp_quad", *_STOCK_REVIEW_CONFLICTS))):
            spans = suggest_inward_print_edge(page_image, src_points, boundary_refinement, neighbor_polygons)
            if spans:
                full_mode["suggested_edge_traces"] = [_vertices_on_page(span, width, height) for span in spans]
                full_mode["suggested_edge_source"] = POSSIBLE_INWARD_PRINT_EDGE
        if (enable_classical_retry and not manual and all(mode["status"] == "review_needed" for mode in modes.values())
                and not any(full_mode.get(key) for key in ("suggested_edge_traces", "suggested_warp_quad", "outline_preview"))
                and not any(mask.get(key) for key in ("warp_quad", *_STOCK_REVIEW_CONFLICTS))):
            preview_source, measure = None, None
            if (protection["passed"] and mask.get("source") == "classical_border_quad"
                    and mask.get("detector_source") == "edge_contour" and "PRINT_BOUNDARY_UNSUPPORTED" in full_reasons
                    and set(full_reasons) <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"}):
                preview_source, measure = POSSIBLE_BOWED_PRINT_EDGE, suggest_bowed_print_preview
            elif ((protection["passed"] or protection.get("review_untrimmed")) and mask.get("source") == "yolo_seg_mask"
                  and mask.get("detector_source") == "yolo_onnx" and "PRINT_BOUNDARY_AMBIGUOUS" in full_reasons
                  and set(full_reasons) <= {"PRINT_BOUNDARY_AMBIGUOUS", "POSSIBLE_CONTENT_CUTOFF", "PRINT_ASPECT_UNCERTAIN"}):
                preview_source, measure = POSSIBLE_DISPLACED_PRINT_EDGES, suggest_displaced_print_preview
            elif (protection["passed"] and mask.get("source") == "yolo_seg_mask"
                  and mask.get("detector_source") == "yolo_onnx"
                  and {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN"} <= set(full_reasons)
                  and set(full_reasons) <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_CORNERS_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"}):
                preview_source, measure = POSSIBLE_ALBUM_LAYER_EDGE, suggest_album_layer_preview
            elif ((protection["passed"] or protection.get("review_untrimmed")) and mask.get("source") == "yolo_seg_mask"
                  and mask.get("detector_source") == "yolo_onnx"
                  and {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_CORNERS_UNCERTAIN"} <= set(full_reasons)
                  and set(full_reasons) <= {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_CORNERS_UNCERTAIN",
                                           "POSSIBLE_CONTENT_CUTOFF", "PRINT_ASPECT_UNCERTAIN"}):
                preview_source, measure = POSSIBLE_LOCAL_PRINT_EDGE, suggest_local_print_edge_preview
            if measure is not None:
                try:
                    options = {"protected_regions": regions} if preview_source in (
                        POSSIBLE_ALBUM_LAYER_EDGE, POSSIBLE_DISPLACED_PRINT_EDGES,
                    ) else {}
                    observed = measure(page_image, src_points, boundary_refinement, neighbor_polygons, **options)
                    if observed is not None:
                        preview, evidence = _render_outline_preview(
                            page_image, observed.outline, np.asarray(full_mode["page_to_output"]),
                        )
                        full_mode["outline_preview"] = {
                            **evidence, "source": preview_source, "source_side": observed.side,
                            "source_search_band": _vertices_on_page(observed.search_band, width, height),
                        }
                        if observed.inset_band is not None:
                            full_mode["outline_preview"]["source_inset_band"] = _vertices_on_page(observed.inset_band, width, height)
                        full_mode["suggested_edge_traces"] = [
                            _vertices_on_page(path, width, height) for path in observed.traces
                        ]
                        full_mode["suggested_edge_source"] = preview_source
                        result = replace(result, full_print_preview=preview)
                except (ValueError, cv2.error, np.linalg.LinAlgError):
                    pass  # Keep the original held preview when local recovery cannot be rendered.
        if (cuts := full_mode.get("shared_rim_cut_traces")) and not any(
            full_mode.get(key) for key in ("suggested_edge_traces", "suggested_warp_quad", "outline_preview")
        ):
            full_mode["suggested_edge_traces"] = cuts
            full_mode["suggested_edge_source"] = SHARED_PRINT_RIM_SOURCE
        if (enable_classical_retry and not manual and protection["passed"]
                and mask.get("source") == "yolo_seg_mask" and mask.get("detector_source") == "yolo_onnx"
                and (mask.get("boundary_evidence") or {}).get("score", 0.) >= .65
                and "PRINT_BOUNDARY_UNSUPPORTED" in uncertain_edges
                and uncertain_edges <= {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_POSSIBLY_OCCLUDED",
                                        "PRINT_CORNERS_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"}
                and all(mode["status"] == "review_needed" for mode in modes.values())
                and not any(mask.get(key) for key in ("warp_quad", *_STOCK_REVIEW_CONFLICTS))
                and not any(full_mode.get(key) for key in ("suggested_edge_traces", "suggested_warp_quad", "outline_preview"))):
            try:
                spans = embossed_carrier_paper_traces(page_image, src_points, boundary_refinement)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                spans = []
            if spans:
                full_mode["suggested_edge_traces"] = [_vertices_on_page(span, width, height) for span in spans]
                full_mode["suggested_edge_source"] = POSSIBLE_EMBOSSED_ALBUM_PAPER_EDGES
        if (not manual and visible is not None and protection.get("review_untrimmed")
                and set(full_reasons) <= {"PRINT_BOUNDARY_AMBIGUOUS", "POSSIBLE_CONTENT_CUTOFF", "PRINT_ASPECT_UNCERTAIN"}
                and not full_mode.get("outline_preview") and not any(mask.get(key) for key in _STOCK_REVIEW_CONFLICTS)):
            try:
                marks = off_print_advisory_regions(page_image, src_points, regions, protection["violations"], neighbor_polygons)
            except (ValueError, cv2.error):
                marks = []
            if marks:
                result.metadata["off_print_advisory_regions"] = marks
        if (not manual and visible is not None and mask.get("detector_source") == "yolo_onnx"
                and set(full_reasons) == {"PRINT_BOUNDARY_UNSUPPORTED", "PRINT_ASPECT_UNCERTAIN"}
                and all(mode["status"] == "review_needed" for mode in modes.values())
                and not any(full_mode.get(key) for key in ("suggested_edge_traces", "suggested_warp_quad", "outline_preview"))
                and not any(mask.get(key) for key in ("warp_quad", *_STOCK_REVIEW_CONFLICTS))):
            try:
                opening = album_opening_traces(page_image, visible, boundary_refinement)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                opening = []
            if opening:
                full_mode["suggested_edge_traces"] = [_vertices_on_page(trace, width, height) for trace in opening]
                full_mode["suggested_edge_source"] = POSSIBLE_ALBUM_OPENING_EDGE
        strip_reasons = set(full_reasons)
        if (not manual and visible is not None and mask.get("detector_source") == "yolo_onnx"
                and full_mode["status"] == "review_needed" and not full_mode.get("outline_preview")
                and len(strip_reasons) == 3
                and strip_reasons - {"PRINT_BOUNDARY_AMBIGUOUS", "PRINT_BOUNDARY_UNSUPPORTED"}
                == {"PRINT_ASPECT_UNCERTAIN", "POSSIBLE_CONTENT_CUTOFF"}
                and not any(mask.get(key) for key in _STOCK_REVIEW_CONFLICTS)):
            try:
                strip = source_strip_outline(page_image, src_points, boundary_refinement, neighbor_polygons)
                if strip is not None:
                    _validate_quad(strip, width, height)
                    full_mode["source_strip_review"] = {
                        "source_size": {"width": width, "height": height},
                        "source_outline": _vertices_on_page(strip, width, height),
                    }
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                pass  # The existing held crop already supplies its ordinary review.
        extent_traces = mask.get(PARTIAL_PRINT_EXTENT_TRACES)
        if not manual and extent_traces and not any(
            review_traces_intersect_claim(extent_traces, neighbor, width, height) for neighbor in neighbor_masks or []
        ):
            full_mode[PARTIAL_PRINT_EXTENT_TRACES] = extent_traces
        return result
    rectified_untrimmed = corrected.copy()
    if trim_border and protection["passed"]:
        before_shape = corrected.shape[:2]
        page_regions_in_crop = _rectified_protected_regions(
            regions,
            page_width=width,
            page_height=height,
            matrix=matrix,
            target_width=target_width,
            target_height=target_height,
        )
        local_regions = detect_protected_regions(corrected) if protect_regions else []
        border_result = find_physical_border(
            analysis,
            reference_box=(left, top, left + target_width, top + target_height),
            protected_regions=merge_protected_regions(page_regions_in_crop, local_regions),
        )
        trim_metadata = border_result.metadata()
        if border_result.crop_quad is not None:
            quad = border_result.crop_quad
            print_quad = cv2.perspectiveTransform(quad.reshape(1, 4, 2), rectified_to_page)[0]
            # The sampled context bounds also enclose every permitted manual crop.
            in_context = np.all(quad >= [-left, -top]) and np.all(
                quad <= [target_width + right - 1, target_height + bottom - 1]
            )
            veto = None
            if not in_context or not cv2.isContourConvex(quad) or cv2.contourArea(quad, oriented=True) <= 0:
                veto = "BORDER_TRIM_GEOMETRY_VETO"
            elif protect_regions and not validate_protected_regions(
                print_quad, regions, width=width, height=height
            )["passed"]:
                veto = "BORDER_PROTECTED_CONTENT_VETO"
            if veto:
                trim_metadata.update(
                    applied=False, confidence=0.0, reason_codes=[veto], crop_quad=None,
                    offsets=dict.fromkeys(("top", "bottom", "left", "right"), 0),
                )
            else:
                corrected = rectify_crop_quad(page_image, quad, page_to_rectified=matrix)
                metadata["warp_quad"] = _vertices_on_page(print_quad, width, height)
                metadata["target_height"], metadata["target_width"] = corrected.shape[:2]
                density = np.sqrt(cv2.contourArea(print_quad) / (corrected.shape[0] * corrected.shape[1]))
                metadata["source_pixels_per_output_pixel"] = round(float(density), 4)
        metadata["border_trim"] = {
            "before": {"height": before_shape[0], "width": before_shape[1]},
            "after": {"height": corrected.shape[0], "width": corrected.shape[1]},
            **trim_metadata,
        }
        # Preserving an ambiguous edge is safe; confidence describes only a
        # mutation that was actually applied.
        metadata["border_trim_confidence"] = border_result.confidence if trim_metadata["applied"] else 1.0
    else:
        metadata["border_trim"] = {
            "method": "disabled",
            "applied": False,
            "reason_codes": [
                "BORDER_TRIM_UNCERTAIN" if trim_border else "BORDER_TRIM_DISABLED_PRESERVE_SOURCE"
            ],
        }
        metadata["border_trim_confidence"] = None
    corrected, orientation = _orient_crop(
        corrected, auto_orient, orientation_model_path, manual_rotation=manual_rotation,
    )
    rectified_untrimmed = rotate_image(rectified_untrimmed, int(orientation.get("rotation_degrees") or 0))
    metadata["orientation"] = orientation
    metadata["orientation_confidence"] = orientation.get("confidence")
    metadata["untrimmed_shape"] = {
        "height": int(rectified_untrimmed.shape[0]),
        "width": int(rectified_untrimmed.shape[1]),
    }
    metadata["final_shape"] = {"height": int(corrected.shape[0]), "width": int(corrected.shape[1])}
    return CropResult(rectified_untrimmed=rectified_untrimmed, image=corrected, metadata=metadata)


def crop_and_correct_photo_with_metadata(
    page_image: np.ndarray,
    bounding_box: dict,
    mask: dict | None = None,
    *,
    trim_border: bool = True,
    auto_orient: bool = True,
    orientation_model_path: Path | None = None,
    protected_regions: list[dict] | None = None,
    protect_regions: bool = True,
) -> tuple[np.ndarray, dict]:
    """Backward-compatible tuple facade around the typed crop result."""
    result = correct_photo_crop(
        page_image,
        bounding_box,
        mask,
        trim_border=trim_border,
        auto_orient=auto_orient,
        orientation_model_path=orientation_model_path,
        protected_regions=protected_regions,
        protect_regions=protect_regions,
    )
    return result.image, result.metadata


def save_corrected_photo(page_image: np.ndarray, bounding_box: dict, mask: dict | None, output_path: Path) -> None:
    corrected = crop_and_correct_photo(page_image, bounding_box, mask)
    save_jpeg(output_path, corrected)
