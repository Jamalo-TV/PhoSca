from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.album_windows import ALBUM_WINDOW_OCCLUDED, ALBUM_WINDOW_SOURCE, ALBUM_WINDOW_STOCK_UNCERTAIN
from app.pipeline.border_removal import find_physical_border, wide_paper_motif_proposal
from app.pipeline.crop_sampling import OWNED_BILINEAR, warp_owned_crop
from app.pipeline.geometry import (
    normalized_polygon_overlap,
    normalized_vertices_to_pixels,
    order_quad_cyclic,
    pixel_vertices_to_normalized,
    quad_from_polygon,
    rectify_crop_quad,
)
from app.pipeline.image_ops import normalized_box_to_pixels, save_jpeg
from app.pipeline.orientation import correct_photo_orientation, rotate_image
from app.pipeline.page_plane import PRINT_ASPECT_REFERENCE_REVIEW, MeasuredPrintReference, measured_page_aspect
from app.pipeline.print_edges import (
    continuous_motif_rim,
    missing_corner_material,
    observed_rim_points,
    print_corner_coverage,
    refine_print_boundary,
)
from app.pipeline.protected_regions import detect_protected_regions, merge_protected_regions, validate_protected_regions
from app.pipeline.segmentation_filter import refine_visible_paper_rim

_PRINT_BOUNDARY_SOURCE = "physical_print_boundary"
_PAPER_RIM_SOURCE = "visible_paper_rim"
_REFINED_VISIBILITY_SOURCES = {_PRINT_BOUNDARY_SOURCE, _PAPER_RIM_SOURCE}
_LOCALIZED_OCCLUSION_REASONS = {"PRINT_CORNERS_UNCERTAIN", "PRINT_POSSIBLY_OCCLUDED"}


@dataclass(frozen=True)
class CropResult:
    """Faithful rectified pixels plus any optional, reversible derivative."""

    rectified_untrimmed: np.ndarray
    image: np.ndarray
    metadata: dict
    image_only: np.ndarray | None = None


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


def validate_review_quads(mask: dict, width: int, height: int) -> None:
    """Validate authoritative reviewer vertices before storing a correction revision."""
    measured = mask.get("print_size_mm") is not None
    outer = _explicit_quad(mask, "warp_quad", width, height, preserve_width_edge=measured)
    if measured and outer is None:
        raise ValueError("measured_print_requires_warp_quad")
    inner = _explicit_quad(mask, "image_quad", width, height)
    if inner is None:
        return
    if outer is None or any(cv2.pointPolygonTest(outer, tuple(point), True) < -0.02 for point in inner):
        raise ValueError("image_quad_outside_print")


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
_PAGE_FACE_SUGGESTIONS = {"opencv_haarcascade_frontalface_default", "opencv_haarcascade_profileface"}


def _advisory_face_overlap(protection: dict, allow_page_face_overlap: bool) -> bool:
    return allow_page_face_overlap and all(
        violation["source"] in _PAGE_FACE_SUGGESTIONS for violation in protection["violations"]
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


def _crop_dimensions(
    points: np.ndarray, width: int, height: int, *, metric: bool,
    camera_matrix: np.ndarray | None, print_aspect_ratio: float | None,
) -> tuple[int, int, dict, list[str]]:
    top, right, bottom, left = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    target_width = round(float(max(top, bottom))) + 1
    target_height = round(float(max(left, right))) + 1
    aspect = None
    method = "projected_side_lengths"
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
        # A physical measurement fixes the span between the rectified corner pixel centers.
        offset = 1 if print_aspect_ratio is not None else 0
        scale = max((target_width - offset) / aspect, target_height - offset)
        target_width, target_height = round(scale * aspect) + offset, round(scale) + offset
    return target_width, target_height, {"method": method, "estimated_ratio": aspect}, reasons


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
    if mask.get("visible_polygon_source") in _REFINED_VISIBILITY_SOURCES:
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
    if mask is None:
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


def _neighbor_occlusion(
    page_image: np.ndarray, src_points: np.ndarray, own: np.ndarray | None, neighbors: list[np.ndarray],
    *, inferred_sides: tuple[int, ...] = (), observed_sides: dict[int, tuple[np.ndarray, float]] | None = None,
) -> bool:
    if not neighbors:
        return False
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
                    return True
                rows, columns = np.nonzero(overlap)
                claimed = np.column_stack((columns, rows)) / ratio + low
                confirmed_stock = observed_rim_points(page_image, src_points, claimed, observed_sides)
                if np.count_nonzero(~confirmed_stock) >= minimum_overlap:
                    return True
            # A single exterior contour can cover an interior occluder hole.
            # A distinct, substantially smaller enclosed proposal is review evidence.
            if np.all(lower > low) and np.all(upper < high):
                area = cv2.contourArea(polygon) * ratio ** 2
                if minimum_overlap <= area <= 0.80 * footprint_area and all(
                    cv2.pointPolygonTest(src_points, tuple(point), True) >= reach for point in polygon
                ):
                    return True
        # Paper color can continue across two touching prints and make the other
        # print's ink edge appear to be this print's physical boundary.
        if any(
            sum(cv2.pointPolygonTest(polygon, tuple(point), True) >= -reach for point in samples) >= 8
            for samples in inferred_edge_samples
        ):
            return True
    return False


def _orient_crop(image: np.ndarray, enabled: bool, model_path: Path | None) -> tuple[np.ndarray, dict]:
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


def _dual_crop_result(
    page: np.ndarray, full_image: np.ndarray, source_quad: np.ndarray, matrix: np.ndarray, metadata: dict,
    inner: np.ndarray | None, full_reasons: list[str], *, trim_border: bool, auto_orient: bool,
    orientation_model_path: Path | None, page_regions: list[dict], protect_regions: bool, owned_sampling: bool,
    neighbors: list[np.ndarray], corner_mount_detected: bool,
) -> CropResult:
    page_height, page_width = page.shape[:2]
    height, width = full_image.shape[:2]
    image_only = None
    image_mode = {"status": "review_needed", "reason_codes": full_reasons.copy()}
    diagnostics = {"method": "deferred_until_print_boundary_is_verified", "applied": False}
    adjusted, orientation = _orient_crop(full_image, auto_orient, orientation_model_path)
    rotation = int(orientation.get("rotation_degrees") or 0) % 360
    full_mode = _geometry_for_mode(source_quad, matrix, full_image, page_width, page_height, rotation, full_reasons)
    if owned_sampling:
        full_mode["sampling"] = OWNED_BILINEAR

    independent = corner_mount_detected and set(full_reasons) <= _LOCALIZED_OCCLUSION_REASONS
    reference_review = full_reasons == [PRINT_ASPECT_REFERENCE_REVIEW]
    if not full_reasons or independent or reference_review:
        quad, diagnostics, image_reasons = _inner_crop_plan(
            full_image, matrix, inner, trim_border=trim_border, page_width=page_width, page_height=page_height,
            page_regions=page_regions, protect_regions=protect_regions, allow_wide_search=not independent,
        )
        wide_rim = diagnostics["method"] == "wide_paper_rim"
        suggested_quad = None
        all_edges = diagnostics.get("edges", {})
        four_rim_edges = len(all_edges) == 4 and all(edge.get("is_physical_border") for edge in all_edges.values())
        if independent and (quad is None or (inner is None and not four_rim_edges)):
            image_reasons.append("IMAGE_VISIBILITY_UNCERTAIN")
            quad = None
        if not image_reasons and quad is None:
            image_only = adjusted
            image_mode = {**full_mode, "method": diagnostics["method"]}
        elif quad is not None:
            try:
                _validate_quad(quad, width, height)
                source = cv2.perspectiveTransform(quad.reshape(1, 4, 2), np.linalg.inv(matrix))[0]
                if independent or wide_rim or (inner is None and four_rim_edges):
                    # The rectified scan corroborates the paper rim. Locate its last
                    # boundary in source pixels, where resampling cannot blur the seam.
                    radius = max(3, min(16, round(max(width, height) / 450))) if wide_rim else 3
                    refined, native_evidence, native_reasons = refine_print_boundary(page, source, search_radius=radius)
                    diagnostics["native_boundary_refinement"] = native_evidence
                    native_verified = refined is not None and not native_reasons
                    if (independent or wide_rim) and (
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
                measured = metadata["aspect_ratio_estimation"]["method"] in {"known_print_aspect", "measured_page_reference"}
                if (independent or wide_rim) and not image_reasons and not _independent_motif_visible(page, source, neighbors):
                    image_reasons.append("IMAGE_VISIBILITY_UNCERTAIN")
                if (measured or independent or wide_rim) and not _is_rectangular_in_measured_plane(quad):
                    image_reasons.append("IMAGE_RECTANGLE_INCONSISTENT")
                if protect_regions and not validate_protected_regions(source, page_regions, width=page_width, height=page_height)["passed"]:
                    image_reasons.append("BORDER_PROTECTED_CONTENT_VETO")
                if not image_reasons and wide_rim:
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
                if not image_reasons:
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
                            full_reasons if reference_review else [],
                        )
                        image_mode["sampling"] = OWNED_BILINEAR
                        if independent:
                            image_mode["independent_of_full_print"] = True
                        image_only = rotate_image(pixels, rotation)
                    else:
                        image_reasons.append("IMAGE_BOUNDARY_SAMPLING_UNCERTAIN")
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                image_reasons.append("IMAGE_QUAD_UNSTABLE")
        if image_reasons:
            diagnostics["applied"] = False
            image_mode = {"status": "review_needed", "reason_codes": list(dict.fromkeys(image_reasons))}
            if suggested_quad is not None:
                image_mode["suggested_warp_quad"] = suggested_quad

    metadata.update(
        crop_modes={"full_print": full_mode, "image_only": image_mode},
        orientation=orientation,
        orientation_confidence=orientation.get("confidence"),
        final_shape=full_mode["shape"],
        untrimmed_shape=full_mode["shape"],
        border_trim={**diagnostics, "applied": bool(diagnostics.get("applied") and image_only is not None)},
        border_trim_confidence=diagnostics.get("confidence") if diagnostics.get("applied") else None,
    )
    return CropResult(rectified_untrimmed=adjusted, image=adjusted, metadata=metadata, image_only=image_only)


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
    print_aspect_ratio: float | None = None,
    measured_page_references: list[MeasuredPrintReference] | None = None,
    neighbor_masks: list[dict] | None = None,
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
    quad_metrics = _validate_quad(src_points, width, height)
    boundary_refinement: dict = {}
    full_reasons: list[str] = []
    visible = None
    neighbor_polygons: list[np.ndarray] = []
    corner_mount_detected = False
    manual = not allow_border_expansion or str(mask.get("source") or "").startswith("manual")
    regions = [] if not protect_regions else (
        protected_regions if protected_regions is not None else detect_protected_regions(page_image)
    )
    if dual_mode and not manual and mask.get("warp_quad") is None:
        refined, boundary_refinement, full_reasons = refine_print_boundary(page_image, src_points)
        if enable_classical_retry and "PRINT_OUTER_RIM_UNCERTAIN" in full_reasons:
            hypothesis = _outer_contour_hypothesis(page_image, src_points)
            if hypothesis is not None:
                recovered, evidence, remaining = refine_print_boundary(page_image, hypothesis)
                if recovered is not None and not remaining:
                    refined, full_reasons = recovered, []
                    boundary_refinement = {**evidence, "seed": "local_outer_contour"}
        if refined is not None and "PRINT_OUTER_RIM_UNCERTAIN" not in full_reasons:
            refined_protection = validate_protected_regions(refined, regions, width=width, height=height)
            if not refined_protection["passed"] and not _advisory_face_overlap(refined_protection, allow_page_face_overlap):
                full_reasons.append("PRINT_BOUNDARY_PROTECTED_CONTENT_VETO")
            else:
                src_points = refined
                quad_metrics = _validate_quad(src_points, width, height)
    if dual_mode:
        if mask.get("album_window_occluded") is True:
            full_reasons.append(ALBUM_WINDOW_OCCLUDED)
        elif mask.get("source") == ALBUM_WINDOW_SOURCE:
            full_reasons.append(ALBUM_WINDOW_STOCK_UNCERTAIN)
        if not manual and np.any((src_points < 0.5) | (src_points > np.array([width, height]) - 1.5)):
            full_reasons.append("PRINT_AT_CAPTURE_EDGE")
        visible = _proposal_polygon(mask, width, height)
        neighbor_polygons = [
            points for neighbor in neighbor_masks or []
            if (points := _proposal_polygon(neighbor, width, height)) is not None
        ]
        edge_evidence = [boundary_refinement.get(side, {}) for side in ("top", "right", "bottom", "left")]
        inferred_sides = tuple(
            index for index, evidence in enumerate(edge_evidence) if evidence.get("matched_print_rim")
        )
        observed_sides = {
            index: (np.array(evidence["observed_rim_lab"]), evidence["observed_rim_depth_px"])
            for index, evidence in enumerate(edge_evidence)
            if not full_reasons and evidence.get("support", 0) >= 0.9
            and (evidence.get("corner_support") or 0) >= 0.8
            and "observed_rim_lab" in evidence
            and index not in inferred_sides
        }
        neighbor_occlusion = _neighbor_occlusion(
            page_image, src_points, visible, neighbor_polygons, inferred_sides=inferred_sides,
            observed_sides=observed_sides,
        )
        if _clipped_visible_corner(src_points, visible) or neighbor_occlusion:
            full_reasons.append("PRINT_POSSIBLY_OCCLUDED")
        if not full_reasons and missing_corner_material(page_image, src_points):
            full_reasons.append("PRINT_CORNERS_UNCERTAIN")
        if not neighbor_occlusion and set(full_reasons) <= _LOCALIZED_OCCLUSION_REASONS:
            covered, corner_mount_detected = print_corner_coverage(page_image, src_points)
            if covered and not full_reasons:
                full_reasons.append("PRINT_POSSIBLY_OCCLUDED")
    protection = (
        validate_protected_regions(src_points, regions, width=width, height=height)
        if protect_regions
        else {"passed": True, "regions_checked": 0, "violations": []}
    )
    if not protection["passed"]:
        # A page-level face box can extend beyond the physical edge of a print.
        if not _advisory_face_overlap(protection, allow_page_face_overlap):
            raise ValueError("protected_region_cut")
        protection["review_untrimmed"] = True
        full_reasons.append("POSSIBLE_CONTENT_CUTOFF")
    boundary_verified = bool(boundary_refinement and not full_reasons and visible is not None)
    review_visible_rim = bool(
        protection["passed"] and boundary_refinement and visible is not None and full_reasons
        and "PRINT_BOUNDARY_PROTECTED_CONTENT_VETO" not in full_reasons
    )
    target_width, target_height, aspect, aspect_reasons = _crop_dimensions(
        src_points, width, height, metric=dual_mode, camera_matrix=camera_matrix, print_aspect_ratio=print_aspect_ratio,
    )
    if (not full_reasons and aspect_reasons == ["PRINT_ASPECT_UNCERTAIN"] and camera_matrix is None
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
    owned_sampling = dual_mode and (not full_reasons or full_reasons == [PRINT_ASPECT_REFERENCE_REVIEW])
    if owned_sampling:
        analysis, supported = warp_owned_crop(page_image, matrix, (target_width, target_height))
        if not supported:
            full_reasons.append("PRINT_BOUNDARY_SAMPLING_UNCERTAIN")
    else:
        analysis = cv2.warpPerspective(
            page_image, page_to_analysis, (left + target_width + right, top + target_height + bottom),
            borderMode=cv2.BORDER_CONSTANT if dual_mode else cv2.BORDER_REPLICATE,
        )
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
        "auto_orient": auto_orient,
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
        if boundary_verified:
            raw_vertices = pixel_vertices_to_normalized(visible, width, height)
            measured_vertices = pixel_vertices_to_normalized(src_points, width, height)
            overlap = normalized_polygon_overlap(raw_vertices, measured_vertices)
            if overlap.first_containment >= 0.95:
                metadata["refined_visible_polygon"] = normalized_quad
                metadata["refined_visibility_source"] = _PRINT_BOUNDARY_SOURCE
        elif review_visible_rim:
            rim = None
            primary = _proposal_polygon({"polygon": mask.get("polygon")}, width, height)
            if primary is not None and not np.array_equal(primary, visible):
                primary_neighbors = [
                    points for neighbor in neighbor_masks or []
                    if (points := _proposal_polygon({"polygon": neighbor.get("polygon")}, width, height)) is not None
                ]
                observed = refine_visible_paper_rim(page_image, primary, primary_neighbors)
                if observed is not None:
                    agreement = normalized_polygon_overlap(
                        pixel_vertices_to_normalized(visible, width, height),
                        pixel_vertices_to_normalized(observed, width, height),
                    )
                    if agreement.first_containment >= .90 and agreement.second_containment >= .90:
                        rim = observed
            if rim is None:
                rim = refine_visible_paper_rim(page_image, visible, neighbor_polygons)
            if rim is not None:
                metadata["refined_visible_polygon"] = _vertices_on_page(rim, width, height)
                metadata["refined_visibility_source"] = _PAPER_RIM_SOURCE
        return _dual_crop_result(
            page_image, corrected, src_points, matrix, metadata, _explicit_quad(mask, "image_quad", width, height),
            full_reasons, trim_border=trim_border, auto_orient=auto_orient, orientation_model_path=orientation_model_path,
            page_regions=regions, protect_regions=protect_regions, owned_sampling=owned_sampling,
            neighbors=neighbor_polygons, corner_mount_detected=corner_mount_detected,
        )
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
    if auto_orient:
        corrected, orientation_metadata = correct_photo_orientation(
            corrected,
            model_path=orientation_model_path,
        )
        metadata["orientation"] = orientation_metadata
        rotation = int(orientation_metadata.get("rotation_degrees") or 0)
        if rotation:
            rectified_untrimmed = rotate_image(rectified_untrimmed, rotation)
        metadata["orientation_confidence"] = orientation_metadata.get("confidence")
    else:
        metadata["orientation"] = {"rotation_degrees": 0, "method": "disabled", "decision": "preserve"}
        metadata["orientation_confidence"] = None
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
