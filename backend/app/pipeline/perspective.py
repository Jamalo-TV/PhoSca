from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.border_removal import trim_physical_border
from app.pipeline.geometry import normalized_vertices_to_pixels, order_quad_cyclic, quad_from_polygon
from app.pipeline.image_ops import normalized_box_to_pixels, save_jpeg
from app.pipeline.orientation import correct_photo_orientation, rotate_image
from app.pipeline.protected_regions import detect_protected_regions, merge_protected_regions, validate_protected_regions


@dataclass(frozen=True)
class CropResult:
    """Faithful rectified pixels plus any optional, reversible derivative."""

    rectified_untrimmed: np.ndarray
    image: np.ndarray
    metadata: dict


def _points_from_mask(mask: dict | None, width: int, height: int) -> np.ndarray | None:
    if not mask:
        return None
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
        (points[:, 0] < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= height).any()
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
) -> CropResult:
    height, width = page_image.shape[:2]
    src_points = _points_from_mask(mask, width, height)
    if src_points is None:
        x1, y1, x2, y2 = normalized_box_to_pixels(bounding_box, width, height)
        # Bounding boxes use half-open maximum edges; warp vertices use
        # inclusive pixel centers. Clamp the right/bottom edge accordingly so
        # a legitimate normalized 1.0 never becomes an out-of-range pixel.
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        src_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    src_points = _order_points(src_points)
    quad_metrics = _validate_quad(src_points, width, height)
    regions = protected_regions if protected_regions is not None else detect_protected_regions(page_image)
    protection = (
        validate_protected_regions(src_points, regions, width=width, height=height)
        if protect_regions
        else {"passed": True, "regions_checked": 0, "violations": []}
    )
    if not protection["passed"]:
        raise ValueError("protected_region_cut")
    top_width = np.linalg.norm(src_points[1] - src_points[0])
    bottom_width = np.linalg.norm(src_points[2] - src_points[3])
    left_height = np.linalg.norm(src_points[3] - src_points[0])
    right_height = np.linalg.norm(src_points[2] - src_points[1])
    target_width = max(1, int(round(max(top_width, bottom_width))))
    target_height = max(1, int(round(max(left_height, right_height))))
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
    corrected = cv2.warpPerspective(page_image, matrix, (target_width, target_height), borderMode=cv2.BORDER_REPLICATE)
    if corrected.size == 0 or min(corrected.shape[:2]) < 2:
        raise ValueError("empty_perspective_output")
    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    reprojection_error = float(np.max(np.linalg.norm(projected - destination, axis=1)))
    if not np.isfinite(reprojection_error) or reprojection_error > 0.05:
        raise ValueError("perspective_reprojection_error")
    rectified_untrimmed = corrected.copy()
    source_pixels_per_output_pixel = float(
        np.sqrt(quad_metrics["area"] / max(float(target_width * target_height), 1.0))
    )
    normalized_quad = [
        {
            "x": round(float(point[0]) / max(width - 1, 1), 8),
            "y": round(float(point[1]) / max(height - 1, 1), 8),
        }
        for point in src_points
    ]
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
    if trim_border:
        before_shape = corrected.shape[:2]
        page_regions_in_crop = _rectified_protected_regions(
            regions,
            page_width=width,
            page_height=height,
            matrix=matrix,
            target_width=target_width,
            target_height=target_height,
        )
        local_regions = detect_protected_regions(corrected)
        border_result = trim_physical_border(
            corrected,
            protected_regions=merge_protected_regions(page_regions_in_crop, local_regions),
        )
        corrected = border_result.image
        metadata["border_trim"] = {
            "before": {"height": before_shape[0], "width": before_shape[1]},
            "after": {"height": corrected.shape[0], "width": corrected.shape[1]},
            **border_result.metadata(),
        }
        # Preserving an ambiguous edge is safe; confidence describes only a
        # mutation that was actually applied.
        metadata["border_trim_confidence"] = border_result.confidence if border_result.applied else 1.0
    else:
        metadata["border_trim"] = {
            "method": "disabled",
            "applied": False,
            "reason_codes": ["BORDER_TRIM_DISABLED_PRESERVE_SOURCE"],
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
