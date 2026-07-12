from pathlib import Path

import cv2
import numpy as np

from app.pipeline.image_ops import normalized_box_to_pixels, save_jpeg
from app.pipeline.border_removal import remove_uniform_border
from app.pipeline.orientation import correct_photo_orientation
from app.pipeline.protected_regions import detect_protected_regions, validate_protected_regions


def _points_from_mask(mask: dict | None, width: int, height: int) -> np.ndarray | None:
    if not mask:
        return None
    polygon = mask.get("polygon")
    if not polygon:
        return None
    points = np.array(
        [[float(point["x"]) * width, float(point["y"]) * height] for point in polygon],
        dtype=np.float32,
    )
    if len(points) < 4:
        return None
    if len(points) == 4:
        return points

    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
    perimeter = cv2.arcLength(hull, True)
    for epsilon_ratio in (0.01, 0.02, 0.035, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if len(approx) == 4:
            return approx.astype(np.float32)

    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.float32)


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order a quadrilateral clockwise from its top-left corner.

    Sum/difference ordering can assign one point to two corners for rotated or
    symmetric quads.  A centroid-based cyclic order is unambiguous once the
    geometry has been validated.
    """
    if points.shape != (4, 2):
        raise ValueError("Perspective correction requires exactly four points.")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)].astype(np.float32)
    top_left_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -top_left_index, axis=0)


def _validate_quad(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("invalid_perspective_quad")
    if len(np.unique(np.round(points, decimals=3), axis=0)) != 4:
        raise ValueError("duplicate_perspective_corners")
    if (points[:, 0] < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] < 0).any() or (points[:, 1] >= height).any():
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
    corrected, _ = crop_and_correct_photo_with_metadata(
        page_image,
        bounding_box,
        mask,
        trim_border=trim_border,
        auto_orient=auto_orient,
        orientation_model_path=orientation_model_path,
        protected_regions=protected_regions,
        protect_regions=protect_regions,
    )
    return corrected


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
    height, width = page_image.shape[:2]
    src_points = _points_from_mask(mask, width, height)
    if src_points is None:
        x1, y1, x2, y2 = normalized_box_to_pixels(bounding_box, width, height)
        src_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    src_points = _order_points(src_points)
    quad_metrics = _validate_quad(src_points, width, height)
    regions = protected_regions if protected_regions is not None else detect_protected_regions(page_image)
    protection = validate_protected_regions(src_points, regions, width=width, height=height) if protect_regions else {"passed": True, "regions_checked": 0, "violations": []}
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
    normalized_destination = destination / np.array([max(target_width - 1, 1), max(target_height - 1, 1)], dtype=np.float32)
    normalized_matrix = cv2.getPerspectiveTransform(normalized_source.astype(np.float32), normalized_destination.astype(np.float32))
    condition_number = float(np.linalg.cond(matrix))
    normalized_condition_number = float(np.linalg.cond(normalized_matrix))
    if not np.isfinite(normalized_condition_number) or normalized_condition_number > 1e4:
        raise ValueError("unstable_perspective_transform")
    corrected = cv2.warpPerspective(page_image, matrix, (target_width, target_height), borderMode=cv2.BORDER_REPLICATE)
    if corrected.size == 0 or min(corrected.shape[:2]) < 2:
        raise ValueError("empty_perspective_output")
    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    reprojection_error = float(np.max(np.linalg.norm(projected - destination, axis=1)))
    if not np.isfinite(reprojection_error) or reprojection_error > 0.05:
        raise ValueError("perspective_reprojection_error")
    metadata = {
        "source": mask.get("source") if mask else "bounding_box",
        "target_width": target_width,
        "target_height": target_height,
        "trim_border": trim_border,
        "auto_orient": auto_orient,
        "quad_validation": {
            **quad_metrics,
            "transform_condition_number": condition_number,
            "normalized_transform_condition_number": normalized_condition_number,
            "reprojection_error": reprojection_error,
        },
        "protected_regions": protection,
    }
    if trim_border:
        before_shape = corrected.shape[:2]
        corrected = remove_uniform_border(corrected)
        metadata["border_trim"] = {
            "before": {"height": before_shape[0], "width": before_shape[1]},
            "after": {"height": corrected.shape[0], "width": corrected.shape[1]},
        }
    if auto_orient:
        corrected, orientation_metadata = correct_photo_orientation(
            corrected,
            model_path=orientation_model_path,
        )
        metadata["orientation"] = orientation_metadata
    return corrected, metadata


def save_corrected_photo(page_image: np.ndarray, bounding_box: dict, mask: dict | None, output_path: Path) -> None:
    corrected = crop_and_correct_photo(page_image, bounding_box, mask)
    save_jpeg(output_path, corrected)
