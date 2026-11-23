"""Shared geometry and coordinate-space primitives for photo extraction.

Polygon vertices use an inclusive pixel-center convention: normalized ``0``
maps to the first pixel center and normalized ``1`` maps to the last pixel
center.  Bounding boxes remain the existing normalized half-open API contract.
Keeping the vertex conversion here prevents the two conventions from being
silently mixed by segmentation and perspective correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PolygonOverlap:
    iou: float
    first_containment: float
    second_containment: float


def _point_array(points: np.ndarray | list | tuple) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("invalid_polygon_vertices")
    return array


def normalized_vertices_to_pixels(
    points: np.ndarray | list | tuple,
    width: int,
    height: int,
) -> np.ndarray:
    """Map normalized polygon vertices to inclusive source pixel centers."""
    if width <= 0 or height <= 0:
        raise ValueError("invalid_coordinate_space_dimensions")
    array = _point_array(points)
    clipped = np.clip(array, 0.0, 1.0)
    return clipped * np.array([max(width - 1, 0), max(height - 1, 0)], dtype=np.float32)


def pixel_vertices_to_normalized(
    points: np.ndarray | list | tuple,
    width: int,
    height: int,
) -> np.ndarray:
    """Map inclusive source pixel centers to normalized polygon vertices."""
    if width <= 0 or height <= 0:
        raise ValueError("invalid_coordinate_space_dimensions")
    array = _point_array(points)
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.clip(array / scale, 0.0, 1.0)


def rectify_crop_quad(
    image: np.ndarray,
    crop_quad: np.ndarray,
    *,
    page_to_rectified: np.ndarray | None = None,
) -> np.ndarray:
    """Sample an ordered print quad, optionally straight from its original album page."""
    quad = _point_array(crop_quad)
    if quad.shape != (4, 2) or not cv2.isContourConvex(quad) or cv2.contourArea(quad, oriented=True) <= 0:
        raise ValueError("invalid_print_quad")

    top, right, bottom, left = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    # Opposite sides estimate the same print dimension. Pixel centers include both endpoints.
    width = round(float((top + bottom) / 2)) + 1
    height = round(float((left + right) / 2)) + 1
    if min(width, height) < 2:
        raise ValueError("degenerate_print_quad")
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    if page_to_rectified is None:
        if np.any(quad < 0) or np.any(quad > [image.shape[1] - 1, image.shape[0] - 1]):
            raise ValueError("print_quad_outside_image")
        if np.array_equal(quad, quad[0] + destination) and np.array_equal(quad[0], np.rint(quad[0])):
            x, y = quad[0].astype(int)
            return image[y : y + height, x : x + width].copy()

    transform = cv2.getPerspectiveTransform(quad, destination)
    if page_to_rectified is not None:
        transform = transform @ page_to_rectified
    return cv2.warpPerspective(image, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)


def order_quad_cyclic(points: np.ndarray | list | tuple) -> np.ndarray:
    """Return four unique convex points in clockwise image-space order.

    The historical sum/difference corner assignment can choose the same point
    twice for a diamond.  Angle sorting is a permutation, so it cannot invent
    or duplicate corners.  Sum is used only to choose a stable starting point.
    """
    array = _point_array(points)
    if array.shape != (4, 2):
        raise ValueError("quad_requires_four_vertices")
    if len(np.unique(np.round(array, decimals=4), axis=0)) != 4:
        raise ValueError("duplicate_quad_vertices")

    hull = cv2.convexHull(array.reshape(-1, 1, 2), clockwise=False).reshape(-1, 2)
    if hull.shape != (4, 2):
        raise ValueError("non_convex_quad_vertices")
    center = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    ordered = hull[np.argsort(angles)].astype(np.float32)

    # In image coordinates (positive y down), increasing polar angle follows
    # the visible clockwise cycle.  Rotate, never reassign, to start at the
    # upper-left-like vertex.  Lexicographic tie-breaking makes diamonds stable.
    sums = ordered[:, 0] + ordered[:, 1]
    minimum = float(np.min(sums))
    tied = np.flatnonzero(np.isclose(sums, minimum, rtol=0.0, atol=1e-4))
    start = int(min(tied, key=lambda index: (float(ordered[index, 1]), float(ordered[index, 0]))))
    return np.roll(ordered, -start, axis=0)


def quad_from_polygon(points: np.ndarray | list | tuple) -> tuple[np.ndarray, str]:
    """Create a conservative convex warp quad from an arbitrary polygon."""
    array = _point_array(points)
    if len(array) < 4:
        raise ValueError("polygon_requires_four_vertices")
    if len(array) == 4:
        return order_quad_cyclic(array), "polygon_quad"

    hull = cv2.convexHull(array.reshape(-1, 1, 2)).reshape(-1, 2)
    perimeter = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
    hull_area = max(1e-6, abs(float(cv2.contourArea(hull.reshape(-1, 1, 2)))))
    for epsilon_ratio in (0.01, 0.02, 0.035, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), epsilon_ratio * perimeter, True).reshape(-1, 2)
        if len(approx) != 4:
            continue
        approx_area = abs(float(cv2.contourArea(approx.reshape(-1, 1, 2))))
        if approx_area >= hull_area * 0.82:
            return order_quad_cyclic(approx), "convex_polygon_quad"

    # A minimum-area rectangle encloses the visible hull, so it is safer than
    # an aggressively simplified polygon when four supported sides are absent.
    return order_quad_cyclic(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)), "minimum_area_rect"


def normalized_polygon_overlap(
    first: np.ndarray | list | tuple,
    second: np.ndarray | list | tuple,
    *,
    raster_size: int = 256,
) -> PolygonOverlap:
    """Calculate mask-aware overlap for normalized, possibly concave polygons."""
    if raster_size < 32:
        raise ValueError("polygon_overlap_raster_too_small")
    try:
        first_points = normalized_vertices_to_pixels(first, raster_size, raster_size)
        second_points = normalized_vertices_to_pixels(second, raster_size, raster_size)
    except ValueError:
        return PolygonOverlap(0.0, 0.0, 0.0)
    if len(first_points) < 3 or len(second_points) < 3:
        return PolygonOverlap(0.0, 0.0, 0.0)

    first_mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    second_mask = np.zeros_like(first_mask)
    cv2.fillPoly(first_mask, [np.rint(first_points).astype(np.int32)], 1)
    cv2.fillPoly(second_mask, [np.rint(second_points).astype(np.int32)], 1)
    first_area = int(first_mask.sum())
    second_area = int(second_mask.sum())
    if first_area == 0 or second_area == 0:
        return PolygonOverlap(0.0, 0.0, 0.0)
    intersection = int(np.logical_and(first_mask, second_mask).sum())
    union = first_area + second_area - intersection
    return PolygonOverlap(
        iou=float(intersection / union) if union else 0.0,
        first_containment=float(intersection / first_area),
        second_containment=float(intersection / second_area),
    )
