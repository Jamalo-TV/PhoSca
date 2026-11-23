"""Identify reflective sleeve seams that a local print mask followed."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, quad_from_polygon
from app.pipeline.print_edges import _sample_lab


def _outside_crossing(polygon: np.ndarray, along: np.ndarray) -> np.ndarray:
    end = np.roll(polygon, -1, axis=0)
    x, y = polygon.T
    dx, dy = (end - polygon).T
    crosses = (np.minimum(x, end[:, 0]) <= along[:, None]) & (along[:, None] < np.maximum(x, end[:, 0]))
    fraction = np.divide(
        along[:, None] - x, dx, out=np.zeros((len(along), len(polygon))), where=dx != 0,
    )
    return np.min(np.where(crosses, y + fraction * dy, np.inf), axis=1)


def _bright_ridges(image: np.ndarray, centers: np.ndarray, inward: np.ndarray) -> np.ndarray:
    offsets = np.arange(-15, 16)
    points = centers[:, None, :] + offsets[None, :, None] * inward
    upper = np.array(image.shape[1::-1]) - 1
    in_frame = ((points >= 0) & (points <= upper)).all(axis=(1, 2))
    ridges = np.zeros((len(centers), 2))
    if in_frame.any():
        lab = _sample_lab(image, points[in_frame].reshape(-1, 2)).reshape(-1, len(offsets), 3)
        luminance = lab[:, :, 0]
        center = luminance[:, 5:-5]
        shoulders = np.maximum(luminance[:, :-10], luminance[:, 10:])
        pale = (center >= 212) & (np.max(np.abs(lab[:, 5:-5, 1:] - 128), axis=2) <= 18)
        strength = np.where(pale, center - shoulders, 0)
        best = np.argmax(strength, axis=1)
        ridges[in_frame] = np.column_stack((offsets[5:-5][best], strength[np.arange(len(best)), best]))
    return ridges


def _ridge_shoulder(
    image: np.ndarray, centers: np.ndarray, inward: np.ndarray, depths: np.ndarray,
) -> np.ndarray | None:
    points = centers[:, None, :] + depths[None, :, None] * inward
    upper = np.array(image.shape[1::-1]) - 1
    valid = ((points >= 0) & (points <= upper)).all(axis=(1, 2))
    if np.count_nonzero(valid) < 10:
        return None
    lab = _sample_lab(image, points[valid].reshape(-1, 2)).reshape(-1, len(depths), 3)
    return np.median(lab, axis=0)


def _ridge_continues_into_album(
    image: np.ndarray, extensions: np.ndarray, inward: np.ndarray, occupied: list[np.ndarray],
    depths: np.ndarray, material: np.ndarray,
) -> bool:
    for centers in extensions:
        outside = np.array([
            all(cv2.pointPolygonTest(polygon, tuple(point), True) < -5 for polygon in occupied)
            for point in centers
        ])
        if np.count_nonzero(outside) < 10:
            continue
        centers = centers[outside]
        offsets, strength = _bright_ridges(image, centers, inward).T
        supported = (strength >= 7) & (np.abs(offsets) <= 8)
        if np.mean(supported) < .75:
            continue
        crest = centers[supported] + offsets[supported, None] * inward
        backing = _ridge_shoulder(image, crest, inward, depths)
        if backing is not None and np.max(np.linalg.norm(backing - material, axis=1)) <= 10:
            return True
    return False


def sleeve_reflection_sides(
    image: np.ndarray, original: np.ndarray, refined: np.ndarray, occupied: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Report sides where an added band follows a fold in the album sleeve."""
    if len(original) < 4 or normalized_polygon_overlap(original, refined).iou < .90:
        return []
    upper = np.array(image.shape[1::-1]) - 1
    original_pixels = original * upper
    refined_pixels = refined * upper
    try:
        quad, _ = quad_from_polygon(original_pixels)
    except ValueError:
        return []
    if np.any(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1) < 40):
        return []
    sides = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        length = float(np.linalg.norm(end - start))
        direction = (end - start) / length
        inward = np.array([-direction[1], direction[0]])
        basis = np.column_stack((direction, inward))
        along = np.linspace(.04, .96, 80) * length
        near = _outside_crossing((original_pixels - start) @ basis, along)
        proposed = _outside_crossing((refined_pixels - start) @ basis, along)
        expanded = np.isfinite(near) & np.isfinite(proposed) & (proposed < near - 5)
        if np.count_nonzero(expanded) < 20:
            continue
        along, near, proposed = along[expanded], near[expanded], proposed[expanded]
        base = start + along[:, None] * direction
        displacement, strength = _bright_ridges(image, base + proposed[:, None] * inward, inward).T
        ridge = proposed + displacement
        supported = (strength >= 7) & (ridge < near - 7)
        if np.count_nonzero(supported) < 16 or np.mean(supported) < .65:
            continue
        along, ridge, near, base = along[supported], ridge[supported], near[supported], base[supported]
        if np.ptp(along) < .22 * length:
            continue
        slope, offset = np.polyfit(along, ridge, 1)
        if abs(slope) > .11 or np.percentile(np.abs(ridge - offset - slope * along), 80) > 6:
            continue
        # A distinct print edge must still be visible before the fold.
        points = base[:, None, :] + (near[:, None] + [-7., -3., 3., 7.])[:, :, None] * inward
        in_frame = ((points >= 0) & (points <= upper)).all(axis=(1, 2))
        if np.count_nonzero(in_frame) < 16:
            continue
        lab = _sample_lab(image, points[in_frame].reshape(-1, 2)).reshape(-1, 4, 3)
        contrast = np.linalg.norm(lab[:, :2].mean(axis=1) - lab[:, 2:].mean(axis=1), axis=1)
        if np.median(contrast) < 12:
            continue
        # Compare the material between the two borders with exposed backing
        # at the same offset from the reflection. Real print stock can lie
        # behind a fold without belonging to the album.
        depths = np.array([3.5, min(11.5, .64 * np.median(near - ridge))])
        material = _ridge_shoulder(image, base + ridge[:, None] * inward, inward, depths)
        if material is None:
            continue
        tail = np.array([np.linspace(-.19, -.025, 20), np.linspace(1.025, 1.19, 20)]) * length
        extensions = start + tail[..., None] * direction + (offset + slope * tail)[..., None] * inward
        if _ridge_continues_into_album(image, extensions, inward, occupied, depths, material):
            sides.append((start, end))
    return sides
