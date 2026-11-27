"""Measure a common image boundary for comparing two held proposals."""

import json
from hashlib import sha256

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.print_edges import _edge_band, _line_endpoints, _quad_from_lines, _side_line

SHARED_MOTIF_REVIEW = "shared_motif_review"
SHARED_MOTIF_REVIEW_REVISION = 1


def motif_review_id(outline: list, polygons: list[list]) -> str:
    members = sorted(json.dumps(polygon, sort_keys=True, separators=(",", ":")) for polygon in polygons)
    return sha256(json.dumps([outline, members], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def near_nested_model_masks(first: np.ndarray, second: np.ndarray) -> bool:
    areas = sorted((cv2.contourArea(first), cv2.contourArea(second)))
    if areas[1] <= 0 or not .79 <= areas[0] / areas[1] <= .89:
        return False
    overlap = normalized_polygon_overlap(first, second, raster_size=1024)
    return (
        .79 <= overlap.iou <= .84
        and max(overlap.first_containment, overlap.second_containment) >= .96
        and min(overlap.first_containment, overlap.second_containment) <= .86
    )


def _supported_motif(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    lines = []
    side_colors = []
    stations = np.linspace(.05, .95, 96, dtype=np.float32)
    offsets = [-5, -4, -3, -2, 2, 3, 4, 5]
    radius = int(np.clip(round(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() * .02), 3, 8))
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        line, evidence = _side_line(image, start, end, radius)
        if (line is None or evidence["ambiguous"] or evidence["support"] < .91
                or (evidence.get("corner_support") or 0) < .82):
            return None
        measured = _line_endpoints(start, end, line)
        band = _edge_band(image, *measured, offsets, fractions=stations)
        if band is None:
            return None
        outside = np.median(band[:, :4], axis=1)
        inside = np.median(band[:, 4:], axis=1)
        palette = np.median(outside, axis=0)
        support = ((outside[:, 0] >= 181) & (outside[:, 0] - inside[:, 0] >= 19)
                   & (np.linalg.norm(outside - palette, axis=1) <= 16))
        if support.mean() < .84 or min(part.mean() for part in np.array_split(support, 6)) < .66:
            return None
        lines.append(line)
        side_colors.append(palette)
    colors = np.float32(side_colors)
    if np.max(np.linalg.norm(colors - np.median(colors, axis=0), axis=1)) > 22:
        return None
    corners = _quad_from_lines(np.asarray(lines))
    if corners is None or not cv2.isContourConvex(corners) or np.max(np.linalg.norm(corners - quad, axis=1)) > 5:
        return None
    return corners


def measure_shared_motif(
    image: np.ndarray, polygons: tuple[np.ndarray, np.ndarray], other_claims: list[np.ndarray],
) -> np.ndarray | None:
    """Return one native frame that both proposals cover, for visual review."""
    scale = np.float32(image.shape[1::-1]) - 1
    vertices = np.vstack(polygons) * scale
    lower = np.maximum(0, np.floor(vertices.min(axis=0) - 12).astype(int))
    upper = np.minimum(np.int32(scale) + 1, np.ceil(vertices.max(axis=0) + 12).astype(int) + 1)
    size = upper - lower
    if size.min() < 80 or size.max() > 1600 or size.prod() > .20 * np.prod(scale + 1):
        return None
    left, top = lower
    right, bottom = upper
    patch = image[top:bottom, left:right]
    lightness = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[..., 0]
    blurred = cv2.GaussianBlur(lightness, (3, 3), .6)
    by_threshold = []
    for low, high in ((18, 37), (30, 62)):
        contours, _ = cv2.findContours(cv2.Canny(blurred, low, high), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if not .40 <= cv2.contourArea(contour) / size.prod() <= .85:
                continue
            quad = cv2.approxPolyDP(contour, .012 * cv2.arcLength(contour, True), True).reshape(-1, 2)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            quad = order_quad_cyclic(np.float32(quad + lower))
            overlaps = [normalized_polygon_overlap(quad / scale, polygon, raster_size=1024) for polygon in polygons]
            if any(overlap.iou < .78 or overlap.first_containment < .94 for overlap in overlaps):
                continue
            measured = _supported_motif(image, quad)
            if measured is not None and ((measured >= 0) & (measured <= scale)).all():
                candidates.append(measured)
        if not candidates:
            return None
        by_threshold.extend(candidates)
    result = np.median(by_threshold, axis=0).astype(np.float32)
    if np.max(np.linalg.norm(by_threshold - result, axis=2)) > 3:
        return None
    if any(normalized_polygon_overlap(result / scale, claim, raster_size=1024).first_containment > .001
           for claim in other_claims):
        return None
    return result
