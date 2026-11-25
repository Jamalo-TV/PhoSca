"""Find a measured outer outline to compare with an uncertain crop."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, order_quad_cyclic
from app.pipeline.print_edges import _edge_band


PRINT_OUTLINE_SUGGESTION_REVISION = 1


def _outline_at_distance(
    distance: np.ndarray, offset: np.ndarray, seed: np.ndarray, scale: np.ndarray, threshold: int,
) -> np.ndarray | None:
    mask = cv2.morphologyEx(np.uint8(distance <= threshold), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seed_area = cv2.contourArea(seed)
    outlines = []
    for contour in contours:
        if not 1.02 * seed_area <= cv2.contourArea(contour) <= 1.27 * seed_area:
            continue
        points = contour[:, 0]
        if np.any(points.min(axis=0) < 3) or np.any(points.max(axis=0) > np.array(distance.shape[1::-1]) - 4):
            continue
        hull = cv2.convexHull(contour)
        corners = cv2.approxPolyDP(hull, .012 * cv2.arcLength(hull, True), True)[:, 0]
        if len(corners) != 4:
            continue
        quad = order_quad_cyclic(corners + offset)
        match = normalized_polygon_overlap(seed / scale, quad / scale)
        if .81 <= match.iou <= .96 and match.first_containment >= .985:
            outlines.append(quad)
    return outlines[0] if len(outlines) == 1 else None


def _visible_stock_perimeter(image: np.ndarray, outline: np.ndarray, stock: np.ndarray) -> bool:
    for start, end in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        length = float(np.linalg.norm(end - start))
        count = int(np.clip(np.ceil(length), 64, 1600))
        inset = 6 / length
        fractions = np.linspace(inset, 1 - inset, count, dtype=np.float32)
        band = _edge_band(image, start, end, [-8, -5, -3, 3, 5, 8], fractions=fractions)
        if band is None or np.median(band[:, 3:, 0]) < 180:
            return False
        distance = np.linalg.norm(band[:, :, 1:] - stock, axis=2)
        votes = (distance[:, :3].mean(axis=1) >= 15) & (distance[:, 3:].mean(axis=1) <= 10)
        body = votes[(fractions >= .035) & (fractions <= .965)]
        if body.mean() < .83 or any(part.mean() < .50 for part in np.array_split(body, 4)):
            return False
        # A separating gap across two adjacent prints can disappear in the
        # median and closing operations. Check the interior rim in the source.
        interrupted = (distance[:, 3:] > 16).all(axis=1)
        if (interrupted[:-1] & interrupted[1:]).any():
            return False
    return True


def suggest_outer_print_outline(
    image: np.ndarray, seed: np.ndarray, preview: np.ndarray, neighbors: list[np.ndarray],
) -> np.ndarray | None:
    """Return one plausible larger stock outline for review."""
    extent = float(np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).min())
    if extent < 180:
        return None
    scale = np.float32(image.shape[1::-1]) - 1
    pad = .20 * extent
    low = np.maximum(0, np.floor(seed.min(axis=0) - pad).astype(int))
    high = np.minimum(scale, np.ceil(seed.max(axis=0) + pad)).astype(int) + 1
    patch = image[low[1]:high[1], low[0]:high[0]]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = lab[..., 1:]
    initial = cv2.fillConvexPoly(np.zeros(lab.shape[:2], np.uint8), np.rint(seed - low).astype(np.int32), 1)
    pale = (initial != 0) & (lab[..., 0] >= 180) & (np.linalg.norm(chroma - 128, axis=2) <= 16)
    if np.count_nonzero(pale) < .10 * cv2.contourArea(seed):
        return None
    stock = np.median(chroma[pale], axis=0)
    distance = cv2.medianBlur(np.linalg.norm(chroma - stock, axis=2), 5)
    narrow = _outline_at_distance(distance, low, seed, scale, 8)
    outline = _outline_at_distance(distance, low, seed, scale, 12)
    if narrow is None or outline is None or np.max(np.linalg.norm(narrow - outline, axis=1)) > 5:
        return None
    normalized = outline / scale
    if (np.max(np.linalg.norm(outline - preview, axis=1)) < max(7., .018 * extent)
            or normalized_polygon_overlap(preview / scale, normalized).first_containment < .98):
        return None
    for neighbor in neighbors:
        other = neighbor / scale
        match = normalized_polygon_overlap(normalized, other)
        if max(match.first_containment, match.second_containment) <= .02:
            continue
        before = normalized_polygon_overlap(seed / scale, other)
        if not (before.second_containment >= .92 and match.second_containment >= .985
                and match.first_containment <= .68):
            return None
    return outline if _visible_stock_perimeter(image, outline, stock) else None
