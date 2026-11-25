"""Find review outlines where a classical proposal crosses visible paper edges."""

from collections.abc import Callable

import cv2
import numpy as np

from app.pipeline.edge_prints import _photo_texture
from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _edge_band, refine_print_boundary


CROSSING_PRINT_REVISION = 1
CROSSING_PRINT_REVIEW = "CROSSING_PRINT_OUTLINE_REVIEW"
REVIEWED_CROSSING_OUTLINE = "reviewed_crossing_outline"


def _pale_leaf_foreground(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Separate dark regions inside the observed pale page without closing gaps."""
    height, width = image.shape[:2]
    short = min(height, width)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    coarse = lab[::4, ::4]
    smooth = cv2.GaussianBlur(lab, (0, 0), 1.1)[::4, ::4]
    pale = ((coarse[:, :, 0] > 180) & (np.linalg.norm(coarse[:, :, 1:] - 128, axis=2) < 13)
            & (np.linalg.norm(coarse - smooth, axis=2) < 5))
    if pale.mean() < .12:
        return None
    light = np.quantile(coarse[:, :, 0][pale], .64)
    pale &= (coarse[:, :, 0] >= light - 7) & (coarse[:, :, 0] < light + 8)
    ys, xs = np.ogrid[:height, :width]
    x, y = xs.astype(np.float32) / width - .5, ys.astype(np.float32) / height - .5
    sample_y, sample_x = np.nonzero(pale)
    terms = np.column_stack((x[0, 4 * sample_x], y[4 * sample_y, 0], np.ones(len(sample_x), np.float32)))
    fit, _, rank, _ = np.linalg.lstsq(terms, coarse[pale], rcond=None)
    if rank != 3:
        return None
    lab -= fit[2]
    lab -= x[..., None] * fit[0]
    lab -= y[..., None] * fit[1]
    distance_sq = np.einsum("hwi,hwi->hw", lab, lab)
    count, components, stats, _ = cv2.connectedComponentsWithStats(np.uint8(distance_sq <= 13**2))
    major = (np.arange(count) > 0) & (stats[:, cv2.CC_STAT_AREA] > .035 * height * width)
    points = cv2.findNonZero(major[components].astype(np.uint8))
    if points is None:
        return None
    hull = cv2.convexHull(points)
    if not .50 <= cv2.contourArea(hull) / (width * height) <= .97:
        return None
    leaf = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(leaf, hull, 1)
    leaf[[0, -1], :] = leaf[:, [0, -1]] = 0
    inset = max(9., .032 * short)
    distance = cv2.distanceTransform(leaf, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    mask = np.uint8((distance_sq > 28**2) & (distance > inset))
    size = max(3, round(.0036 * short) | 1)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    # An inner strip is needed on all four sides. Cutting a region at the
    # hull inset would invent a paper edge.
    return opened, distance > inset + 2


def _outside_pale_paper(image: np.ndarray, start: np.ndarray, stop: np.ndarray, scale: float) -> bool:
    offsets = np.float32([-13, -19, -27, -36, -48, 4, 8, 13, 19, 27]) * scale
    samples = _edge_band(image, start, stop, offsets, fractions=np.linspace(.08, .92, 72, dtype=np.float32))
    if samples is None:
        return False
    paper, scene = np.split(samples.astype(np.float32), 2, axis=1)
    stock = np.median(paper, axis=1)
    support = ((stock[:, 0] > 171) & (np.linalg.norm(stock[:, 1:] - 128, axis=1) < 17)
               & (np.linalg.norm(paper - stock[:, None], axis=2).max(axis=1) < 11)
               & (np.linalg.norm(scene - stock[:, None], axis=2).mean(axis=1) > 9))
    return support.mean() >= .83 and min(part.mean() for part in np.array_split(support, 4)) >= .65


def _shadow_on_pale_paper(image: np.ndarray, start: np.ndarray, stop: np.ndarray, scale: float) -> bool:
    offsets = np.float32([-35, -26, -20, -14, -11, -8, -5, -3, 4, 7]) * scale
    samples = _edge_band(image, start, stop, offsets, fractions=np.linspace(.11, .89, 112, dtype=np.float32))
    if samples is None:
        return False
    light = samples[:, :, 0]
    far, shoulder, near, scene = (strip.mean(axis=1) for strip in np.split(light, [3, 6, 8], axis=1))
    support = (far >= 180) & (far - shoulder >= 3.2) & (shoulder - near >= 3) & (near - scene >= 11)
    return support.mean() >= .75 and min(part.mean() for part in np.array_split(support, 4)) >= .65


def crossing_print_outlines(
    image: np.ndarray, quad_from_contour: Callable[[np.ndarray], tuple[np.ndarray, str]],
) -> list[np.ndarray]:
    height, width = image.shape[:2]
    if min(height, width) < 480:
        return []
    ratio = min(1., 2048 / max(height, width))
    dimensions = (round(width * ratio), round(height * ratio))
    reduced = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA) if ratio < 1 else image
    foreground = _pale_leaf_foreground(reduced)
    if foreground is None:
        return []
    mask, interior = foreground
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = dimensions[0] * dimensions[1]
    short = min(dimensions)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not .06 * frame_area < area < .26 * frame_area:
            continue
        xs, ys = contour[:, 0].T
        if not interior[ys, xs].all():
            continue
        seed, kind = quad_from_contour(contour)
        if kind != "classical_border_quad" or area < .72 * cv2.contourArea(seed):
            continue
        lengths = np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1)
        if min(lengths) < .17 * short or not .50 < lengths[::2].sum() / lengths[1::2].sum() < 2:
            continue
        candidates.append(order_quad_cyclic(seed * (np.float32([width - 1, height - 1]) / (np.array(dimensions) - 1))))
    if len(candidates) > 4:
        return []
    scale = np.clip(min(width, height) / 1920, .5, 1.5)
    outlines = []
    for seed in candidates:
        fitted, sides, reasons = refine_print_boundary(image, seed, search_radius=round(18 * scale))
        if (fitted is None or reasons or any(
            side["support"] < .86 or (side["corner_support"] or 0) < .65 for side in sides.values()
        )):
            continue
        backed = [_outside_pale_paper(image, start, stop, scale)
                  for start, stop in zip(fitted, np.roll(fitted, -1, axis=0), strict=True)]
        if not any(backed[i] and backed[i + 2] for i in (0, 1)):
            continue
        if not any(_shadow_on_pale_paper(image, start, stop, scale)
                   for start, stop in zip(fitted, np.roll(fitted, -1, axis=0), strict=True)):
            continue
        if _photo_texture(image, fitted):
            # A fit can follow a neighbour a few pixels away. Use the
            # observed component as the editable starting outline.
            outlines.append(seed)
    return outlines
