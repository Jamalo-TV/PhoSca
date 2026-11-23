from collections.abc import Sequence

import cv2
import numpy as np

from app.pipeline.geometry import normalized_vertices_to_pixels, quad_from_polygon


SOURCE_CONTOUR_REVISION = 3
CONTRAST_COMPONENT_SOURCE = "contrast_component"


_FRAME_SIZE = 96
_FRAME_QUAD = np.float32([[15, 15], [80, 15], [80, 80], [15, 80]])
_FRAME_CORNERS = np.float32([[0, 0], [95, 0], [95, 95], [0, 95]])
_FRAME_Y, _FRAME_X = np.mgrid[:_FRAME_SIZE, :_FRAME_SIZE]
_FRAME_PLANE = np.stack(
    (np.ones_like(_FRAME_X), (_FRAME_X - 47.5) / _FRAME_SIZE, (_FRAME_Y - 47.5) / _FRAME_SIZE), axis=2,
).astype(np.float32)
_FRAME_EXTERIOR = (
    ((_FRAME_X >= 25) & (_FRAME_X <= 70) & ((_FRAME_Y <= 7) | (_FRAME_Y >= 88)))
    | ((_FRAME_Y >= 25) & (_FRAME_Y <= 70) & ((_FRAME_X <= 7) | (_FRAME_X >= 88)))
)
_FRAME_EXTERIOR_FIT = np.linalg.pinv(_FRAME_PLANE[_FRAME_EXTERIOR]).astype(np.float32)


def _empty_drawn_frame(image: np.ndarray, polygon: np.ndarray) -> bool:
    height, width = image.shape[:2]
    try:
        quad, _ = quad_from_polygon(normalized_vertices_to_pixels(polygon, width, height))
        if np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min() < 30:
            return False
        mapping = cv2.getPerspectiveTransform(_FRAME_QUAD, quad)
        if np.any(_FRAME_CORNERS @ mapping[2, :2] + mapping[2, 2] <= 1e-6):
            return False
        outside = cv2.perspectiveTransform(_FRAME_CORNERS[None], mapping)[0]
    except (ValueError, cv2.error):
        return False
    if not np.isfinite(outside).all() or np.any((outside < 0) | (outside > [width - 1, height - 1])):
        return False
    patch = cv2.warpPerspective(image, mapping, (_FRAME_SIZE, _FRAME_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Fit the surrounding paper so a small lighting gradient is not mistaken for a stock change.
    difference = lab - _FRAME_PLANE @ (_FRAME_EXTERIOR_FIT @ lab[_FRAME_EXTERIOR])
    error = np.linalg.norm(difference, axis=2)
    interior = np.s_[23:73, 23:73]
    if (np.quantile(error[_FRAME_EXTERIOR], .95) > 3 or np.mean(error[interior] > 5) > .012
            or np.any(error[interior] > 11) or np.linalg.norm(np.median(difference[interior], axis=(0, 1))) > 1.2):
        return False
    # A printed stroke has the same backing on both sides and traces all four edges.
    strokes = np.array([np.rot90(error, turn)[8:23, 25:71] >= 13 for turn in range(4)])
    thickness = strokes.sum(axis=1)
    runs = strokes[:, 0] + np.sum(strokes[:, 1:] & ~strokes[:, :-1], axis=1)
    return bool(np.all(np.mean((runs == 1) & (thickness <= 8), axis=1) >= .83))


def _paper_reference(lab: np.ndarray) -> np.ndarray | None:
    shortest = min(lab.shape[:2])
    near, far = max(1, round(shortest * 0.035)), max(2, round(shortest * 0.12))
    band = np.concatenate(
        (
            lab[near:far, near:-near].reshape(-1, 3),
            lab[-far:-near, near:-near].reshape(-1, 3),
            lab[far:-far, near:far].reshape(-1, 3),
            lab[far:-far, -far:-near].reshape(-1, 3),
        )
    )
    bright = band[band[:, 0] >= np.percentile(band[:, 0], 60)]
    paper = np.median(bright, axis=0)
    # Neutral scan borders and bright monochrome prints are ambiguous.
    if paper[0] < 185 or paper[2] < 136:
        return None
    # Pale prints on dark album stock can dominate the bright tail without
    # matching enough of the actual page border to establish a paper color.
    nearby = ((band - paper) ** 2).sum(axis=1) <= 12**2
    return paper if nearby.mean() >= 0.50 else None


def paper_and_edge_rejection_reasons(
    image: np.ndarray,
    polygons: Sequence[Sequence[dict[str, float]]],
    *,
    source_image: np.ndarray | None = None,
) -> dict[int, str]:
    """Identify paper boxes, edge strips and source-confirmed empty ink frames."""
    height, width = image.shape[:2]
    if not polygons or min(height, width) < 16:
        return {}
    scale = min(1.0, 800 / max(height, width))
    dimensions = (round(width * scale), round(height * scale))
    if min(dimensions) < 16:
        return {}
    preview = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(preview, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    paper = _paper_reference(lab)
    kernel = np.ones((5, 5), np.uint8)

    rejected = {}
    for index, polygon in enumerate(polygons):
        points = np.asarray([(point["x"], point["y"]) for point in polygon], dtype=np.float32)
        if len(points) < 3 or not np.isfinite(points).all():
            continue
        points = np.clip(points, 0.0, 1.0)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        span_x, span_y = x2 - x1, y2 - y1
        horizontal_strip = (
            (y1 <= 0.005 or y2 >= 0.995)
            and span_y <= 0.017
            and span_x >= 0.18
            and span_x * width >= 12 * span_y * height
        )
        vertical_strip = (
            (x1 <= 0.005 or x2 >= 0.995)
            and span_x <= 0.017
            and span_y >= 0.18
            and span_y * height >= 12 * span_x * width
        )
        if horizontal_strip or vertical_strip:
            rejected[index] = "thin_image_edge_fragment"
            continue
        if source_image is not None and _empty_drawn_frame(source_image, points):
            rejected[index] = "empty_drawn_frame"
            continue
        if paper is None:
            continue

        pixels = np.rint(points * (width - 1, height - 1)).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pixels)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pixels - (x, y)], 1)
        mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        interior = lab[y : y + h, x : x + w][mask != 0]
        if len(interior) < 32:
            continue
        close = ((interior - paper) ** 2).sum(axis=1) <= 12**2
        if close.mean() >= 0.82:
            dark, light = np.percentile(interior[:, 0], (10, 90))
            if light - dark <= 30:
                rejected[index] = "paper_colored_region"
    return rejected


def paper_around_nested_print(image: np.ndarray, parent: np.ndarray, child: np.ndarray) -> bool:
    """Verify that a weak outer outline is mostly plain backing around a print."""
    height, width = image.shape[:2]
    scale = min(1.0, 800 / max(height, width))
    width, height = round(width * scale), round(height * scale)
    if min(width, height) < 16:
        return False
    preview = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    parent_mask = np.zeros((height, width), np.uint8)
    child_mask = np.zeros_like(parent_mask)
    dimensions = (width - 1, height - 1)
    cv2.fillPoly(parent_mask, [np.rint(parent * dimensions).astype(np.int32)], 1)
    cv2.fillPoly(child_mask, [np.rint(child * dimensions).astype(np.int32)], 1)
    kernel = np.ones((9, 9), np.uint8)
    parent_mask = cv2.erode(parent_mask, kernel)
    child_mask = cv2.dilate(child_mask, kernel)
    remainder = (parent_mask != 0) & (child_mask == 0)
    if np.count_nonzero(remainder) < 0.05 * width * height:
        return False
    lab = cv2.cvtColor(preview, cv2.COLOR_BGR2LAB)[remainder].astype(np.float32)
    dark, median, light = np.percentile(lab[:, 0], (10, 50, 90))
    chroma_distance = np.abs(lab[:, 1:] - np.median(lab[:, 1:], axis=0))
    return bool(median >= 155 and light - dark <= 35 and np.percentile(chroma_distance, 90) <= 12)


def refine_visible_paper_rim(
    image: np.ndarray, proposal: np.ndarray, neighbors: Sequence[np.ndarray],
) -> np.ndarray | None:
    """Extend the instance only into observed pale print material near its mask."""
    x, y, width, height = cv2.boundingRect(proposal)
    source_radius = max(4, min(96, round(0.10 * min(width, height))))
    pad = round(2.5 * source_radius) + 2
    left, top = max(0, x - pad), max(0, y - pad)
    right, bottom = min(image.shape[1], x + width + pad), min(image.shape[0], y + height + pad)
    bounds = np.float32([right - left - 1, bottom - top - 1])
    if min(bounds) < 12:
        return None
    roi = image[top:bottom, left:right]
    scale = min(1.0, 900 / max(roi.shape[:2]))
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    origin = np.float32([left, top])
    gain = np.float32([roi.shape[1] - 1, roi.shape[0] - 1]) / bounds
    radius = max(3, round(float(source_radius * min(gain))))

    def raster(points: np.ndarray) -> np.ndarray:
        vertices = np.rint((points - origin) * gain).astype(np.int32)
        return cv2.fillPoly(np.zeros(roi.shape[:2], np.uint8), [vertices], 1)

    def distance(mask: np.ndarray) -> np.ndarray:
        return cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    core = raster(proposal)
    own_distance = distance(core)
    apron = (own_distance >= 1.3 * radius) & (own_distance <= 2.5 * radius)
    rival_distance = np.full(roi.shape[:2], np.inf, dtype=np.float32)
    for neighbor in neighbors:
        neighbor_low, neighbor_high = neighbor.min(axis=0), neighbor.max(axis=0)
        if np.any(neighbor_high < origin) or np.any(neighbor_low > (right - 1, bottom - 1)):
            continue
        rival = distance(raster(neighbor))
        neighbor_radius = max(3, round(0.10 * min((neighbor_high - neighbor_low) * gain)))
        apron &= rival >= 1.25 * neighbor_radius
        np.minimum(rival_distance, rival, out=rival_distance)
    if np.count_nonzero(apron) < 80:
        return None

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    exterior = lab[apron]
    backing = np.median(exterior, axis=0)
    residual = np.percentile(np.linalg.norm(exterior - backing, axis=1), 90)
    if residual > 9:
        return None
    contrast = np.linalg.norm(lab - backing, axis=2)
    foreground = contrast > max(12, residual + 7)
    kernel = np.ones((3, 3), np.uint8)
    substrate_edge = cv2.dilate((~foreground).astype(np.uint8), kernel) > 0
    pale = (lab[:, :, 0] >= max(165, backing[0] + 8)) & (np.linalg.norm(lab[:, :, 1:] - 128, axis=2) <= 24)
    plausible = foreground & pale & (own_distance <= radius) & (rival_distance >= own_distance + max(3, 0.25 * radius))
    rim_samples = lab[plausible & substrate_edge]
    if len(rim_samples) < radius:
        return None
    rim = np.median(rim_samples, axis=0)
    plausible &= np.linalg.norm(lab - rim, axis=2) <= 14
    count, components = cv2.connectedComponents((plausible & (core == 0)).astype(np.uint8))
    touches_core = np.bincount(components[cv2.dilate(core, kernel) != 0], minlength=count) > 0
    touches_backing = np.bincount(components[substrate_edge], minlength=count) >= 0.10 * radius
    keep = touches_core & touches_backing
    keep[0] = False
    addition = keep[components]
    added = np.count_nonzero(addition)
    core_area = np.count_nonzero(core)
    if added < max(1, 0.01 * core_area) or added > 0.32 * core_area:
        return None
    contours, _ = cv2.findContours(core | addition.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    points = cv2.approxPolyDP(contour, 0.45, True).reshape(-1, 2)
    return points.astype(np.float32) / gain + origin if len(points) >= 4 else None
