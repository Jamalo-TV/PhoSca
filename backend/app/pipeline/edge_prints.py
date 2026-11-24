"""Visible outlines of bright prints cut by a dark-backed capture edge."""

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _side_line


CAPTURE_EDGE_REVISION = 1
CAPTURE_EDGE_SOURCE = "capture_edge_boundary"
CAPTURE_EDGE_HOLD = "PRINT_AT_CAPTURE_EDGE"


def _photo_texture(image: np.ndarray, quad: np.ndarray) -> bool:
    target = np.float32([[0, 0], [191, 0], [191, 191], [0, 191]])
    view = cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (192, 192))
    light = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)[16:-16, 16:-16].astype(np.float32)
    dx = np.abs(cv2.Sobel(light, cv2.CV_32F, 1, 0) / 8)
    dy = np.abs(cv2.Sobel(light, cv2.CV_32F, 0, 1) / 8)
    # Scanner cards, flat backing, and ruled paper lack distributed texture in
    # two directions even when their outline looks like a photograph.
    light, dx, dy = [values.reshape(4, 40, 4, 40).transpose(0, 2, 1, 3).reshape(16, -1)
                     for values in (light, dx, dy)]
    spread = np.ptp(np.percentile(light, [10, 90], axis=1), axis=0)
    return np.count_nonzero((spread > 15) & (np.percentile(dx, 90, axis=1) > 3)
                            & (np.percentile(dy, 90, axis=1) > 3)) >= 4


def _edge_seed(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    _, width = shape
    x, y, span_x, span_y = cv2.boundingRect(contour)
    component = np.zeros((span_y, span_x), np.uint8)
    cv2.drawContours(component, [contour - np.int32([[[x, y]]])], -1, 1, cv2.FILLED)
    rows = np.arange(round(.18 * span_y), round(.82 * span_y))
    rows = rows[component[rows].any(axis=1)]
    if len(rows) < .50 * span_y:
        return None
    left = np.argmax(component[rows], axis=1)
    vx, vy, x0, y0 = cv2.fitLine(np.float32(np.column_stack((left, rows))), cv2.DIST_HUBER, 0, .01, .01).ravel()
    if abs(vy) < .97:
        return None
    line = x0 + (np.arange(span_y) - y0) * (vx / vy)
    if np.mean(np.abs(left - line[rows]) <= 2.5) < .82:
        return None
    # A bright scanner strip may join the photo near a cropped corner. Use
    # the stable physical side to seed the outline and measure it again below.
    component[np.arange(span_x)[None, :] < line[:, None] - 2] = 0
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    if not cv2.contourArea(hull) or cv2.countNonZero(component) < .90 * cv2.contourArea(hull):
        return None
    points = cv2.approxPolyDP(hull, .008 * cv2.arcLength(hull, True), True)[:, 0] + [x, y]
    if len(points) != 4 or np.count_nonzero(points[:, 0] == width - 1) != 2:
        return None
    return order_quad_cyclic(points)


def _visible_edge_supported(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, line: np.ndarray, backing: float,
) -> bool:
    tangent = end - start
    length = np.linalg.norm(tangent)
    inward = np.float32([-tangent[1], tangent[0]]) / length
    alignment = line[:2] @ inward
    if abs(alignment) < .9:
        return False
    middle = start + np.linspace(.10, .90, 96, dtype=np.float32)[:, None] * tangent
    distance = -(middle @ line[:2] + line[2]) / alignment
    points = middle[:, None, :] + (distance[:, None, None] + np.float32([-2.5, 2.5])[None, :, None]) * inward
    if np.any((points < 0) | (points > np.array(image.shape[1::-1]) - 1)):
        return False
    x, y = np.moveaxis(points.astype(np.float32), -1, 0)
    colors = cv2.cvtColor(cv2.remap(image, x.copy(), y.copy(), cv2.INTER_LINEAR), cv2.COLOR_BGR2LAB)
    outside, inside = colors[:, :, 0].astype(np.float32).T
    return np.mean((outside <= backing + 42) & (inside - outside >= 16)) >= .78


def _measure_clipped_outline(image: np.ndarray, seed: np.ndarray, backing: float) -> np.ndarray | None:
    height, width = image.shape[:2]
    radius = int(np.clip(round(min(height, width) * .004), 4, 18))
    lines = []
    supported = 0
    for start, end in zip(seed, np.roll(seed, -1, axis=0), strict=True):
        is_capture_edge = any(start[axis] == end[axis] and start[axis] in (0, limit)
                              for axis, limit in enumerate((width - 1, height - 1)))
        if is_capture_edge:
            lines.append(np.cross([*start, 1.], [*end, 1.]))
            continue
        line, evidence = _side_line(image, start, end, radius)
        if (line is None or evidence["support"] < .80
                or not _visible_edge_supported(image, start, end, line, backing)):
            return None
        lines.append(line)
        supported += 1
    if supported < 2:
        return None
    vertices = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(np.abs(vertices[:, 2]) < 1e-9):
        return None
    quad = (vertices[:, :2] / vertices[:, 2:3]).astype(np.float32)
    upper = np.float32([width - 1, height - 1])
    if np.linalg.norm(seed - quad, axis=1).max() > 1.8 * radius or np.any((quad < -1) | (quad > upper + 1)):
        return None
    quad = np.clip(quad, 0, upper)
    cut = np.sort(quad[quad[:, 0] == width - 1, 1])
    if len(cut) != 2 or not cv2.isContourConvex(quad):
        return None
    rows = np.rint(cut[0] + np.linspace(.1, .9, 160) * np.ptp(cut)).astype(int)
    native_edge = cv2.cvtColor(image[rows, -1, None, :], cv2.COLOR_BGR2LAB)[:, 0, 0]
    # Check the last source column itself. A thin scanner border can vanish
    # in the working image and would make a fully captured print look cut.
    if np.mean(native_edge > backing + 38) < .68 or not _photo_texture(image, quad):
        return None
    return quad


def clipped_print_outlines(image: np.ndarray) -> list[np.ndarray]:
    """Find visible source quads touching the right side of a dark-backed scan."""
    height, width = image.shape[:2]
    if min(height, width) < 48:
        return []
    reduction = min(1., 1280 / max(height, width))
    dimensions = (round(width * reduction), round(height * reduction))
    if min(dimensions) < 48:
        return []
    reduced = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    small_height, small_width = reduced.shape[:2]
    light = cv2.cvtColor(reduced, cv2.COLOR_BGR2LAB)[:, :, 0]
    backing = float(np.median(light[:, :max(1, small_width // 9)]))
    if backing > 65:
        return []
    mask = cv2.morphologyEx(np.uint8(light > backing + 38), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for contour in contours:
        x, y, span_x, span_y = cv2.boundingRect(contour)
        if (x + span_x != small_width or span_x < .15 * small_width or span_y < .12 * small_height
                or not .025 <= cv2.contourArea(contour) / (small_width * small_height) <= .4):
            continue
        middle = light[y + span_y // 8:y + span_y - span_y // 8, -3:]
        if np.mean(middle > backing + 38) < .72:
            continue
        seed = _edge_seed(contour, (small_height, small_width))
        if seed is None:
            continue
        # Use the raster's actual endpoints after the coarse search.
        high = np.float32([width - 1, height - 1])
        small_high = np.float32([small_width - 1, small_height - 1])
        seed = np.where(seed == small_high, high, seed * (high / small_high))
        quad = _measure_clipped_outline(image, seed, backing)
        if quad is not None:
            outlines.append(quad)
    return outlines
