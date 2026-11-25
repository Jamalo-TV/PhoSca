"""Visible outlines of bright prints cut by a dark-backed capture edge."""

from typing import Literal

import cv2
import numpy as np

from app.pipeline.geometry import order_quad_cyclic
from app.pipeline.print_edges import _side_line


CAPTURE_EDGE_REVISION = 6
CAPTURE_EDGE_SOURCE = "capture_edge_boundary"
CAPTURE_EDGE_HOLD = "PRINT_AT_CAPTURE_EDGE"
SCANNER_BACKING_REVIEW = "SCANNER_BACKING_UNCERTAIN"
CLIPPED_PRINT_OVERLAP_REVIEW = "CLIPPED_PRINT_OVERLAP_UNCERTAIN"


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


def _frame_sides(quad: np.ndarray, ends: np.ndarray, image: np.ndarray) -> np.ndarray:
    limit = np.array(image.shape[1::-1]) - 1
    return ((quad == ends) & ((quad == 0) | (quad == limit))).any(axis=1)


def clipped_print_overlap_uncertain(image: np.ndarray, quad: np.ndarray, clipped: np.ndarray) -> bool:
    """Check whether a contained contour reuses a visible side of another print."""
    height, width = image.shape[:2]
    ends = np.roll(clipped, -1, axis=0)
    sides = ends - clipped
    norms = sides[:, [1, 0]] * np.float32([-1, 1]) / np.linalg.norm(sides, axis=1)[:, None]
    # The capture edge contains no physical paper boundary to share.
    frame_side = _frame_sides(clipped, ends, image)
    if frame_side.all():
        return False
    distances = np.abs(np.einsum("vsi,si->vs", quad[:, None] - clipped, norms))[:, ~frame_side]
    shared = (np.maximum(distances, np.roll(distances, -1, axis=0)) <= 3).any(axis=1)
    if not shared.any() or shared.all():
        return False
    radius = int(np.clip(round(min(height, width) * .008), 4, 18))
    for index in np.flatnonzero(~shared):
        line, evidence = _side_line(image, quad[index], quad[(index + 1) % 4], radius)
        if line is None or evidence["support"] < .80 or (evidence["corner_support"] or 0) < .60:
            return True
    return False


def clipped_fragment_needs_review(image: np.ndarray, quad: np.ndarray) -> bool:
    """Check for three proposed sides with no supported separate paper edge."""
    ends = np.roll(quad, -1, axis=0)
    visible = np.flatnonzero(~_frame_sides(quad, ends, image))
    if len(visible) != 3:
        return False
    radius = int(np.clip(round(min(image.shape[:2]) * .006), 5, 18))
    for index in visible:
        line, evidence = _side_line(image, quad[index], ends[index], radius)
        if line is not None or evidence["ambiguous"] or evidence["support"] >= .55:
            return False
    return True


def _sample_backing(
    image: np.ndarray, occluded: np.ndarray, points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [np.ascontiguousarray(axis) for axis in np.moveaxis(points, -1, 0)]
    colors = cv2.cvtColor(cv2.remap(image, x, y, cv2.INTER_LINEAR), cv2.COLOR_BGR2LAB)
    blocked = cv2.remap(occluded, x, y, cv2.INTER_NEAREST, borderValue=1)
    height, width = occluded.shape
    outside = (x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)
    return colors.astype(np.float32), (blocked != 0) | outside


def _quiet_backing_across_side(
    image: np.ndarray, occluded: np.ndarray, start: np.ndarray, end: np.ndarray, normal: np.ndarray,
) -> bool:
    centers = start + np.linspace(.035, .965, 120, dtype=np.float32)[:, None] * (end - start)
    offsets = np.arange(-8, 9, dtype=np.float32)
    samples, blocked = _sample_backing(image, occluded, centers[:, None, :] + offsets[:, None] * normal)
    visible = ~blocked.any(axis=1)
    if np.count_nonzero(visible) < 18:
        return False
    samples = samples[visible]
    quiet = (samples[..., 0].max(axis=1) <= 76) & (np.linalg.norm(samples[:, -1] - samples[:, 0], axis=1) <= 4)
    # A weak but consistently signed material change is still evidence for a
    # physical edge. Texture noise in a single surface is much less coherent.
    step = samples[:, 9:].mean(axis=1) - samples[:, :8].mean(axis=1)
    return np.mean(quiet) >= .80 and np.linalg.norm(np.median(step, axis=0)) <= 1.25


def _continuous_rail_across_side(
    image: np.ndarray, occluded: np.ndarray, start: np.ndarray, end: np.ndarray, axis: int,
) -> bool:
    across = 1 - axis
    first, last = np.sort([start[across], end[across]])
    inset = .07 * (last - first)
    # One native station per possible rail, capped for very large captures.
    coordinates = np.arange(np.ceil(first + inset), np.floor(last - inset) + 1,
                            max(1, int(np.ceil((last - first) / 3000))), dtype=np.float32)
    if not len(coordinates):
        return False
    centers = start + ((coordinates - start[across]) / (end[across] - start[across]))[:, None] * (end - start)
    along = np.linspace(-32, 32, 17, dtype=np.float32)
    across_offsets = np.arange(-4, 5, dtype=np.float32)
    points = np.broadcast_to(centers[:, None, None, :], (len(centers), len(along), len(across_offsets), 2)).copy()
    points[..., axis] += along[None, :, None]
    points[..., across] += across_offsets[None, None, :]
    colors, blocked = _sample_backing(image, occluded, points.reshape(len(centers), -1, 2))
    light = colors[..., 0].reshape(points.shape[:3])
    available = ~blocked.any(axis=1) & (light.max(axis=(1, 2)) <= 76)
    if not available.any():
        return False
    step = light[available, :, 5:].mean(axis=-1) - light[available, :, :4].mean(axis=-1)
    median = np.median(step, axis=1)
    signed = step * np.sign(median[:, None])
    # Verify the same side of the same line stays brighter both before and
    # after it crosses the supposed print edge.
    stable = (signed >= 2)
    return bool(np.any(
        (np.abs(median) >= 3.5) & stable[:, 0] & stable[:, -1]
        & (stable[:, :8].sum(axis=1) >= 7) & (stable[:, 9:].sum(axis=1) >= 7)
    ))


def _has_compact_tone_change(light: np.ndarray, surface: np.ndarray, reduction: float) -> bool:
    width, height = (max(1, round(size * reduction)) for size in light.shape[::-1])
    reduced = cv2.resize(light, (width, height), interpolation=cv2.INTER_AREA)
    clear = np.uint8(cv2.resize(255 * surface, (width, height), interpolation=cv2.INTER_AREA) == 255)
    horizontal = np.ones((1, 75), np.uint8)
    vertical = horizontal.T
    inspected = (cv2.erode(clear, horizontal, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                 & cv2.erode(clear, vertical, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    # Compare a patch to the backing in both axes. A dark rectangle with a
    # blurred edge can have a coherent tone but barely any texture.
    for operation, fill, threshold in ((cv2.MORPH_BLACKHAT, 0, 5), (cv2.MORPH_TOPHAT, 255, 8)):
        tones = np.where(clear, reduced, fill).astype(np.uint8)
        differences = np.minimum(cv2.morphologyEx(tones, operation, horizontal),
                                 cv2.morphologyEx(tones, operation, vertical))
        changed = np.uint8((differences >= threshold) & (inspected != 0))
        if np.any(cv2.boxFilter(changed, cv2.CV_32F, (17, 17), normalize=False) >= 160):
            return True
    return False


def _has_local_tonal_corner(light: np.ndarray, surface: np.ndarray) -> bool:
    reduction = min(.5, 1400 / max(light.shape))
    width, height = (round(size * reduction) for size in light.shape[::-1])
    span = 9
    if min(width, height) < 3 * span:
        return False
    size = (width, height)
    reduced = cv2.resize(light, size, interpolation=cv2.INTER_AREA)
    clear = np.uint8(cv2.resize(255 * surface, size, interpolation=cv2.INTER_AREA) == 255)
    # Include the complete sample windows in the clearance test, even near
    # an adjacent print. The outer radius is span + span // 2.
    kernel = np.ones((3 * span, 3 * span), np.uint8)
    inspect = cv2.erode(clear, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)[span:-span, span:-span]
    min_area = 130
    if np.count_nonzero(inspect) < min_area:
        return False
    average = cv2.boxFilter(reduced, cv2.CV_32F, (span, span), normalize=True)
    tl = average[:-2 * span, :-2 * span]
    tr = average[:-2 * span, 2 * span:]
    bl = average[2 * span:, :-2 * span]
    br = average[2 * span:, 2 * span:]
    corner = np.zeros_like(inspect, dtype=bool)
    # An isolated light or dark quadrant differs from both adjacent ones.
    # Charge the change against the variation among the other three so a
    # single long scanner rail or a smooth gradient contributes little.
    for target, across, down, diagonal in ((tl, tr, bl, br), (tr, tl, br, bl),
                                           (bl, br, tl, tr), (br, bl, tr, tl)):
        step_x, step_y = target - across, target - down
        variation = np.maximum(np.abs(diagonal - across), np.abs(diagonal - down))
        corner |= ((step_x * step_y > 0)
                   & (np.minimum(np.abs(step_x), np.abs(step_y)) >= 3 + variation))
    supported = np.uint8(corner) & inspect
    if np.count_nonzero(supported) < min_area:
        return False
    _, _, components, _ = cv2.connectedComponentsWithStats(supported)
    return bool(np.any(components[1:, cv2.CC_STAT_AREA] >= min_area))


def _has_independent_content(image: np.ndarray, occluded: np.ndarray, quad: np.ndarray) -> bool:
    x, y, width, height = cv2.boundingRect(quad)
    surface = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(surface, np.rint(quad - np.float32([x, y])).astype(np.int32), 1)
    region = np.s_[y:y + height, x:x + width]
    surface[occluded[region] != 0] = 0
    interior = cv2.erode(surface, np.ones((17, 17), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    light = cv2.cvtColor(image[region], cv2.COLOR_BGR2LAB)[..., 0]
    if _has_compact_tone_change(light, interior, min(1., 544 / min(image.shape[:2]))):
        return True

    # A rail can be strong in one direction. A small patch of photographic
    # texture within the proposed outline cannot be dismissed as backing.
    observed = False
    window = (97, 97)
    # Use bounded strips on large scans, with native context for the
    # box window, Gaussian filter and gradients on strip boundaries.
    for top in range(0, height, 256):
        bottom = min(top + 256, height)
        start, end = max(0, top - 53), min(height, bottom + 53)
        core = slice(top - start, bottom - start)
        clear = interior[start:end]
        inspected = cv2.boxFilter(clear, cv2.CV_32F, window, normalize=False,
                                 borderType=cv2.BORDER_CONSTANT)[core] >= 8200
        if not inspected.any():
            continue
        observed = True
        smoothed = cv2.GaussianBlur(light[start:end].astype(np.float32), (0, 0), 1)
        dx = np.abs(cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, scale=1 / 8))
        dy = np.abs(cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, scale=1 / 8))
        detail = np.uint8((np.minimum(dx, dy) > 1.45) & (clear != 0))
        counts = cv2.boxFilter(detail, cv2.CV_32F, window, normalize=False, borderType=cv2.BORDER_CONSTANT)[core]
        if np.any(counts[inspected] > 275):
            return True
    return not observed or _has_local_tonal_corner(light, surface)


def scanner_backing_evidence(
    image: np.ndarray, quad: np.ndarray, clipped: list[np.ndarray], *, allow_rejection: bool = True,
) -> Literal["backing", "uncertain"] | None:
    """Check whether a classical outline follows scanner hardware across neighboring prints."""
    sides = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(sides, axis=1)
    if np.any(lengths < 100):
        return None
    directions = sides / lengths[:, None]
    aligned = np.abs(directions).max(axis=1) >= .998
    oblique = np.flatnonzero(~aligned)
    if (len(oblique) != 2 or (oblique[1] - oblique[0]) % 2 != 1
            or len(set(np.abs(directions[aligned]).argmax(axis=1))) != 2
            or np.abs(directions[oblique]).min() < .25):
        return None
    occluded = np.zeros(image.shape[:2], np.uint8)
    for print_outline in clipped:
        cv2.fillConvexPoly(occluded, np.rint(print_outline).astype(np.int32), 1)
    occluded = cv2.dilate(occluded, np.ones((13, 13), np.uint8))
    normals = directions[:, [1, 0]] * np.float32([-1, 1])
    sides_to_test = [(quad[index], quad[(index + 1) % 4]) for index in oblique]
    if not all(_quiet_backing_across_side(image, occluded, start, end, normals[index])
               for index, (start, end) in zip(oblique, sides_to_test, strict=True)):
        return None
    tracks = [{axis for axis in (0, 1) if _continuous_rail_across_side(image, occluded, start, end, axis)}
              for start, end in sides_to_test]
    if not tracks[0] or not tracks[1] or len(tracks[0] | tracks[1]) != 2:
        return None
    return "uncertain" if not allow_rejection or _has_independent_content(image, occluded, quad) else "backing"
