from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from app.pipeline.geometry import rectify_crop_quad
from app.pipeline.protected_regions import region_box_in_pixels


@dataclass(frozen=True)
class BorderEdgeAssessment:
    edge: str
    trim_pixels: int
    confidence: float
    is_physical_border: bool
    reason_codes: tuple[str, ...]
    measurements: dict[str, float]
    rim_bgr: tuple[int, int, int] | None = None
    # Signed inward distance to the first picture pixel at either end of the side.
    insets: tuple[float, float] = (0.0, 0.0)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BorderTrimPlan:
    offsets: dict[str, int]
    confidence: float
    applied: bool
    reason_codes: tuple[str, ...]
    edges: dict[str, BorderEdgeAssessment]
    crop_quad: np.ndarray | None

    def metadata(self) -> dict:
        return {
            "method": "fitted_print_edges_v3",
            "crop_quad": self.crop_quad.tolist() if self.crop_quad is not None else None,
            "coordinate_space": "untrimmed_rectified_pixel_centers_before_orientation",
            "offsets_kind": "maximum_inward_depth_per_edge",
            "offsets": dict(self.offsets),
            "confidence": self.confidence,
            "applied": self.applied,
            "reason_codes": list(self.reason_codes),
            "edges": {name: edge.as_dict() for name, edge in self.edges.items()},
        }


@dataclass(frozen=True)
class BorderTrimResult(BorderTrimPlan):
    image: np.ndarray


def _oriented(image: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return image
    if edge == "bottom":
        return image[::-1]
    if edge == "left":
        return np.transpose(image, (1, 0, 2))
    if edge == "right":
        return np.transpose(image[:, ::-1], (1, 0, 2))
    raise ValueError(f"Unsupported edge: {edge}")


def _protected_edge_veto(
    edge: BorderEdgeAssessment, width: int, height: int, protected_regions: list[dict] | None
) -> bool:
    if not edge.is_physical_border or not protected_regions:
        return False
    start, end = edge.insets
    for region in protected_regions:
        box = region_box_in_pixels(region, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        horizontal = edge.edge in {"top", "bottom"}
        span = width if horizontal else height
        along_box = (x1, x2 - 1) if horizontal else (y1, y2 - 1)
        boundary = max(start + (end - start) * pos / (span - 1) for pos in along_box)
        clearance = {"left": x1, "right": width - x2, "top": y1, "bottom": height - y2}
        if clearance[edge.edge] < boundary - 1e-4:
            return True
    return False


def _seam_candidates(strip: np.ndarray, max_depth: int) -> set[tuple[int, int]]:
    # Replicate the outer row so the line detector can see even a one-pixel rim.
    pad = 5
    span = strip.shape[1]
    proposal_width = min(span, 900)
    work = strip
    if span > proposal_width:
        work = cv2.resize(strip, (proposal_width, strip.shape[0]), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(work, pad, 0, 0, 0, cv2.BORDER_REPLICATE)
    segments = cv2.createLineSegmentDetector().detect(padded)[0]
    candidates: set[tuple[int, int]] = set()
    if segments is None:
        return candidates
    for x1, y1, x2, y2 in segments[:, 0]:
        if abs(x2 - x1) < max(12, proposal_width * 0.035):
            continue
        # Resize uses pixel centers. Restore their native positions before fitting.
        x1, x2 = (x1 + 0.5) * span / proposal_width - 0.5, (x2 + 0.5) * span / proposal_width - 0.5
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > 0.10:
            continue
        start = round(float(y1 - pad - x1 * slope))
        end = round(float(y1 - pad + (span - 1 - x1) * slope))
        if 0 <= max(start, end) < max_depth:
            candidates.add((start, end))
    return candidates


def _fit_seam(samples: np.ndarray, columns: np.ndarray, start: int, end: int) -> tuple[float, float]:
    # A segment may only describe a short, nearly level part of a sloping seam.
    # Refit to the transitions along its length before extrapolating to the corners.
    span = columns[-1]
    positions = np.rint(start + (end - start) * columns / span).astype(int)
    visible = np.flatnonzero(positions >= 0)
    if len(visible) < len(columns) * 0.42:
        return start, end
    rows = np.maximum(positions[visible] + np.arange(-3, 4)[:, None], 0)
    gradient = samples[rows, visible] - samples[rows + 1, visible]
    direction = np.sign(np.median(samples[rows[0], visible] - samples[rows[-1] + 1, visible]))
    gradient *= direction
    peaks = np.argmax(gradient, axis=0)
    indices = np.arange(len(visible))
    peak_strength = gradient[peaks, indices]
    strong = peak_strength >= 4
    if np.count_nonzero(strong) < len(visible) * 0.42:
        return start, end
    # Interpolate the gradient peak between samples. A hard paper/picture
    # step at pixel d lies at d - 0.5 in pixel-center coordinates.
    before = gradient[np.maximum(0, peaks - 1), indices]
    after = gradient[np.minimum(len(rows) - 1, peaks + 1), indices]
    shift = 0.5 * (after - before) / np.maximum(2 * peak_strength - before - after, 1.0)
    centers = rows[peaks, indices] + 0.5 + np.clip(shift, -0.5, 0.5)
    points = np.column_stack((columns[visible[strong]], centers[strong]))
    vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vy) > 0.10 * vx:
        return start, end
    return float(y - x * vy / vx), float(y + (span - x) * vy / vx)


def _is_pale_paper(color: tuple[int, int, int]) -> bool:
    return min(color) >= 145 and max(color) - min(color) <= 55


def _picture_side_line(
    smooth: np.ndarray, columns: np.ndarray, line: tuple[float, float], paper_tone: float, direction: float,
    sample_step: int = 1,
) -> tuple[float, float] | None:
    """Locate the paper-to-picture crossing, excluding peaks that lead back into paper."""
    start, end = line
    predicted = start + (end - start) * columns / columns[-1]
    valid = predicted >= -0.5
    x = columns[valid]
    indices = np.flatnonzero(valid)
    center = np.maximum(0, np.rint(predicted[valid]).astype(int))
    rows = np.clip(center + np.arange(-4 * sample_step, 4 * sample_step + 1)[:, None], 0, len(smooth) - 2)
    inner_rows = np.minimum(center + sample_step * np.arange(3, 8)[:, None], len(smooth) - 1)
    inner = np.median(smooth[inner_rows, indices], axis=0)
    # At a decorative ruling the largest gradient may point toward the ink.
    # A print boundary crosses to the local scene tone and stays on that side.
    level = (paper_tone + inner) * 0.5
    before = direction * (smooth[rows, indices] - level)
    after = direction * (level - smooth[rows + 1, indices])
    stays_inside = direction * (smooth[np.minimum(rows + 2 * sample_step, len(smooth) - 1), indices] - level) <= 1
    crosses = (before >= 0) & (after >= 0) & (before + after >= 2 / sample_step) & stays_inside
    usable = np.any(crosses, axis=0) & (direction * (paper_tone - inner) >= 6)
    if np.count_nonzero(usable) < 0.26 * len(columns):
        return None
    locations = rows + before / np.maximum(before + after, 1)
    distance = np.where(crosses, np.abs(locations - predicted[valid]), np.inf)
    closest = np.argmin(distance, axis=0)
    points = np.column_stack((x[usable], locations[closest[usable], np.flatnonzero(usable)]))
    vx, vy, cx, cy = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vy) > 0.10 * vx:
        return None
    return float(cy - cx * vy / vx), float(cy + (columns[-1] - cx) * vy / vx)


def _measure_strip(
    view: np.ndarray,
    edge: str,
    *,
    origin: int,
    allowed: int,
    matching_paper: np.ndarray | None = None,
) -> BorderEdgeAssessment:
    span = view.shape[1]
    if allowed < 2 or span < 16:
        return BorderEdgeAssessment(edge, 0, 0.0, False, ("BORDER_IMAGE_TOO_SMALL",), {})
    max_depth = origin + allowed

    gray = cv2.cvtColor(view[: max_depth + 6], cv2.COLOR_BGR2GRAY)
    columns = np.linspace(0, span - 1, min(160, span)).astype(int)
    samples = gray[:, columns].astype(np.float32)
    # Reduce grain along the seam, keeping its native depth and one-pixel edges.
    smooth = cv2.blur(gray, (5, 1))[:, columns].astype(np.float32)
    best_score = 0.0
    best_color: tuple[int, int, int] | None = None
    winning_line = None
    winning_tone, winning_direction = 0.0, 0.0
    candidate_measurements: dict[str, float] = {}
    candidates = _seam_candidates(gray, max_depth)
    fits = {pair: _fit_seam(smooth, columns, *pair) for pair in candidates}
    candidates.update((round(start), round(end)) for start, end in fits.values())
    for start, end in sorted(candidates):
        if not 0 <= max(start, end) < max_depth:
            continue
        positions = np.rint(start + (end - start) * columns / (span - 1)).astype(int)
        visible = np.flatnonzero(positions >= 0)
        if len(visible) < 0.42 * len(columns):
            continue
        seam = positions[visible]
        outside = np.maximum(seam - 2, 0)
        if matching_paper is not None:
            band = np.maximum(0, seam + np.arange(-3, 1)[:, None])
            colors = view[band, columns[visible]]
            nearest = np.argmin(np.linalg.norm(colors - matching_paper, axis=2), axis=0)
            outside = band[nearest, np.arange(len(visible))]
        rim = samples[outside, visible]
        tone = np.median(rim)
        near_match = np.mean(np.abs(rim - tone) <= 13)
        if matching_paper is not None:
            near_match = np.mean(np.max(np.abs(view[outside, columns[visible]] - matching_paper), axis=1) <= 22)
        # A decorative margin may contain ink, but paper should recur all the
        # way to the frame. This rules out long lines inside textured subjects.
        # The halo can contain the adjacent album decoration or another print.
        # Require paper between the seam and the predicted edge, plus its near surround.
        outer = np.minimum(origin, np.maximum(0, outside - 8))
        exterior_rows = outer + (outside - outer) * np.array([[0], [.25], [.50], [.75]])
        exterior = samples[exterior_rows.astype(int), visible]
        frame_match = np.mean(np.abs(exterior - tone) <= 18)
        if near_match < 0.68 or (matching_paper is None and frame_match < 0.33):
            continue
        near_delta = smooth[outside, visible] - smooth[seam + 2, visible]
        inner_delta = tone - smooth[seam + 4, visible]
        informative = np.abs(near_delta) >= 6
        direction = np.sign(np.median(near_delta[informative])) if informative.any() else 0.0
        support = np.mean((direction * near_delta >= 6) & (direction * inner_delta >= 6))
        if support < (0.32 if matching_paper is not None else 0.54):
            continue
        if matching_paper is not None:
            # Evidence may stop where a pale sky meets pale paper. Only the
            # corroborated pass can ignore these locally indistinguishable pixels.
            distinct = np.abs(inner_delta) >= 6
            support = np.mean(direction * near_delta[distinct] >= 6) if distinct.any() else 0.0
        # Ink ruled across a margin can have a stronger edge than the print.
        # If the band beyond it returns to the same paper, keep looking inward.
        if np.mean(direction * inner_delta > 32) > 0.80:
            probe_end = min(max_depth + 1 - seam.max(), 32)
            if probe_end > 5:
                beyond = smooth[seam + np.arange(5, probe_end)[:, None], visible]
                if np.any(np.mean(np.abs(beyond - tone) <= 6, axis=1) >= 0.80):
                    continue
        score = float(0.55 * support + 0.25 * near_match + 0.20 * frame_match)
        if score < 0.68:
            continue
        blue, green, red = np.median(view[outside, columns[visible]], axis=0)
        color = (int(blue), int(green), int(red))
        # A printed outline can be sharper than the inner edge of its paper.
        if best_color is not None and (_is_pale_paper(color), score) <= (_is_pale_paper(best_color), best_score):
            continue
        best_color = color
        best_score = score
        winning_line = (start, end)
        winning_tone, winning_direction = float(tone), float(direction)
        candidate_measurements = {
            "boundary_support": round(float(support), 4),
            "outer_tone_support": round(float(frame_match), 4),
            "rim_luminance": winning_tone,
            "transition_direction": winning_direction,
        }

    if winning_line is None:
        return BorderEdgeAssessment(edge, 0, 0.0, False, ("NO_SUPPORTED_PHYSICAL_BORDER",), {})

    fitted = _picture_side_line(smooth, columns, winning_line, winning_tone, winning_direction)
    if fitted is None:
        fitted = fits.get(winning_line) or _fit_seam(smooth, columns, *winning_line)
    # Move from the measured transition to the first picture pixel center.
    # Keep the two distances independent when the paper narrows to a corner.
    insets = np.asarray(fitted) + 0.5 - origin
    return BorderEdgeAssessment(
        edge,
        max(0, int(np.ceil(insets.max()))),
        round(best_score, 4),
        True,
        ("PHYSICAL_BORDER_TRANSITION_SUPPORTED",),
        candidate_measurements,
        best_color,
        (float(insets[0]), float(insets[1])),
    )


def _assess_edge(
    view: np.ndarray, edge: str, *, origin: int, max_crop_ratio: float, padding: int, sample_step: int,
    matching_paper: np.ndarray | None = None,
) -> BorderEdgeAssessment:
    depth = view.shape[0] - origin
    allowed = min(round(depth * max_crop_ratio), depth // 3)
    strip = view[:origin + allowed + 9 * sample_step]
    if sample_step > 1 and allowed >= 2 * sample_step and view.shape[1] >= 16 * sample_step:
        # Compare paper and scene at a common scale, then localize in the native
        # scan. A fixed two-pixel probe cannot resolve a soft high-resolution rim.
        preview = cv2.resize(strip, None, fx=1 / sample_step, fy=1 / sample_step, interpolation=cv2.INTER_AREA)
        coarse_origin = round(origin / sample_step)
        result = _measure_strip(
            preview, edge, origin=coarse_origin, allowed=round(allowed / sample_step), matching_paper=matching_paper,
        )
        if result.is_physical_border and max(result.insets) + coarse_origin >= 1.5:
            columns = np.linspace(0, strip.shape[1] - 1, min(160, strip.shape[1])).astype(int)
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            smooth = cv2.blur(gray, (5 * sample_step, 1))[:, columns].astype(np.float32)
            # The resize maps pixel centers by (x + 0.5) / scale - 0.5.
            a, b = (v + coarse_origin for v in result.insets)
            x = (np.array([0, strip.shape[1] - 1]) + 0.5) / sample_step - 0.5
            prediction = (a + (b - a) * x / (preview.shape[1] - 1)) * sample_step - 0.5
            fitted = _picture_side_line(
                smooth, columns, tuple(prediction), result.measurements["rim_luminance"],
                result.measurements["transition_direction"], sample_step,
            )
            insets = np.asarray(fitted if fitted is not None else prediction) + 0.5 - origin
        else:
            # A single native row can disappear or take on scene color in an area average.
            near_origin = min(origin, 3 * sample_step)
            result = _measure_strip(
                view[origin - near_origin:], edge, origin=near_origin,
                allowed=min(allowed, 3 * sample_step), matching_paper=matching_paper,
            )
            insets = np.array(result.insets)
    else:
        result = _measure_strip(strip, edge, origin=origin, allowed=allowed, matching_paper=matching_paper)
        insets = np.array(result.insets)
    if not result.is_physical_border:
        return result
    measurements = {**result.measurements, "seam_start": float(insets[0] - 0.5), "seam_end": float(insets[1] - 0.5)}
    insets = np.clip(insets - max(padding, 0), -origin, allowed)
    insets = np.where(np.abs(insets - np.rint(insets)) < 0.035, np.rint(insets), insets)
    return replace(
        result, trim_pixels=max(0, int(np.ceil(insets.max()))), is_physical_border=bool(np.any(insets)),
        insets=(float(insets[0]), float(insets[1])), measurements=measurements,
    )


def _blended_paper_edge(view: np.ndarray, paper: np.ndarray) -> float:
    # The warp can blend a subpixel rim with different scene colors along a
    # whole edge. Compare the outer row to known paper and the local texture.
    strip = cv2.blur(view[:5].astype(np.float32), (5, 1))
    distance = np.linalg.norm(strip - paper, axis=2)
    steps = np.diff(distance, axis=0)
    local_texture = np.median(np.abs(steps[1:]), axis=0)
    return float(np.mean(steps[0] > np.maximum(14.0, 1.65 * local_texture)))


def _edge_quad(edges: dict[str, BorderEdgeAssessment], width: int, height: int) -> np.ndarray:
    # Lines in the untrimmed ROI. Intersect neighboring sides instead of
    # assuming that a seam ends on the original frame's axis-aligned side.
    lines = {}
    for name, edge in edges.items():
        start, end = edge.insets if edge.is_physical_border else (0.0, 0.0)
        if name in {"top", "bottom"}:
            if name == "bottom":
                start, end = height - 1 - start, height - 1 - end
            lines[name] = [(start - end) / (width - 1), 1.0, -start]
        else:
            if name == "right":
                start, end = width - 1 - start, width - 1 - end
            lines[name] = [1.0, (start - end) / (height - 1), -start]
    horizontal = [lines[name] for name in ("top", "top", "bottom", "bottom")]
    vertical = [lines[name] for name in ("left", "right", "right", "left")]
    intersections = np.cross(horizontal, vertical)
    return (intersections[:, :2] / intersections[:, 2:]).astype(np.float32)


def find_physical_border(
    image: np.ndarray,
    *,
    reference_box: tuple[int, int, int, int] | None = None,
    max_crop_ratio: float = 0.12,
    padding: int = 0,
    protected_regions: list[dict] | None = None,
) -> BorderTrimPlan:
    """Fit the inner print outline. The quad and protected boxes use the reference ROI."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Border classification expects a three-channel BGR image.")
    analysis_height, analysis_width = image.shape[:2]
    x1, y1, x2, y2 = reference_box if reference_box is not None else (0, 0, analysis_width, analysis_height)
    if reference_box is not None and not (0 <= x1 < x2 <= analysis_width and 0 <= y1 < y2 <= analysis_height):
        raise ValueError("Border reference box must be inside the analysis image.")
    width, height = x2 - x1, y2 - y1
    sample_step = max(1, round(max(width, height) / 700))
    reference = image[y1:y2, x1:x2]
    # Include only real neighboring page pixels, and keep each side's span in
    # reference coordinates so its two depths can be intersected with its neighbors.
    strips = {
        "top": (image[:y2, x1:x2], y1),
        "bottom": (_oriented(image[y1:, x1:x2], "bottom"), analysis_height - y2),
        "left": (_oriented(image[y1:y2, :x2], "left"), x1),
        "right": (_oriented(image[y1:y2, x1:], "right"), analysis_width - x2),
    }
    edges = {
        name: _assess_edge(view, name, origin=origin, max_crop_ratio=max_crop_ratio, padding=padding, sample_step=sample_step)
        for name, (view, origin) in strips.items()
    }
    paper_colors = [edge.rim_bgr for edge in edges.values() if edge.rim_bgr and _is_pale_paper(edge.rim_bgr)]
    paper = np.median(paper_colors, axis=0) if paper_colors else None
    for name, edge in edges.items():
        view, origin = strips[name]
        if paper is not None and origin > 0 and edge.reason_codes == ("NO_SUPPORTED_PHYSICAL_BORDER",):
            edge = _assess_edge(
                view, name, origin=origin, max_crop_ratio=max_crop_ratio, padding=padding, sample_step=sample_step, matching_paper=paper,
            )
        color = edge.rim_bgr
        if paper is not None and color is None and edge.reason_codes == ("NO_SUPPORTED_PHYSICAL_BORDER",):
            support = _blended_paper_edge(_oriented(reference, name), paper)
            if support >= 0.70 and padding <= 0:
                edge = BorderEdgeAssessment(
                    name, 1, round(support, 4), True, ("RESAMPLED_PAPER_EDGE",),
                    {"boundary_support": support}, insets=(1.0, 1.0),
                )
        elif color is not None and not _is_pale_paper(color):
            # A single dark wall or saturated sky can also end in a sharp line.
            if not any(
                other.edge != name and other.rim_bgr is not None
                and max(abs(a - b) for a, b in zip(color, other.rim_bgr)) <= 18
                for other in edges.values()
            ):
                edge = replace(edge, trim_pixels=0, is_physical_border=False, reason_codes=("BORDER_TRIM_UNCERTAIN",))
        if _protected_edge_veto(edge, width, height, protected_regions):
            edge = replace(edge, trim_pixels=0, is_physical_border=False, reason_codes=("BORDER_PROTECTED_CONTENT_VETO",))
        edges[name] = edge

    offsets = {name: edge.trim_pixels for name, edge in edges.items()}
    reasons = tuple(dict.fromkeys(reason for edge in edges.values() for reason in edge.reason_codes))
    supported = [edge.confidence for edge in edges.values() if edge.is_physical_border]
    if not supported:
        return BorderTrimPlan(offsets, 0.0, False, reasons, edges, None)
    quad = _edge_quad(edges, width, height)
    quad = np.clip(quad, [-x1, -y1], [analysis_width - x1 - 1, analysis_height - y1 - 1]).astype(np.float32)
    widths = quad[[1, 2], 0] - quad[[0, 3], 0] + 1
    heights = quad[[3, 2], 1] - quad[[0, 1], 1] + 1
    if min(widths) < width * 0.62 or min(heights) < height * 0.62:
        return BorderTrimPlan(dict.fromkeys(edges, 0), 0.0, False, ("BORDER_TRIM_CONTENT_AREA_VETO",), edges, None)
    return BorderTrimPlan(offsets, min(supported), True, reasons, edges, quad)


def trim_physical_border(
    image: np.ndarray,
    *,
    max_crop_ratio: float = 0.12,
    padding: int = 0,
    protected_regions: list[dict] | None = None,
) -> BorderTrimResult:
    plan = find_physical_border(image, max_crop_ratio=max_crop_ratio, padding=padding, protected_regions=protected_regions)
    trimmed = rectify_crop_quad(image, plan.crop_quad) if plan.crop_quad is not None else image.copy()
    return BorderTrimResult(**vars(plan), image=trimmed)


def remove_uniform_border(
    image: np.ndarray,
    *,
    max_crop_ratio: float = 0.12,
    min_content_ratio: float = 0.012,
    color_distance_threshold: float = 18.0,
    padding: int = 2,
) -> np.ndarray:
    """Backward-compatible wrapper around the content-aware classifier.

    The legacy parameters remain accepted for callers, but trimming decisions
    no longer rely on a global pixel-count threshold.
    """
    del min_content_ratio, color_distance_threshold
    return trim_physical_border(image, max_crop_ratio=max_crop_ratio, padding=padding).image
