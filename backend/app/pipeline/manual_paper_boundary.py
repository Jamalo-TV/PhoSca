"""Keep a manual rectangle from replacing a measured, non-straight paper edge."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic
from app.pipeline.print_edges import _edge_band, _side_line, bowed_print_sides, refine_print_boundary


MANUAL_PAPER_BOUNDARY_REVISION = 2
MANUAL_PAPER_BOUNDARY_REVIEW = "PRINT_CURVED_OUTLINE_REVIEW"


def paper_curve_review_reference(correction: dict) -> dict | None:
    full = (correction.get("crop_modes") or {}).get("full_print") or {}
    if "PRINT_BOUNDARY_CURVED" not in (full.get("reason_codes") or []) or len(full.get("warp_quad") or []) != 4:
        return None
    sides = bowed_print_sides(correction.get("boundary_refinement") or {})
    return {"warp_quad": full["warp_quad"], "sides": sides or list(range(4)),
            "traces": full.get("suggested_edge_traces") or []}


def manual_paper_reference(metadata: dict, crop_quad: list[dict] | None) -> dict | None:
    review = metadata.get("review_provenance") or {}
    original = review.get("original_prediction") or {}
    if len(crop_quad or []) != 4 or original.get("segmentation_confidence") is None:
        return None
    mask = original.get("segmentation_mask") or {}
    curve = review.get("paper_curve_review")
    outline = mask.get("visible_polygon") or mask.get("polygon") or []
    seed = mask.get("warp_quad") or mask.get("polygon") or []
    if len(seed) != 4:
        seed = original.get("print_warp_quad") or (curve or {}).get("warp_quad") or []
    legacy_extended = curve is None and mask.get("extended_print_outline") is True
    if len(seed) != 4 or (not curve and not legacy_extended and len(outline) < 8):
        return None
    # Ignore a different print after the reviewer replaces a bad proposal.
    # A smaller rectangle inside the same print still needs the original edge.
    overlap = normalized_polygon_overlap([(p["x"], p["y"]) for p in seed], [(p["x"], p["y"]) for p in crop_quad])
    if overlap.iou < .82 and max(overlap.first_containment, overlap.second_containment) < .90:
        return None
    return {"warp_quad": seed, "outline": outline, "curve": curve, "legacy_extended": legacy_extended}


def manual_paper_download_needs_review(metadata: dict, crop_quad: list[dict] | None) -> bool:
    reference = manual_paper_reference(metadata, crop_quad)
    if reference is None:
        return False
    check = (metadata.get("correction") or {}).get("manual_paper_boundary") or {}
    curve_recheck = bool(reference["curve"] or reference["legacy_extended"])
    if check.get("revision") == MANUAL_PAPER_BOUNDARY_REVISION or (not curve_recheck and check.get("revision") == 1):
        return check.get("requires_review") is not False
    if curve_recheck:
        return True
    # Old ready rows may have retained the original reference but skipped the
    # check. Collinear samples or a clipped inner corner do not suggest an
    # outward bow and can keep their existing download.
    correction = metadata.get("correction") or {}
    native_area = (correction.get("quad_validation") or {}).get("area")
    current = order_quad_cyclic(np.float32([(p["x"], p["y"]) for p in crop_quad]))
    normalized_area = cv2.contourArea(current)
    scale = np.sqrt(native_area / normalized_area) if native_area and normalized_area else 4096
    seed = order_quad_cyclic(np.float32([(p["x"], p["y"]) for p in reference["warp_quad"]])) * scale
    outline = _bounded_outline(np.float32([(p["x"], p["y"]) for p in reference["outline"]]) * scale)
    return outline is not None and any(_side_samples(seed, outline, side) is not None for side in range(4))


def _bounded_outline(outline: np.ndarray) -> np.ndarray | None:
    if len(outline) > 512:
        outline = cv2.approxPolyDP(outline, .20, True)[:, 0]
    return outline if len(outline) <= 512 else None


def _side_samples(quad: np.ndarray, outline: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray] | None:
    start, end = quad[side], quad[(side + 1) % 4]
    tangent = end - start
    length = np.linalg.norm(tangent)
    if length < 40:
        return None
    axis = tangent / length
    inward = np.float32([-axis[1], axis[0]])
    fraction = np.linspace(.06, .94, 96, dtype=np.float32)
    local = np.column_stack(((outline - start) @ axis, (outline - start) @ inward))
    first, last = local, np.roll(local, -1, axis=0)
    distances = fraction[:, None] * length
    between = (distances >= np.minimum(first[:, 0], last[:, 0])) & (distances < np.maximum(first[:, 0], last[:, 0]))
    dx = last[:, 0] - first[:, 0]
    slope = np.divide(last[:, 1] - first[:, 1], dx, out=np.zeros_like(dx), where=dx != 0)
    depth = first[:, 1] + (distances - first[:, 0]) * slope
    crosses = between & (np.abs(depth) <= 10)
    if not (crosses.sum(axis=1) == 1).all():
        return None
    offsets = depth[crosses]
    if (offsets < -2.5).mean() < .22:
        return None
    curve = np.polyval(np.polyfit(fraction, offsets, 2), fraction)
    chord = np.polyval(np.polyfit(fraction, offsets, 1), fraction)
    if np.quantile(np.abs(curve - offsets), .9) > 1 or np.quantile(np.abs(chord - offsets), .7) < .65:
        return None
    return start + fraction[:, None] * tangent + offsets[:, None] * inward, inward


def _native_edge_support(image: np.ndarray, stations: np.ndarray, inward: np.ndarray) -> np.ndarray | None:
    samples = stations[:, None] + np.arange(-17, 11, dtype=np.float32)[None, :, None] * inward
    if np.any((samples < 0) | (samples > np.array(image.shape[1::-1]) - 1)):
        return None
    x, y = np.moveaxis(samples, -1, 0)
    colors = cv2.cvtColor(cv2.remap(image, x.copy(), y.copy(), cv2.INTER_LINEAR), cv2.COLOR_BGR2LAB)
    steps = np.linalg.norm(np.diff(colors.astype(np.float32), axis=1), axis=2)
    edge = np.max(steps[:, 15:19], axis=1)
    support = edge >= 12
    if support.mean() < .90 or min(part.mean() for part in np.array_split(support, 4)) < .80:
        return None
    # An alternative parallel transition can be a paper rim beside printed ink.
    peaks = (steps[:, 1:-1] >= steps[:, :-2]) & (steps[:, 1:-1] > steps[:, 2:])
    rival = peaks & (steps[:, 1:-1] >= np.maximum(13, .58 * edge[:, None]))
    rival[:, 13:19] = False
    if any(rival[:, begin:begin + 3].any(axis=1).mean() >= .47 for begin in range(rival.shape[1] - 2)):
        return None
    # A faint paper-to-page step need not be as strong as an ink boundary.
    faint = peaks[:, :12] & (steps[:, 1:13] >= 4)
    for begin in range(8):
        supported = faint[:, begin:begin + 5].any(axis=1)
        if supported.mean() >= .70 and min(part.mean() for part in np.array_split(supported, 4)) >= .55:
            return None
    return support


def _dense_curve_traces(image: np.ndarray, reference: dict) -> tuple[list, set[int]]:
    width, height = image.shape[1::-1]
    quad = order_quad_cyclic(normalized_vertices_to_pixels(
        [(p["x"], p["y"]) for p in reference["warp_quad"]], width, height,
    ))
    outline = _bounded_outline(normalized_vertices_to_pixels(
        [(p["x"], p["y"]) for p in reference["outline"]], width, height,
    ))
    if outline is None or len(outline) < 4:
        return [], set()
    if np.any(np.min(np.linalg.norm(quad[:, None] - outline[None], axis=2), axis=1) > 2.25):
        return [], set()
    traces = []
    sides = set()
    scale = np.float32([width - 1, height - 1])
    for side in range(4):
        measured = _side_samples(quad, outline, side)
        if measured is None:
            continue
        support = _native_edge_support(image, *measured)
        if support is None:
            continue
        hits = np.flatnonzero(support)
        for run in np.split(hits, np.flatnonzero(np.diff(hits) != 1) + 1):
            if len(run) >= 4:
                samples = cv2.approxPolyDP(np.float32(measured[0][run]), .28, False)[:, 0] / scale
                traces.append([{"x": float(x), "y": float(y)} for x, y in samples])
                sides.add(side)
    return traces, sides


def _straight_paper_side(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    line, evidence = _side_line(image, start, end, 3)
    bow = evidence.get("measured_bow_px")
    residual = evidence["maximum_residual_px"]
    if evidence["ambiguous"] or (
        evidence["support"] >= .82 and (abs(bow or 0) > .5 or (residual or 0) > .9)
    ):
        return False
    if line is not None and np.max(np.abs(np.float32([start, end]) @ line[:2] + line[2])) <= 1:
        if bow is not None and evidence["support"] >= .90 and evidence["corner_support"] >= .8:
            return True
    # A faint straight cut between neutral papers may lose its local peak in
    # JPEG. Look for the same signed colour change across short, fixed collars.
    colors = _edge_band(image, start, end, np.arange(-5., 5.25, .5),
                        fractions=np.linspace(.04, .96, 112, dtype=np.float32))
    if colors is None:
        return False
    colors = colors.astype(np.float32)
    for shift in range(-2, 3):
        outside, inside = (colors[:, offset + shift:offset + shift + 3].mean(axis=1) for offset in (4, 14))
        wide = outside - inside
        direction = np.median(wide, axis=0)
        magnitude = np.linalg.norm(direction)
        if magnitude < 1:
            continue
        direction /= magnitude
        close = colors[:, 7 + shift:9 + shift].mean(axis=1) - colors[:, 12 + shift:14 + shift].mean(axis=1)
        strength, local = wide @ direction, close @ direction
        pale = ((np.minimum(outside[:, 0], inside[:, 0]) >= 162)
                & (np.linalg.norm(outside[:, 1:] - 128, axis=1) < 14)
                & (np.linalg.norm(inside[:, 1:] - 128, axis=1) < 14))
        votes = pale & (strength >= 2.6) & (local >= np.maximum(1.35, .43 * strength))
        # Equal signs on wide collars can also straddle a shallow bow. Track
        # the half-height of the actual transition in each part of the side.
        profile = (outside[:, None] - colors) @ direction
        lower, upper = profile[:, 5 + shift:15 + shift], profile[:, 6 + shift:16 + shift]
        half = strength[:, None] * .5
        crossing = (lower < half) & (upper >= half)
        votes &= crossing.sum(axis=1) == 1
        if votes.mean() < .84 or min(part.mean() for part in np.array_split(votes, 4)) < .65:
            continue
        rows, columns = np.nonzero(crossing & votes[:, None])
        positions = np.full(len(votes), np.nan)
        positions[rows] = -2.5 + .5 * (shift + columns) + .5 * (
            half[rows, 0] - lower[rows, columns]
        ) / (upper[rows, columns] - lower[rows, columns])
        bands = [part[np.isfinite(part)] for part in np.array_split(positions, 8)]
        if min(map(len, bands)) < 7:
            continue
        centers = np.array([np.median(part) for part in bands])
        if np.ptp(centers) <= .75 and abs(np.median(centers) + .5) <= 1:
            return True
    return False


def _edge_ends_at_print_corners(image: np.ndarray, quad: np.ndarray, side: int) -> bool:
    offsets = [-2.5, -1.5, 1.5, 2.5]
    start, end = quad[[side, (side + 1) % 4]]
    beyond = np.linspace(6, 16, 5, dtype=np.float32) / np.linalg.norm(end - start)
    extension = _edge_band(image, start, end, offsets, fractions=np.r_[-beyond, 1 + beyond])
    if extension is None:
        return False
    delta = np.linalg.norm(extension[:, :2].mean(axis=1) - extension[:, 2:].mean(axis=1), axis=1)
    # A nearby album rail can form a much stronger straight line than the print.
    if any(np.count_nonzero(part >= 2.7) >= 3 for part in np.split(delta, 2)):
        return False
    for neighbor, reverse in (((side - 1) % 4, True), ((side + 1) % 4, False)):
        start, end = quad[[neighbor, (neighbor + 1) % 4]]
        near = np.float32([2.5, 4.5, 6.5, 8.5]) / np.linalg.norm(end - start)
        arms = _edge_band(image, start, end, offsets, fractions=1 - near if reverse else near)
        if arms is None or np.count_nonzero(np.linalg.norm(
            arms[:, :2].mean(axis=1) - arms[:, 2:].mean(axis=1), axis=1,
        ) >= 2.6) < 3:
            return False
    return True


def _confirmed_straight_outline(image: np.ndarray, previous: np.ndarray, current: np.ndarray, sides: list[int]) -> bool:
    current = order_quad_cyclic(current)
    current = min((np.roll(current, i, axis=0) for i in range(4)),
                  key=lambda quad: np.linalg.norm(quad - previous))
    unchanged_sides = 0
    for index in range(4):
        end = (index + 1) % 4
        delta = previous[end] - previous[index]
        inward = np.float32([-delta[1], delta[0]]) / np.linalg.norm(delta)
        separation = (current[[index, end]] - previous[[index, end]]) @ inward
        if index not in sides:
            unchanged_sides += int(np.max(np.abs(separation)) <= 2)
            continue
        # Dimensions or a movement along the same curved line are not new edge evidence.
        if separation.mean() > -2.1 or separation.max() > -1.25 or separation.min() < -15:
            return False
        if not _edge_ends_at_print_corners(image, current, index):
            return False
    # A surrounding mat can have four straight sides and four valid corners.
    # Keep independent side lines that already associate this edit with the print.
    if unchanged_sides < min(2, 4 - len(sides)):
        return False
    return all(_straight_paper_side(image, start, end)
               for start, end in zip(current, np.roll(current, -1, axis=0), strict=True))


def check_manual_paper_boundary(image: np.ndarray, current: np.ndarray, reference: dict) -> tuple[bool, list]:
    traces, dense_sides = _dense_curve_traces(image, reference) if len(reference["outline"]) >= 8 else ([], set())
    curve = reference["curve"]
    if traces and curve is None:
        return True, traces
    if curve is None and not reference["legacy_extended"]:
        return False, []
    width, height = image.shape[1::-1]
    seed = (curve or reference)["warp_quad"]
    previous = order_quad_cyclic(normalized_vertices_to_pixels([(p["x"], p["y"]) for p in seed], width, height))
    if curve is None:
        fitted, evidence, reasons = refine_print_boundary(image, previous, collect_curve_segments=True)
        if fitted is not None and not reasons:
            return False, []
        # Older edits of an extended outline retain the original proposal but
        # may have overwritten the original correction and its review reason.
        curve = paper_curve_review_reference({
            "crop_modes": {"full_print": {"warp_quad": seed, "reason_codes": reasons}},
            "boundary_refinement": evidence,
        })
        if curve is None:
            return True, []
        scale = np.float32([width - 1, height - 1])
        curve["traces"] = [
            [{"x": float(x), "y": float(y)} for x, y in np.float32(segment) / scale]
            for side in evidence.values() for segment in side.get("curve_segments", [])
        ]
    if _confirmed_straight_outline(image, previous, current, sorted(dense_sides | set(curve["sides"]))):
        return False, []
    return True, traces or curve["traces"]
