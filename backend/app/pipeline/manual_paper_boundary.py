"""Verify source paper boundaries that a manual rectangle may cut or overrun."""

import cv2
import numpy as np

from app.pipeline.bowed_edge_review import POSSIBLE_BOWED_PRINT_EDGE
from app.pipeline.edge_prints import CAPTURE_EDGE_HOLD
from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic
from app.pipeline.print_edges import (
    POSSIBLE_BACKED_PAPER_EDGE, PRINT_SHARED_RIM_UNCERTAIN, SHARED_PRINT_RIM_SOURCE,
    _backing_color, _edge_band, _side_line, _stock_ring, bowed_print_sides,
    refine_print_boundary,
)
from app.pipeline.scalloped_prints import SCALLOPED_PRINT_CONFLICT


MANUAL_PAPER_BOUNDARY_REVISION = 2
MANUAL_PAPER_BOUNDARY_REVIEW = "PRINT_CURVED_OUTLINE_REVIEW"
MANUAL_CAPTURE_OUTLINE_REVIEW = "PRINT_CAPTURE_OUTLINE_REVIEW"
MANUAL_CONTRAST_OUTLINE_CHECK = "manual_contrast_outline_check"
MANUAL_CONTRAST_OUTLINE_REVISION = 1
PAPER_GRID_OVERLAP_REVIEW = "PAPER_GRID_OVERLAP_REVIEW"
PAPER_CLAIM_OVERLAP_CHECKS = {
    PAPER_GRID_OVERLAP_REVIEW: ("paper_grid_overlap_revision", 1),
    SCALLOPED_PRINT_CONFLICT: ("scalloped_ownership_overlap_revision", 1),
}


def paper_claim_groups(mask: dict | None, original_mask: dict | None = None) -> dict:
    current, original = mask or {}, original_mask or {}
    groups = {}
    if grid := current.get("paper_grid_review") or original.get("paper_grid_review"):
        groups[PAPER_GRID_OVERLAP_REVIEW] = grid.get("outline")
    if scalloped := current.get("scalloped_ownership_group") or original.get("scalloped_ownership_group"):
        groups[SCALLOPED_PRINT_CONFLICT] = scalloped
    return groups


def original_prediction_mask(metadata: dict) -> dict | None:
    return ((metadata.get("review_provenance") or {}).get("original_prediction") or {}).get("segmentation_mask")


def confirmed_contrast_outline(metadata: dict) -> bool:
    mask = original_prediction_mask(metadata) or {}
    if mask.get("detector_source") != "contrast_region":
        return False
    revisions = (metadata.get("review_provenance") or {}).get("correction_revisions") or []
    return any(revision.get("contrast_print_outline_confirmed") is True for revision in revisions)


def contrast_outline_check_pending(metadata: dict) -> bool:
    correction = metadata.get("correction") or {}
    return (correction.get(MANUAL_CONTRAST_OUTLINE_CHECK) != MANUAL_CONTRAST_OUTLINE_REVISION
            and confirmed_contrast_outline(metadata))


def paper_claim_overlap_checks(groups: dict) -> dict:
    return dict(PAPER_CLAIM_OVERLAP_CHECKS[reason] for reason in groups)


def paper_claims_need_overlap_check(mask: dict, metadata: dict, *, automatic: bool = False) -> bool:
    groups = paper_claim_groups(mask, None if automatic else original_prediction_mask(metadata))
    if automatic:
        groups.pop(PAPER_GRID_OVERLAP_REVIEW, None)
    checks = paper_claim_overlap_checks(groups)
    correction = metadata.get("correction") or {}
    return any(correction.get(key) != revision for key, revision in checks.items())


def overlapping_paper_claims(
    mask: dict, groups: dict, neighbor: dict, neighbor_groups: dict, width: int, height: int,
) -> set[str]:
    shared_groups = {reason for reason, outline in groups.items() if outline and outline == neighbor_groups.get(reason)}
    if not shared_groups:
        return set()
    polygons = []
    for candidate in (mask, neighbor):
        vertices = candidate.get("warp_quad") or candidate.get("visible_polygon") or candidate.get("polygon") or []
        if len(vertices) < 3:
            return shared_groups
        try:
            points = np.float32([(point["x"], point["y"]) for point in vertices])
            if not np.isfinite(points).all() or ((points < 0) | (points > 1)).any():
                return shared_groups
            polygons.append(cv2.convexHull(normalized_vertices_to_pixels(points, width, height)))
        except (KeyError, TypeError, ValueError, cv2.error):
            return shared_groups
    first, second = polygons
    shared, _ = cv2.intersectConvexConvex(first, second)
    return shared_groups if shared >= max(6.0, .002 * min(cv2.contourArea(first), cv2.contourArea(second))) else set()


def hold_overlapping_paper_claim_modes(modes: dict, reasons: set[str]) -> None:
    for mode in modes.values():
        if mode.get("status") == "ready":
            mode["geometry_status"] = "ready"
        mode["status"] = "review_needed"
        mode["reason_codes"] = list(dict.fromkeys([*sorted(reasons), *(mode.get("reason_codes") or [])]))


def manual_scalloped_outline(metadata: dict, crop_quad: list[dict] | None) -> list[dict] | None:
    mask = original_prediction_mask(metadata) or {}
    if mask.get("scalloped_print") is True:
        outline, seed = mask.get("visible_polygon"), mask.get("warp_quad")
    else:
        outline, seed = mask.get("scalloped_ownership_outline"), mask.get("scalloped_ownership_group")
    if not outline or len(seed or []) != 4 or len(crop_quad or []) != 4:
        return None
    first, second = (np.float32([(p["x"], p["y"]) for p in quad]) for quad in (seed, crop_quad))
    overlap = normalized_polygon_overlap(first, second)
    if overlap.iou >= .82 or max(overlap.first_containment, overlap.second_containment) >= .90:
        return outline
    return None


def capture_outline_review_required(print_mode: dict) -> bool:
    return CAPTURE_EDGE_HOLD in (print_mode.get("reason_codes") or [])


def paper_curve_review_reference(correction: dict) -> dict | None:
    full = (correction.get("crop_modes") or {}).get("full_print") or {}
    if len(full.get("warp_quad") or []) != 4:
        return None
    previews = full.get("outline_preview") or {}
    if (previews.get("source") == POSSIBLE_BOWED_PRINT_EDGE and previews.get("source_side") in range(4)
            and full.get("suggested_edge_traces")):
        return {"warp_quad": full["warp_quad"], "sides": [previews["source_side"]],
                "traces": full["suggested_edge_traces"]}
    if "PRINT_BOUNDARY_CURVED" not in (full.get("reason_codes") or []):
        return None
    sides = bowed_print_sides(correction.get("boundary_refinement") or {})
    return {"warp_quad": full["warp_quad"], "sides": sides or list(range(4)),
            "traces": full.get("suggested_edge_traces") or []}


def backed_edge_review_reference(print_mode: dict) -> dict | None:
    if print_mode.get("status") != "review_needed" or print_mode.get("suggested_edge_source") != POSSIBLE_BACKED_PAPER_EDGE:
        return None
    quad, traces = print_mode.get("warp_quad") or [], print_mode.get("suggested_edge_traces") or []
    if len(quad) != 4 or not traces or any(len(trace) < 2 for trace in traces):
        return None
    return {"warp_quad": quad, "traces": traces}


def shared_rim_cut_is_inside(quad: list[dict], traces: list[list[dict]], width: int, height: int) -> bool:
    """A paper correction must preserve the previously disputed cut plus a native pixel."""
    contour = order_quad_cyclic(normalized_vertices_to_pixels([(point["x"], point["y"]) for point in quad], width, height))
    points = normalized_vertices_to_pixels(
        [(point["x"], point["y"]) for trace in traces for point in trace], width, height,
    )
    return all(cv2.pointPolygonTest(contour, tuple(point), True) >= 1. for point in points)


def shared_rim_cut_segments(quad: list[dict], boundary: dict) -> list[list[dict]]:
    if len(quad) != 4:
        return []
    return [
        [{axis: round(first[axis] + fraction * (last[axis] - first[axis]), 8) for axis in ("x", "y")}
         for fraction in (.12, .88)]
        for side, first, last in zip(("top", "right", "bottom", "left"), quad, [*quad[1:], quad[0]], strict=True)
        if (boundary.get(side) or {}).get("shared_rim_unresolved")
    ]


def shared_rim_cut_reference(correction: dict) -> list[list[dict]]:
    full_print = ((correction.get("crop_modes") or {}).get("full_print") or {})
    if PRINT_SHARED_RIM_UNCERTAIN not in (full_print.get("reason_codes") or []):
        return []
    if traces := full_print.get("shared_rim_cut_traces"):
        return traces
    if full_print.get("suggested_edge_source") == SHARED_PRINT_RIM_SOURCE:
        return full_print.get("suggested_edge_traces") or []
    return shared_rim_cut_segments(full_print.get("warp_quad") or [], correction.get("boundary_refinement") or {})


def manual_paper_reference(metadata: dict, crop_quad: list[dict] | None) -> dict | None:
    review = metadata.get("review_provenance") or {}
    trace_curve = review.get("paper_trace_review")
    if trace_curve and len(crop_quad or []) == 4:
        seed, edited = ([(point["x"], point["y"]) for point in corners]
                        for corners in (trace_curve["warp_quad"], crop_quad))
        overlap = normalized_polygon_overlap(seed, edited)
        if overlap.iou < .82 and max(overlap.first_containment, overlap.second_containment) < .90:
            trace_curve = None
    else:
        trace_curve = None
    original = review.get("original_prediction") or {}
    mask = original.get("segmentation_mask") or {}
    previous_print = original.get("print_warp_quad") or []
    capture_outline = original.get("capture_outline_unverified")
    if capture_outline is None:
        # Older edits saved the rendered outline but not its reasons. Only the
        # outline itself, not an expanded detection box, shows frame contact.
        capture_outline = original.get("status") == "review_needed" and any(
            point.get(axis) in (0., 1.) for point in previous_print for axis in ("x", "y")
        )
    curve = trace_curve or review.get("paper_curve_review")
    outline = mask.get("visible_polygon") or mask.get("polygon") or []
    seed = mask.get("warp_quad") or mask.get("polygon") or []
    if capture_outline and original.get("segmentation_confidence") is not None:
        return {"capture_outline": True, "warp_quad": previous_print or seed, "outline": outline, "curve": curve}
    if (backed_edge := review.get("backed_edge_review")) and len(crop_quad or []) == 4:
        prior, edited = ([(point["x"], point["y"]) for point in shape] for shape in (backed_edge["warp_quad"], crop_quad))
        overlap = normalized_polygon_overlap(prior, edited)
        if overlap.iou >= .82 or max(overlap.first_containment, overlap.second_containment) >= .90:
            return {"backed_edge_review": backed_edge["traces"]}
    if trace_curve:
        return {"warp_quad": trace_curve["warp_quad"], "outline": [], "curve": trace_curve,
                "legacy_extended": False, "trace_review": True}
    if original.get("segmentation_confidence") is None:
        return None
    if len(crop_quad or []) != 4:
        return None
    if len(seed) != 4:
        seed = previous_print or (curve or {}).get("warp_quad") or []
    legacy_extended = curve is None and mask.get("extended_print_outline") is True
    if len(seed) != 4 or (not curve and not legacy_extended and len(outline) < 8):
        return None
    # Ignore a different print after the reviewer replaces a bad proposal.
    # A smaller rectangle inside the same print still needs the original edge.
    overlap = normalized_polygon_overlap([(p["x"], p["y"]) for p in seed], [(p["x"], p["y"]) for p in crop_quad])
    if overlap.iou < .82 and max(overlap.first_containment, overlap.second_containment) < .90:
        return None
    return {"warp_quad": seed, "outline": outline, "curve": curve, "legacy_extended": legacy_extended}


def paper_curve_from_review_traces(
    quad: list[dict], traces: list, width: int, height: int,
) -> dict | None:
    outline = order_quad_cyclic(normalized_vertices_to_pixels(
        [(point["x"], point["y"]) for point in quad], width, height,
    ))
    sides = set()
    matching_traces = []
    for trace in traces:
        points = normalized_vertices_to_pixels([(point["x"], point["y"]) for point in trace], width, height)
        for side, (start, end) in enumerate(zip(outline, np.roll(outline, -1, axis=0), strict=True)):
            length = np.linalg.norm(end - start)
            tangent = (end - start) / length
            inward = np.float32([-tangent[1], tangent[0]])
            distance = (points - start) @ tangent
            depth = (points - start) @ inward
            span = distance[-1] - distance[0]
            if abs(span) < .65 * length or distance.min() < -5 or distance.max() > length + 5:
                continue
            if np.max(np.abs(depth)) > 12:
                continue
            chord = depth[0] + (depth[-1] - depth[0]) * (distance - distance[0]) / span
            # Sustained departure of observed points from their own chord is
            # not repaired by entering dimensions or by shifting all corners.
            if np.quantile(np.abs(depth - chord), .65) > 2.5:
                sides.add(side)
                matching_traces.append(trace)
                break
    if not sides:
        return None
    return {"warp_quad": quad, "sides": sorted(sides), "traces": matching_traces}


def manual_paper_download_needs_review(metadata: dict, crop_quad: list[dict] | None) -> bool:
    reference = manual_paper_reference(metadata, crop_quad)
    if reference is None:
        return False
    check = (metadata.get("correction") or {}).get("manual_paper_boundary") or {}
    if reference.get("backed_edge_review"):
        return (check.get("revision") != MANUAL_PAPER_BOUNDARY_REVISION
                or check.get("backed_edge_verified") is not True or check.get("requires_review") is not False)
    if reference.get("capture_outline"):
        return (
            check.get("revision") != MANUAL_PAPER_BOUNDARY_REVISION
            or check.get("capture_outline_verified") is not True
            or check.get("requires_review") is not False
        )
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


def _competing_inner_stock(image: np.ndarray, quad: np.ndarray, radius: int) -> bool:
    # A mount and the print inside it can each form a pale rectangle.
    # A regular darker motif inside the selected stock does not satisfy this test.
    sides = list(zip(quad, np.roll(quad, -1, axis=0), strict=True))
    rim = [_edge_band(image, start, end, [2, 4, 6]) for start, end in sides]
    if any(samples is None for samples in rim):
        return False
    outer_stock = np.median(np.asarray(rim), axis=(0, 1, 2))
    offsets = np.arange(5, radius, .5, dtype=np.float32)
    fractions = np.linspace(.13, .87, 80, dtype=np.float32)
    layers = []
    for start, end in sides:
        band = _edge_band(image, start, end, offsets, fractions=fractions)
        if band is None:
            return False
        stations = np.arange(len(offsets) - 26)[:, None]
        outside = band[:, stations + [0, 1, 2]].mean(axis=2)
        inside = band[:, stations + [9, 10, 11, 12]].mean(axis=2)
        interior = band[:, stations + [16, 18, 20, 22, 24, 26]]
        before = np.linalg.norm(outside - outer_stock, axis=2)
        after = np.linalg.norm(inside - outer_stock, axis=2)
        change = after - before
        votes = (before <= 7) & (after >= 12) & (change >= 9)
        votes &= (inside[..., 0] >= 185) & (np.max(np.abs(inside[..., 1:] - 128), axis=2) <= 21)
        # A dark motif can have pale antialiasing at its edge. Look well
        # beyond that transition for the same distinct, pale material.
        votes &= (interior[..., 0] >= 185).all(axis=2)
        votes &= (np.linalg.norm(interior - inside[:, :, None], axis=3) <= 8).all(axis=2)
        quarters = np.stack([part.mean(axis=0) for part in np.array_split(votes, 4)])
        supported = (votes.mean(axis=0) >= .88) & (quarters.min(axis=0) >= .77)
        if not supported.any():
            return False
        best = np.argmax(np.where(supported, np.where(votes, change, 0).mean(axis=0), 0))
        layers.append(np.median(inside[votes[:, best], best], axis=0))
    return bool((np.linalg.norm(layers - np.median(layers, axis=0), axis=1) <= 10).all())


def _unexplained_capture_material(
    image: np.ndarray, quad: np.ndarray, previous: np.ndarray, backing: np.ndarray,
) -> bool:
    # Check the old footprint beyond the near-paper collars. Thin album ink
    # such as a rail is different from a broad patch that may still be photo.
    size = 128
    unit_range = np.linspace(0, 1, size, dtype=np.float32)
    x_unit, y_unit = np.meshgrid(unit_range, unit_range)
    features = np.stack((np.ones_like(x_unit), x_unit, y_unit), axis=-1)
    low, high = previous.min(axis=0), previous.max(axis=0)
    for axis in (0, 1):
        for before in (True, False):
            first, last = low.copy(), high.copy()
            if before:
                last[axis] = min(last[axis], quad[:, axis].min() - 36)
            else:
                first[axis] = max(first[axis], quad[:, axis].max() + 36)
            extent = last - first
            if extent.min() < 10:
                continue
            xs, ys = first[0] + x_unit * extent[0], first[1] + y_unit * extent[1]
            pixels = cv2.remap(image, xs, ys, cv2.INTER_LINEAR)
            lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).astype(np.float32)
            mask = np.zeros((size, size), np.uint8)
            previous_in_grid = np.rint((previous - first) * (size - 1) / extent).astype(np.int32)
            cv2.fillConvexPoly(mask, previous_in_grid, 1)
            active = mask > 0
            if np.count_nonzero(active) < 256:
                continue
            distance = np.linalg.norm(lab - backing, axis=2)
            plane = np.linalg.lstsq(features[active], lab[active], rcond=None)[0]
            residual = np.linalg.norm(lab - features @ plane, axis=2)
            near = active & (features[..., axis + 1] >= .88 if before else features[..., axis + 1] <= .12)
            # An independently backed near edge can lead into gradual
            # lighting drift. Test for local non-backing material even then.
            if near.any() and np.mean(distance[near] <= 9) >= .8 and np.percentile(residual[active], 90) <= 3.5:
                foreground = active & (residual > 7)
            else:
                foreground = active & ((distance > 20) | ((distance > 10) & (residual > 4.5)))
            kernel_size = np.maximum(3, np.round(10 * (size - 1) / extent).astype(int) | 1)
            opened = cv2.morphologyEx(
                foreground.astype(np.uint8), cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size)),
            )
            _, _, stats, _ = cv2.connectedComponentsWithStats(opened)
            if np.any(stats[1:, cv2.CC_STAT_AREA] >= max(65, .045 * np.count_nonzero(mask))):
                return True
    return False


def _confirmed_capture_outline(image: np.ndarray, quad: np.ndarray, previous: np.ndarray) -> bool:
    # A sharp line can be printed ink and a straight album rail can extend
    # past a print. Require backing, a coherent contrasting stock ring,
    # independent corner terminations and a check for wider exterior edges.
    # A pale rectangular mat can still pass those tests. Keep a corroborated
    # side of the original proposal attached to the same physical print.
    previous = min((np.roll(previous, i, axis=0) for i in range(4)),
                   key=lambda corners: np.linalg.norm(corners - quad))
    retained = np.linalg.norm(previous - quad, axis=1) <= 2.
    if not any(retained[i] and retained[(i + 1) % 4] and _straight_paper_side(
        image, previous[i], previous[(i + 1) % 4]
    ) for i in range(4)):
        return False
    backing = _backing_color(image)
    if not _stock_ring(image, quad, backing):
        return False
    sides = np.roll(quad, -1, axis=0) - quad
    radius = min(130, max(36, round(float(np.linalg.norm(sides, axis=1).min()) * .112)))
    for index, (start, end) in enumerate(zip(quad, np.roll(quad, -1, axis=0), strict=True)):
        outside = _edge_band(image, start, end, [-6, -12, -22, -36])
        if outside is None or not (np.mean(np.linalg.norm(outside - backing, axis=2) <= 10, axis=0) >= .80).all():
            return False
        if not _straight_paper_side(image, start, end) or not _edge_ends_at_print_corners(image, quad, index):
            return False
        # A printed frame inside confirmed stock is expected. Only a broader
        # transition toward the album can compete with the outer edge.
        dx, dy = end - start
        outset = 4 * np.float32([dy, -dx]) / np.hypot(dx, dy)
        _, exterior = _side_line(image, start + outset, end + outset, radius, outward_only=True)
        if exterior["support"] >= .75:
            return False
    return not (_competing_inner_stock(image, quad, radius) or _unexplained_capture_material(image, quad, previous, backing))


def check_manual_paper_boundary(image: np.ndarray, current: np.ndarray, reference: dict) -> tuple[bool, list]:
    if backed_traces := reference.get("backed_edge_review"):
        points = normalized_vertices_to_pixels(
            [(point["x"], point["y"]) for trace in backed_traces for point in trace], *image.shape[1::-1],
        )
        # The measured spans must all lie against the same edited side. A
        # change elsewhere or in the output aspect cannot align that side.
        for start, stop in zip(current, np.roll(current, -1, axis=0), strict=True):
            tangent = stop - start
            length = float(np.linalg.norm(tangent))
            axis = tangent / length
            along = (points - start) @ axis
            across = (points - start) @ np.float32([-axis[1], axis[0]])
            if along.min() >= -2 and along.max() <= length + 2 and np.max(np.abs(across)) <= 2:
                return False, []
        return True, backed_traces
    if reference.get("capture_outline"):
        current = order_quad_cyclic(current)
        curve = reference.get("curve")
        seed = reference["warp_quad"]
        if len(seed) != 4:
            return True, (curve or {}).get("traces", [])
        previous = order_quad_cyclic(normalized_vertices_to_pixels(
            [(point["x"], point["y"]) for point in seed], *image.shape[1::-1],
        ))
        if not _confirmed_capture_outline(image, current, previous):
            return True, (curve or {}).get("traces", [])
        if not curve:
            return False, []
        # If prior curved-paper evidence also exists, preserve its own
        # verification below after confirming the replacement paper edge.
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
