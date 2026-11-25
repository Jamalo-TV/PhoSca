"""Keep a manual rectangle from replacing a measured, non-straight paper edge."""

import cv2
import numpy as np

from app.pipeline.geometry import normalized_polygon_overlap, normalized_vertices_to_pixels, order_quad_cyclic


MANUAL_PAPER_BOUNDARY_REVISION = 1
MANUAL_PAPER_BOUNDARY_REVIEW = "PRINT_CURVED_OUTLINE_REVIEW"


def manual_paper_reference(metadata: dict, crop_quad: list[dict] | None) -> dict | None:
    review = metadata.get("review_provenance") or {}
    original = review.get("original_prediction") or {}
    if len(crop_quad or []) != 4 or original.get("segmentation_confidence") is None:
        return None
    mask = original.get("segmentation_mask") or {}
    outline = mask.get("visible_polygon") or mask.get("polygon") or []
    seed = mask.get("warp_quad") or mask.get("polygon") or []
    if len(seed) != 4 or len(outline) < 8:
        return None
    # Ignore a different print after the reviewer replaces a bad proposal.
    # A smaller rectangle inside the same print still needs the original edge.
    overlap = normalized_polygon_overlap([(p["x"], p["y"]) for p in seed], [(p["x"], p["y"]) for p in crop_quad])
    if overlap.iou < .82 and max(overlap.first_containment, overlap.second_containment) < .90:
        return None
    return {"warp_quad": seed, "outline": outline}


def manual_paper_download_needs_review(metadata: dict, crop_quad: list[dict] | None) -> bool:
    reference = manual_paper_reference(metadata, crop_quad)
    if reference is None:
        return False
    check = (metadata.get("correction") or {}).get("manual_paper_boundary") or {}
    if check.get("revision") == MANUAL_PAPER_BOUNDARY_REVISION:
        return check.get("requires_review") is not False
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


def measure_manual_paper_boundary(image: np.ndarray, reference: dict) -> list:
    """Return closed-source evidence only as separate edge traces for review."""
    width, height = image.shape[1::-1]
    quad = order_quad_cyclic(normalized_vertices_to_pixels(
        [(p["x"], p["y"]) for p in reference["warp_quad"]], width, height,
    ))
    outline = _bounded_outline(normalized_vertices_to_pixels(
        [(p["x"], p["y"]) for p in reference["outline"]], width, height,
    ))
    if outline is None or len(outline) < 4:
        return []
    if np.any(np.min(np.linalg.norm(quad[:, None] - outline[None], axis=2), axis=1) > 2.25):
        return []
    traces = []
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
    return traces
