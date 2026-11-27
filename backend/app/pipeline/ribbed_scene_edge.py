"""Open scene transitions beside a repeating album surface."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.pipeline.album_backing import _repeating_profile
from app.pipeline.geometry import normalized_vertices_to_pixels, order_quad_cyclic, pixel_vertices_to_normalized
from app.pipeline.manual_paper_boundary import review_source_quads_match
from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line
from app.pipeline.projective_prints import PROJECTIVE_PRINT_OUTLINE_REVIEW
from app.pipeline.protected_regions import region_box_in_pixels
from app.pipeline.tile_print_review import CONTESTED_TILE_FOLLOWUP, TILED_PRINT_PROPOSAL, tiled_outline_conflicts

RIBBED_SCENE_EDGE = "ribbed_scene_edge"
RIBBED_SCENE_EDGE_SCOPE = "ribbed_scene_edge_scope"
_STATIONS = np.linspace(.08, .92, 180, dtype=np.float32)
_COLLARS = ((-15, -14, -13), (-11, -10, -9), (-7, -6, -5), (2, 3, 4))


@dataclass(frozen=True)
class RibbedSceneEdge:
    spans: list[np.ndarray]
    scopes: list[np.ndarray]


def _unit_inward(side: np.ndarray) -> np.ndarray:
    dx, dy = side[1] - side[0]
    return np.float32([-dy, dx]) / np.hypot(dx, dy)


def _correlation(first: np.ndarray, second: np.ndarray, max_lag: int) -> float:
    best = 0.
    a = first[14:-14].copy()
    a -= a.mean()
    for lag in range(-max_lag, max_lag + 1):
        b = second[14 + lag:len(second) - 14 + lag].copy()
        b -= b.mean()
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm > 0:
            corr = float(a @ b / norm)
            best = max(best, corr)
    return best


def _band_scope(side: np.ndarray, first: float, last: float, outer: float, inner: float) -> np.ndarray:
    start, end = side[0] + np.float32([[first], [last]]) * (side[1] - side[0])
    normal = _unit_inward(side)
    return np.float32([start + outer * normal, end + outer * normal, end + inner * normal, start + inner * normal])


def _propose_line(image: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    length = np.linalg.norm(side[1] - side[0])
    if not 200 <= length <= 450:
        return None
    # Search only a bounded interior corridor. The same palette rule must
    # win along separated parts of the side.
    low = max(11, int(np.floor(.043 * length)))
    high = min(27, int(np.ceil(.090 * length)))
    band = _edge_band(image, *side, range(low - 7, high + 7), fractions=_STATIONS)
    if band is None:
        return None
    columns = np.arange(high - low + 1) + 7
    near = band[:, columns - 1] - band[:, columns + 1]
    wide = (np.median(band[:, columns[:, None] + [-7, -5, -3]], axis=2)
            - np.median(band[:, columns[:, None] + [2, 4, 6]], axis=2))
    middles, slopes = np.meshgrid(np.arange(low, high + .1, .5), np.arange(-.040, .041, .0025))
    middles, slopes = middles.ravel(), slopes.ravel()
    end_shifts = middles[:, None] + slopes[:, None] * length * np.float32([-.5, .5])
    options = end_shifts[:, :1] + np.diff(end_shifts, axis=1) * _STATIONS
    retained = (options.min(axis=1) >= low) & (options.max(axis=1) <= high)
    if not retained.any():
        return None
    offsets = np.rint(options[retained]).astype(np.intp) - low
    near, wide = near[np.arange(len(_STATIONS)), offsets], wide[np.arange(len(_STATIONS)), offsets]
    signed = (near[..., 0] >= 9) & (wide[..., 0] >= 22) & (wide[..., 2] >= 4)
    body = signed.mean(axis=1)
    eighth = np.stack([sector.mean(axis=1) for sector in np.array_split(signed, 8, axis=1)], axis=1).min(axis=1)
    # Do not silently select a smaller passing contour when a competing line dominates.
    strength = (.46 * body + .30 * eighth + .14 * np.minimum(near[..., 0] / 45., 1).mean(axis=1)
                + .10 * np.minimum(wide[..., 0] / 70., 1).mean(axis=1))
    winner = int(np.argmax(strength))
    if body[winner] < .955 or eighth[winner] < .82:
        return None

    hypothesized = np.float32(side + end_shifts[retained][winner, :, None] * _unit_inward(side))
    line, evidence = _side_line(image, *hypothesized, 3)
    if line is None or evidence["support"] < .88 or evidence["ambiguous"]:
        return None
    fitted = np.float32(_line_endpoints(*hypothesized, line))
    shifts = (fitted - side) @ _unit_inward(side)
    if np.min(shifts) < 12 or np.max(shifts) > 24:
        return None
    return fitted, _band_scope(side, .07, .93, low - 9, high + 8)


def _backs_against_repeating_leaf(image: np.ndarray, local: np.ndarray, opposing: np.ndarray) -> bool:
    length = np.linalg.norm(local[1] - local[0])
    stations = np.linspace(.08, .92, round(.84 * length), dtype=np.float32)
    sampled = [_edge_band(image, *local, offsets, fractions=stations) for offsets in _COLLARS]
    if any(value is None for value in sampled):
        return False
    profiles = [_repeating_profile(np.median(band[..., 0], axis=1)) for band in sampled]
    if any(signal.std() < 8 or repeat < .76 for signal, repeat in profiles[:3]) or profiles[-1][1] > .37:
        return False
    # Corroborate against the separated page material beyond the opposite
    # side, retaining a bounded lag for the unknown rib phase.
    ordinary = np.linspace(.14, .86, round(.72 * length), dtype=np.float32)
    a = _edge_band(image, *local, [-12, -11, -10], fractions=ordinary)
    b = _edge_band(image, *local, [-8, -7, -6], fractions=ordinary)
    c = _edge_band(image, *local, [3, 4, 5], fractions=ordinary)
    peer = _edge_band(image, *opposing, [14, 15, 16], fractions=ordinary)
    if any(value is None for value in (a, b, c, peer)):
        return False
    lab = [np.median(band.reshape(-1, 3), axis=0) for band in (a, peer)]
    if np.linalg.norm(lab[0] - lab[1]) > 6:
        return False
    outer, inner, scene, reference = [_repeating_profile(np.median(band[..., 0], axis=1))[0]
                                     for band in (a, b, c, peer)]
    return (_correlation(outer, inner, 4) >= .84 and _correlation(outer, scene, 4) <= .30
            and _correlation(outer, reference, 13) >= .78)


def _observed_open_spans(image: np.ndarray, measured: np.ndarray) -> list[np.ndarray]:
    length = np.linalg.norm(measured[1] - measured[0])
    fractions = np.linspace(.12, .88, round(.76 * length), dtype=np.float32)
    samples = _edge_band(image, *measured, [-2, -1, 0, 1, 2], fractions=fractions)
    if samples is None:
        return []
    # Refined geometry represents first scene centers. Permit the physical signed
    # transition directly outside the first center or up to one native pixel either way.
    supported = (samples[:, :3, 0] - samples[:, 2:, 0]).max(axis=1) >= 13
    observed = np.flatnonzero(supported)
    pieces = np.split(observed, np.flatnonzero(np.diff(observed) > 2) + 1)
    spans = []
    for piece in pieces:
        if len(piece) < 20:
            continue
        f = fractions[piece[[0, -1]]]
        if (f[1] - f[0]) * length >= 20:
            spans.append(np.float32(measured[0] + f[:, None] * (measured[1] - measured[0])))
    if sum(np.linalg.norm(b - a) for a, b in spans) < .55 * length:
        return []
    return spans


def measure_ribbed_scene_edge(image: np.ndarray, held_quad: np.ndarray) -> RibbedSceneEdge | None:
    """Measure a single open scene transition and its original-pixel witnesses."""
    quad = np.asarray(held_quad, dtype=np.float32)
    if (quad.shape != (4, 2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad)
            or cv2.contourArea(quad, oriented=True) <= 0):
        return None
    candidates = []
    for i in range(4):
        proposed = _propose_line(image, quad[[i, (i + 1) % 4]])
        if proposed is None:
            continue
        measured, search = proposed
        opposing = quad[[(i + 3) % 4, (i + 2) % 4]]
        if not _backs_against_repeating_leaf(image, measured, opposing):
            continue
        spans = _observed_open_spans(image, measured)
        if spans:
            scopes = [search, _band_scope(measured, .07, .93, -17, 7), _band_scope(opposing, .13, .87, 12, 18)]
            candidates.append(RibbedSceneEdge(spans, scopes))
    return candidates[0] if len(candidates) == 1 else None


def ribbed_scene_edge_review(
    image: np.ndarray, mask: dict, provenance: dict, page_sha256: str, modes: dict,
    claims: list[dict], protected_regions: list[dict],
) -> dict | None:
    """Describe original-pixel traces for an exact, still-held follow-up outline."""
    proposal = provenance.get("deferred_print_proposal") or {}
    creation = provenance.get("manual_creation") or {}
    prediction = provenance.get("prediction_provenance") or {}
    full, motif = (modes.get(key) or {} for key in ("full_print", "image_only"))
    if (proposal.get("source") != CONTESTED_TILE_FOLLOWUP or proposal.get("reason") != TILED_PRINT_PROPOSAL
            or proposal.get("id") != creation.get("deferred_proposal_id") or creation.get("confirm_standalone_print") is not True
            or not page_sha256 or prediction.get("page_sha256") != page_sha256
            or mask.get("manual_added") is not True or mask.get("image_quad")
            or full.get("status") != "review_needed" or motif.get("status") != "review_needed"
            or PROJECTIVE_PRINT_OUTLINE_REVIEW not in (full.get("reason_codes") or [])
            or set(full["reason_codes"]) - {PROJECTIVE_PRINT_OUTLINE_REVIEW, "PRINT_ASPECT_UNCERTAIN"}
            or any(full.get(key) for key in ("suggested_edge_traces", "suggested_warp_quad", "outline_preview"))):
        return None
    height, width = image.shape[:2]
    outline = mask.get("warp_quad")
    if not review_source_quads_match(outline, full.get("warp_quad"), creation.get("warp_quad"), proposal.get("polygon")):
        return None
    try:
        current = order_quad_cyclic(normalized_vertices_to_pixels([(p["x"], p["y"]) for p in outline], width, height))
        measured = measure_ribbed_scene_edge(image, current)
        if measured is None:
            return None
        scale = np.float32([width - 1, height - 1])
        if any(np.any((scope < 0) | (scope > scale)) for scope in measured.scopes):
            return None

        def normalized(points):
            return [{"x": float(x), "y": float(y)} for x, y in pixel_vertices_to_normalized(points, width, height)]

        scopes = [normalized(scope) for scope in measured.scopes]
        if any(tiled_outline_conflicts(scope, claims, width, height) for scope in scopes):
            return None
        for region in protected_regions:
            if box := region_box_in_pixels(region, width, height):
                left, top, right, bottom = box
                pad = float(region.get("margin_ratio", .08)) * max(right - left, bottom - top)
                protected = np.float32([[left - pad, top - pad], [right + pad, top - pad],
                                        [right + pad, bottom + pad], [left - pad, bottom + pad]])
                if any(cv2.intersectConvexConvex(scope, protected)[0] > 1 for scope in measured.scopes):
                    return None
    except (cv2.error, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return {"suggested_edge_source": RIBBED_SCENE_EDGE,
            "suggested_edge_traces": [normalized(span) for span in measured.spans],
            RIBBED_SCENE_EDGE_SCOPE: scopes}
