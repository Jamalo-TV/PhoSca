"""Locate supported pieces of a pale paper edge without filling in gaps."""

import cv2
import numpy as np

from app.pipeline.motif_boundary import motif_seams_intersect_claim
from app.pipeline.print_edges import _edge_band, _line_endpoints, _side_line


OPEN_PAPER_EDGE_SPANS = "open_paper_edge_spans"
OPEN_PAPER_EDGE_ANCHOR = "open_paper_edge_anchor"
OPEN_PAPER_EDGE_MARGIN = 16.
_SIDES = ("top", "right", "bottom", "left")


def _measured_runs(
    image: np.ndarray, cap: np.ndarray, end: np.ndarray, palette: np.ndarray, *, cap_at_end: bool,
) -> list[np.ndarray]:
    side = np.float32([end, cap] if cap_at_end else [cap, end])
    length = float(np.linalg.norm(end - cap))
    distances = np.arange(22, .93 * length, 4, dtype=np.float32)
    fractions = distances / length
    offsets = np.arange(-13, 18.5, .5, dtype=np.float32)
    samples = _edge_band(image, *side, offsets, fractions=1 - fractions if cap_at_end else fractions)
    if samples is None:
        return []
    drift = np.arange(-3.5, 3.51, .5, dtype=np.float32)
    columns = np.rint((drift - offsets[0]) * 2).astype(int)

    def across(depths: list[float]) -> np.ndarray:
        indices = columns[:, None] + np.rint(np.float32(depths) * 2).astype(int)
        return np.median(samples[:, indices], axis=2)

    outside, paper = across([-9, -7, -5]), across([6, 9, 12])
    contact = across([2, 3, 4])
    close_out, close_in = across([-2, -1.5, -1]), across([1, 1.5, 2])
    transition = paper - outside
    step = close_in[..., 0] - close_out[..., 0]
    supports = ((paper[..., 0] >= 208) & (np.linalg.norm(paper[..., 1:] - palette[1:], axis=2) <= 9)
                & (transition[..., 0] >= 11) & (np.linalg.norm(transition, axis=2) >= 12)
                & (step >= 5.5) & (np.linalg.norm(contact - paper, axis=2) <= 13))
    evidence = np.where(supports, step + .18 * np.minimum(transition[..., 0], 50) - .35 * np.abs(drift), 0)
    measured = np.flatnonzero(evidence.max(axis=1) > 0)
    displacement = drift[evidence.argmax(axis=1)]
    tangent = (side[1] - side[0]) / length
    normal = np.float32([-tangent[1], tangent[0]])
    observed = cap + fractions[:, None] * (end - cap) + displacement[:, None] * normal
    # A missed station or a jump to a different transition breaks the guide.
    breaks = (np.diff(measured) != 1) | (np.abs(np.diff(displacement[measured])) > 2.5)
    groups = np.split(measured, np.flatnonzero(breaks) + 1)
    upper_length = lower_length = 0.
    traces = []
    for group in groups:
        if len(group) < 10:
            continue
        start, stop = fractions[group[[0, -1]]]
        if start < .66 and stop > .43:
            return []
        # An isolated matching colour farther from the observed ends can
        # belong to different stock. Keep only long, local runs at the ends.
        group = group[(fractions[group] <= .26) | (fractions[group] >= .70)]
        if len(group) < 21:
            continue
        if fractions[group[-1]] <= .26:
            upper_length += float(distances[group[-1]] - distances[group[0]])
        else:
            lower_length += float(distances[group[-1]] - distances[group[0]])
        simplified = cv2.approxPolyDP(np.float32(observed[group]), .9, False)
        traces.append(simplified[:, 0])
    return traces if upper_length >= .15 * length and lower_length >= .083 * length else []


def _corner_palette(
    image: np.ndarray, cap: np.ndarray, along_top: np.ndarray, *, cap_at_end: bool,
) -> np.ndarray | None:
    toward_paper = cap + along_top * 100.
    rail = np.float32([cap, toward_paper] if cap_at_end else [toward_paper, cap])
    fractions = np.float32(np.r_[np.linspace(.12, .44, 17), np.linspace(-.44, -.12, 17)])
    samples = _edge_band(image, *rail, [-8, -6, -4, -2, 2, 4, 6, 8],
                         fractions=fractions if cap_at_end else 1 - fractions)
    if samples is None:
        return None
    exterior, paper = (np.median(pixels, axis=1) for pixels in np.split(samples, 2, axis=1))
    before, past = np.split((paper - exterior)[:, 0], 2)
    palette = np.median(paper[:len(before)], axis=0)
    if np.quantile(before, .20) < 50 or np.median(np.abs(past)) > 22 or palette[0] < 216:
        return None
    return palette


def measure_open_paper_edge(
    image: np.ndarray, quad: np.ndarray, earlier: dict, claims: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Observe the other side of an already held pale stock extension."""
    prior = _SIDES.index(earlier["side"])
    index = (_SIDES.index(earlier["companion"]) + 2) % 4
    if (index - prior) % 2 != 1:
        return None
    old = quad[[index, (index + 1) % 4]]
    cap_at_end = (index + 1) % 4 == prior
    head, tail = old[::-1] if cap_at_end else old
    movement = tail - head
    length = float(np.linalg.norm(movement))
    top = quad[[prior, (prior + 1) % 4]]
    top_length = float(np.linalg.norm(top[1] - top[0]))
    if min(length, top_length) < 512:
        return None
    radius = int(np.clip(round(.029 * min(length, top_length)), 25, 110))
    early = head + np.float32([.04, .275])[:, None] * movement
    seed = early[::-1] if cap_at_end else early
    fits = [_side_line(image, *seed, reach, outward_only=True) for reach in (radius, radius + 8)]
    if any(line is None or evidence["ambiguous"] or evidence["support"] < .93 or evidence["corner_support"] < .85
           for line, evidence in fits):
        return None
    (line, _), (independent, _) = fits
    observed = np.float32(_line_endpoints(*seed, line))
    if np.max(np.abs(observed @ independent[:2] + independent[2])) > .85:
        return None
    top_line = np.cross(*np.column_stack((top, np.ones(2))))
    intersection = np.cross(line, top_line)
    if abs(intersection[2]) < 1e-8:
        return None
    cap = np.float32(intersection[:2] / intersection[2])
    along = (top[1] - top[0]) / top_length
    if not cap_at_end:
        along = -along
    if not 15 <= (head - cap) @ along <= .043 * length:
        return None
    palette = _corner_palette(image, cap, along, cap_at_end=cap_at_end)
    if palette is None:
        return None
    end = np.float32(_line_endpoints(*old, line)[0 if cap_at_end else 1])
    traces = _measured_runs(image, cap, end, palette, cap_at_end=cap_at_end)
    if not traces:
        return None
    anchor = cap + np.float32([-8., 8.])[:, None] * along
    if any(motif_seams_intersect_claim([*traces, anchor], owner, margin_px=OPEN_PAPER_EDGE_MARGIN) for owner in claims):
        return None
    return traces, anchor
