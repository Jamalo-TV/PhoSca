"""Measure an adjacent scene edge after locating the end of a stacked photo."""

import cv2
import numpy as np

from app.pipeline.faint_margin import _glyph_groups, _joined_glyph_groups
from app.pipeline.print_edges import _edge_band, _line_endpoints, _quad_from_lines, _sample_lab, _side_line


def _locate_scene_line(
    image: np.ndarray, side: np.ndarray, short: float, palette: np.ndarray,
) -> np.ndarray | None:
    fractions = np.linspace(.115, .968, 164, dtype=np.float32)
    spacing = max(1., short / 1300)
    offsets = np.arange(.05 * short, .32 * short, spacing, dtype=np.float32)
    band = _edge_band(image, *side, offsets, fractions=fractions)
    if band is None:
        return None
    field = np.median(band[:, :45, 1:], axis=1)
    if np.mean(np.linalg.norm(field - palette, axis=1) <= 8) < .92:
        return None
    columns = np.arange(33, len(offsets) - 16)
    if len(columns) < 4:
        return None
    near = np.median(band[:, columns[:, None] - [4, 7, 10, 13]], axis=2)
    far = np.median(band[:, columns[:, None] - [18, 23, 30]], axis=2)
    inside = np.median(band[:, columns[:, None] + [4, 7, 9, 12]], axis=2)
    contrast = near - inside
    step = band[:, columns - 2] - band[:, columns + 2]
    votes = ((np.linalg.norm(near[..., 1:] - field[:, None], axis=2) <= 7)
             & (np.linalg.norm(far[..., 1:] - field[:, None], axis=2) <= 6)
             & ((contrast[..., 0] >= 23) | (np.linalg.norm(contrast[..., 1:], axis=2) >= 15))
             & ((step[..., 0] >= 5) | (np.linalg.norm(step[..., 1:], axis=2) >= 5)))
    tilts = np.linspace(-.06, .06, 79, dtype=np.float32) * short / spacing
    guesses = np.rint(np.arange(len(columns))[None, :, None]
                      + tilts[:, None, None] * (fractions - .5)).astype(int).clip(1, len(columns) - 2)
    hits = np.logical_or.reduce([votes[np.arange(len(fractions)), guesses + shift] for shift in (-1, 0, 1)])
    support = hits.mean(axis=2)
    weakest = np.min([part.mean(axis=2) for part in np.array_split(hits, 8, axis=2)], axis=0)
    tilt, depth = np.unravel_index(np.argmax(support + .10 * weakest), support.shape)
    rival = np.abs(offsets[columns] - offsets[columns[depth]]) >= .016 * short
    if support[tilt, depth] < .88 or weakest[tilt, depth] < .44 or np.any(support[:, rival] >= .72):
        return None

    edge = side[1] - side[0]
    inward = np.float32([-edge[1], edge[0]]) / np.linalg.norm(edge)
    ends = fractions[[0, -1]]
    shifts = offsets[columns[depth]] + tilts[tilt] * spacing * (ends - .5)
    seed = side[0] + ends[:, None] * edge + shifts[:, None] * inward
    fits = []
    for radius in (max(5, round(short * ratio)) for ratio in (.0047, .0071, .0094)):
        line, evidence = _side_line(image, *seed, radius)
        if line is not None and not evidence["ambiguous"] and evidence["support"] >= .67:
            fits.append(_line_endpoints(*seed, line))
    if len(fits) < 2:
        return None
    consensus = np.median(np.float32(fits), axis=0)
    return consensus if np.max(np.linalg.norm(fits - consensus, axis=2)) <= 1. else None


def _observed_scene_spans(
    image: np.ndarray, line: np.ndarray, short: float, palette: np.ndarray,
) -> list[np.ndarray]:
    fractions = np.linspace(.025, .975, 196, dtype=np.float32)
    offsets = np.float32([-34, -25, -17, -10, -6, -3, 3, 7, 12, 19, 26]) * (short / 2120.)
    band = _edge_band(image, *line, offsets, fractions=fractions)
    if band is None:
        return []
    exterior = np.median(band[:, :3], axis=1)
    change = np.median(band[:, 3:6], axis=1) - np.median(band[:, 7:], axis=1)
    transition = np.linalg.norm(band[:, 5] - band[:, 6], axis=1)
    supported = ((np.linalg.norm(exterior[:, 1:] - palette, axis=1) <= 8)
                 & ((change[:, 0] >= 21) | (np.linalg.norm(change[:, 1:], axis=1) >= 14))
                 & (transition >= 8))
    if supported.mean() < .62:
        return []
    stations = line[0] + fractions[:, None] * (line[1] - line[0])
    runs = np.flatnonzero(np.diff(np.r_[False, supported, False])).reshape(-1, 2)
    return [stations[[first, last - 1]] for first, last in runs if last - first >= 8]


def _continues_to_scene_end(
    image: np.ndarray, line: np.ndarray, end_line: np.ndarray, short: float, palette: np.ndarray, *, at_end: bool,
) -> bool:
    cross = np.cross(_line(line), _line(end_line))
    if abs(cross[2]) < 1e-6:
        return False
    cap = np.float32(cross[:2] / cross[2])
    direction = line[1] - line[0]
    direction /= np.linalg.norm(direction)
    inward = np.float32([-direction[1], direction[0]])
    toward = direction if at_end else -direction
    scale = short / 2120.
    along = np.float32([*np.linspace(-23, -5, 13), *np.linspace(8, 21, 11)]) * scale
    across = np.float32([-23, -16, -9, 11, 16, 22, 27]) * scale
    points = cap + along[:, None, None] * toward + across[None, :, None] * inward
    if np.any((points < 0) | (points > np.float32(image.shape[1::-1]) - 1)):
        return False
    lab = _sample_lab(image, points.reshape(-1, 2)).reshape(len(along), len(across), 3)
    outside = np.median(lab[:, :3], axis=1)
    inside = np.median(lab[:, 3:], axis=1)
    difference = outside - inside
    before, after = difference[:13], difference[13:]
    return bool(np.mean((before[:, 0] >= 26) & (np.linalg.norm(before[:, 1:], axis=1) >= 9)) >= .72
                and np.mean((np.abs(after[:, 0]) <= 19) & (np.linalg.norm(after[:, 1:], axis=1) <= 8)) >= .82
                and np.mean(np.linalg.norm(outside[:, 1:] - palette, axis=1) <= 8) >= .94)


def _same_material_strip(image: np.ndarray, outer: np.ndarray, inner: np.ndarray, palette: np.ndarray) -> bool:
    length = np.linalg.norm(outer[1] - outer[0])
    width = np.linalg.norm(outer - inner, axis=1).max()
    along = np.linspace(0., 1., int(np.clip(length, 96, 1600)), dtype=np.float32)
    across = np.linspace(.005, .995, int(np.clip(width, 32, 360)), dtype=np.float32)
    first = inner[0] + along[:, None] * (inner[1] - inner[0])
    last = outer[0] + along[:, None] * (outer[1] - outer[0])
    points = first[:, None] + across[None, :, None] * (last - first)[:, None]
    values = _sample_lab(image, points.reshape(-1, 2)).reshape(len(along), len(across), 3)
    field = np.median(values[..., 1:], axis=(0, 1))
    if np.linalg.norm(field - palette) > 8:
        return False
    unlike = np.linalg.norm(values[..., 1:] - field, axis=2) > 11
    if unlike.any():
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(unlike), connectivity=8)
        if stats[1:, cv2.CC_STAT_AREA].max(initial=0) >= 6:
            return False
    light = values[..., 0]
    highpass = np.abs(light - cv2.GaussianBlur(light, (0, 0), 2.))
    varied = (np.abs(light - np.median(light, axis=1)[:, None]) >= 18) & (highpass >= 4.3)
    patches = cv2.morphologyEx(np.uint8(varied), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if patches.any():
        _, _, stats, _ = cv2.connectedComponentsWithStats(patches, connectivity=8)
        width, length, area = stats[1:, 2:].T
        if np.any((width >= 13) & (length >= 16) & (area >= 84)):
            return False
    # Preserve thin marks of the same hue as the material. Local windows keep
    # isolated lettering from being swamped by unrelated paper texture.
    smooth = cv2.GaussianBlur(light, (0, 0), .8)
    residual = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) - smooth
    last_window = max(0, len(along) - 210)
    for threshold in (4, 10):
        for start in [*range(0, last_window, 120), last_window]:
            for turn in (0, 1):
                candidate = np.rot90(residual[start:start + 210] >= threshold, turn)
                if _glyph_groups(candidate, False)[1] or _joined_glyph_groups(candidate)[1]:
                    return False
    return True


def _line(points: np.ndarray) -> np.ndarray:
    return np.cross(*np.column_stack((points, np.ones(2))))


def measure_stacked_scene_companion(
    image: np.ndarray, original: np.ndarray, outline: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Shorten one adjacent side when it ends at the observed scene and has the same outer stock."""
    moved = np.linalg.norm(outline - original, axis=1) > 2.
    changed = np.flatnonzero(moved & np.roll(moved, -1))
    if len(changed) != 1 or np.count_nonzero(moved) != 2:
        return None
    edge, = changed
    short = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1).min()
    original_short = np.linalg.norm(np.roll(original, -1, axis=0) - original, axis=1).min()
    bottom = outline[[edge, (edge + 1) % 4]]
    tangent = bottom[1] - bottom[0]
    inward = np.float32([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    end_line = bottom + .0035 * original_short * inward
    stock = _edge_band(image, *end_line, np.float32([-27, -23, -19, -16]) * (short / 2120.))
    if stock is None:
        return None
    palette = np.median(stock[..., 1:], axis=(0, 1))
    matches = []
    for index in ((edge - 1) % 4, (edge + 1) % 4):
        ids = [index, (index + 1) % 4]
        side = outline[ids]
        measured = _locate_scene_line(image, side, short, palette)
        if measured is None or not (spans := _observed_scene_spans(image, measured, short, palette)):
            continue
        if not _continues_to_scene_end(image, measured, end_line, short, palette, at_end=ids[1] == edge):
            continue
        # Keep source pixels around the transition where the foil or a pale
        # paper shoulder makes the exact material boundary ambiguous.
        tangent = measured[1] - measured[0]
        outward = np.float32([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
        lines = np.stack([_line(outline[[i, (i + 1) % 4]]) for i in range(4)])
        lines[index] = _line(measured + .010 * short * outward)
        revised = _quad_from_lines(lines)
        if revised is None or not cv2.isContourConvex(revised):
            continue
        if _same_material_strip(image, side, revised[ids], palette):
            matches.append((revised, spans))
    return matches[0] if len(matches) == 1 else None
