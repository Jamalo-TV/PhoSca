"""Find continuous embossing across the sides of a pale print proposal."""

import cv2
import numpy as np

from app.pipeline.print_edges import _PRINT_SIDES, _edge_band, _side_line


EMBOSSED_POCKET_PROPOSAL = "possible_embossed_album_pocket"
POSSIBLE_EMBOSSED_ALBUM_PAPER_EDGES = "possible_embossed_album_paper_edges"


def _highpass(image: np.ndarray, sigma: float) -> np.ndarray:
    light = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return light - cv2.GaussianBlur(light, (0, 0), sigma)


def _repeat_vectors(signal: np.ndarray, depth: float, radius: int) -> list[np.ndarray]:
    """Measure two independent, repeating phases in the candidate interior."""
    template = signal[radius:-radius, radius:-radius]
    if template.std() < 3:
        return []
    correlations = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    directions = (
        (x >= .009 * depth) & (x <= .027 * depth) & (np.abs(y) <= .006 * depth),
        (y >= .014 * depth) & (y <= .032 * depth) & (np.abs(x) <= .008 * depth),
    )
    vectors = []
    for direction in directions:
        scores = np.where(direction, correlations, -1)
        row, col = np.unravel_index(scores.argmax(), scores.shape)
        dx, dy = col - radius, row - radius
        if scores[row, col] < .76 or correlations[radius + round(.5 * dy), radius + round(.5 * dx)] > .50:
            return []
        vectors.append(np.float32([dx, dy]))
    return vectors


def _crossing_trace(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, repeat: np.ndarray, sigma: float,
) -> np.ndarray | None:
    fractions = np.linspace(.12, .88, 23, dtype=np.float32)
    stations = start + fractions[:, None] * (end - start)
    tangent = (end - start) / np.linalg.norm(end - start)
    normal = np.float32([-tangent[1], tangent[0]])
    along = np.arange(-round(2.4 * sigma), round(2.4 * sigma) + 1, dtype=np.float32)
    across = np.arange(-round(.55 * sigma), round(.55 * sigma) + 1, dtype=np.float32)
    jitter = max(1., round(.22 * sigma))
    if .5 * abs(normal @ repeat) <= across[-1] + jitter * np.sqrt(2):
        return None
    grid = stations[:, None, None] + across[None, :, None, None] * normal + along[None, None, :, None] * tangent
    margin = .5 * abs(repeat) + 5 * sigma
    low = np.floor(grid.min(axis=(0, 1, 2)) - margin).astype(int)
    high = np.ceil(grid.max(axis=(0, 1, 2)) + margin).astype(int) + 1
    if np.any(low < 0) or np.any(high > image.shape[1::-1]) or np.any(high - low >= 32767):
        return None
    signal = _highpass(image[low[1]:high[1], low[0]:high[0]], sigma)
    grid = (grid - low).astype(np.float32).reshape(-1, len(along), 2)

    def sample(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = cv2.remap(signal, position[..., 0], position[..., 1], cv2.INTER_LINEAR).reshape(len(fractions), -1)
        values -= values.mean(axis=1, keepdims=True)
        return values, np.linalg.norm(values, axis=1)

    first, first_norm = sample(grid - .5 * repeat)
    valid = np.zeros(len(fractions), dtype=bool)
    minimum_amplitude = 3 * np.sqrt(len(along) * len(across))
    for dx in (-jitter, 0., jitter):
        for dy in (-jitter, 0., jitter):
            second, second_norm = sample(grid + .5 * repeat + np.float32([dx, dy]))
            valid |= ((first * second).sum(axis=1) >= .70 * first_norm * second_norm) & (
                first_norm >= minimum_amplitude) & (second_norm >= minimum_amplitude)
    supported = np.flatnonzero(valid)
    longest = max(np.split(supported, np.flatnonzero(np.diff(supported) != 1) + 1), key=len)
    if len(longest) < 15:
        return None
    return stations[longest[[0, -1]]]


def _is_pale_interior(image: np.ndarray, quad: np.ndarray) -> bool:
    """Veto coherent content after averaging the embossing cells."""
    center = quad.mean(axis=0)
    inset = center + .79 * (quad - center)
    left, top = np.ceil(inset.min(axis=0)).astype(int)
    right, bottom = np.floor(inset.max(axis=0)).astype(int)
    tile = cv2.resize(image[top:bottom, left:right], (144, 108), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(tile, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[..., 0]
    chroma = np.linalg.norm(lab[..., 1:] - 128, axis=-1)
    if (np.percentile(light, 3) < 218 or np.percentile(chroma, 97) > 10
            or np.ptp(np.percentile(light, [10, 90])) > 29):
        return False
    # Ignore broad illumination differences and test the remaining low
    # contrast scene structure away from the welded rims.
    residual = (light - cv2.GaussianBlur(light, (0, 0), 9.))[6:-6, 6:-6]
    if np.ptp(np.percentile(residual, [5, 95])) > 3.6:
        return False
    # A small visible detail can be lost in a percentile over the whole
    # pocket. Test overlapping windows of the same already smoothed tile.
    windows = np.lib.stride_tricks.sliding_window_view(residual, (22, 24))[::10, ::12]
    local_spans = np.ptp(np.percentile(windows, [10, 90], axis=(-2, -1)), axis=0)
    return bool(local_spans.max() <= 6.)


def embossed_pocket_traces(image: np.ndarray, quad: np.ndarray) -> list[np.ndarray]:
    """Mark repeated source phase; translucent film can also cover a print."""
    edges = np.roll(quad, -1, axis=0) - quad
    width, depth = abs(edges[[0, 2], 0]).min(), abs(edges[[1, 3], 1]).min()
    if (depth < 450 or width < 1.08 * depth
            or max(abs(edges[[0, 2], 1])) > .045 * width or max(abs(edges[[1, 3], 0])) > .045 * depth
            or not _is_pale_interior(image, quad)):
        return []
    radius = round(.037 * depth)
    half = min(320, round(.105 * depth))
    if radius >= half - 24:
        return []
    x, y = np.rint(quad.mean(axis=0)).astype(int)
    sigma = np.clip(.004 * depth, 1.8, 12.)
    vectors = _repeat_vectors(_highpass(image[y - half:y + half, x - half:x + half], sigma), depth, radius)
    if not vectors:
        return []
    horizontal_repeat = vectors[0]
    traces = [_crossing_trace(image, quad[a], quad[b], horizontal_repeat, sigma) for a, b in ((1, 2), (3, 0))]
    return [trace for trace in traces if trace is not None] if all(trace is not None for trace in traces) else []


def _carrier_phase_reference(
    image: np.ndarray, quad: np.ndarray, extent: float, sigma: float,
) -> np.ndarray | None:
    half, radius = round(.075 * extent), round(.037 * extent)
    if half > 320 or radius >= half - 24:
        return None
    references = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        edge = end - start
        outward = np.float32([edge[1], -edge[0]]) / np.linalg.norm(edge)
        repeated = []
        for fraction in (.40, .60):
            x, y = np.rint(start + fraction * edge + 1.13 * half * outward).astype(int)
            if half <= x <= image.shape[1] - half and half <= y <= image.shape[0] - half:
                patch = image[y - half:y + half, x - half:x + half]
                if periods := _repeat_vectors(_highpass(patch, sigma), extent, radius):
                    repeated.append(periods)
        references.append(repeated)
    for index in range(2):
        opposed = references[index] + references[index + 2]
        if not references[index] or not references[index + 2]:
            continue
        if np.max(np.ptp(opposed, axis=0)) <= max(2., .0014 * extent):
            return np.median(opposed, axis=0).astype(np.float32)
    return None


def _phase_break_on_rail(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, periods: np.ndarray, sigma: float,
) -> list[np.ndarray]:
    fractions = np.linspace(.08, .92, 49, dtype=np.float32)
    stations = start + fractions[:, None] * (end - start)
    tangent = (end - start) / np.linalg.norm(end - start)
    inward = np.float32([-tangent[1], tangent[0]])
    repeat = periods[int(abs(tangent[1]) > abs(tangent[0]))]
    jitter = max(1., round(.22 * sigma))
    distance = float(1.25 * sigma)
    along = np.arange(-round(2.4 * sigma), round(2.4 * sigma) + 1, dtype=np.float32)
    across = np.arange(-round(.5 * sigma), round(.5 * sigma) + 1, dtype=np.float32)
    if abs(repeat @ inward) / 2 + across[-1] + jitter * np.sqrt(2) >= distance:
        return []
    grid = stations[:, None, None] + across[None, :, None, None] * inward + along[None, None, :, None] * tangent
    margin = .5 * abs(repeat) + 7 * sigma
    low = np.floor(grid.min(axis=(0, 1, 2)) - margin).astype(int)
    high = np.ceil(grid.max(axis=(0, 1, 2)) + margin).astype(int) + 1
    if np.any(low < 0) or np.any(high > image.shape[1::-1]) or np.any(high - low >= 32767):
        return []
    signal = _highpass(image[low[1]:high[1], low[0]:high[0]], sigma)
    grid = (grid - low).astype(np.float32)

    def sample(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = coordinates.reshape(-1, len(along), 2)
        values = cv2.remap(signal, points[..., 0], points[..., 1], cv2.INTER_LINEAR).reshape(grid.shape[:3])
        # A straight shadow or scene edge alone is not a repeating texture.
        values -= values.mean(axis=2, keepdims=True)
        values = values.reshape(len(fractions), -1)
        norms = np.linalg.norm(values, axis=1)
        return values / np.maximum(norms[:, None], 1e-6), norms

    minimum_norm = 3 * np.sqrt(len(along) * len(across))
    matches = []
    for depth in (-distance, distance):
        center = grid + depth * inward
        first, first_norm = sample(center - .5 * repeat)
        best = np.zeros(len(fractions), np.float32)
        for dx in (-jitter, 0., jitter):
            for dy in (-jitter, 0., jitter):
                second, second_norm = sample(center + .5 * repeat + np.float32([dx, dy]))
                valid = np.minimum(first_norm, second_norm) >= minimum_norm
                best = np.maximum(best, np.where(valid, (first * second).sum(axis=1), 0.))
        matches.append(best)
    band = _edge_band(image, start, end, [-3, -1, 1, 3], fractions=fractions)
    if band is None:
        return []
    transition = np.linalg.norm(np.diff(band.astype(np.float32), axis=1), axis=2).max(axis=1)
    outside, inside = matches
    valid = (outside >= .70) & (inside <= .45) & (transition >= 10)
    if valid.mean() < .72 or min(part.mean() for part in np.array_split(valid, 4)) < .50:
        return []
    intervals = np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)
    return [stations[[first, stop - 1]] for first, stop in intervals if stop - first >= 3]


def embossed_carrier_paper_traces(image: np.ndarray, quad: np.ndarray, boundary: dict) -> list[np.ndarray]:
    """Trace measured phase breaks around an uncertain photo in an embossed carrier."""
    sides = [boundary.get(name) or {} for name in _PRINT_SIDES]
    weak = [index for index, side in enumerate(sides) if side.get("support", 0) <= .36]
    extent = float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
    if (len(weak) < 2 or not any(side.get("support", 0) >= .92 for side in sides)
            or any(side.get("ambiguous") for side in sides) or extent < 620):
        return []
    radius = min(180, round(.070 * extent))
    lines = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        line, facts = _side_line(image, start, end, radius)
        if (line is None or facts["ambiguous"] or facts["support"] < .80
                or (facts["corner_support"] or 0.) < .74 or facts["maximum_residual_px"] > 1.75):
            return []
        lines.append(line)
    crosses = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(abs(crosses[:, 2]) < .25):
        return []
    measured = np.float32(crosses[:, :2] / crosses[:, 2, None])
    if (not cv2.isContourConvex(measured) or np.linalg.norm(measured - quad, axis=1).max() > .091 * extent
            or not .82 <= cv2.contourArea(measured) / cv2.contourArea(quad) <= 1.06):
        return []
    sigma = np.clip(.004 * extent, 2., 12.)
    periods = _carrier_phase_reference(image, measured, extent, sigma)
    if periods is None:
        return []
    measured_sides = [_phase_break_on_rail(image, start, end, periods, sigma)
                      for start, end in zip(measured, np.roll(measured, -1, axis=0), strict=True)]
    if not all(measured_sides):
        return []
    return [span for index in weak for span in measured_sides[index]]
