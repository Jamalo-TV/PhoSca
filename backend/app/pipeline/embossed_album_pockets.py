"""Find continuous embossing across the sides of a pale print proposal."""

import cv2
import numpy as np


EMBOSSED_POCKET_PROPOSAL = "possible_embossed_album_pocket"


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
