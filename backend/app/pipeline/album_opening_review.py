"""Trace visible transitions near a held crop with a wide side apron."""

import cv2
import numpy as np


ALBUM_OPENING_REVIEW_REVISION = 1
POSSIBLE_ALBUM_OPENING_EDGE = "possible_album_opening_edge"
_SIDES = ("top", "right", "bottom", "left")


def _dark_transition(lightness, stations, offsets, unit, *, vertical):
    # Sample the pale side and the dark side of the same source transition.
    samples = offsets[:, None] + unit * np.float32([-8, -4, 0, 4])
    shape = (len(stations), *samples.shape)
    along = np.broadcast_to(stations[:, None, None], shape)
    across = np.broadcast_to(samples[None], shape)
    x, y = (across, along) if vertical else (along, across)
    values = cv2.remap(
        lightness, np.ascontiguousarray(x.reshape(len(stations), -1), np.float32),
        np.ascontiguousarray(y.reshape(len(stations), -1), np.float32), cv2.INTER_LINEAR,
    ).reshape(shape).astype(np.float32)
    pale = values[..., :2].mean(axis=-1)
    dark = values[..., 2:].mean(axis=-1)
    return np.clip(pale - dark, 0, 80) * ((pale >= 113) & (dark <= 140))


def _proposal_left(polygon, rows):
    start, end = polygon, np.roll(polygon, -1, axis=0)
    intersects = (start[:, 1] > rows[:, None]) != (end[:, 1] > rows[:, None])
    rise = end[:, 1] - start[:, 1]
    slopes = np.divide(end[:, 0] - start[:, 0], rise, out=np.zeros_like(rise), where=rise != 0)
    positions = start[:, 0] + (rows[:, None] - start[:, 1]) * slopes
    return np.min(np.where(intersects, positions, np.inf), axis=1)


def _measure_side(lightness, proposal, low, high, unit):
    width, height = high - low
    rows = np.linspace(low[1] + .14 * height, high[1] - .14 * height, 96, dtype=np.float32)
    columns = np.arange(low[0] + .025 * width, low[0] + .25 * width, unit, dtype=np.float32)
    if len(columns) < 8:
        return None
    strength = _dark_transition(lightness, rows, columns, unit, vertical=True)
    shifts = np.linspace(-.065 * width, .065 * width, 41)
    candidates = columns[:, None, None] + shifts[None, :, None] * np.linspace(-.5, .5, len(rows))
    indices = np.rint((candidates - columns[0]) / unit).astype(np.int32)
    inside = (indices >= 0) & (indices < len(columns))
    row_indices = np.arange(len(rows))
    samples = strength[row_indices, np.clip(indices, 0, len(columns) - 1)] * inside
    ranked = samples.mean(axis=-1) + 16 * (samples >= 18).mean(axis=-1)
    estimate = candidates[np.unravel_index(ranked.argmax(), ranked.shape)]
    local = abs(columns[None] - estimate[:, None]) <= 3.5 * unit
    best = np.argmax(np.where(local, strength, -1), axis=1)
    supported = strength[row_indices, best] * local[row_indices, best] >= 18
    if supported.mean() < .86 or min(part.mean() for part in np.array_split(supported, 4)) < .75:
        return None
    line = np.polynomial.Polynomial.fit(rows[supported], columns[best[supported]], 1)
    gap = line(rows) - _proposal_left(proposal, rows)
    if not np.isfinite(gap).all() or np.quantile(gap, .10) < .085 * width or np.median(gap) < .105 * width:
        return None
    return line, rows[[0, -1]]


def _measure_bow(lightness, low, high, line, unit):
    width, height = high - low
    x = np.linspace(line(low[1] + .035 * height), high[0], 128, dtype=np.float32)
    rows = np.arange(low[1] - .105 * height, low[1] + .09 * height, unit, dtype=np.float32)
    if rows[0] - 8 * unit < 0 or rows[-1] + 4 * unit > lightness.shape[0] - 1:
        return None
    strength = _dark_transition(lightness, x, rows, unit, vertical=False)
    scores = strength + 12 * (strength >= 15)
    offsets = np.arange(-3, 4)
    indices = np.arange(len(rows))
    previous = indices[None] - offsets[:, None]
    inside = (previous >= 0) & (previous < len(rows))
    previous = np.clip(previous, 0, len(rows) - 1)
    penalties = 4.8 * abs(offsets[:, None]) + 1.9 * offsets[:, None] ** 2
    backtrack = np.empty((len(x) - 1, len(rows)), np.int32)
    cumulative = scores[0] - .6 * abs(rows - (low[1] + .045 * height)) / unit
    for station, score in enumerate(scores[1:]):
        prior = np.where(inside, cumulative[previous] - penalties, -np.inf)
        backtrack[station] = prior.argmax(axis=0)
        cumulative = prior[backtrack[station], indices] + score
    end = int((cumulative - .6 * abs(rows - low[1]) / unit).argmax())
    path = [end]
    for step in backtrack[::-1]:
        end = previous[step[end], end]
        path.append(end)
    path.reverse()
    y = rows[path]
    supported = strength[np.arange(len(x)), path] >= 15
    if supported.mean() < .92 or min(part.mean() for part in np.array_split(supported, 4)) < .82:
        return None
    chord = np.linspace(np.median(y[:8]), np.median(y[-8:]), len(y))
    if (np.median((chord - y)[40:88]) < .020 * width or np.max(abs(np.gradient(y, x))) > .90
            or abs(line(y[0]) - x[0]) > 8 * unit):
        return None
    trace = np.float32(np.column_stack((x, y)))
    return cv2.approxPolyDP(trace, 1.5 * unit, False).reshape(-1, 2)


def album_opening_traces(page: np.ndarray, proposal: np.ndarray, boundary: dict) -> list[np.ndarray]:
    """Find two opening-like transitions in original pixels for a held proposal."""
    supports = [(boundary.get(side) or {}).get("support", 0) for side in _SIDES]
    turns = next((index for index in range(4)
                  if max(supports[index], supports[(index - 1) % 4]) < .45
                  and min(supports[(index + 1) % 4], supports[(index + 2) % 4]) >= .88), None)
    if turns is None:
        return []
    original_width, original_height = page.shape[1::-1]
    lightness = cv2.cvtColor(page, cv2.COLOR_BGR2LAB)[..., 0]
    dimensions = (original_width, original_height)
    for _ in range(turns):
        proposal = np.column_stack((proposal[:, 1], dimensions[0] - 1 - proposal[:, 0]))
        dimensions = dimensions[::-1]
    lightness = np.ascontiguousarray(np.rot90(lightness, turns))
    low, high = proposal.min(axis=0), proposal.max(axis=0)
    width, height = high - low
    if min(width, height) < 150 or not .55 <= width / height <= 1.45:
        return []
    unit = float(np.clip(height / 660, .5, 3.))
    side = _measure_side(lightness, proposal, low, high, unit)
    if side is None:
        return []
    line, rows = side
    top = _measure_bow(lightness, low, high, line, unit)
    if top is None:
        return []
    traces = [top, np.float32(np.column_stack((line(rows), rows)))]
    for _ in range(turns):
        traces = [np.column_stack((dimensions[1] - 1 - points[:, 1], points[:, 0])) for points in traces]
        dimensions = dimensions[::-1]
    return traces
