"""Measure continuities in the exposed background beside a capture-edge proposal."""

import cv2
import numpy as np

from app.pipeline.crop_sampling import _outline_mask


CAPTURE_FABRIC_PROPOSAL = "possible_capture_fabric"


def _warm_album_area(lab: np.ndarray) -> np.ndarray | None:
    lightness, _, yellow = lab.transpose(2, 0, 1)
    frame = np.concatenate((yellow[:20].ravel(), yellow[-20:].ravel(),
                            yellow[:, :20].ravel(), yellow[:, -20:].ravel()))
    warm = np.uint8((yellow >= np.median(frame) + 8) & (lightness >= 105))
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(warm)
    if count <= 1:
        return None
    largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    contours, _ = cv2.findContours(np.uint8(labels == largest), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0]
    if not .35 <= cv2.contourArea(hull) / yellow.size <= .95:
        return None
    return _outline_mask(hull, yellow.shape)


def _sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.remap(field, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR)


def _paper_rim_or_color_change(light: np.ndarray, chroma: np.ndarray, valid: np.ndarray) -> bool:
    stations = np.arange(len(light))[:, None]
    for drift in (-6., -3., 0., 3., 6.):
        shift = np.rint(np.linspace(-.5, .5, len(light))[:, None] * drift).astype(int)
        strip = light[stations, np.arange(3, 30)[None, :] + shift][valid]
        spread = np.ptp(np.quantile(strip, [.10, .90], axis=0), axis=0)
        center = np.median(strip, axis=0)
        outer = np.maximum(np.roll(spread, 5), np.roll(spread, -5))
        near = np.minimum(np.roll(spread, 4), np.roll(spread, -4))
        uniform = (spread <= 15) & (np.abs(center - np.roll(center, 1)) <= 7) & (outer >= 36)
        thin = ((spread <= 19) & (near >= 50) & (spread <= .38 * outer)
                | (spread <= 24) & (near >= 62) & (spread <= .16 * near))
        if np.any(uniform[3:-4] & uniform[4:-3]) or np.any(thin[4:-4]):
            return True
        strip = chroma[stations, np.arange(6, 27)[None, :] + shift][valid]
        shift_in_color = np.median(strip[:, 12:] - strip[:, :9], axis=0)
        if np.max(np.linalg.norm(shift_in_color, axis=1)) >= 6:
            return True
    return False


def capture_fabric_traces(image: np.ndarray, proposals: list[np.ndarray]) -> dict[int, list[np.ndarray]]:
    """Find review evidence while guarding narrow visible rims of clipped prints."""
    height, width = image.shape[:2]
    if not proposals or min(height, width) < 100 or height * width > 12_000_000:
        return {}
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    album = _warm_album_area(lab)
    if album is None:
        return {}
    clear = np.uint8(cv2.dilate(album, np.ones((13, 13), np.uint8)) == 0)
    lightness = lab[..., 0]
    fine = lightness - cv2.GaussianBlur(lightness, (0, 0), 1.55)
    roughness = np.sqrt(cv2.blur(fine * fine, (19, 19)))
    soft = cv2.GaussianBlur(lab, (0, 0), 3.2)
    result = {}
    for index, proposal in enumerate(proposals):
        span = np.ptp(proposal, axis=0)
        if span.min() <= 0 or span.max() < 3.5 * span.min():
            continue
        inside = _outline_mask(proposal, (height, width)) != 0
        rail = [inside[:5].mean(axis=1).max(), inside[:, :5].mean(axis=0).max(),
                inside[-5:].mean(axis=1).max(), inside[:, -5:].mean(axis=0).max()]
        if (max(rail) < .48 or np.mean(album[inside] == 0) < .80
                or np.median(roughness[inside]) < 11
                or np.mean(np.linalg.norm(lab[inside, 1:] - 128, axis=1) <= 8) < .78):
            continue
        normal_axis = 1 if int(np.argmax(rail)) % 2 == 0 else 0
        traces = []
        for start, stop in zip(proposal, np.roll(proposal, -1, axis=0), strict=True):
            vector = stop - start
            length = np.linalg.norm(vector)
            if length < 38 or abs(vector[normal_axis]) < .55 * length:
                continue
            tangent = vector / length
            normal = np.float32([-tangent[1], tangent[0]])
            centers = start + np.linspace(.09, .91, 28, dtype=np.float32)[:, None] * vector
            grid = centers[:, None] + np.arange(-16, 17, dtype=np.float32)[None, :, None] * normal
            valid = ((grid >= 1) & (grid < [width - 2, height - 2])).all(axis=(1, 2))
            valid &= (_sample(clear, grid) == 1).mean(axis=1) >= .95
            if valid.sum() < 10:
                continue
            profile = _sample(soft, grid)
            spread = _sample(roughness, grid)
            chroma_change = np.linalg.norm(
                np.median(profile[valid, :6, 1:], axis=1) - np.median(profile[valid, -6:, 1:], axis=1),
                axis=1,
            )
            near = np.median(spread[valid, :6])
            far = np.median(spread[valid, -6:])
            if _paper_rim_or_color_change(_sample(lightness, grid), profile[..., 1:], valid):
                traces = []
                break
            if (np.median(chroma_change) <= 2.8 and np.quantile(chroma_change, .95) <= 4.8
                    and min(near, far) >= 12 and .72 <= far / near <= 1.38):
                observed = np.flatnonzero(valid)
                runs = np.split(observed, np.flatnonzero(np.diff(observed) > 1) + 1)
                traces.extend(centers[run[[0, -1]]] for run in runs if len(run) >= 2)
        if traces:
            result[index] = traces
    return result
