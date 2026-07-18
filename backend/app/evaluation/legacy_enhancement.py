"""Frozen July-2026 deterministic enhancement baseline.

This module exists only so before/after evaluation remains reproducible after
the production planner changes.  It is never imported by the application
pipeline and must not be used to generate release assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LegacyReport:
    blur: float
    noise: float
    yellowing: float
    fading: float
    compression: float
    severity: str
    preset: str


def _dynamic_range(image: np.ndarray) -> float:
    values = []
    for channel in cv2.split(image):
        low, high = np.percentile(channel, [5, 95])
        values.append((float(high) - float(low)) / 255.0)
    return float(np.mean(values))


def _compression(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if min(gray.shape) < 16:
        return 0.0
    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))
    boundaries = [vertical[:, 7::8], horizontal[7::8, :]]
    regular = [
        np.delete(vertical, np.arange(7, vertical.shape[1], 8), axis=1),
        np.delete(horizontal, np.arange(7, horizontal.shape[0], 8), axis=0),
    ]
    boundary_mean = float(np.mean([item.mean() for item in boundaries if item.size]))
    regular_mean = float(np.mean([item.mean() for item in regular if item.size]))
    return max(0.0, min(1.0, (boundary_mean - regular_mean) / 18.0))


def _analyze(image: np.ndarray) -> LegacyReport:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    noise = float(np.median(np.abs(residual)) * 1.4826)
    mean_b = float(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2].mean())
    yellowing = max(0.0, min(1.0, (mean_b - 132.0) / 42.0))
    fading = max(0.0, min(1.0, 1.0 - _dynamic_range(image)))
    compression = _compression(image)
    blur_component = max(0.0, min(1.0, (120.0 - blur) / 120.0))
    severity_score = max(min(1.0, noise / 22.0), yellowing, fading, blur_component * 0.65)
    if severity_score >= 0.66:
        severity, preset = "severe", "aggressive"
    elif severity_score >= 0.34:
        severity, preset = "moderate", "balanced"
    else:
        severity, preset = "minimal", "light"
    return LegacyReport(blur, noise, yellowing, fading, compression, severity, preset)


def _quality(image: np.ndarray) -> tuple[float, float, float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    noise = float(np.median(np.abs(residual)) * 1.4826)
    dynamic = _dynamic_range(image)
    clipping = float(np.mean((image <= 2) | (image >= 253)))
    blockiness = _compression(image)
    sharpness_score = 1.0 - float(np.exp(-sharpness / 140.0))
    noise_score = 1.0 - min(1.0, noise / 24.0)
    range_score = min(1.0, max(0.0, (dynamic - 0.08) / 0.55))
    overall = (
        sharpness_score * 0.38
        + noise_score * 0.24
        + range_score * 0.38
        - min(0.25, clipping * 3.0)
        - min(0.2, blockiness * 0.4)
    )
    return sharpness, noise, dynamic, clipping, max(0.0, min(1.0, overall))


def _revert(before: np.ndarray, after: np.ndarray, step: str, min_delta: float) -> bool:
    before_quality = _quality(before)
    after_quality = _quality(after)
    sharpness_delta = after_quality[0] - before_quality[0]
    noise_delta = after_quality[1] - before_quality[1]
    range_delta = after_quality[2] - before_quality[2]
    clipping_delta = after_quality[3] - before_quality[3]
    overall_delta = after_quality[4] - before_quality[4]
    if (
        step == "denoise"
        and noise_delta < -0.4
        and abs(sharpness_delta) / max(before_quality[0], 1.0) <= 0.45
        and clipping_delta < 0.04
    ):
        return False
    if step == "color" and range_delta > 0.015 and clipping_delta < 0.04:
        return False
    return overall_delta < min_delta or clipping_delta > 0.06


def _white_balance(image: np.ndarray) -> np.ndarray:
    channels = cv2.split(image.astype(np.float32))
    means = [float(channel.mean()) for channel in channels]
    gray_mean = float(np.mean(means))
    balanced = []
    for channel, mean in zip(channels, means, strict=True):
        scale = gray_mean / max(mean, 1.0)
        balanced.append(np.clip(channel * (1.0 + (scale - 1.0) * 0.2), 0, 255).astype(np.uint8))
    return cv2.merge(balanced)


def _yellow(image: np.ndarray, score: float, strength: float) -> np.ndarray:
    if score <= 0.05:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = image.shape[:2]
    local_b = cv2.resize(
        lab[:, :, 2],
        (max(2, min(10, width // 80 or 2)), max(2, min(10, height // 80 or 2))),
        interpolation=cv2.INTER_AREA,
    )
    correction = np.clip((local_b - 132.0) / 42.0, 0.0, 1.0) * min(20.0, score * 30.0) * strength
    correction_map = cv2.resize(correction, (width, height), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    protection = np.clip((hsv[:, :, 1].astype(np.float32) - 60.0) / 100.0, 0.0, 0.65)
    highlight_bias = np.clip((hsv[:, :, 2].astype(np.float32) - 150.0) / 100.0, 0.0, 0.35)
    lab[:, :, 2] = np.clip(lab[:, :, 2] - correction_map * (1.0 - protection + highlight_bias), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _fade(image: np.ndarray, score: float, strength: float) -> np.ndarray:
    if score <= 0.2:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), 9)
    local_std = np.sqrt(np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 9) - mean * mean, 0.0))
    strength_map = strength * (0.35 + np.clip((28.0 - local_std) / 28.0, 0.0, 1.0) * 0.65)
    channels = []
    for channel in cv2.split(image):
        low, high = np.percentile(channel, [2, 98])
        if high - low < 12:
            channels.append(channel)
            continue
        stretched = (channel.astype(np.float32) - float(low)) * (255.0 / (float(high) - float(low)))
        channels.append(
            np.clip(stretched * strength_map + channel.astype(np.float32) * (1.0 - strength_map), 0, 255).astype(
                np.uint8
            )
        )
    return cv2.bilateralFilter(cv2.merge(channels), d=5, sigmaColor=24, sigmaSpace=24)


def legacy_conservative_enhance_photo(image: np.ndarray) -> tuple[np.ndarray, dict]:
    report = _analyze(image)
    preset = {
        "light": (0.18, 0.25, 0.18, -0.015),
        "balanced": (0.35, 0.55, 0.35, -0.03),
        "aggressive": (0.58, 0.75, 0.55, -0.045),
    }[report.preset]
    working = image.copy()
    steps: list[dict] = []
    if report.noise >= 3.5 or report.compression >= 0.18:
        recommended = max(0.0, min(1.0, (max(report.noise, report.compression * 24.0) - 2.0) / 18.0))
        strength = max(1, min(9, int(round(2 + max(preset[0], recommended) * 8))))
        candidate = cv2.fastNlMeansDenoisingColored(working, None, strength, strength, 7, 21)
        reverted = _revert(working, candidate, "denoise", preset[3])
        working = working if reverted else candidate
        steps.append({"operation": "denoise", "reverted": reverted, "strength": strength})
    if report.yellowing > 0.05 or report.fading > 0.2:
        candidate = _fade(_yellow(_white_balance(working), report.yellowing, preset[1]), report.fading, preset[2])
        reverted = _revert(working, candidate, "color", preset[3])
        working = working if reverted else candidate
        steps.append({"operation": "color", "reverted": reverted})
    return working, {"method": "legacy_2026_07", "report": report.__dict__, "steps": steps}
