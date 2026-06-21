from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.degradation import DegradationReport
from app.pipeline.presets import EnhancementConfig


def _auto_white_balance(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    channels = cv2.split(image.astype(np.float32))
    means = [float(channel.mean()) for channel in channels]
    gray_mean = float(np.mean(means))
    balanced = []
    for channel, mean in zip(channels, means, strict=True):
        scale = gray_mean / max(mean, 1.0)
        adjusted = channel * (1.0 + (scale - 1.0) * strength)
        balanced.append(np.clip(adjusted, 0, 255).astype(np.uint8))
    return cv2.merge(balanced)


def _remove_yellowing_spatial(image: np.ndarray, score: float, strength: float) -> np.ndarray:
    if score <= 0.05 or strength <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = image.shape[:2]
    grid_w = max(2, min(10, width // 80 or 2))
    grid_h = max(2, min(10, height // 80 or 2))

    local_b = cv2.resize(lab[:, :, 2], (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    local_excess = np.clip((local_b - 132.0) / 42.0, 0.0, 1.0)
    correction = local_excess * min(20.0, score * 30.0) * strength
    correction_map = cv2.resize(correction, (width, height), interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    warm_subject_protection = np.clip((saturation - 60.0) / 100.0, 0.0, 0.65)
    highlight_yellowing_bias = np.clip((value - 150.0) / 100.0, 0.0, 0.35)
    correction_map *= 1.0 - warm_subject_protection + highlight_yellowing_bias

    lab[:, :, 2] = np.clip(lab[:, :, 2] - correction_map, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _restore_fading_adaptive(image: np.ndarray, score: float, strength: float) -> np.ndarray:
    if score <= 0.2 or strength <= 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), 9)
    mean_sq = cv2.GaussianBlur(gray * gray, (0, 0), 9)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    flatness = np.clip((28.0 - local_std) / 28.0, 0.0, 1.0)
    strength_map = strength * (0.35 + flatness * 0.65)

    restored_channels = []
    for channel in cv2.split(image):
        low, high = np.percentile(channel, [2, 98])
        if high - low < 12:
            restored_channels.append(channel)
            continue
        stretched = (channel.astype(np.float32) - float(low)) * (255.0 / (float(high) - float(low)))
        blended = (stretched * strength_map) + (channel.astype(np.float32) * (1.0 - strength_map))
        restored_channels.append(np.clip(blended, 0, 255).astype(np.uint8))
    restored = cv2.merge(restored_channels)
    return cv2.bilateralFilter(restored, d=5, sigmaColor=24, sigmaSpace=24)


def apply_color_corrections(image: np.ndarray, report: DegradationReport, config: EnhancementConfig) -> tuple[np.ndarray, dict]:
    if not config.color_correction_enabled:
        return image, {"color_correction": False}

    corrected = image.copy()
    steps: list[str] = []
    if config.auto_white_balance:
        corrected = _auto_white_balance(corrected, strength=0.2)
        steps.append("white_balance")
    if report.yellowing_score > 0.05:
        corrected = _remove_yellowing_spatial(corrected, report.yellowing_score, config.yellowing_removal_strength)
        steps.append("spatial_yellowing_removal")
    if report.fading_score > 0.2:
        corrected = _restore_fading_adaptive(corrected, report.fading_score, config.fading_recovery_strength)
        steps.append("adaptive_fading_recovery")

    return corrected, {
        "color_correction": bool(steps),
        "steps": steps,
        "yellowing_score": report.yellowing_score,
        "fading_score": report.fading_score,
    }
