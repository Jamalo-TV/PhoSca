from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityReport:
    sharpness: float
    noise_level: float
    dynamic_range: float
    clipping_ratio: float
    blockiness: float
    overall_score: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualityDelta:
    before: QualityReport
    after: QualityReport
    sharpness_delta: float
    noise_delta: float
    dynamic_range_delta: float
    clipping_delta: float
    blockiness_delta: float
    overall_delta: float
    is_improvement: bool

    def as_dict(self) -> dict:
        data = asdict(self)
        data["before"] = self.before.as_dict()
        data["after"] = self.after.as_dict()
        return data


def _as_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _noise_level(gray: np.ndarray) -> float:
    baseline = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - baseline.astype(np.float32)
    return float(np.median(np.abs(residual)) * 1.4826)


def _dynamic_range(image: np.ndarray) -> float:
    ranges = []
    for channel in cv2.split(image):
        low, high = np.percentile(channel, [5, 95])
        ranges.append((float(high) - float(low)) / 255.0)
    return float(np.mean(ranges))


def _blockiness(gray: np.ndarray) -> float:
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0

    gray_f = gray.astype(np.float32)
    vertical = np.abs(np.diff(gray_f, axis=1))
    horizontal = np.abs(np.diff(gray_f, axis=0))

    v_boundary = vertical[:, 7::8]
    h_boundary = horizontal[7::8, :]
    if v_boundary.size == 0 or h_boundary.size == 0:
        return 0.0

    v_regular = np.delete(vertical, np.arange(7, vertical.shape[1], 8), axis=1)
    h_regular = np.delete(horizontal, np.arange(7, horizontal.shape[0], 8), axis=0)
    regular_mean = float(np.mean([v_regular.mean() if v_regular.size else 0.0, h_regular.mean() if h_regular.size else 0.0]))
    boundary_mean = float(np.mean([v_boundary.mean(), h_boundary.mean()]))
    return max(0.0, min(1.0, (boundary_mean - regular_mean) / 18.0))


class QualityAssessor:
    """Lightweight no-reference quality checks for restoration gating."""

    def assess(self, image: np.ndarray) -> QualityReport:
        image = _as_bgr_uint8(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise = _noise_level(gray)
        dynamic_range = _dynamic_range(image)
        clipping_ratio = float(np.mean((image <= 2) | (image >= 253)))
        blockiness = _blockiness(gray)

        sharpness_score = 1.0 - float(np.exp(-sharpness / 140.0))
        noise_score = 1.0 - min(1.0, noise / 24.0)
        range_score = min(1.0, max(0.0, (dynamic_range - 0.08) / 0.55))
        clipping_penalty = min(0.25, clipping_ratio * 3.0)
        blockiness_penalty = min(0.2, blockiness * 0.4)
        overall = (
            (sharpness_score * 0.38)
            + (noise_score * 0.24)
            + (range_score * 0.38)
            - clipping_penalty
            - blockiness_penalty
        )

        return QualityReport(
            sharpness=sharpness,
            noise_level=noise,
            dynamic_range=dynamic_range,
            clipping_ratio=clipping_ratio,
            blockiness=blockiness,
            overall_score=max(0.0, min(1.0, overall)),
        )

    def compare(self, before: np.ndarray, after: np.ndarray) -> QualityDelta:
        if before.shape[:2] != after.shape[:2]:
            after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)

        before_report = self.assess(before)
        after_report = self.assess(after)
        overall_delta = after_report.overall_score - before_report.overall_score
        clipping_delta = after_report.clipping_ratio - before_report.clipping_ratio
        blockiness_delta = after_report.blockiness - before_report.blockiness

        return QualityDelta(
            before=before_report,
            after=after_report,
            sharpness_delta=after_report.sharpness - before_report.sharpness,
            noise_delta=after_report.noise_level - before_report.noise_level,
            dynamic_range_delta=after_report.dynamic_range - before_report.dynamic_range,
            clipping_delta=clipping_delta,
            blockiness_delta=blockiness_delta,
            overall_delta=overall_delta,
            is_improvement=overall_delta >= -0.01 and clipping_delta < 0.05 and blockiness_delta < 0.08,
        )

    def should_revert(self, delta: QualityDelta, min_quality_delta: float = -0.03) -> bool:
        if delta.overall_delta < min_quality_delta:
            return True
        if delta.clipping_delta > 0.06:
            return True
        if delta.blockiness_delta > 0.12:
            return True
        return False
