from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from app.pipeline.enhancement_types import normalize_bgr_uint8


@dataclass(frozen=True)
class QualityReport:
    sharpness: float
    noise_level: float
    dynamic_range: float
    clipping_ratio: float
    blockiness: float
    overall_score: float
    shadow_clipping: tuple[float, ...] = ()
    highlight_clipping: tuple[float, ...] = ()
    saturation_mean: float = 0.0
    saturation_p95: float = 0.0
    texture_energy: float = 0.0
    halo_score: float = 0.0
    mean_lab: tuple[float, ...] = ()

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
    color_delta: float = 0.0
    luminance_delta: float = 0.0
    saturation_delta: float = 0.0
    local_structure_min: float = 1.0
    local_structure_mean: float = 1.0
    texture_ratio: float = 1.0
    halo_delta: float = 0.0
    aspect_ratio_error: float = 0.0
    scale_factors: tuple[float, float] = (1.0, 1.0)
    channel_shadow_delta: tuple[float, ...] = ()
    channel_highlight_delta: tuple[float, ...] = ()

    def as_dict(self) -> dict:
        data = asdict(self)
        data["before"] = self.before.as_dict()
        data["after"] = self.after.as_dict()
        return data


def _noise_level(gray: np.ndarray) -> float:
    baseline = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - baseline.astype(np.float32)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    flat = gradient <= np.percentile(gradient, 65)
    values = residual[flat] if int(flat.sum()) >= 64 else residual.ravel()
    return float(np.median(np.abs(values)) * 1.4826)


def _dynamic_range(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, [5, 95])
    return float((high - low) / 255.0)


def _blockiness(gray: np.ndarray) -> float:
    if gray.shape[0] < 24 or gray.shape[1] < 24:
        return 0.0
    gray_f = gray.astype(np.float32)
    vertical = np.abs(np.diff(gray_f, axis=1))
    horizontal = np.abs(np.diff(gray_f, axis=0))
    scores: list[float] = []
    for phase in range(8):
        vb = vertical[:, phase::8]
        hb = horizontal[phase::8, :]
        if not vb.size or not hb.size:
            continue
        vr = np.delete(vertical, np.arange(phase, vertical.shape[1], 8), axis=1)
        hr = np.delete(horizontal, np.arange(phase, horizontal.shape[0], 8), axis=0)
        regular = float(np.mean([vr.mean() if vr.size else 0.0, hr.mean() if hr.size else 0.0]))
        boundary = float(np.mean([vb.mean(), hb.mean()]))
        scores.append(max(0.0, (boundary - regular) / 18.0))
    return float(np.clip(max(scores, default=0.0), 0.0, 1.0))


def _texture_energy(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    threshold = np.percentile(magnitude, 55)
    textured = magnitude[magnitude >= threshold]
    return float(np.mean(textured)) if textured.size else 0.0


def _halo_score(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float32)
    local_min = cv2.erode(gray, np.ones((3, 3), np.uint8)).astype(np.float32)
    local_max = cv2.dilate(gray, np.ones((3, 3), np.uint8)).astype(np.float32)
    # Isolated one-pixel extrema are a useful conservative halo/ringing proxy.
    overshoot = np.maximum(gray_f - local_max + 1.0, local_min - gray_f + 1.0)
    return float(np.mean(np.clip(overshoot, 0.0, None)) / 255.0)


def _local_ssim(source: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = source_gray.shape
    tile_h = max(16, height // 4)
    tile_w = max(16, width // 4)
    values: list[float] = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            a = source_gray[y : min(height, y + tile_h), x : min(width, x + tile_w)]
            b = candidate_gray[y : min(height, y + tile_h), x : min(width, x + tile_w)]
            if a.size < 16:
                continue
            mean_a, mean_b = float(a.mean()), float(b.mean())
            var_a, var_b = float(a.var()), float(b.var())
            covariance = float(np.mean((a - mean_a) * (b - mean_b)))
            numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
            denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
            values.append(float(np.clip(numerator / max(denominator, 1e-6), -1.0, 1.0)))
    if not values:
        return 1.0, 1.0
    return float(min(values)), float(np.mean(values))


def _source_overshoot(source: np.ndarray, candidate: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    source_min = cv2.erode(source, kernel).astype(np.float32)
    source_max = cv2.dilate(source, kernel).astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    excess = np.maximum(candidate_f - source_max - 3.0, source_min - candidate_f - 3.0)
    return float(np.mean(np.clip(excess, 0.0, None)) / 255.0)


class QualityAssessor:
    """Deterministic, source-relative safety measurements for candidate gating."""

    def assess(self, image: np.ndarray) -> QualityReport:
        image, _ = normalize_bgr_uint8(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise = _noise_level(gray)
        dynamic_range = _dynamic_range(image)
        shadow = tuple(float(np.mean(channel <= 2)) for channel in cv2.split(image))
        highlight = tuple(float(np.mean(channel >= 253)) for channel in cv2.split(image))
        clipping_ratio = float(np.mean((image <= 2) | (image >= 253)))
        blockiness = _blockiness(gray)
        saturation_mean = float(np.mean(hsv[:, :, 1]) / 255.0)
        saturation_p95 = float(np.percentile(hsv[:, :, 1], 95) / 255.0)
        texture = _texture_energy(gray)

        sharpness_score = 1.0 - float(np.exp(-sharpness / 140.0))
        noise_score = 1.0 - min(1.0, noise / 24.0)
        range_score = min(1.0, max(0.0, (dynamic_range - 0.08) / 0.55))
        overall = (
            sharpness_score * 0.30
            + noise_score * 0.28
            + range_score * 0.28
            + min(1.0, texture / 80.0) * 0.14
            - min(0.3, clipping_ratio * 4.0)
            - min(0.2, blockiness * 0.35)
        )
        return QualityReport(
            sharpness=sharpness,
            noise_level=noise,
            dynamic_range=dynamic_range,
            clipping_ratio=clipping_ratio,
            blockiness=blockiness,
            overall_score=float(np.clip(overall, 0.0, 1.0)),
            shadow_clipping=shadow,
            highlight_clipping=highlight,
            saturation_mean=saturation_mean,
            saturation_p95=saturation_p95,
            texture_energy=texture,
            halo_score=_halo_score(gray),
            mean_lab=tuple(float(value) for value in lab.mean(axis=(0, 1))),
        )

    def compare(self, before: np.ndarray, after: np.ndarray) -> QualityDelta:
        before, _ = normalize_bgr_uint8(before)
        after, _ = normalize_bgr_uint8(after)
        before_height, before_width = before.shape[:2]
        after_height, after_width = after.shape[:2]
        scale_factors = (after_width / max(before_width, 1), after_height / max(before_height, 1))
        before_aspect = before_width / max(before_height, 1)
        after_aspect = after_width / max(after_height, 1)
        aspect_error = abs(after_aspect / max(before_aspect, 1e-6) - 1.0)
        comparable = after
        if comparable.shape[:2] != before.shape[:2]:
            comparable = cv2.resize(comparable, (before_width, before_height), interpolation=cv2.INTER_AREA)

        before_report = self.assess(before)
        after_report = self.assess(comparable)
        overall_delta = after_report.overall_score - before_report.overall_score
        clipping_delta = after_report.clipping_ratio - before_report.clipping_ratio
        blockiness_delta = after_report.blockiness - before_report.blockiness
        local_min, local_mean = _local_ssim(before, comparable)
        lab_delta = np.asarray(after_report.mean_lab) - np.asarray(before_report.mean_lab)
        luminance_delta = float(lab_delta[0])
        color_delta = float(np.linalg.norm(lab_delta[1:]))
        texture_ratio = after_report.texture_energy / max(before_report.texture_energy, 1e-6)
        halo_delta = _source_overshoot(before, comparable)
        channel_shadow_delta = tuple(
            a - b for a, b in zip(after_report.shadow_clipping, before_report.shadow_clipping, strict=True)
        )
        channel_highlight_delta = tuple(
            a - b for a, b in zip(after_report.highlight_clipping, before_report.highlight_clipping, strict=True)
        )
        safe = (
            overall_delta >= -0.02
            and clipping_delta <= 0.02
            and color_delta <= 10.0
            and abs(after_report.saturation_mean - before_report.saturation_mean) <= 0.08
            and local_min >= 0.78
            and 0.70 <= texture_ratio <= 1.55
            and halo_delta <= 0.025
            and aspect_error <= 0.005
        )
        return QualityDelta(
            before=before_report,
            after=after_report,
            sharpness_delta=after_report.sharpness - before_report.sharpness,
            noise_delta=after_report.noise_level - before_report.noise_level,
            dynamic_range_delta=after_report.dynamic_range - before_report.dynamic_range,
            clipping_delta=clipping_delta,
            blockiness_delta=blockiness_delta,
            overall_delta=overall_delta,
            is_improvement=safe,
            color_delta=color_delta,
            luminance_delta=luminance_delta,
            saturation_delta=after_report.saturation_mean - before_report.saturation_mean,
            local_structure_min=local_min,
            local_structure_mean=local_mean,
            texture_ratio=texture_ratio,
            halo_delta=halo_delta,
            aspect_ratio_error=aspect_error,
            scale_factors=scale_factors,
            channel_shadow_delta=channel_shadow_delta,
            channel_highlight_delta=channel_highlight_delta,
        )

    def should_revert(self, delta: QualityDelta, min_quality_delta: float = -0.03) -> bool:
        if delta.overall_delta < min_quality_delta:
            return True
        if (
            delta.clipping_delta > 0.02
            or max(delta.channel_shadow_delta + delta.channel_highlight_delta, default=0.0) > 0.03
        ):
            return True
        if delta.blockiness_delta > 0.10:
            return True
        if delta.color_delta > 10.0 or abs(delta.saturation_delta) > 0.08:
            return True
        if delta.local_structure_min < 0.76:
            return True
        if delta.texture_ratio < 0.68 or delta.texture_ratio > 1.65:
            return True
        if delta.halo_delta > 0.025 or delta.aspect_ratio_error > 0.005:
            return True
        return False
