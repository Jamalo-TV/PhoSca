from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from app.pipeline.enhancement_types import DegradationSignal, normalize_bgr_uint8


@dataclass(frozen=True)
class DegradationReport:
    # Compatibility fields retained for API consumers and historical rows.
    blur_score: float
    noise_level: float
    yellowing_score: float
    fading_score: float
    scratch_density: float
    overall_severity: str
    recommended_preset: str
    face_count: int = 0
    face_sizes: tuple[float, ...] = ()
    compression_artifacts: float = 0.0
    dynamic_range: float = 0.0
    should_denoise: bool = False
    should_correct_color: bool = False
    should_inpaint: bool = False
    should_enhance_faces: bool = False
    denoise_strength_recommended: float = 0.0
    # Evidence-based conservative planner fields.
    luma_noise: float = 0.0
    chroma_noise: float = 0.0
    contrast_score: float = 0.0
    underexposure_score: float = 0.0
    overexposure_score: float = 0.0
    shadow_clipping_ratio: float = 0.0
    highlight_clipping_ratio: float = 0.0
    channel_shadow_clipping: tuple[float, ...] = ()
    channel_highlight_clipping: tuple[float, ...] = ()
    saturation_mean: float = 0.0
    saturation_p95: float = 0.0
    monochrome_score: float = 0.0
    sepia_ambiguity: float = 0.0
    cast_score: float = 0.0
    cast_confidence: float = 0.0
    cast_lab: tuple[float, float] = (0.0, 0.0)
    neutral_fraction: float = 0.0
    should_adjust_tone: bool = False
    should_sharpen: bool = False
    analysis_confidence: float = 0.0
    input_contract: dict = field(default_factory=dict)
    signals: dict[str, DegradationSignal] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        data = asdict(self)
        data["signals"] = {name: signal.as_dict() for name, signal in self.signals.items()}
        return data


def _analysis_view(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )


def _robust_noise(gray: np.ndarray, chroma: np.ndarray) -> tuple[float, float, float]:
    gray_f = gray.astype(np.float32)
    smooth = cv2.medianBlur(gray, 3).astype(np.float32)
    residual = gray_f - smooth
    gradient = cv2.magnitude(
        cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3),
    )
    flat = gradient <= np.percentile(gradient, 65)
    if int(flat.sum()) < 64:
        flat = np.ones_like(gray, dtype=bool)
    luma_noise = float(np.median(np.abs(residual[flat])) * 1.4826)

    chroma_f = chroma.astype(np.float32)
    chroma_smooth = cv2.GaussianBlur(chroma_f, (3, 3), 0.8)
    chroma_residual = np.linalg.norm(chroma_f - chroma_smooth, axis=2)
    chroma_noise = float(np.median(chroma_residual[flat]) * 1.4826)
    flat_fraction = float(np.mean(flat))
    return luma_noise, chroma_noise, flat_fraction


def _compression_artifacts(gray: np.ndarray) -> float:
    if gray.shape[0] < 24 or gray.shape[1] < 24:
        return 0.0
    gray_f = gray.astype(np.float32)
    vertical = np.abs(np.diff(gray_f, axis=1))
    horizontal = np.abs(np.diff(gray_f, axis=0))
    phase_scores: list[float] = []
    for phase in range(8):
        v_boundary = vertical[:, phase::8]
        h_boundary = horizontal[phase::8, :]
        if not v_boundary.size or not h_boundary.size:
            continue
        v_regular = np.delete(vertical, np.arange(phase, vertical.shape[1], 8), axis=1)
        h_regular = np.delete(horizontal, np.arange(phase, horizontal.shape[0], 8), axis=0)
        regular = float(
            np.mean([v_regular.mean() if v_regular.size else 0.0, h_regular.mean() if h_regular.size else 0.0])
        )
        boundary = float(np.mean([v_boundary.mean(), h_boundary.mean()]))
        phase_scores.append(max(0.0, (boundary - regular) / 18.0))
    return float(np.clip(max(phase_scores, default=0.0), 0.0, 1.0))


def _color_summary(image: np.ndarray, lab: np.ndarray, hsv: np.ndarray) -> dict[str, Any]:
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    sat_mean = float(np.mean(saturation))
    sat_p95 = float(np.percentile(saturation, 95))

    b, g, r = cv2.split(image.astype(np.float32) / 255.0)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    monochrome = float(np.clip(1.0 - colorfulness / 0.12, 0.0, 1.0))

    warm_order = float(np.mean((r > g + 0.025) & (g > b + 0.015)))
    lab_b_mean = float(lab[:, :, 2].mean() - 128.0)
    sepia = float(np.clip(warm_order * 0.7 + max(0.0, lab_b_mean / 35.0) * 0.3, 0.0, 1.0))

    lightness = lab[:, :, 0]
    neutral_mask = (hsv[:, :, 1] <= 38) & (lightness >= 35) & (lightness <= 225)
    neutral_fraction = float(np.mean(neutral_mask))
    if int(neutral_mask.sum()) >= 64:
        neutral_a = lab[:, :, 1][neutral_mask].astype(np.float32) - 128.0
        neutral_b = lab[:, :, 2][neutral_mask].astype(np.float32) - 128.0
        cast_a = float(np.median(neutral_a))
        cast_b = float(np.median(neutral_b))
        dispersion = float(np.median(np.sqrt((neutral_a - cast_a) ** 2 + (neutral_b - cast_b) ** 2)))
        cast_score = float(np.clip(np.hypot(cast_a, cast_b) / 24.0, 0.0, 1.0))
        cast_confidence = float(np.clip((neutral_fraction / 0.25) * (1.0 - dispersion / 35.0), 0.0, 1.0))
    else:
        cast_a = cast_b = cast_score = cast_confidence = 0.0

    return {
        "saturation_mean": sat_mean,
        "saturation_p95": sat_p95,
        "monochrome_score": monochrome,
        "sepia_ambiguity": sepia,
        "neutral_fraction": neutral_fraction,
        "cast_score": cast_score,
        "cast_confidence": cast_confidence,
        "cast_lab": (cast_a, cast_b),
        "lab_b_mean": lab_b_mean,
    }


def _face_summary(image: np.ndarray) -> tuple[int, tuple[float, ...]]:
    try:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            return 0, ()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    except Exception:
        return 0, ()
    image_area = max(1, image.shape[0] * image.shape[1])
    sizes = tuple(float((w * h) / image_area) for _, _, w, h in faces)
    return len(sizes), sizes


def _scratch_density(image: np.ndarray) -> float:
    try:
        from app.pipeline.inpainting import generate_scratch_mask

        mask = generate_scratch_mask(image)
        return float(cv2.countNonZero(mask) / mask.size)
    except Exception:
        return 0.0


def analyze_degradation(image: np.ndarray) -> DegradationReport:
    normalized, contract = normalize_bgr_uint8(image)
    view = _analysis_view(normalized)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)

    p01, p05, median, p95, p99 = (float(value) for value in np.percentile(gray, [1, 5, 50, 95, 99]))
    dynamic_range = (p95 - p05) / 255.0
    low_contrast = float(np.clip((0.46 - dynamic_range) / 0.34, 0.0, 1.0))
    shadow_ratio = float(np.mean(gray <= 3))
    highlight_ratio = float(np.mean(gray >= 252))
    # A scan can be materially underexposed without piling pixels at black.
    # Require both depressed midtones or highlight headroom; this catches a
    # globally dark capture while leaving normally exposed warm/sepia scenes.
    underexposure = float(
        max(
            np.clip((125.0 - median) / 55.0, 0.0, 1.0),
            np.clip((175.0 - p95) / 80.0, 0.0, 1.0),
        )
    )
    overexposure = float(
        np.clip((median - 178.0) / 68.0, 0.0, 1.0) * np.clip((0.12 + float(np.mean(gray > 230))) / 0.32, 0.0, 1.0)
    )

    channel_shadow = tuple(float(np.mean(channel <= 2)) for channel in cv2.split(view))
    channel_highlight = tuple(float(np.mean(channel >= 253)) for channel in cv2.split(view))
    luma_noise, chroma_noise, flat_fraction = _robust_noise(gray, lab[:, :, 1:3])
    compression = _compression_artifacts(gray)
    color = _color_summary(view, lab, hsv)

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = laplacian_variance
    blur_signal = float(np.clip((95.0 - laplacian_variance) / 95.0, 0.0, 1.0))
    blur_confidence = float(np.clip(1.0 - max(luma_noise / 18.0, compression * 0.5), 0.0, 1.0))
    scratch_density = _scratch_density(view)
    face_count, face_sizes = _face_summary(view)

    monochrome = float(color["monochrome_score"])
    sepia = float(color["sepia_ambiguity"])
    cast_score = float(color["cast_score"])
    cast_confidence = float(color["cast_confidence"])
    tone_confidence = float(np.clip(1.0 - max(underexposure, overexposure) * 0.45, 0.25, 1.0))
    should_adjust_tone = (
        (low_contrast >= 0.28 or underexposure >= 0.28)
        and tone_confidence >= 0.55
        and underexposure < 0.82
        and overexposure < 0.72
    )
    # Preserve the historical "color review suggested" flag for warm/cast
    # evidence, while the typed planner below requires confidence and vetoes
    # monochrome/sepia ambiguity before changing pixels.
    should_correct_color = cast_score >= 0.18 and (cast_confidence >= 0.62 or sepia >= 0.58)
    should_denoise = (luma_noise >= 4.5 or chroma_noise >= 3.8 or compression >= 0.22) and flat_fraction >= 0.25
    should_sharpen = blur_signal >= 0.42 and blur_confidence >= 0.62 and luma_noise < 5.5 and compression < 0.25
    should_inpaint = scratch_density > 0.00015

    denoise_strength = float(np.clip((max(luma_noise, chroma_noise, compression * 18.0) - 3.0) / 15.0, 0.0, 1.0))
    yellowing = float(np.clip(max(0.0, float(color["lab_b_mean"])) / 42.0, 0.0, 1.0))
    fading = low_contrast * tone_confidence

    actionable = max(
        denoise_strength if should_denoise else 0.0,
        low_contrast * 0.75 if should_adjust_tone else 0.0,
        cast_score * cast_confidence if should_correct_color else 0.0,
        blur_signal * blur_confidence * 0.7 if should_sharpen else 0.0,
    )
    if actionable >= 0.68:
        severity, preset = "severe", "aggressive"
    elif actionable >= 0.30:
        severity, preset = "moderate", "balanced"
    else:
        severity, preset = "minimal", "light"

    reasons: list[str] = []
    if monochrome >= 0.75:
        reasons.append("MONOCHROME_SOURCE")
    if sepia >= 0.55:
        reasons.append("INTENTIONAL_WARM_TONE_AMBIGUOUS")
    if should_denoise:
        reasons.append("NOISE_OR_COMPRESSION_DETECTED")
    if should_adjust_tone:
        reasons.append("LOW_CONTRAST_DETECTED")
    if should_correct_color:
        reasons.append("RELIABLE_NEUTRAL_COLOR_CAST")
    if should_sharpen:
        reasons.append("LOW_EDGE_ACUTANCE")

    signals = {
        "contrast": DegradationSignal(
            low_contrast, tone_confidence, {"p05": p05, "p50": median, "p95": p95, "range": dynamic_range}
        ),
        "underexposure": DegradationSignal(
            underexposure, 0.75, {"p01": p01, "p50": median, "p95": p95, "shadow_ratio": shadow_ratio}
        ),
        "overexposure": DegradationSignal(
            overexposure, 0.75, {"p50": median, "p99": p99, "highlight_ratio": highlight_ratio}
        ),
        "noise": DegradationSignal(
            float(np.clip(max(luma_noise, chroma_noise) / 14.0, 0.0, 1.0)),
            flat_fraction,
            {"luma": luma_noise, "chroma": chroma_noise},
        ),
        "blur": DegradationSignal(blur_signal, blur_confidence, {"laplacian_variance": laplacian_variance}),
        "color_cast": DegradationSignal(
            cast_score, cast_confidence, {"lab": color["cast_lab"], "neutral_fraction": color["neutral_fraction"]}
        ),
        "monochrome": DegradationSignal(monochrome, 0.9, {"saturation_mean": color["saturation_mean"]}),
        "sepia_ambiguity": DegradationSignal(sepia, 0.75, {"lab_b_mean": color["lab_b_mean"]}),
    }

    return DegradationReport(
        blur_score=blur_score,
        noise_level=luma_noise,
        yellowing_score=yellowing,
        fading_score=fading,
        scratch_density=scratch_density,
        overall_severity=severity,
        recommended_preset=preset,
        face_count=face_count,
        face_sizes=face_sizes,
        compression_artifacts=compression,
        dynamic_range=dynamic_range,
        should_denoise=should_denoise,
        should_correct_color=should_correct_color,
        should_inpaint=should_inpaint,
        should_enhance_faces=face_count > 0 and blur_signal >= 0.5 and blur_confidence >= 0.65,
        denoise_strength_recommended=denoise_strength,
        luma_noise=luma_noise,
        chroma_noise=chroma_noise,
        contrast_score=low_contrast,
        underexposure_score=underexposure,
        overexposure_score=overexposure,
        shadow_clipping_ratio=shadow_ratio,
        highlight_clipping_ratio=highlight_ratio,
        channel_shadow_clipping=channel_shadow,
        channel_highlight_clipping=channel_highlight,
        saturation_mean=float(color["saturation_mean"]),
        saturation_p95=float(color["saturation_p95"]),
        monochrome_score=monochrome,
        sepia_ambiguity=sepia,
        cast_score=cast_score,
        cast_confidence=cast_confidence,
        cast_lab=(float(color["cast_lab"][0]), float(color["cast_lab"][1])),
        neutral_fraction=float(color["neutral_fraction"]),
        should_adjust_tone=should_adjust_tone,
        should_sharpen=should_sharpen,
        analysis_confidence=float(np.clip(min(0.95, 0.55 + flat_fraction * 0.35), 0.0, 1.0)),
        input_contract=contract.as_dict(),
        signals=signals,
        reason_codes=tuple(reasons),
    )
