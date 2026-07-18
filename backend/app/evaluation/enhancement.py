"""Fidelity-oriented metrics for deterministic enhancement candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementMetrics:
    psnr: float | None
    ssim: float | None
    lab_error: float | None
    clipping_ratio: float
    shadow_clipping_ratio: float
    highlight_clipping_ratio: float
    saturation_mean: float
    noise_level: float
    sharpness: float
    halo_score: float
    texture_error: float | None
    structural_error: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = np.rint(image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Evaluation expects grayscale, BGR, or BGRA pixels.")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        return np.rint(np.clip(image.astype(np.float32) / maximum, 0.0, 1.0) * 255.0).astype(np.uint8)
    if np.issubdtype(image.dtype, np.floating) and np.isfinite(image).all():
        maximum = float(np.max(image)) if image.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        return np.rint(np.clip(image.astype(np.float32) * scale, 0.0, 255.0)).astype(np.uint8)
    raise ValueError("Unsupported evaluation pixel type.")


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_f = reference.astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_reference = cv2.GaussianBlur(reference_f, (11, 11), 1.5)
    mu_candidate = cv2.GaussianBlur(candidate_f, (11, 11), 1.5)
    sigma_reference = cv2.GaussianBlur(reference_f * reference_f, (11, 11), 1.5) - mu_reference * mu_reference
    sigma_candidate = cv2.GaussianBlur(candidate_f * candidate_f, (11, 11), 1.5) - mu_candidate * mu_candidate
    covariance = cv2.GaussianBlur(reference_f * candidate_f, (11, 11), 1.5) - mu_reference * mu_candidate
    numerator = (2.0 * mu_reference * mu_candidate + c1) * (2.0 * covariance + c2)
    denominator = (mu_reference * mu_reference + mu_candidate * mu_candidate + c1) * (
        sigma_reference + sigma_candidate + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-6)))


def _noise(gray: np.ndarray) -> float:
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    return float(np.median(np.abs(residual)) * 1.4826)


def _halo_score(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(gray_f, (0, 0), 1.2)
    high = gray_f - smooth
    gradient = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    strong = gradient >= np.percentile(gradient, 90)
    if not np.any(strong):
        return 0.0
    overshoot = np.maximum(np.abs(high) - 12.0, 0.0)
    return float(np.mean(overshoot[strong]) / 255.0)


def measure_enhancement(image: np.ndarray, reference: np.ndarray | None = None) -> EnhancementMetrics:
    candidate = _bgr_uint8(image)
    gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)
    low_clip = float(np.mean(np.all(candidate <= 2, axis=2)))
    high_clip = float(np.mean(np.all(candidate >= 253, axis=2)))
    psnr = ssim = lab_error = texture_error = structural_error = None
    if reference is not None:
        reference = _bgr_uint8(reference)
        if reference.shape[:2] != candidate.shape[:2]:
            candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        error = reference.astype(np.float32) - candidate.astype(np.float32)
        mse = float(np.mean(error * error))
        psnr = 100.0 if mse == 0 else float(10.0 * np.log10((255.0 * 255.0) / mse))
        ssim = _ssim(reference, candidate)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_error = float(np.mean(np.linalg.norm(reference_lab - candidate_lab, axis=2)))
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        gradient = cv2.magnitude(
            cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        textured = gradient >= np.percentile(gradient, 65)
        texture_error = (
            float(np.mean(np.abs(reference_gray.astype(np.float32) - gray.astype(np.float32))[textured]) / 255.0)
            if np.any(textured)
            else 0.0
        )
        structural_error = float(1.0 - ssim)
    return EnhancementMetrics(
        psnr=psnr,
        ssim=ssim,
        lab_error=lab_error,
        clipping_ratio=low_clip + high_clip,
        shadow_clipping_ratio=low_clip,
        highlight_clipping_ratio=high_clip,
        saturation_mean=float(np.mean(hsv[:, :, 1]) / 255.0),
        noise_level=_noise(gray),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        halo_score=_halo_score(gray),
        texture_error=texture_error,
        structural_error=structural_error,
    )
