from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DegradationReport:
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

    def as_dict(self) -> dict:
        return asdict(self)


def _blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _noise_level(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    baseline = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - baseline.astype(np.float32)
    return float(np.median(np.abs(residual)) * 1.4826)


def _yellowing_score(image: np.ndarray) -> float:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean_b = float(lab[:, :, 2].mean())
    return max(0.0, min(1.0, (mean_b - 132.0) / 42.0))


def _dynamic_range(image: np.ndarray) -> float:
    ranges = []
    for channel in cv2.split(image):
        low, high = np.percentile(channel, [5, 95])
        ranges.append((float(high) - float(low)) / 255.0)
    return float(np.mean(ranges))


def _fading_score(image: np.ndarray) -> float:
    dynamic_range = _dynamic_range(image)
    return max(0.0, min(1.0, 1.0 - dynamic_range))


def _compression_artifacts(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def _face_summary(image: np.ndarray) -> tuple[int, tuple[float, ...]]:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
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


def _severity(noise_level: float, yellowing_score: float, fading_score: float, scratch_density: float, blur_score: float) -> tuple[str, str]:
    blur_component = max(0.0, min(1.0, (120.0 - blur_score) / 120.0))
    noise_component = max(0.0, min(1.0, noise_level / 22.0))
    scratch_component = max(0.0, min(1.0, scratch_density / 0.02))
    severity_score = max(noise_component, yellowing_score, fading_score, scratch_component, blur_component * 0.65)
    if severity_score >= 0.66:
        return "severe", "aggressive"
    if severity_score >= 0.34:
        return "moderate", "balanced"
    return "minimal", "light"


def analyze_degradation(image: np.ndarray) -> DegradationReport:
    blur = _blur_score(image)
    noise = _noise_level(image)
    yellowing = _yellowing_score(image)
    dynamic_range = _dynamic_range(image)
    fading = max(0.0, min(1.0, 1.0 - dynamic_range))
    scratches = _scratch_density(image)
    compression = _compression_artifacts(image)
    face_count, face_sizes = _face_summary(image)
    severity, preset = _severity(noise, yellowing, fading, scratches, blur)
    should_denoise = noise >= 3.5 or compression >= 0.18
    should_correct_color = yellowing > 0.05 or fading > 0.2
    should_inpaint = scratches > 0.00015
    denoise_strength = max(0.0, min(1.0, (max(noise, compression * 24.0) - 2.0) / 18.0))
    return DegradationReport(
        blur_score=blur,
        noise_level=noise,
        yellowing_score=yellowing,
        fading_score=fading,
        scratch_density=scratches,
        overall_severity=severity,
        recommended_preset=preset,
        face_count=face_count,
        face_sizes=face_sizes,
        compression_artifacts=compression,
        dynamic_range=dynamic_range,
        should_denoise=should_denoise,
        should_correct_color=should_correct_color,
        should_inpaint=should_inpaint,
        should_enhance_faces=face_count > 0,
        denoise_strength_recommended=denoise_strength,
    )
