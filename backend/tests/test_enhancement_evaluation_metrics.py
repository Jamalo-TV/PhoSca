from __future__ import annotations

import cv2
import numpy as np


def test_enhancement_metrics_reward_identity_and_detect_color_shift() -> None:
    from app.evaluation.enhancement import measure_enhancement

    gradient = np.tile(np.linspace(20, 235, 180, dtype=np.uint8), (120, 1))
    reference = cv2.merge((gradient, np.flip(gradient, axis=1), np.full_like(gradient, 128)))
    identity = measure_enhancement(reference, reference)
    shifted = reference.copy()
    shifted[:, :, 2] = np.clip(shifted[:, :, 2].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    changed = measure_enhancement(shifted, reference)

    assert identity.psnr == 100.0
    assert identity.ssim is not None and identity.ssim > 0.999
    assert changed.lab_error is not None and changed.lab_error > 10
    assert changed.psnr is not None and changed.psnr < identity.psnr


def test_enhancement_metrics_detect_clipping_and_halos() -> None:
    from app.evaluation.enhancement import measure_enhancement

    image = np.full((160, 240, 3), 110, dtype=np.uint8)
    cv2.rectangle(image, (50, 40), (190, 120), (190, 190, 190), -1)
    clipped = cv2.convertScaleAbs(image, alpha=3.0, beta=-160)
    sharpened = cv2.addWeighted(image, 2.8, cv2.GaussianBlur(image, (0, 0), 2.0), -1.8, 0)

    source_metrics = measure_enhancement(image)
    clipped_metrics = measure_enhancement(clipped)
    sharpened_metrics = measure_enhancement(sharpened)

    assert clipped_metrics.clipping_ratio > source_metrics.clipping_ratio
    assert sharpened_metrics.halo_score >= source_metrics.halo_score
