from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.degradation import DegradationReport
from app.pipeline.enhancement_types import normalize_bgr_uint8
from app.pipeline.presets import EnhancementConfig


def _neutral_constrained_white_balance(
    image: np.ndarray,
    report: DegradationReport,
    strength: float,
) -> tuple[np.ndarray, dict]:
    """Correct a measured neutral-region Lab cast with bounded chroma motion."""
    if (
        strength <= 0
        or not report.should_correct_color
        or report.cast_confidence < 0.62
        or report.neutral_fraction < 0.04
        or report.monochrome_score >= 0.55
        or report.sepia_ambiguity >= 0.55
    ):
        return image.copy(), {"applied": False, "reason": "neutral_evidence_insufficient"}

    cast_a, cast_b = report.cast_lab
    confidence_weight = float(np.clip(report.cast_confidence, 0.0, 1.0))
    delta_a = float(np.clip(-cast_a * strength * confidence_weight, -8.0, 8.0))
    delta_b = float(np.clip(-cast_b * strength * confidence_weight, -8.0, 8.0))
    if abs(delta_a) < 0.35 and abs(delta_b) < 0.35:
        return image.copy(), {"applied": False, "reason": "cast_below_effective_delta"}

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Strongly saturated pixels are poor illuminant evidence and are protected.
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    chroma_weight = np.clip(1.0 - saturation * 0.72, 0.28, 1.0)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + delta_a * chroma_weight, 0.0, 255.0)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + delta_b * chroma_weight, 0.0, 255.0)
    corrected = cv2.cvtColor(np.rint(lab).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return corrected, {
        "applied": True,
        "method": "neutral_constrained_lab_cast",
        "cast_lab": [float(cast_a), float(cast_b)],
        "applied_lab_delta": [delta_a, delta_b],
        "cast_confidence": report.cast_confidence,
        "neutral_fraction": report.neutral_fraction,
        "max_lab_delta": 8.0,
    }


def apply_color_corrections(
    image: np.ndarray,
    report: DegradationReport,
    config: EnhancementConfig,
) -> tuple[np.ndarray, dict]:
    normalized, contract = normalize_bgr_uint8(image)
    if not config.color_correction_enabled or not config.auto_white_balance:
        return normalized, {
            "color_correction": False,
            "reason": "disabled",
            "input_contract": contract.as_dict(),
        }

    corrected, evidence = _neutral_constrained_white_balance(
        normalized,
        report,
        strength=min(0.55, max(0.0, config.yellowing_removal_strength)),
    )
    return corrected, {
        "color_correction": bool(evidence.get("applied")),
        "steps": ["neutral_constrained_white_balance"] if evidence.get("applied") else [],
        "evidence": evidence,
        "input_contract": contract.as_dict(),
    }
