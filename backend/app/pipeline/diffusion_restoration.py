import logging

import cv2
import numpy as np

from app.pipeline.degradation import DegradationReport
from app.pipeline.dl_enhancement import dl_enhance_photo
from app.utils.device import get_best_device

logger = logging.getLogger(__name__)


def _match_size(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)


def _restore_tone(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    restored_lightness = clahe.apply(lightness)
    restored = cv2.merge((restored_lightness, channel_a, channel_b))
    return cv2.cvtColor(restored, cv2.COLOR_LAB2BGR)


def _unsharp_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.1)
    return cv2.addWeighted(image, 1.18, blurred, -0.18, 0)


def premium_enhance_photo(
    image: np.ndarray,
    base_restored: np.ndarray | None = None,
    base_metadata: dict | None = None,
    preset: str | None = "balanced",
    degradation_report: DegradationReport | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Premium restoration that preserves the original photo structure.

    Generic diffusion image-to-image models are creative, not archival: even at
    low denoise strengths they can replace faces, clothing, and backgrounds.
    This premium path keeps the AI upscaling/face restoration stack, then adds
    conservative tonal cleanup and sharpening with a source-preserving blend.
    """
    try:
        if base_restored is None:
            ai_restored, ai_metadata = dl_enhance_photo(image, preset=preset, degradation_report=degradation_report)
        else:
            ai_restored = base_restored
            ai_metadata = dict(base_metadata or {})
        toned = _restore_tone(ai_restored)
        sharpened = _unsharp_mask(toned)
        reference = _match_size(sharpened, image)
        premium = cv2.addWeighted(sharpened, 0.84, reference, 0.16, 0)
        metadata = {
            "method": "premium_preservation_restoration",
            "models": ai_metadata.get("models", []),
            "device": str(get_best_device()),
            "source_preservation_blend": 0.16,
            "denoise_order": "before_upscale_in_standard_pipeline",
            "diffusion_used": False,
            "ai_metadata": ai_metadata,
        }
        return premium, metadata
    except Exception as exc:  # noqa: BLE001 - restoration should fail closed to source pixels.
        logger.error("Premium enhancement failed: %s. Falling back.", exc)
        return image, {"error": str(exc), "method": "failed"}
