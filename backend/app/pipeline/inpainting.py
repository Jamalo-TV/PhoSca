import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

_lama = None

def get_lama_model():
    global _lama
    if _lama is not None:
        return _lama
    try:
        from simple_lama_inpainting import SimpleLama
        logger.info("Initializing Simple LaMa for inpainting...")
        _lama = SimpleLama()
    except Exception as e:
        logger.error(f"Failed to load LaMa model: {e}")
        raise
    return _lama


def _normalize_response(response: np.ndarray) -> np.ndarray:
    response = response.astype(np.float32)
    high = float(np.percentile(response, 99.7))
    if high <= 1e-6:
        return np.zeros_like(response, dtype=np.float32)
    return np.clip(response / high, 0.0, 1.0)


def _gabor_line_response(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    response = np.zeros_like(gray_f, dtype=np.float32)
    for theta in np.linspace(0.0, np.pi, 8, endpoint=False):
        kernel = cv2.getGaborKernel((15, 15), 3.0, theta, 8.0, 0.25, 0, ktype=cv2.CV_32F)
        kernel -= kernel.mean()
        filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
        response = np.maximum(response, np.abs(filtered))
    return _normalize_response(response)


def _frangi_line_response(gray: np.ndarray) -> np.ndarray:
    try:
        from skimage.filters import frangi
    except Exception:
        return np.zeros_like(gray, dtype=np.float32)

    gray_f = gray.astype(np.float32) / 255.0
    dark = frangi(gray_f, sigmas=(1, 2, 3), black_ridges=True)
    light = frangi(gray_f, sigmas=(1, 2, 3), black_ridges=False)
    return _normalize_response(np.maximum(dark, light))


def generate_scratch_mask(image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """Detect only small line-like defects instead of normal image edges."""
    sensitivity = max(0.0, min(1.0, sensitivity))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dark_lines = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    light_lines = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    morphology_response = _normalize_response(cv2.max(dark_lines, light_lines))
    gabor_response = _gabor_line_response(gray)
    frangi_response = _frangi_line_response(gray)
    response = np.maximum.reduce([morphology_response * 0.95, gabor_response * 0.78, frangi_response])
    threshold = max(0.42, float(np.percentile(response, 99.82 - (sensitivity * 0.5))))

    mask = (response > threshold).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    image_area = image.shape[0] * image.shape[1]
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < 3 or area > max(300, image_area * 0.003):
            continue
        aspect_ratio = max(w, h) / max(1, min(w, h))
        extent = area / max(1, w * h)
        if aspect_ratio < 3.0 and min(w, h) > 4 and extent > 0.08:
            continue
        if extent > 0.75 and min(w, h) > 2:
            continue
        filtered[labels == label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(filtered, kernel, iterations=1)

def inpaint_scratches(image: np.ndarray, sensitivity: float = 0.5, max_mask_ratio: float = 0.03) -> tuple[np.ndarray, dict]:
    """Uses LaMa to inpaint detected scratches."""
    try:
        lama = get_lama_model()
        mask = generate_scratch_mask(image, sensitivity=sensitivity)

        # Check if there are actually scratches to inpaint
        mask_pixels = int(cv2.countNonZero(mask))
        max_mask_pixels = int(image.shape[0] * image.shape[1] * max_mask_ratio)
        if mask_pixels == 0:
            return image, {"scratch_removal": False, "method": "none", "mask_pixels": 0}
        if mask_pixels > max_mask_pixels:
            return image, {"scratch_removal": False, "method": "skipped_large_mask", "mask_pixels": mask_pixels}

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask).convert('L')

        # LaMa requires PIL Images
        result_pil = lama(pil_image, pil_mask)
        result_cv = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        metadata = {
            "scratch_removal": True,
            "method": "LaMa_inpainting",
            "mask_pixels": mask_pixels,
            "scratch_sensitivity": sensitivity,
        }
        return result_cv, metadata
    except Exception as e:
        logger.error(f"LaMa inpainting failed: {e}. Falling back.")
        return image, {"error": str(e), "scratch_removal": False}
