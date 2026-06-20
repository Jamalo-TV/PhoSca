import cv2
import numpy as np


def recover_flash_highlights(image: np.ndarray) -> tuple[np.ndarray, dict]:
    blown_mask = cv2.inRange(image, np.array([250, 250, 250], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    blown_pixels = int(cv2.countNonZero(blown_mask))
    if blown_pixels == 0:
        return image.copy(), {"flash_recovery": False, "blown_pixels": 0}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(blown_mask, kernel, iterations=1)
    recovered = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return recovered, {"flash_recovery": True, "blown_pixels": blown_pixels, "method": "telea_inpaint"}


def reduce_fold_shadows(image: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    threshold = float(np.percentile(gradient, 95))
    fold_mask = (gradient > threshold).astype(np.uint8) * 255
    fold_pixels = int(cv2.countNonZero(fold_mask))
    if fold_pixels == 0:
        return image.copy(), {"fold_removal": False, "fold_pixels": 0}

    blur_layer = cv2.GaussianBlur(image, (25, 25), 0)
    texture = cv2.subtract(image, blur_layer)
    smooth_blur = cv2.GaussianBlur(blur_layer, (25, 25), 0)
    mask_3 = cv2.cvtColor(cv2.GaussianBlur(fold_mask, (15, 15), 0), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended_blur = (smooth_blur.astype(np.float32) * mask_3) + (blur_layer.astype(np.float32) * (1 - mask_3))
    recombined = cv2.add(np.clip(blended_blur, 0, 255).astype(np.uint8), texture)
    return recombined, {"fold_removal": True, "fold_pixels": fold_pixels, "method": "frequency_separation"}


def enhance_photo(image: np.ndarray) -> tuple[np.ndarray, dict]:
    recovered, flash_metadata = recover_flash_highlights(image)
    reduced, fold_metadata = reduce_fold_shadows(recovered)
    return reduced, {**flash_metadata, **fold_metadata}

