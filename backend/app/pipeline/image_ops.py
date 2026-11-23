import hashlib
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from app.utils.image_frames import primary_frame_policy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _working_rgb(image: Image.Image) -> tuple[np.ndarray, dict]:
    source_mode = image.mode
    metadata: dict = {
        "source_mode": source_mode,
        "source_bands": list(image.getbands()),
        "alpha_policy": "none",
        "bit_depth_policy": "preserve_8_bit",
    }
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
        metadata["alpha_policy"] = "composite_over_white_before_working_space"
    elif source_mode.startswith("I;16") or source_mode in {"I", "F"}:
        values = np.asarray(image)
        if not np.isfinite(values).all():
            raise ValueError("Image contains non-finite high-bit-depth pixels.")
        values_f = values.astype(np.float32)
        if source_mode.startswith("I;16") or (source_mode == "I" and float(values_f.max(initial=0.0)) > 255.0):
            maximum = 65535.0 if float(values_f.max(initial=0.0)) <= 65535.0 else float(values_f.max(initial=1.0))
            normalized = np.clip(values_f / max(maximum, 1.0), 0.0, 1.0)
            metadata["bit_depth_policy"] = "scale_integer_range_to_8_bit"
        elif source_mode == "F" and float(values_f.max(initial=0.0)) <= 1.0:
            normalized = np.clip(values_f, 0.0, 1.0)
            metadata["bit_depth_policy"] = "scale_unit_float_to_8_bit"
        else:
            normalized = np.clip(values_f / 255.0, 0.0, 1.0)
            metadata["bit_depth_policy"] = "clip_float_or_integer_to_8_bit"
        gray = np.rint(normalized * 255.0).astype(np.uint8)
        image = Image.fromarray(gray, mode="L").convert("RGB")
    else:
        image = image.convert("RGB")
        if source_mode in {"1", "L", "LA"}:
            metadata["bit_depth_policy"] = "expand_grayscale_to_rgb8"
    return np.asarray(image), metadata


def load_image_with_metadata(path: Path) -> tuple[np.ndarray, dict]:
    """Decode one canonical, orientation-corrected sRGB working image.

    The original bytes remain immutable in storage; this function only defines
    the pixel contract used by legacy OpenCV stages.  ICC conversion is
    explicit so camera profiles do not silently alter segmentation metrics.
    """
    try:
        with Image.open(path) as source:
            frame_policy = primary_frame_policy(source)
            source_mode = source.mode
            source_bands = list(source.getbands())
            exif_orientation = source.getexif().get(274, 1)
            image = ImageOps.exif_transpose(source)
            exif_orientation_applied = exif_orientation not in {None, 1}
            icc_profile = image.info.get("icc_profile")
            icc_converted = False
            if icc_profile:
                try:
                    alpha = image.getchannel("A") if "A" in image.getbands() else None
                    converted = ImageCms.profileToProfile(
                        image.convert("RGB"),
                        ImageCms.ImageCmsProfile(BytesIO(icc_profile)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                    if alpha is not None:
                        converted.putalpha(alpha)
                    image = converted
                    icc_converted = True
                except (ImageCms.PyCMSError, OSError, ValueError):
                    pass
            rgb, contract = _working_rgb(image)
            contract["source_mode"] = source_mode
            contract["source_bands"] = source_bands
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read image: {path}") from exc
    if rgb.size == 0:
        raise ValueError(f"Could not read image: {path}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, {
        **contract,
        "frame_policy": frame_policy,
        "exif_orientation_applied": exif_orientation_applied,
        "icc_profile_present": bool(icc_profile),
        "icc_to_srgb_applied": icc_converted,
        "working_color_space": "sRGB",
        "working_channel_order": "BGR",
        "working_dtype": str(bgr.dtype),
        "working_width": int(bgr.shape[1]),
        "working_height": int(bgr.shape[0]),
    }


def load_image(path: Path) -> np.ndarray:
    image, _ = load_image_with_metadata(path)
    return image


def save_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError(f"Could not write image: {path}")


def calculate_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalized_box_to_pixels(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(width - 1, int(round(float(box["x1"]) * width)))),
        max(0, min(height - 1, int(round(float(box["y1"]) * height)))),
        max(0, min(width, int(round(float(box["x2"]) * width)))),
        max(0, min(height, int(round(float(box["y2"]) * height)))),
    )


def pixel_box_to_normalized(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> dict[str, float]:
    return {
        "x1": max(0.0, min(1.0, x / image_width)),
        "y1": max(0.0, min(1.0, y / image_height)),
        "x2": max(0.0, min(1.0, (x + w) / image_width)),
        "y2": max(0.0, min(1.0, (y + h) / image_height)),
    }


def box_center(box: dict[str, float]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2.0, (float(box["y1"]) + float(box["y2"])) / 2.0)


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    x_left = max(float(a["x1"]), float(b["x1"]))
    y_top = max(float(a["y1"]), float(b["y1"]))
    x_right = min(float(a["x2"]), float(b["x2"]))
    y_bottom = min(float(a["y2"]), float(b["y2"]))
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (float(a["x2"]) - float(a["x1"])) * (float(a["y2"]) - float(a["y1"]))
    area_b = (float(b["x2"]) - float(b["x1"])) * (float(b["y2"]) - float(b["y1"]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
