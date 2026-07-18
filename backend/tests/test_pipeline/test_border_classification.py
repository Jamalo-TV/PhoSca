from __future__ import annotations

import cv2
import numpy as np


def test_content_aware_border_trims_independent_white_edges_with_provenance() -> None:
    from app.pipeline.border_removal import trim_physical_border

    image = np.full((180, 260, 3), 238, dtype=np.uint8)
    image[18:164, 27:241] = (55, 105, 155)
    cv2.line(image, (45, 38), (220, 142), (220, 225, 230), 4)

    result = trim_physical_border(image, max_crop_ratio=0.2)

    assert result.applied is True
    assert result.offsets["top"] >= 14
    assert result.offsets["left"] >= 22
    assert result.image.shape[0] < image.shape[0]
    assert result.image.shape[1] < image.shape[1]
    assert result.confidence >= 0.68
    assert result.metadata()["method"] == "content_aware_independent_edges_v1"


def test_content_aware_border_preserves_quiet_sky_and_dark_scene_edges() -> None:
    from app.pipeline.border_removal import trim_physical_border

    height, width = 180, 280
    gradient = np.tile(np.linspace(215, 120, height, dtype=np.uint8)[:, None], (1, width))
    sky = cv2.merge((gradient, gradient, np.clip(gradient.astype(np.int16) + 8, 0, 255).astype(np.uint8)))
    cv2.rectangle(sky, (85, 105), (205, 170), (28, 42, 55), -1)
    dark = np.full((height, width, 3), (14, 17, 22), dtype=np.uint8)
    cv2.circle(dark, (140, 90), 52, (38, 48, 65), -1)

    sky_result = trim_physical_border(sky, max_crop_ratio=0.2)
    dark_result = trim_physical_border(dark, max_crop_ratio=0.2)

    assert sky_result.offsets["top"] == 0
    assert sky_result.offsets["left"] == 0
    assert dark_result.applied is False


def test_border_trim_protected_region_vetoes_one_edge() -> None:
    from app.pipeline.border_removal import trim_physical_border

    image = np.full((140, 220, 3), 245, dtype=np.uint8)
    image[18:125, 24:202] = (50, 90, 145)
    result = trim_physical_border(
        image,
        max_crop_ratio=0.2,
        protected_regions=[{"type": "text", "box": {"x1": 0.01, "y1": 0.3, "x2": 0.14, "y2": 0.6}}],
    )

    assert result.offsets["left"] == 0
    assert "BORDER_PROTECTED_CONTENT_VETO" in result.edges["left"].reason_codes


def test_typed_crop_result_preserves_untrimmed_rectified_pixels() -> None:
    from app.pipeline.perspective import correct_photo_crop

    page = np.full((180, 260, 3), 238, dtype=np.uint8)
    page[18:164, 27:241] = (55, 105, 155)
    cv2.line(page, (45, 38), (220, 142), (220, 225, 230), 4)

    result = correct_photo_crop(
        page,
        {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        trim_border=True,
        auto_orient=False,
        protect_regions=False,
    )

    assert result.metadata["border_trim"]["applied"] is True
    assert result.metadata["perspective_confidence"] >= 0.72
    assert result.rectified_untrimmed.shape[0] > result.image.shape[0]
    assert result.rectified_untrimmed.shape[1] > result.image.shape[1]
    assert result.metadata["untrimmed_shape"]["height"] == result.rectified_untrimmed.shape[0]
