from __future__ import annotations

import numpy as np
from PIL import Image


def test_transparent_pixels_are_composited_over_white_not_hidden_rgb(tmp_path) -> None:
    from app.pipeline.image_ops import load_image_with_metadata

    rgba = np.zeros((8, 10, 4), dtype=np.uint8)
    rgba[:, :, :3] = (255, 0, 0)
    rgba[:, :, 3] = 0
    rgba[2:6, 3:8] = (20, 40, 200, 255)
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba, "RGBA").save(path)

    bgr, metadata = load_image_with_metadata(path)

    assert np.all(bgr[0, 0] == (255, 255, 255))
    assert np.all(bgr[3, 4] == (200, 40, 20))
    assert metadata["alpha_policy"] == "composite_over_white_before_working_space"


def test_uint16_grayscale_is_scaled_explicitly_to_bgr8(tmp_path) -> None:
    from app.pipeline.image_ops import load_image_with_metadata

    values = np.array([[0, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / "depth16.png"
    Image.fromarray(values, mode="I;16").save(path)

    bgr, metadata = load_image_with_metadata(path)

    assert bgr.dtype == np.uint8
    assert tuple(int(value) for value in bgr[0, 0]) == (0, 0, 0)
    assert 126 <= int(bgr[0, 1, 0]) <= 129
    assert tuple(int(value) for value in bgr[0, 2]) == (255, 255, 255)
    assert metadata["bit_depth_policy"] == "scale_integer_range_to_8_bit"


def test_icc_conversion_preserves_alpha_until_srgb_compositing(tmp_path) -> None:
    from app.pipeline.image_ops import load_image_with_metadata
    from PIL import ImageCms

    path = tmp_path / "profiled-alpha.png"
    image = Image.new("RGBA", (3, 2), (255, 0, 0, 0))
    image.putpixel((1, 0), (255, 0, 0, 255))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(path, icc_profile=profile)

    decoded, metadata = load_image_with_metadata(path)

    assert decoded[0, 0].tolist() == [255, 255, 255]
    assert decoded[0, 1].tolist() == [0, 0, 255]
    assert metadata["icc_to_srgb_applied"] is True
    assert metadata["alpha_policy"] == "composite_over_white_before_working_space"
