from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
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


def test_camera_jpeg_gain_map_does_not_count_as_a_second_page() -> None:
    from app.pipeline.image_ops import load_image_with_metadata
    from app.utils.file_security import inspect_image_content

    path = Path("data/golden_fixtures/page_04.jpg")
    inspected = inspect_image_content(path.read_bytes(), max_image_pixels=5_000_000, max_image_frames=1)
    decoded, metadata = load_image_with_metadata(path)

    assert inspected["frame_policy"] == metadata["frame_policy"]
    assert decoded.shape == (1920, 2560, 3)


def test_crop_calibration_requires_original_bytes_and_decoded_dimensions(tmp_path) -> None:
    from app.pipeline.camera_calibration import load_crop_source
    from app.pipeline.image_ops import sha256_file

    path = tmp_path / "capture.jpg"
    photo = Image.new("RGB", (80, 100))
    exif = photo.getexif()
    exif[274] = 6
    photo.save(path, exif=exif)
    digest = sha256_file(path)
    artifact = tmp_path / "calibration.json"
    record = {"width": 100, "height": 80, "camera_matrix": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
              "calibration_id": "test"}
    artifact.write_text(json.dumps({"schema_version": 1, "captures": {digest: record}}), encoding="utf-8")

    image, calibration = load_crop_source(path, digest, artifact)
    assert image.shape == (80, 100, 3)
    assert calibration is not None
    assert load_crop_source(path, "1" * 64, artifact)[1] is None

    record.update(width=80, height=100)
    artifact.write_text(json.dumps({"schema_version": 1, "captures": {digest: record}}, indent=2), encoding="utf-8")
    assert load_crop_source(path, digest, artifact)[1] is None
    record["camera_matrix"][0][0] = "100"
    artifact.write_text(json.dumps({"schema_version": 1, "captures": {digest: record}}), encoding="utf-8")
    assert load_crop_source(path, digest, artifact)[1] is None

    record.update(width=100, height=80)
    record["camera_matrix"][0][0] = 100
    artifact.write_text(json.dumps({"schema_version": 1, "captures": {digest: record}}) + "\n", encoding="utf-8")
    photo.save(path)
    with pytest.raises(ValueError, match="source hash"):
        load_crop_source(path, digest, artifact)


def test_animated_png_is_rejected_by_upload_and_decoder(tmp_path) -> None:
    from app.pipeline.image_ops import load_image_with_metadata
    from app.utils.file_security import inspect_image_content

    buffer = BytesIO()
    Image.new("RGB", (16, 16), "red").save(
        buffer, format="PNG", save_all=True, append_images=[Image.new("RGB", (16, 16), "blue")]
    )
    with pytest.raises(HTTPException, match="multi-frame images are not supported"):
        inspect_image_content(buffer.getvalue(), max_image_pixels=1024, max_image_frames=1)
    path = tmp_path / "animated.png"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="Could not read image"):
        load_image_with_metadata(path)
