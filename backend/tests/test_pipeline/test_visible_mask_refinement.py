import hashlib
import json

import cv2
import numpy as np

from app.config import Settings
from app.pipeline import segmentation, visible_mask_refinement


def _rect(left, top, right, bottom, scale):
    return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]]) / scale


def test_visible_mask_resolution_leaves_the_tie_between_neighboring_prints_unowned():
    shape = (100, 151, 3)
    scale = np.float32([150, 99])
    original = [_rect(20, 25, 69, 80, scale), _rect(81, 25, 130, 80, scale)]
    candidates = [_rect(20, 25, 88, 80, scale), _rect(62, 25, 130, 80, scale)]

    first, second = visible_mask_refinement.resolve_mask_ownership(shape, original, candidates)

    assert np.rint(first[:, 0].max() * scale[0]) == 74
    assert np.rint(second[:, 0].min() * scale[0]) == 76
    assert np.rint(first[:, 0].min() * scale[0]) == 20
    assert np.rint(second[:, 0].max() * scale[0]) == 130


def test_optional_refiner_uses_source_pixels_and_preserves_initial_crop_geometry(monkeypatch, tmp_path):
    model = tmp_path / "visible.onnx"
    model.write_bytes(b"mask-refinement-fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {model.name: {"sha256": digest, "license": "test"}}}))
    settings = Settings(
        DATABASE_URL="sqlite+aiosqlite://", REDIS_URL="redis://localhost:6379/0",
        VISIBLE_MASK_MODEL_PATH=model, MODEL_MANIFEST_PATH=manifest, ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=False,
    )
    box = {"x1": .3, "y1": .3, "x2": .7, "y2": .7}
    polygon = [{"x": x, "y": y} for x, y in _rect(.3, .3, .7, .7, 1)]
    seed = segmentation.SegmentationDetection(box, {"polygon": polygon}, .78, 1.2, True, [])
    initial = segmentation.SegmentationResult([seed], {})
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: initial)

    class Session:
        def run(self, outputs, feeds):
            assert outputs == ["mask_logits"]
            tensor = feeds["image_and_mask"]
            assert tensor.shape == (1, 4, 320, 320)
            assert np.allclose(tensor[0, :3, 0, 0], [65 / 255, 108 / 255, 195 / 255])
            expanded = cv2.dilate(tensor[0, 3], np.ones((7, 7), np.uint8))
            return [expanded[None, None] - .5]

    monkeypatch.setattr(visible_mask_refinement, "_load_session", lambda *_: Session())
    original = np.full((138, 160, 3), [65, 108, 195], dtype=np.uint8)
    prepared = np.zeros_like(original)
    result = segmentation.segment_album_page(prepared, settings, source_image=original)

    assert "error" not in result.metadata["visible_mask_refinement"]
    assert result.metadata["visible_mask_refinement"] == {"model_sha256": digest, "refined_detections": 1}
    refined, = result.detections
    assert refined.bounding_box == box
    assert refined.mask["polygon"] == polygon
    assert min(point["x"] for point in refined.mask["visible_polygon"]) < box["x1"]
    assert "visible_polygon" not in seed.mask

    model.write_bytes(b"different-model")
    unavailable = segmentation.segment_album_page(prepared, settings, source_image=original)

    assert unavailable.detections == initial.detections
    assert "hash mismatch" in unavailable.metadata["visible_mask_refinement"]["error"]
