import hashlib
import json

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.pipeline import segmentation, visible_mask_refinement
from app.pipeline.crop_validation import crop_validation_profile, segmentation_matches_crop_profile


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


@pytest.mark.parametrize("inner", [(40, 30, 90, 70), (10, 10, 140, 90)], ids=["interior-hole", "fully-disputed"])
def test_unrepresentable_visibility_preserves_proposals_for_review(monkeypatch, tmp_path, inner):
    model = tmp_path / "visible.onnx"
    model.write_bytes(b"mask-refinement-fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {model.name: {"sha256": digest, "license": "test"}}}))
    settings = Settings(
        DATABASE_URL="sqlite+aiosqlite://", REDIS_URL="redis://localhost:6379/0",
        VISIBLE_MASK_MODEL_PATH=model, MODEL_MANIFEST_PATH=manifest, ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=False,
    )
    scale = np.float32([150, 99])
    outlines = [_rect(10, 10, 140, 90, scale), _rect(*inner, scale)]
    detections = [
        segmentation.SegmentationDetection(
            {"x1": float(outline[0, 0]), "y1": float(outline[0, 1]),
             "x2": float(outline[2, 0]), "y2": float(outline[2, 1])},
            {"polygon": [{"x": float(x), "y": float(y)} for x, y in outline]}, .90, 1.2, True, [],
        )
        for outline in outlines
    ]
    monkeypatch.setattr(segmentation, "_segment_album_page", lambda *_: segmentation.SegmentationResult(detections, {}))
    monkeypatch.setattr(visible_mask_refinement, "_load_session", lambda *_: object())
    monkeypatch.setattr(visible_mask_refinement, "_local_outline", lambda _, __, polygon: polygon)

    result = segmentation.segment_album_page(np.zeros((100, 151, 3), np.uint8), settings)

    assert result.detections == detections
    assert result.metadata["visible_mask_refinement"] == {"ownership_unresolved": True}
    profile = crop_validation_profile(settings)
    assert not segmentation_matches_crop_profile(settings, {**result.metadata, "validation_profile": profile})


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


@pytest.mark.parametrize("scene", ["sleeve_fold", "print_surface_glare", "pale_print_border", "pale_border_under_fold"])
def test_only_reflections_continuing_onto_the_album_can_shorten_learned_masks(scene):
    image = np.full((240, 380, 3), 160, dtype=np.uint8)
    image[32:177, 74:302] = (76, 110, 150)
    if scene == "sleeve_fold":
        image[189:192] = 247
    elif scene == "print_surface_glare":
        image[177:192, 74:302] = (76, 110, 150)
        image[189:192, 74:302] = 247
    else:
        image[177:192, 74:302] = 230
        image[192:, :] = 222
        if scene == "pale_border_under_fold":
            image[189:192] = 247
    scale = np.float32([379, 239])
    primary = _rect(74, 32, 301, 176, scale)
    learned = _rect(74, 32, 301, 191, scale)
    occupied = [primary * scale, learned * scale]

    refined = visible_mask_refinement._remove_sleeve_fold(image, primary, learned, occupied)

    assert min(refined[:, 1]) == pytest.approx(min(primary[:, 1]))
    expected = primary if scene == "sleeve_fold" else learned
    assert max(refined[:, 1]) == pytest.approx(max(expected[:, 1]))
