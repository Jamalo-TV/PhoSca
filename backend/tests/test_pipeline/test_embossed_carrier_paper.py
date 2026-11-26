import cv2
import numpy as np
import pytest

from app.pipeline.embossed_album_pockets import (
    POSSIBLE_EMBOSSED_ALBUM_PAPER_EDGES, embossed_carrier_paper_traces,
)
from app.pipeline.perspective import correct_photo_crop


_LOOSE_OUTLINE = np.float32([[211, 300], [1530, 300], [1529, 1346], [207, 1343]])
_BOUNDARY = {
    name: {"support": support} for name, support in zip(
        ("top", "right", "bottom", "left"), (.96, .09, .16, .22), strict=True,
    )
}


def _album_with_print(surface: str) -> np.ndarray:
    y, x = np.ogrid[:1690, :1760]
    vertical = np.cos(2 * np.pi * y / 23)
    embossing = 14 * np.cos(2 * np.pi * x / 15) * vertical + 6 * (2 * vertical**2 - 1)
    light = np.uint8(np.rint(232 + embossing))
    page = cv2.cvtColor(np.full_like(light, 232) if surface == "plain" else light, cv2.COLOR_GRAY2BGR)
    photo = page[300:1300, 253:1483]
    noise = np.random.default_rng(823).normal(0, 9, (*photo.shape[:2], 1))
    photo[:] = np.uint8(np.clip(noise + [58, 72, 128], 0, 255))
    cv2.ellipse(photo, (530, 480), (250, 315), 25, 0, 360, (37, 93, 144), -1)
    cv2.rectangle(photo, (130, 190), (260, 540), (191, 179, 155), -1)
    if surface == "continuous_film":
        photo[:] = np.uint8(np.rint(np.clip(photo + embossing[300:1300, 253:1483, None], 0, 255)))
    return page


@pytest.mark.parametrize("surface", ["plain", "continuous_film"])
def test_album_paper_traces_need_a_measured_break_in_carrier_phase(surface):
    assert embossed_carrier_paper_traces(_album_with_print(surface), _LOOSE_OUTLINE, _BOUNDARY) == []


def test_embossed_carrier_traces_explain_held_crop_without_changing_either_mode(monkeypatch):
    from app.pipeline import perspective

    image = _album_with_print("embossed")
    polygon = [{"x": x / 1759, "y": y / 1689} for x, y in _LOOSE_OUTLINE.tolist()]
    mask = {"source": "yolo_seg_mask", "detector_source": "yolo_onnx", "polygon": polygon,
            "boundary_evidence": {"score": .78}}
    options = {"dual_mode": True, "enable_classical_retry": True, "auto_orient": False, "protect_regions": False}
    with monkeypatch.context() as baseline:
        baseline.setattr(perspective, "embossed_carrier_paper_traces", lambda *_: [])
        prior = correct_photo_crop(image, {}, mask, **options)
    result = correct_photo_crop(image, {}, mask, **options)
    modes = result.metadata["crop_modes"]
    full = modes["full_print"].copy()
    assert full.pop("suggested_edge_source") == POSSIBLE_EMBOSSED_ALBUM_PAPER_EDGES
    right, bottom, left = full.pop("suggested_edge_traces")
    np.testing.assert_allclose([point["x"] * 1759 for point in right], 1482, atol=1.)
    np.testing.assert_allclose([point["y"] * 1689 for point in bottom], 1299, atol=1.)
    np.testing.assert_allclose([point["x"] * 1759 for point in left], 253, atol=1.)
    assert full == prior.metadata["crop_modes"]["full_print"]
    assert modes["image_only"] == prior.metadata["crop_modes"]["image_only"]
    assert all(mode["status"] == "review_needed" for mode in modes.values())
    assert result.full_print_preview is None and result.image_only is None
    np.testing.assert_array_equal(result.image, prior.image)
