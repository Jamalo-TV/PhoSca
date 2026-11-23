import cv2
import numpy as np
import pytest

from app.evaluation.cropping import evaluate_rendered_crop
from app.pipeline.geometry import normalized_vertices_to_pixels
from app.pipeline.perspective import correct_photo_crop


def _wide_print(tilted: bool, asymmetric: bool = False, detail: str = ""):
    side = 480
    rim = (229, 237, 245)
    photo = np.full((side, side, 3), rim, np.uint8)
    x1, y1, x2, y2 = (29, 36, 390, 391) if asymmetric else (81, 92, 376, 394)
    yy, xx = np.indices((y2 - y1 + 1, x2 - x1 + 1), dtype=np.float32)
    tone = np.uint8(52 + .13 * xx + .08 * yy + 18 * np.sin(xx / 17) * np.cos(yy / 29))
    photo[y1:y2 + 1, x1:x2 + 1] = np.stack((tone + 4, tone + 25, tone + 43), axis=2)
    if detail == "rule":
        cv2.rectangle(photo, (x1 - 14, y1 - 14), (x2 + 14, y2 + 14), (50, 67, 95), 7)
    elif detail == "caption":
        cv2.putText(photo, "Berlin", (102, 444), cv2.FONT_HERSHEY_SIMPLEX, .5, (224, 231, 239), 1)
    elif detail == "open_sky":
        photo[y2 + 1:, x1 + 20:x2] = (87, 103, 139)

    frame = np.float32([[0, 0], [479, 0], [479, 479], [0, 479]])
    outer = np.float32([[147, 87], [670, 145], [603, 581], [192, 530]]) if tilted else frame + [117, 91]
    matrix = cv2.getPerspectiveTransform(frame, outer.astype(np.float32))
    shape = (792, 664)
    page = np.full((shape[1], shape[0], 3), (65, 87, 113), np.uint8)
    print_mask = cv2.warpPerspective(np.ones(photo.shape[:2], np.uint8), matrix, shape, flags=cv2.INTER_NEAREST)
    photo_on_page = cv2.warpPerspective(photo, matrix, shape, borderMode=cv2.BORDER_REPLICATE)
    page[print_mask != 0] = photo_on_page[print_mask != 0]
    motif = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    binary = np.zeros(photo.shape[:2], np.uint8)
    binary[y1:y2 + 1, x1:x2 + 1] = 1
    truth = cv2.warpPerspective(binary, matrix, shape, flags=cv2.INTER_NEAREST)
    corners = cv2.perspectiveTransform(motif[None], matrix)[0]
    vertices = [{"x": float(x) / (shape[0] - 1), "y": float(y) / (shape[1] - 1)} for x, y in outer]
    return page, {"source": "manual_quad", "warp_quad": vertices}, truth, corners, (y2 - y1 + 1, x2 - x1 + 1)


@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_wide_rim_suggestion_is_precise_and_requires_an_explicit_motif(tilted, asymmetric):
    page, mask, truth, corners, shape = _wide_print(tilted, asymmetric)
    options = dict(dual_mode=True, auto_orient=False, protect_regions=False, print_aspect_ratio=1.0)

    proposed = correct_photo_crop(page, {}, mask, **options)

    modes = proposed.metadata["crop_modes"]
    assert modes["full_print"]["status"] == "ready"
    image_only = modes["image_only"]
    assert proposed.image_only is None and image_only["status"] == "review_needed"
    assert image_only["reason_codes"] == ["WIDE_MOTIF_CONFIRMATION_REQUIRED"] and "warp_quad" not in image_only
    vertices = image_only["suggested_warp_quad"]
    quad = [(point["x"], point["y"]) for point in vertices]
    assert np.linalg.norm(normalized_vertices_to_pixels(quad, *page.shape[1::-1]) - corners, axis=1).max() < .5

    accepted = correct_photo_crop(page, {}, {**mask, "image_quad": vertices}, **options)
    mode = accepted.metadata["crop_modes"]["image_only"]
    assert mode["status"] == "ready"
    assert np.array_equal(accepted.image, proposed.image)
    metrics = evaluate_rendered_crop(truth, quad, np.array(mode["page_to_output"]), accepted.image_only.shape[:2],
                                     expected_corners=corners, expected_shape=shape, sampling=mode["sampling"])
    assert metrics["foreign_beyond_1px"] == metrics["cutoff_beyond_1px"] == 0
    assert metrics["aspect_ratio_relative_error"] < .01
    if not tilted:
        assert metrics["foreign_source_samples"] == 0


@pytest.mark.parametrize("detail", ["rule", "open_sky"])
def test_wide_rim_fallback_rejects_unenclosed_or_decorated_content(detail):
    page, mask, *_ = _wide_print(False, detail=detail)

    crop = correct_photo_crop(page, {}, mask, dual_mode=True, auto_orient=False, protect_regions=False)

    mode = crop.metadata["crop_modes"]["image_only"]
    assert crop.image_only is None and "suggested_warp_quad" not in mode


def test_wide_rim_cannot_suggest_removing_a_protected_caption(monkeypatch):
    from app.pipeline import perspective

    page, mask, *_ = _wide_print(False, detail="caption")
    monkeypatch.setattr(perspective, "detect_protected_regions", lambda *_: [])
    provenance = {"type": "text", "box": {"x1": 211, "y1": 520, "x2": 290, "y2": 545}}

    crop = correct_photo_crop(page, {}, mask, dual_mode=True, auto_orient=False, protected_regions=[provenance])

    mode = crop.metadata["crop_modes"]["image_only"]
    assert crop.image_only is None and "suggested_warp_quad" not in mode
    assert "BORDER_PROTECTED_CONTENT_VETO" in mode["reason_codes"]
