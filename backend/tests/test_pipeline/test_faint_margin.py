import cv2
import numpy as np
import pytest

from app.pipeline.decisions import MOTIF_MARGIN_MARKS_REVIEW
from app.pipeline.faint_margin import _source_strokes_supported
from app.pipeline.motif_margin import motif_margin_marks
from app.pipeline.perspective import correct_photo_crop


def _white_print() -> np.ndarray:
    page = np.full((230, 340, 3), 236, np.uint8)
    page[40:192, 45:285] = 249
    y, x = np.indices((128, 216))
    page[52:180, 57:273] = np.stack([85 + x % 40, 45 + y % 46, 126 + (x + y) % 49], axis=2)
    return page


def _margin_regions(page: np.ndarray) -> list[np.ndarray]:
    motif = np.float32([[12, 12], [227, 12], [227, 139], [12, 139]])
    inverse = np.float32([[1, 0, 45], [0, 1, 40], [0, 0, 1]])
    return motif_margin_marks(page, page[40:192, 45:285], motif, inverse)


def test_joined_faint_date_in_the_readme_is_held_with_source_coordinates() -> None:
    from scripts.render_crop_examples import faint_caption

    page, _, image, state = faint_caption()
    assert image is None and state == "Motiv in Prüfung"
    seed = [(84, 76), (574, 74), (572, 388), (86, 386)]
    mask = {"polygon": [{"x": x / 679, "y": y / 459} for x, y in seed]}
    crop = correct_photo_crop(page, {}, mask, dual_mode=True, auto_orient=False, enable_classical_retry=True)
    full, motif = (crop.metadata["crop_modes"][mode] for mode in ("full_print", "image_only"))
    assert full["status"] == "ready" and motif["reason_codes"] == [MOTIF_MARGIN_MARKS_REVIEW]
    region, = motif["margin_mark_regions"]
    points = [(vertex["x"] * 679, vertex["y"] * 459) for vertex in region]
    assert np.allclose(points, [[162, 86], [243, 86], [243, 99], [162, 99]], atol=1.)


@pytest.mark.parametrize("contrast", [(38, 38, 38), (7, 54, 42)], ids=["graphite", "blue"])
def test_faint_date_is_held_and_located_on_the_original(contrast) -> None:
    page = _white_print()
    glyphs = np.zeros((12, 100), np.uint8)
    cv2.putText(glyphs, "SEP 1954", (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.26, 255, 1, cv2.LINE_AA)
    page[41:53, 79:179] -= np.rint(glyphs[..., None] / 255 * np.asarray(contrast)).astype(np.uint8)
    region, = _margin_regions(page)
    np.testing.assert_allclose(region, [[81, 43], [121, 43], [121, 49], [81, 49]], atol=1)

    polygon = [{"x": x / 339, "y": y / 229} for x, y in [(42, 38), (287, 37), (286, 194), (43, 193)]]
    crop = correct_photo_crop(page, {}, {"polygon": polygon}, dual_mode=True, auto_orient=False)
    full, motif = (crop.metadata["crop_modes"][mode] for mode in ("full_print", "image_only"))
    assert full["status"] == "ready"
    assert MOTIF_MARGIN_MARKS_REVIEW in motif["reason_codes"]
    assert len(motif["margin_mark_regions"]) == 1
    assert crop.image_only is None and crop.image_only_preview is None


@pytest.mark.parametrize("control", ["separate_pips", "blue_ruling", "paper_tone_step"])
def test_faint_non_text_patterns_do_not_mark_the_paper(control) -> None:
    page = _white_print()
    if control == "separate_pips":
        for i, center in enumerate([(80, 46), (88, 47), (97, 46), (107, 45), (117, 46), (128, 47)]):
            cv2.ellipse(page, center, (2, 1), 18 * (i % 2), 0, 360, (220, 219, 220), -1, cv2.LINE_AA)
    elif control == "blue_ruling":
        for i, tint in enumerate([(246, 223, 231), (243, 218, 223), (244, 225, 232)]):
            cv2.line(page, (75 + 37 * i, 42 + i), (104 + 37 * i, 42 + i), tint, 1, cv2.LINE_AA)
    else:
        page[40:52, 100:285] = (241, 216, 226)
    assert not _margin_regions(page)


@pytest.mark.parametrize(("ridge_y", "motif_top"), [(8, 9), (2, 17)])
@pytest.mark.parametrize("reverse_winding", [False, True])
def test_native_evidence_requires_paper_on_both_sides_of_a_stroke(ridge_y, motif_top, reverse_winding) -> None:
    page = np.full((73, 91, 3), 249, np.uint8)
    page[17 + ridge_y, 19:70] = 206
    native = np.float32([(x, 17 + ridge_y) for x in range(27, 58)])
    motif = np.float32([[3, motif_top], [56, motif_top], [56, 35], [3, 35]])
    forward = np.float32([[1, 0, -15], [0, 1, -17], [0, 0, 1]])
    contrast, stock = (np.full(len(native), level) for level in (35, 246))
    if reverse_winding:
        motif = motif[::-1]
    assert not _source_strokes_supported(page, forward, (60, 40), motif, native, contrast, stock, False)
    # Move only the crossed boundary while keeping the original pixels fixed.
    if ridge_y == 8:
        motif[motif[:, 1] == motif_top, 1] += 9
    else:
        forward[1, 2] += 5
    assert _source_strokes_supported(page, forward, (60, 40), motif, native, contrast, stock, False)
