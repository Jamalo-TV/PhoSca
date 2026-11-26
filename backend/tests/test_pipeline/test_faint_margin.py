import cv2
import numpy as np
import pytest

from app.pipeline.decisions import MOTIF_MARGIN_MARKS_REVIEW
from app.pipeline.faint_margin import _source_strokes_supported, faint_margin_region
from app.pipeline.motif_margin import held_print_motif_margin_marks, motif_margin_marks
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


def test_held_print_caption_requires_visible_paper_and_an_unclaimed_native_mark() -> None:
    full = np.full((1440, 1120, 3), 245, np.uint8)
    rows = np.arange(108, 1349)
    fraction = (rows - rows[0]) / (rows[-1] - rows[0])
    bow = np.rint(20 * fraction * (1 - fraction))
    left = np.column_stack((65 + bow, rows))
    right = np.column_stack((1054 - bow, rows))[::-1]
    cv2.fillPoly(full, [np.concatenate((left, right)).astype(np.int32)], (48, 70, 85))
    blank = full.copy()
    cv2.putText(full, "CONTACT NYC 1954", (250, 73), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (48, 48, 48), 3, cv2.LINE_AA)
    paper = np.float32([[0, 0], [1119, 0], [1119, 1439], [0, 1439]])
    matrix = np.eye(3)

    mark, = held_print_motif_margin_marks(full, full, matrix, paper, paper, [])
    np.testing.assert_allclose(mark, [[251, 46], [605, 46], [605, 76], [251, 76]], atol=1.)
    assert not held_print_motif_margin_marks(blank, full, matrix, paper, paper, [])
    peer = np.float32([[560, 35], [665, 35], [665, 90], [560, 90]])
    assert not held_print_motif_margin_marks(full, full, matrix, paper, paper, [peer])
    visibility = paper.copy()
    visibility[:2, 1] = 43
    assert not held_print_motif_margin_marks(full, full, matrix, paper, visibility, [])


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


@pytest.mark.parametrize("case", ["same_rim", "opposite_rim", "pips", "ruling"])
def test_visible_heading_does_not_hide_an_independent_faint_date(case) -> None:
    page = _white_print()
    cv2.putText(page, "AUGUST 1954", (86, 49), cv2.FONT_HERSHEY_SIMPLEX, .26, (62, 61, 60), 1, cv2.LINE_AA)
    expected = [[114, 46.5]]
    if case in {"same_rim", "opposite_rim"}:
        glyphs = np.zeros((13, 85), np.uint8)
        cv2.putText(glyphs, "SEP 1954", (2, 9), cv2.FONT_HERSHEY_SIMPLEX, .26, 255, 1, cv2.LINE_AA)
        if case == "opposite_rim":
            glyphs = np.rot90(glyphs, 2)
            position = np.s_[179:192, 151:236]
            expected.append([213, 185])
        else:
            position = np.s_[40:53, 171:256]
            expected.append([193, 46])
        page[position] -= np.rint(glyphs[..., None] * (38 / 255)).astype(np.uint8)
    elif case == "pips":
        for index in range(6):
            cv2.ellipse(page, (170 + 9 * index, 46 - index % 2), (2, 1), 0, 0, 360, (223, 224, 222), -1, cv2.LINE_AA)
    else:
        cv2.line(page, (168, 43), (217, 44), (227, 223, 229), 1, cv2.LINE_AA)

    polygon = [{"x": x / 339, "y": y / 229} for x, y in [(42, 38), (287, 37), (286, 194), (43, 193)]]
    crop = correct_photo_crop(page, {}, {"polygon": polygon}, dual_mode=True, auto_orient=False)
    full, motif = (crop.metadata["crop_modes"][name] for name in ("full_print", "image_only"))
    assert full["status"] == "ready" and "margin_mark_regions" not in full
    assert motif["reason_codes"] == [MOTIF_MARGIN_MARKS_REVIEW]
    assert crop.image_only is None and crop.image_only_preview is None
    centers = [np.mean([[point["x"] * 339, point["y"] * 229] for point in region], axis=0)
               for region in motif["margin_mark_regions"]]
    np.testing.assert_allclose(centers, expected, atol=1.5)


@pytest.mark.parametrize("missing", [None, "dash", "first_glyph", "second_glyph", "visible_gap"])
def test_dashed_motif_note_requires_each_part_on_the_source_paper(missing) -> None:
    height, width = 90, 250
    page = np.full((height, width, 3), 245, np.uint8)
    ink = (74, 74, 74)
    for text, x in [("269617", 16), ("2", 122), ("8", 140)]:
        cv2.putText(page, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, .68, ink, 2, cv2.LINE_AA)
    cv2.rectangle(page, (104, 23), (110, 25), ink, -1)

    source = page.copy()
    if missing in {"dash", "visible_gap"}:
        source[22:27, 103:112] = 245
    elif missing == "first_glyph":
        source[12:34, 121:139] = 245
    elif missing == "second_glyph":
        source[12:34, 139:157] = 245
    # Keep the proposal unchanged when dropping only the native witness.
    proposal = source if missing == "visible_gap" else page
    lab = cv2.cvtColor(proposal[:37], cv2.COLOR_BGR2LAB)
    owned = np.ones(lab.shape[:2], dtype=bool)
    motif = np.float32([[3, 40], [245, 40], [245, 80], [3, 80]])
    args = (source, lab, owned, lambda points: points.astype(np.float32), np.eye(3), (width, height), motif)

    prefix = faint_margin_region(*args)
    assert prefix is not None and prefix[:, 0].max() < 100
    extended = faint_margin_region(*args, extend_dashed_note=True)
    if missing is None:
        assert extended is not None and extended[:, 0].max() >= 154
    else:
        np.testing.assert_array_equal(extended, prefix)


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
