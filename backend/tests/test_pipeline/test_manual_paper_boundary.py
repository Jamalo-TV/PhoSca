import cv2
import numpy as np
import pytest

from app.pipeline.manual_paper_boundary import check_manual_paper_boundary, paper_curve_review_reference


def test_an_unrelated_small_curve_does_not_become_a_required_correction():
    quad = [{"x": x, "y": y} for x, y in ((.2, .2), (.8, .2), (.8, .8), (.2, .8))]
    correction = {
        "crop_modes": {"full_print": {"warp_quad": quad, "reason_codes": ["PRINT_BOUNDARY_CURVED"]}},
        "boundary_refinement": {"top": {"measured_bow_px": 1.4}, "bottom": {"measured_bow_px": .6}},
    }
    assert paper_curve_review_reference(correction)["sides"] == [0]


@pytest.mark.parametrize(("blur", "quality"), [(0, 100), (.65, 85), (.95, 75)])
def test_corrected_straight_paper_is_distinct_from_a_continuing_album_edge(blur, quality):
    height, width = 1050, 1300
    y, x = np.indices((height, width), dtype=np.float32)
    quad = np.float32([[225, 165], [1079, 165], [1079, 935], [225, 935]])
    along = np.clip((y - 165) / 770, 0, 1)
    print_rows = (x <= 1079) & (y >= 165) & (y <= 935)
    figure = print_rows & (x >= 225)
    density = 109 + 33 * np.cos(x / 75) * np.cos(y / 40) + 8 * np.sin(x / 11 + y / 23)
    ink = np.uint8(np.dstack((density, density + 19, density + 28)))

    # The prior detector can follow the curved ink instead of the faint paper.
    weak_paper = np.full_like(ink, 224)
    weak_paper[figure] = ink[figure]
    weak_paper[figure & (x < 232 - 6 * along * (1 - along))] = 219
    estimated = quad.copy()
    estimated[[0, 3], 0] += 6

    bowed = np.full_like(ink, 224)
    curved = print_rows & (x >= 225 - 16 * along * (1 - along))
    bowed[curved] = ink[curved]
    bowed[:, :216] = (115, 119, 122)
    album_edge = quad.copy()
    album_edge[[0, 3], 0] = 216
    mat = np.full_like(ink, 224)
    outer_mat = quad + np.float32([[-10, -10], [10, -10], [10, 10], [-10, 10]])
    mat[155:946, 215:1090] = 185
    mat[curved] = ink[curved]
    pale_bow = np.full_like(ink, 224)
    pale_mask = print_rows & (x >= 225 - 12 * along * (1 - along))
    pale_bow[pale_mask] = ink[pale_mask]
    pale_bow[pale_mask & (x <= 239)] = 200
    pale_chord = quad.copy()
    pale_chord[[0, 3], 0] -= 2.5

    def source(image):
        if blur:
            image = cv2.GaussianBlur(image, (0, 0), blur)
        if quality < 100:
            payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        return image

    def reference(previous):
        saved = [{"x": float(a / (width - 1)), "y": float(b / (height - 1))} for a, b in previous]
        return {"outline": saved, "warp_quad": saved, "legacy_extended": False,
                "curve": {"sides": [3], "traces": [], "warp_quad": saved}}

    weak_paper = source(weak_paper)
    assert check_manual_paper_boundary(weak_paper, estimated, reference(estimated))[0]
    assert not check_manual_paper_boundary(weak_paper, quad, reference(estimated))[0]
    assert check_manual_paper_boundary(source(bowed), album_edge, reference(quad))[0]
    assert check_manual_paper_boundary(source(mat), outer_mat, reference(quad))[0]
    assert check_manual_paper_boundary(source(pale_bow), pale_chord, reference(quad))[0]
