import cv2
import numpy as np
import pytest

from app.pipeline.manual_paper_boundary import _competing_inner_stock, _confirmed_capture_outline


def _rect(left, top, right, bottom):
    return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])


def _source_frame(case):
    y, x = np.indices((950, 1300), dtype=np.float32)
    page = np.float32([198, 205, 212]) + (2 * np.sin(x / 65 + y / 120) + np.cos(x / 21 - y / 43))[..., None]
    outer = _rect(220, 166, 999, 801)
    ink = _rect(251, 193, 970, 773)
    if case in {"pale_mat", "lighter_mat"}:
        page[139:829, 190:1031] = [226, 231, 235] if case == "pale_mat" else [238, 232, 226]
    else:
        page[:, 1087:1094] = [135, 146, 161]
        page[:, 1200:] = [182, 190, 200]
    page[166:802, 220:1000] = [210, 211, 218] if case == "lighter_mat" else [250, 251, 252]
    tone = 85 + 36 * np.cos(x / 46) * np.cos(y / 21) + 7 * np.sin(x / 24 + y / 33)
    page[193:774, 251:971] = np.dstack((tone, tone + 14, tone + 30))[193:774, 251:971]
    previous = outer.copy()
    previous[[1, 2], 0] = 1299
    selected = outer
    if case == "ink":
        selected = ink
    elif case == "rail":
        selected = _rect(220, 166, 1086, 801)
    elif case in {"pale_mat", "lighter_mat"}:
        selected = _rect(190, 139, 1030, 828)
        previous = _rect(190, 139, 1299, 828)
    return np.rint(page).astype(np.uint8), selected, previous


@pytest.mark.parametrize("capture", ["raw", "jpeg", "perspective"])
@pytest.mark.parametrize("case", ["paper", "ink", "rail", "pale_mat", "lighter_mat"])
def test_capture_correction_distinguishes_print_stock_from_ink_rails_and_mounts(capture, case):
    page, selected, previous = _source_frame(case)
    if capture == "perspective":
        # A small projective change preserves the original left registration.
        transform = cv2.getPerspectiveTransform(
            _rect(0, 0, 1299, 949),
            np.float32([[17, 8], [1272, 19], [1285, 930], [19, 938]]),
        )
        page = cv2.warpPerspective(page, transform, page.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
        selected, previous = cv2.perspectiveTransform(np.float32([selected, previous]), transform)
    if capture != "raw":
        page = cv2.GaussianBlur(page, (0, 0), .65)
        encoded = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        page = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if case in {"pale_mat", "lighter_mat", "paper"}:
        assert _competing_inner_stock(page, selected, 65) is (case != "paper")
    assert _confirmed_capture_outline(page, selected, previous) is (case == "paper")


@pytest.mark.parametrize("material", ["thin_rail", "light_drift", "subtle_photo"])
def test_old_capture_area_preserves_review_for_unexplained_photo_content(material):
    page = np.full((560, 720, 3), 231, dtype=np.uint8)
    page[93:475, 103:605] = 202
    page[123:445, 133:575] = [120, 151, 193]
    candidate = _rect(103, 93, 604, 474)
    previous = _rect(103, 93, 719, 474)
    if material == "thin_rail":
        page[92:475, 668:676] = 105
    elif material == "light_drift":
        page[:, 641:] = np.rint(np.linspace(231, 206, 79)).astype(np.uint8)[None, :, None]
    else:
        rows, cols = np.mgrid[110:461, 652:720]
        tones = 231 - 13 - 11 * np.sin(cols * .02) * np.cos(rows * .043)
        page[110:461, 652:] = tones.astype(np.uint8)[..., None]
    assert _confirmed_capture_outline(page, candidate, previous) is (material != "subtle_photo")
