import cv2
import numpy as np
import pytest

from app.pipeline.printed_illustration_review import find_illustration_frame


def _ribbon_path(*, inward_corners: bool) -> np.ndarray:
    left, top, right, bottom = 340, 277, 1110, 1325
    radius = 86
    if inward_corners:
        corners = (
            (right, top, 180, 90),
            (right, bottom, 270, 180),
            (left, bottom, 0, -90),
            (left, top, 90, 0),
        )
    else:
        corners = (
            (right - radius, top + radius, 270, 360),
            (right - radius, bottom - radius, 0, 90),
            (left + radius, bottom - radius, 90, 180),
            (left + radius, top + radius, 180, 270),
        )
    arcs = []
    for center_x, center_y, start, end in corners:
        angles = np.deg2rad(np.linspace(start, end, 90))
        arcs.append([center_x, center_y] + radius * np.column_stack((np.cos(angles), np.sin(angles))))
    return np.rint(np.concatenate(arcs)).astype(np.int32)


def _illustrated_cover(
    *, inward_corners: bool = True, ribbon_lightness: int = 243, occluded: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    lab = np.empty((1600, 1382, 3), np.uint8)
    lab[:] = (79, 128, 128)
    lab[125:1475, 123:1255] = (193, 131, 145)
    ribbon = _ribbon_path(inward_corners=inward_corners)

    # Leave warm board on both sides of the pale line, with colored drawing
    # shapes farther inside. The proposed fragments do not define the ribbon.
    plate = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillPoly(plate, [ribbon], 1)
    interior = cv2.erode(plate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39)))
    lab[interior != 0] = (222, 128, 131)
    cv2.ellipse(lab, (684, 790), (145, 300), -23, 0, 360, (113, 149, 150), -1)
    cv2.ellipse(lab, (862, 713), (74, 216), 23, 0, 360, (160, 122, 164), -1)
    cv2.polylines(lab, [ribbon], True, (ribbon_lightness, 126, 128), 7, cv2.LINE_AA)
    if occluded:
        lab[236:318, 668:837] = (157, 127, 140)

    bounds = ((408, 375, 999, 907), (420, 464, 1030, 1160), (429, 523, 1007, 1200), (426, 673, 1004, 1240))
    proposals = [np.float32(((left, top), (right, top), (right, bottom), (left, bottom)))
                 for left, top, right, bottom in bounds]
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), proposals, ribbon.astype(np.float32)


def _assert_follows_ribbon(outline: np.ndarray | None, ribbon: np.ndarray) -> None:
    assert outline is not None
    assert max(abs(cv2.pointPolygonTest(ribbon, tuple(point), True)) for point in outline) < 3.5
    ribbon_distances = [abs(cv2.pointPolygonTest(outline, tuple(point), True)) for point in ribbon]
    assert np.quantile(ribbon_distances, .95) < 4.


def test_illustration_frame_follows_the_native_inward_ribbon_after_rotation():
    source, proposals, ribbon = _illustrated_cover()
    _assert_follows_ribbon(find_illustration_frame(source, proposals), ribbon)

    def rotate_points(points: np.ndarray) -> np.ndarray:
        return np.column_stack((source.shape[0] - 1 - points[:, 1], points[:, 0]))

    rotated = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
    result = find_illustration_frame(rotated, [rotate_points(polygon) for polygon in proposals])
    _assert_follows_ribbon(result, rotate_points(ribbon))


@pytest.mark.parametrize(
    "cover_options",
    [{"ribbon_lightness": 205}, {"inward_corners": False}, {"occluded": True}],
    ids=["weak-ribbon", "convex-corners", "occluded-ribbon"],
)
def test_illustration_frame_abstains_without_a_supported_inward_ribbon(cover_options):
    source, proposals, _ = _illustrated_cover(**cover_options)
    assert find_illustration_frame(source, proposals) is None
