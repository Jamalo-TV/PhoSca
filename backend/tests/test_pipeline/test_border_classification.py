from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("padding", [0, 3])
def test_axis_aligned_print_preserves_its_exact_pixels_and_requested_padding(padding: int) -> None:
    from app.pipeline.border_removal import trim_physical_border

    image = np.full((180, 260, 3), 238, dtype=np.uint8)
    image[18:164, 27:241] = (55, 105, 155)
    cv2.line(image, (45, 38), (220, 142), (220, 225, 230), 4)

    result = trim_physical_border(image, max_crop_ratio=0.2, padding=padding)

    assert result.applied is True
    left, top, right, bottom = 27 - padding, 18 - padding, 240 + padding, 163 + padding
    np.testing.assert_array_equal(result.image, image[top : bottom + 1, left : right + 1])
    corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
    np.testing.assert_allclose(result.crop_quad, corners, atol=0.05)
    assert result.confidence >= 0.68
    metadata = result.metadata()
    assert metadata["method"] == "fitted_print_edges_v3"
    np.testing.assert_allclose(metadata["crop_quad"], result.crop_quad)


@pytest.mark.parametrize(
    ("quarter_turns", "start_depth", "end_depth", "patterned"),
    [(0, 16, 0, True), (1, 2, 2, False), (2, 7, 0, False), (3, 15, 1, False), (2, 12, 25, True)],
)
def test_tapered_and_thin_print_edges_keep_nearby_picture_detail(
    quarter_turns: int, start_depth: int, end_depth: int, patterned: bool
) -> None:
    from app.pipeline.border_removal import trim_physical_border

    height, width = 340, 540
    grain = np.random.default_rng(7).normal(115, 23, (height, width)).clip(55, 175).astype(np.uint8)
    image = cv2.merge((grain - 15, grain, grain + 16))
    rows, cols = np.indices((height, width))
    depths = np.rint(np.linspace(start_depth, end_depth, width)).astype(int)
    paper = rows < depths
    image[paper] = (211, 230, 240)
    if patterned:
        ink = (rows < depths - 4) & ((rows + cols // 3) % 8 < 3)
        image[ink] = (45, 55, 65)
    # Place a visible detail immediately below the shallow end of the rim.
    detail_x = width // 12 if start_depth < end_depth else 11 * width // 12
    detail = (np.abs(cols - detail_x) <= 4) & (rows >= depths + 2) & (rows < depths + 6)
    image[detail] = (19, 31, 244)
    image = np.rot90(image, quarter_turns).copy()
    edge = ("top", "left", "bottom", "right")[quarter_turns]

    result = trim_physical_border(image)

    assert result.applied
    observed_detail = (result.image[:, :, 2] > 195) & (result.image[:, :, 0] < 55)
    assert np.count_nonzero(observed_detail) >= 0.7 * np.count_nonzero(detail)
    assert not np.any(np.all(result.image == (211, 230, 240), axis=2))
    assert not np.any(np.all(result.image == (45, 55, 65), axis=2))
    assert all(pixels == 0 for name, pixels in result.offsets.items() if name != edge)
    expected = np.array([[0, start_depth], [width - 1, end_depth], [width - 1, height - 1], [0, height - 1]])
    for _ in range(quarter_turns):
        expected = np.roll(np.column_stack((expected[:, 1], width - 1 - expected[:, 0])), -1, axis=0)
        height, width = width, height
    np.testing.assert_allclose(result.crop_quad, expected, atol=1.25)


def test_rectified_corners_follow_all_four_inner_print_edges() -> None:
    from app.pipeline.border_removal import trim_physical_border

    height, width = 320, 480
    expected = np.array([[11, 18], [466, 2], [477, 306], [2, 318]], dtype=np.int32)
    scene = np.random.default_rng(25).integers(65, 160, size=(height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, expected, 1)
    scene[mask == 0] = 240

    result = trim_physical_border(cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR))

    assert result.applied
    np.testing.assert_allclose(result.crop_quad, expected, atol=1.5)
    assert not np.any(result.image[:, :, 0] == 240)


def test_neighboring_page_pixels_recover_a_print_edge_crossing_the_initial_crop() -> None:
    from app.pipeline.border_removal import find_physical_border

    grain = np.random.default_rng(15).integers(80, 150, size=(236, 416), dtype=np.uint8)
    x = np.arange(grain.shape[1]) - 8
    top = np.rint(12 + 15 - 21 * x / 399)
    grain[np.arange(len(grain))[:, None] < top] = 230

    plan = find_physical_border(cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR), reference_box=(8, 12, 408, 232))

    assert plan.applied
    np.testing.assert_allclose(plan.crop_quad, [[0, 15], [399, -6], [399, 219], [0, 219]], atol=1)


@pytest.mark.parametrize("scale", [1, 4])
def test_soft_decorated_rim_tracks_the_picture_at_different_scan_sizes(scale: int) -> None:
    from app.pipeline.border_removal import find_physical_border

    y, x = np.indices((230, 512))
    depths = np.rint(np.linspace(12, 23, 512))
    texture = np.random.default_rng(923).normal(0, 4, x.shape)
    gray = np.rint(198 + 6 * np.cos(x / 36) + texture).clip(60, 215).astype(np.uint8)
    gray[y < depths] = 230
    gray[(y < depths - 4) & ((y + x // 5) % 11 < 2)] = 128
    gray[y == depths - 6] = 128
    gray = cv2.GaussianBlur(gray, (5, 5), 0.9)
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    plan = find_physical_border(cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR))

    assert plan.applied
    np.testing.assert_allclose(plan.crop_quad[:2, 1] / scale, [12, 23], atol=0.65)


def test_single_native_paper_row_on_large_scan_retains_exact_photo_pixels() -> None:
    from app.pipeline.border_removal import trim_physical_border

    gray = np.random.default_rng(14).integers(85, 160, size=(240, 2000), dtype=np.uint8)
    gray[0] = 235
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = trim_physical_border(image)

    assert result.offsets == {"top": 1, "bottom": 0, "left": 0, "right": 0}
    np.testing.assert_array_equal(result.image, image[1:])


def test_content_aware_border_preserves_quiet_sky_and_dark_scene_edges() -> None:
    from app.pipeline.border_removal import trim_physical_border

    height, width = 180, 280
    gradient = np.tile(np.linspace(215, 120, height, dtype=np.uint8)[:, None], (1, width))
    sky = cv2.merge((gradient, gradient, np.clip(gradient.astype(np.int16) + 8, 0, 255).astype(np.uint8)))
    cv2.rectangle(sky, (85, 105), (205, 170), (28, 42, 55), -1)
    dark = np.full((height, width, 3), (14, 17, 22), dtype=np.uint8)
    cv2.circle(dark, (140, 90), 52, (38, 48, 65), -1)

    sky_result = trim_physical_border(sky, max_crop_ratio=0.2)
    dark_result = trim_physical_border(dark, max_crop_ratio=0.2)

    assert sky_result.offsets["top"] == 0
    assert sky_result.offsets["left"] == 0
    assert dark_result.applied is False


def test_long_line_inside_textured_photo_does_not_remove_the_outer_scene() -> None:
    from app.pipeline.border_removal import trim_physical_border

    gray = np.random.default_rng(17).normal(140, 28, (220, 320)).clip(55, 215).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(image, (0, 8), (319, 17), (35, 35, 35), 2)

    result = trim_physical_border(image)

    assert result.applied is False
    np.testing.assert_array_equal(result.image, image)


def test_rule_printed_across_margin_does_not_become_the_cut_line() -> None:
    from app.pipeline.border_removal import trim_physical_border

    image = np.full((300, 500, 3), 125, dtype=np.uint8)
    image[:22] = 228
    image[7:14] = 30

    result = trim_physical_border(image)

    assert 22 <= result.offsets["top"] <= 24
    assert np.all(result.image == 125)


def test_warp_blended_paper_pixel_on_varied_scene_edge() -> None:
    from app.pipeline.border_removal import trim_physical_border

    tones = np.linspace(20, 154, 240, dtype=np.uint8)
    photo = cv2.cvtColor(np.tile(tones, (160, 1)), cv2.COLOR_GRAY2BGR)
    photo[:, :4] = 235
    photo[-1] = np.rint(photo[-1] * 0.80 + 235 * 0.20).astype(np.uint8)

    result = trim_physical_border(photo)

    assert result.offsets["bottom"] == 1
    np.testing.assert_array_equal(result.image, photo[:-1, 4:])
    assert result.edges["bottom"].reason_codes == ("RESAMPLED_PAPER_EDGE",)
    assert result.offsets["right"] == result.offsets["top"] == 0


def test_blue_sky_above_a_sloping_horizon_is_preserved() -> None:
    from app.pipeline.border_removal import trim_physical_border

    rng = np.random.default_rng(29)
    rows, columns = np.indices((340, 540))
    field = rng.normal(118, 22, rows.shape).clip(65, 178).astype(np.uint8)
    image = cv2.merge((field - 25, field + 5, field - 20))
    sky = rows < 12 + columns * 0.04
    shade = (7 * np.sin(columns / 47) + 4 * np.cos(rows / 13) + rng.normal(0, 2, rows.shape))
    image[sky] = np.array([235, 200, 125]) + shade[sky, None]

    result = trim_physical_border(image)

    np.testing.assert_array_equal(result.image, image)


def test_single_dark_scene_edge_is_preserved_while_dark_frame_is_removed() -> None:
    from app.pipeline.border_removal import trim_physical_border

    gray = np.random.default_rng(26).normal(115, 18, (220, 320)).clip(50, 180).astype(np.uint8)
    wall_scene = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    wall_scene[:, :18] = (18, 16, 16)
    framed = wall_scene.copy()
    framed[:12] = framed[-14:] = framed[:, -16:] = (18, 16, 16)

    np.testing.assert_array_equal(trim_physical_border(wall_scene).image, wall_scene)
    paper_on_top = wall_scene.copy()
    paper_on_top[:12] = (219, 229, 239)
    mixed = trim_physical_border(paper_on_top)
    assert mixed.offsets["top"] >= 12
    assert mixed.offsets["left"] == 0

    result = trim_physical_border(framed)
    assert not np.any(np.all(result.image == (18, 16, 16), axis=2))
    assert result.image.shape[0] >= 220 - 12 - 14 - 6
    assert result.image.shape[1] >= 320 - 18 - 16 - 6


@pytest.mark.parametrize(
    "box",
    [
        {"x1": 0.01, "y1": 0.3, "x2": 0.14, "y2": 0.6},
        {"x1": 2, "y1": 42, "x2": 31, "y2": 84},
    ],
)
def test_border_trim_protected_region_vetoes_one_edge(box: dict) -> None:
    from app.pipeline.border_removal import trim_physical_border

    image = np.full((140, 220, 3), 245, dtype=np.uint8)
    image[18:125, 24:202] = (50, 90, 145)
    result = trim_physical_border(
        image,
        max_crop_ratio=0.2,
        protected_regions=[{"type": "text", "box": box}],
    )

    assert result.offsets["left"] == 0
    assert "BORDER_PROTECTED_CONTENT_VETO" in result.edges["left"].reason_codes
    assert result.offsets["right"] >= 18


def test_sloped_trim_checks_protected_content_at_its_actual_position() -> None:
    from app.pipeline.border_removal import trim_physical_border

    grain = np.random.default_rng(5).integers(75, 160, size=(220, 400), dtype=np.uint8)
    image = cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR)
    depths = np.rint(np.linspace(19, 1, image.shape[1]))
    image[np.arange(image.shape[0])[:, None] < depths] = 228

    def trim_around(x: int):
        box = {"x1": x, "y1": 7, "x2": x + 20, "y2": 30}
        return trim_physical_border(image, protected_regions=[{"type": "text", "box": box}])

    clear = trim_around(350)
    assert clear.applied
    np.testing.assert_allclose(clear.crop_quad[:2, 1], [19, 1], atol=1.25)

    crossing = trim_around(20)
    assert crossing.edges["top"].reason_codes == ("BORDER_PROTECTED_CONTENT_VETO",)
    np.testing.assert_array_equal(crossing.image, image)


def test_typed_crop_result_preserves_untrimmed_rectified_pixels() -> None:
    from app.pipeline.perspective import correct_photo_crop

    page = np.full((180, 260, 3), 238, dtype=np.uint8)
    page[18:164, 27:241] = (55, 105, 155)
    cv2.line(page, (45, 38), (220, 142), (220, 225, 230), 4)

    result = correct_photo_crop(
        page,
        {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        trim_border=True,
        auto_orient=False,
        protect_regions=False,
    )

    assert result.metadata["border_trim"]["applied"] is True
    assert result.metadata["perspective_confidence"] >= 0.72
    assert result.rectified_untrimmed.shape[0] > result.image.shape[0]
    assert result.rectified_untrimmed.shape[1] > result.image.shape[1]
    assert result.metadata["untrimmed_shape"]["height"] == result.rectified_untrimmed.shape[0]
