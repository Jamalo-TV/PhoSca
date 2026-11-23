import cv2
import numpy as np
import pytest

from app.pipeline.geometry import normalized_vertices_to_pixels
from app.pipeline.perspective import correct_photo_crop, validate_review_quads


def _vertices(points, image):
    height, width = image.shape[:2]
    return [{"x": float(x / (width - 1)), "y": float(y / (height - 1))} for x, y in points]


def _box(x1, y1, x2, y2, image):
    height, width = image.shape[:2]
    return {"x1": x1 / width, "y1": y1 / height, "x2": x2 / width, "y2": y2 / height}


def _rect(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def _crop(image, box, mask=None, **kwargs):
    return correct_photo_crop(image, box, mask, auto_orient=False, protect_regions=False, **kwargs)


def test_half_open_box_and_explicit_outer_quad_have_exact_pixel_centers() -> None:
    page = np.random.default_rng(830).integers(0, 256, (100, 150, 3), dtype=np.uint8)
    box = _box(10, 11, 90, 76, page)
    expected = page[11:76, 10:90]
    box_result = _crop(page, box, trim_border=False)
    quad_result = _crop(page, box, {
        "polygon": _vertices(_rect(12, 12, 68, 63), page),
        "warp_quad": _vertices(_rect(10, 11, 89, 75), page),
    }, trim_border=False)
    assert np.array_equal(box_result.image, expected)
    assert np.array_equal(quad_result.image, expected)


def test_manual_motif_crop_is_independent_and_rejects_corners_outside_print() -> None:
    page = np.random.default_rng(830).integers(0, 256, (120, 180, 3), dtype=np.uint8)
    outer = _rect(10, 11, 149, 109)
    inner = _rect(23, 21, 134, 97)
    mask = {"warp_quad": _vertices(outer, page), "image_quad": _vertices(inner, page), "source": "manual_quad"}
    result = _crop(page, {}, mask, dual_mode=True, trim_border=False)
    modes = result.metadata["crop_modes"]
    assert modes["full_print"]["status"] == modes["image_only"]["status"] == "ready"
    assert np.array_equal(result.image, page[11:110, 10:150])
    assert np.array_equal(result.image_only, page[21:98, 23:135])
    projected = cv2.perspectiveTransform(inner.reshape(1, 4, 2), np.array(modes["image_only"]["page_to_output"]))
    assert np.allclose(projected[0], _rect(0, 0, 111, 76), atol=1e-5)

    mask["image_quad"] = _vertices(_rect(4, 9, 135, 96), page)
    with pytest.raises(ValueError, match="image_quad_outside_print"):
        validate_review_quads(mask, page.shape[1], page.shape[0])


def test_bilinear_crop_at_stretched_edge_does_not_read_foreign_pixels() -> None:
    from app.pipeline.crop_sampling import warp_owned_crop

    # At the top left, all four taps in the bilinear cell lie outside the quad.
    source = np.float32([[4.48, 4.96], [24.3, 8.2], [22.8, 28.9], [9.4, 22.3]])
    assert all(cv2.pointPolygonTest(source, tap, False) < 0 for tap in ((4, 4), (5, 4), (4, 5), (5, 5)))
    page = np.random.default_rng(50).integers(12, 207, (34, 32, 3), dtype=np.uint8)
    foreign = np.array([[cv2.pointPolygonTest(source, (x, y), False) < 0 for x in range(32)] for y in range(34)])
    alternative = page.copy()
    alternative[foreign] = 255
    size = (187, 174)
    transform = cv2.getPerspectiveTransform(source, _rect(0, 0, size[0] - 1, size[1] - 1))
    plain = cv2.warpPerspective(page, transform, size)
    assert np.any(plain[3] != cv2.warpPerspective(alternative, transform, size)[3])

    cropped, supported = warp_owned_crop(page, transform, size)
    changed_background, other_supported = warp_owned_crop(alternative, transform, size)
    assert supported and other_supported
    assert np.array_equal(cropped, changed_background)
    assert np.array_equal(cropped[48:-48, 48:-48], plain[48:-48, 48:-48])
    eligible = np.argwhere(~foreign)[:, ::-1]
    nearest = eligible[np.argmin(((eligible - source[0]) ** 2).sum(axis=1))]
    assert np.array_equal(cropped[0, 0], page[nearest[1], nearest[0]])


def test_edge_sampler_reports_when_no_nearby_source_value_exists() -> None:
    from app.pipeline.crop_sampling import warp_owned_crop

    source = np.float32([[10.2, 10.4], [70.2, 12.4], [72.7, 24.6], [50.2, 18.4]])
    page = np.full((40, 80), 100, dtype=np.uint8)
    transform = cv2.getPerspectiveTransform(source, _rect(0, 0, 79, 35))
    image, supported = warp_owned_crop(page, transform, (80, 36))
    assert image.shape == (36, 80)
    assert not supported


def test_crop_modes_exclude_actual_foreign_color_at_their_own_boundary() -> None:
    from app.evaluation.crop_fixtures import _projected_corners
    from app.evaluation.cropping import evaluate_rendered_crop
    from app.pipeline.crop_sampling import OWNED_BILINEAR

    outer = _projected_corners(80, 60, (54, 50), (10, -8, 17))
    projection = cv2.getPerspectiveTransform(_rect(0, 0, 79, 59), outer)
    inner = cv2.perspectiveTransform(_rect(10, 10, 69, 49)[None], projection)[0]
    height, width = 100, 108
    own = [np.array([[cv2.pointPolygonTest(quad, (x, y), False) >= 0 for x in range(width)] for y in range(height)])
           for quad in (outer, inner)]
    page = np.full((height, width, 3), (8, 22, 181), np.uint8)
    page[own[0]] = (249, 249, 247)
    page[own[1]] = (29, 122, 70)
    mask = {"warp_quad": _vertices(outer, page), "image_quad": _vertices(inner, page), "source": "manual_quad"}
    crop = _crop(page, {}, mask, dual_mode=True, trim_border=False, print_aspect_ratio=79 / 59)
    for mode, image, truth in zip(("full_print", "image_only"), (crop.image, crop.image_only), own, strict=True):
        state = crop.metadata["crop_modes"][mode]
        assert state["status"] == "ready" and state["sampling"] == OWNED_BILINEAR
        quad = [(point["x"], point["y"]) for point in state["warp_quad"]]
        matrix = np.array(state["page_to_output"])
        baseline = evaluate_rendered_crop(truth, quad, matrix, image.shape[:2])
        sampled = evaluate_rendered_crop(truth, quad, matrix, image.shape[:2], sampling=state["sampling"])
        assert baseline["foreign_fraction"] > .005
        assert baseline["foreign_source_samples"] > 0
        assert sampled["foreign_fraction"] == sampled["foreign_source_samples"] == 0
        assert sampled["footprint_iou"] == baseline["footprint_iou"] == 1
    assert np.all(crop.image_only == (29, 122, 70))


def test_native_white_on_white_boundary_preserves_the_original_print_rim() -> None:
    page = np.full((230, 340, 3), 236, dtype=np.uint8)
    page[40:192, 45:285] = 249
    yy, xx = np.indices((128, 216))
    page[52:180, 57:273] = np.stack([85 + xx % 40, 45 + yy % 46, 126 + (xx + yy) % 49], axis=2)
    seed = np.array([[42, 38], [287, 37], [286, 194], [43, 193]], dtype=np.float32)
    result = _crop(page, {}, {"polygon": _vertices(seed, page)}, dual_mode=True)

    modes = result.metadata["crop_modes"]
    assert modes["full_print"]["status"] == modes["image_only"]["status"] == "ready"
    measured = normalized_vertices_to_pixels(
        [(p["x"], p["y"]) for p in modes["full_print"]["warp_quad"]], page.shape[1], page.shape[0],
    )
    assert np.max(np.linalg.norm(measured - _rect(45, 40, 284, 191), axis=1)) < 0.6
    assert result.image.shape == (152, 240, 3)
    assert result.image_only.shape == (128, 216, 3)
    assert np.all(result.image[1, 1] == 249)
    motif = normalized_vertices_to_pixels(
        [(p["x"], p["y"]) for p in modes["image_only"]["warp_quad"]], page.shape[1], page.shape[0],
    )
    assert np.max(np.linalg.norm(motif - _rect(57, 52, 272, 179), axis=1)) < 0.3
    assert np.mean(np.abs(result.image_only[1:-1, 1:-1].astype(int) - page[53:179, 58:272])) < 0.15


def test_outward_retry_requires_a_distinct_print_rim() -> None:
    page = np.full((310, 440, 3), 135, np.uint8)
    page[50:230, 55:365] = (216, 225, 231)
    page[78:202, 83:337] = (91, 117, 173)
    seed = _rect(69, 64, 350, 215)
    result = _crop(page, {}, {"polygon": _vertices(seed, page)}, dual_mode=True, trim_border=False)
    assert result.metadata["crop_modes"]["full_print"]["status"] == "ready"
    assert np.array_equal(result.image, page[50:230, 55:365])
    assert all(side["outward_search"]["print_rim_supported"] for side in result.metadata["boundary_refinement"].values())

    # A dark nested rectangle has the same geometry and cannot prove a print rim.
    page[50:230, 55:365] = (73, 71, 82)
    page[78:202, 83:337] = (91, 117, 173)
    rejected = _crop(page, {}, {"polygon": _vertices(seed, page)}, dual_mode=True, trim_border=False)
    assert rejected.metadata["crop_modes"]["full_print"]["status"] == "review_needed"


def test_a_supported_motif_edge_cannot_hide_a_coherent_outer_rim() -> None:
    page = np.full((280, 420, 3), 149, np.uint8)
    page[40:240, 45:365] = 244
    page[52:228, 57:353] = (105, 121, 157)
    inner_seed = _rect(57, 52, 352, 227)
    crop = _crop(page, {}, {"polygon": _vertices(inner_seed, page)}, dual_mode=True, trim_border=False)
    assert crop.metadata["crop_modes"]["full_print"]["status"] == "ready"
    assert np.array_equal(crop.image, page[40:240, 45:365])

    # The same strong transitions can enclose only the dark part of a pale scene.
    page[52:228, 57:148] = 247
    partial_seed = _rect(148, 52, 352, 227)
    partial = _crop(page, {}, {"polygon": _vertices(partial_seed, page)}, dual_mode=True, trim_border=False)
    assert "PRINT_OUTER_RIM_UNCERTAIN" in partial.metadata["crop_modes"]["full_print"]["reason_codes"]


def test_uncertain_uniform_motif_keeps_full_print_available() -> None:
    page = np.full((150, 220, 3), 244, np.uint8)
    mask = {"warp_quad": _vertices(_rect(20, 25, 190, 130), page), "source": "manual_quad"}
    result = _crop(page, {}, mask, dual_mode=True)
    assert result.metadata["crop_modes"]["full_print"]["status"] == "ready"
    assert result.metadata["crop_modes"]["image_only"]["reason_codes"] == ["IMAGE_BOUNDARY_UNCERTAIN"]
    assert result.image_only is None


@pytest.mark.parametrize("ranked_contours_available", [True, False])
def test_local_outer_contour_recovers_white_image_content_outside_the_model_seed(
    monkeypatch, ranked_contours_available: bool,
) -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages
    from app.pipeline import segmentation

    if not ranked_contours_available:
        monkeypatch.setattr(segmentation, "detect_photos_classical", lambda *args, **kwargs: segmentation.SegmentationResult([], {}))

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "white_motif")
    reference = fixture.prints[0].reference_modes
    seed = reference["image_only"].corners.copy()
    seed[[0, 3], 0] += 0.33 * (seed[1, 0] - seed[0, 0])
    mask = {"polygon": _vertices(seed, fixture.image), "source": "yolo_seg_mask"}
    crop = _crop(fixture.image, {}, mask, dual_mode=True, enable_classical_retry=True, camera_matrix=fixture.camera_matrix)
    assert crop.metadata["boundary_refinement"]["seed"] == "local_outer_contour"
    height, width = fixture.image.shape[:2]
    for name, mode in crop.metadata["crop_modes"].items():
        assert mode["status"] == "ready"
        corners = normalized_vertices_to_pixels([(p["x"], p["y"]) for p in mode["warp_quad"]], width, height)
        assert np.max(np.linalg.norm(corners - reference[name].corners, axis=1)) < 1


def test_cast_shadow_recovery_uses_print_material_to_choose_the_outer_edge() -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "cast_shadow")
    seed = np.array([[254, 224], [559, 204], [575, 393], [277, 427]], dtype=np.float32)
    mask = {"polygon": _vertices(seed, fixture.image), "source": "yolo_seg_mask"}
    result = _crop(fixture.image, {}, mask, dual_mode=True, camera_matrix=fixture.camera_matrix)
    assert result.metadata["boundary_refinement"]["bottom"]["matched_print_rim"]
    for name, mode in result.metadata["crop_modes"].items():
        assert mode["status"] == "ready"
        measured = normalized_vertices_to_pixels(
            [(point["x"], point["y"]) for point in mode["warp_quad"]], fixture.image.shape[1], fixture.image.shape[0]
        )
        expected = fixture.prints[0].reference_modes[name].corners
        assert np.max(np.linalg.norm(measured - expected, axis=1)) < 1


@pytest.mark.parametrize("case", ["touching", "touching_same_color"])
def test_inferred_rim_does_not_snap_to_a_neighboring_prints_ink(case: str) -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == case)
    neighbor = {"polygon": _vertices(_rect(97, 210, 405, 414), fixture.image)}
    seed = np.array([[425, 236], [690, 236], [688, 419], [426, 419]], dtype=np.float32)
    mask = {"polygon": _vertices(seed, fixture.image), "source": "yolo_seg_mask"}
    result = _crop(fixture.image, {}, mask, dual_mode=True, camera_matrix=fixture.camera_matrix, neighbor_masks=[neighbor])
    modes = result.metadata["crop_modes"]
    if case == "touching_same_color":
        assert "PRINT_POSSIBLY_OCCLUDED" in modes["full_print"]["reason_codes"]
        assert result.image_only is None
    else:
        assert modes["full_print"]["status"] == modes["image_only"]["status"] == "ready"


def test_only_the_fully_visible_print_can_be_recovered_from_an_overlap() -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "overlap")
    proposals = [
        np.float32([[97, 210], [369, 211], [345, 413], [98, 415]]),
        np.float32([[373, 229], [637, 229], [617, 422], [356, 404]]),
    ]
    masks = [{"polygon": _vertices(points, fixture.image), "source": "yolo_seg_mask"} for points in proposals]
    results = [
        _crop(fixture.image, {}, mask, dual_mode=True, camera_matrix=fixture.camera_matrix, neighbor_masks=[masks[1-index]])
        for index, mask in enumerate(masks)
    ]
    rear, front = results
    assert rear.metadata["crop_modes"]["full_print"]["status"] == "review_needed"
    assert rear.image_only is None
    for name, mode in front.metadata["crop_modes"].items():
        assert mode["status"] == "ready"
        measured = normalized_vertices_to_pixels(
            [(point["x"], point["y"]) for point in mode["warp_quad"]], fixture.image.shape[1], fixture.image.shape[0]
        )
        assert np.max(np.linalg.norm(measured - fixture.prints[1].reference_modes[name].corners, axis=1)) < 1


def test_neighbor_mask_spill_on_verified_stock_preserves_both_crop_modes() -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "touching")
    page = fixture.image
    mask = {
        "source": "yolo_seg_mask",
        "polygon": _vertices(_rect(96, 211, 403, 414), page),
        "visible_polygon": _vertices(_rect(85, 200, 404, 425), page),
    }
    neighbor = {"polygon": _vertices(_rect(405, 250, 560, 424), page)}
    baseline = _crop(page, {}, mask, dual_mode=True, camera_matrix=fixture.camera_matrix)
    crop = _crop(page, {}, mask, dual_mode=True, camera_matrix=fixture.camera_matrix, neighbor_masks=[neighbor])

    assert crop.metadata["boundary_refinement"]["right"]["observed_rim_depth_px"] > 10
    assert {name: mode["status"] for name, mode in crop.metadata["crop_modes"].items()} == {
        "full_print": "ready", "image_only": "ready",
    }
    assert np.array_equal(crop.image, baseline.image)
    assert crop.image_only is not None and np.array_equal(crop.image_only, baseline.image_only)


@pytest.mark.parametrize("cover", [None, "edge", "stock_break", "same_stock_outside"])
def test_neighbor_claim_requires_continuous_source_stock_and_an_exterior_transition(cover) -> None:
    from app.pipeline.perspective import _neighbor_occlusion

    page = np.full((300, 420, 3), 65, dtype=np.uint8)
    page[40:261, 45:376] = 235
    if cover == "edge":
        page[94:206, 372:382] = (130, 80, 165)
    elif cover == "stock_break":
        page[80:220, 371:373] = (130, 80, 165)
    elif cover == "same_stock_outside":
        page[75:225, 376:390] = 235

    stock = cv2.cvtColor(page[48:49, 60:61], cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    assert _neighbor_occlusion(
        page, _rect(45, 40, 375, 260), _rect(45, 40, 363, 260), [_rect(364, 80, 400, 220)],
        observed_sides={1: (stock, 12.)},
    ) is (cover is not None)


def test_enclosed_neighbor_keeps_outer_print_in_review_even_with_filled_model_mask() -> None:
    page = np.full((340, 490, 3), 75, np.uint8)
    page[40:301, 50:431] = (100, 140, 173)
    page[96:242, 170:284] = (169, 65, 100)
    outer = _vertices(_rect(50, 40, 430, 300), page)
    front = _vertices(_rect(170, 96, 283, 241), page)
    outer_mask = {"source": "yolo_seg_mask", "polygon": outer, "warp_quad": outer}
    front_mask = {"source": "yolo_seg_mask", "polygon": front, "warp_quad": front, "image_quad": front}

    rear_crop = _crop(page, {}, outer_mask, dual_mode=True, trim_border=False, neighbor_masks=[front_mask])
    assert rear_crop.metadata["crop_modes"]["full_print"]["reason_codes"] == ["PRINT_POSSIBLY_OCCLUDED"]
    assert rear_crop.image_only is None
    visible = _crop(page, {}, front_mask, dual_mode=True, trim_border=False, neighbor_masks=[outer_mask])
    assert visible.metadata["crop_modes"]["full_print"]["status"] == "ready"
    assert np.array_equal(visible.image_only, page[96:242, 170:284])


def test_both_mode_transforms_include_the_same_applied_rotation(monkeypatch) -> None:
    from app.pipeline import perspective

    page = np.random.default_rng(830).integers(0, 256, (120, 180, 3), dtype=np.uint8)
    outer = _rect(10, 11, 149, 109)
    inner = _rect(23, 21, 134, 97)
    mask = {"warp_quad": _vertices(outer, page), "image_quad": _vertices(inner, page), "source": "manual_quad"}
    monkeypatch.setattr(perspective, "correct_photo_orientation", lambda image, **_: (
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), {"rotation_degrees": 90},
    ))
    result = correct_photo_crop(page, {}, mask, protect_regions=False, dual_mode=True)
    for name, image, quad, expected in (
        ("full_print", result.image, outer, page[11:110, 10:150]),
        ("image_only", result.image_only, inner, page[21:98, 23:135]),
    ):
        mode = result.metadata["crop_modes"][name]
        assert np.array_equal(image, cv2.rotate(expected, cv2.ROTATE_90_CLOCKWISE))
        assert tuple(mode["shape"][key] for key in ("height", "width")) == image.shape[:2]
        mapped = cv2.perspectiveTransform(quad.reshape(1, 4, 2), np.array(mode["page_to_output"]))[0]
        assert np.allclose(mapped[0], [image.shape[1] - 1, 0], atol=1e-5)


@pytest.mark.parametrize("case", ["paper_corner_mount", "photo_corners", "touching_same_color"])
def test_independent_motif_crop_requires_an_unbroken_rim_and_no_neighbor_occlusion(case) -> None:
    from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages
    from app.evaluation.cropping import evaluate_rendered_crop

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == case)
    proposals = [geometry_seed(item, fixture.image.shape) for item in fixture.prints]
    for index, (box, mask) in enumerate(proposals):
        neighbors = [other for other_index, (_, other) in enumerate(proposals) if other_index != index]
        result = _crop(fixture.image, box, mask, dual_mode=True, enable_classical_retry=True, neighbor_masks=neighbors)
        modes = result.metadata["crop_modes"]
        assert modes["full_print"]["status"] == "review_needed"
        if case != "paper_corner_mount":
            assert modes["image_only"]["status"] == "review_needed"
            assert result.image_only is None
            continue

        mode = modes["image_only"]
        reference = fixture.prints[index].reference_modes["image_only"]
        assert mode["status"] == "ready" and mode["independent_of_full_print"] is True
        quad = [(point["x"], point["y"]) for point in mode["warp_quad"]]
        metrics = evaluate_rendered_crop(reference.mask, quad, np.array(mode["page_to_output"]), result.image_only.shape[:2],
                                         expected_corners=reference.corners, expected_shape=reference.shape,
                                         sampling=mode["sampling"])
        assert metrics["foreign_source_samples"] == metrics["cutoff_beyond_1px"] == 0
        assert metrics["corner_error_max_px"] < 1


def test_album_color_at_a_missing_corner_does_not_establish_a_holder() -> None:
    from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "paper_corner_mount")
    image = fixture.image.copy()
    image[np.all(image == (45, 53, 64), axis=2)] = 146
    box, mask = geometry_seed(fixture.prints[0], image.shape)
    result = _crop(image, box, mask, dual_mode=True, enable_classical_retry=True)
    assert result.metadata["crop_modes"]["full_print"]["status"] == "review_needed"
    assert result.metadata["crop_modes"]["image_only"]["status"] == "review_needed"
    assert result.image_only is None


def test_learned_visibility_keeps_source_verified_rim_on_a_mounted_print() -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "paper_corner_mount")
    page = fixture.image
    primary = np.float32([[273, 215], [571, 215], [569, 408], [274, 409]])
    learned = np.float32([
        [252, 197], [267, 200], [263, 210], [283, 199], [588, 197], [590, 215],
        [581, 218], [587, 229], [587, 392], [582, 407], [590, 412], [591, 428],
        [277, 425], [272, 429], [252, 429], [252, 409], [262, 409], [254, 398],
        [254, 230], [257, 222], [252, 217],
    ])
    mask = {"source": "yolo_seg_mask", "polygon": _vertices(primary, page)}
    baseline = _crop(page, {}, mask, dual_mode=True, enable_classical_retry=True)
    refined = _crop(page, {}, {**mask, "visible_polygon": _vertices(learned, page)},
                    dual_mode=True, enable_classical_retry=True)

    assert refined.metadata["refined_visibility_source"] == "visible_paper_rim"
    assert refined.metadata["refined_visible_polygon"] == baseline.metadata["refined_visible_polygon"]
    assert refined.metadata["crop_modes"]["full_print"]["status"] == "review_needed"
    assert refined.image_only is not None and np.array_equal(refined.image_only, baseline.image_only)


def test_camera_calibration_undoes_yaw_without_trusting_an_underconstrained_shape() -> None:
    width, height = 960, 780
    page = np.full((height, width, 3), 110, np.uint8)
    camera = np.array([[900, 0, width / 2], [0, 900, height / 2], [0, 0, 1]], dtype=float)
    plane = np.array([[-1.5, -1, 0], [1.5, -1, 0], [1.5, 1, 0], [-1.5, 1, 0]], dtype=float)
    image_points, _ = cv2.projectPoints(plane, np.array([0, 0.65, 0]), np.array([0.0, 0, 7]), camera, None)
    mask = {"warp_quad": _vertices(image_points.reshape(-1, 2), page), "source": "manual_quad"}

    known = _crop(page, {}, mask, dual_mode=True, camera_matrix=camera, trim_border=False)
    unknown = _crop(page, {}, mask, dual_mode=True, trim_border=False)
    assert known.metadata["crop_modes"]["full_print"]["status"] == "ready"
    assert known.image.shape[1] / known.image.shape[0] == pytest.approx(1.5, rel=0.005)
    assert unknown.metadata["crop_modes"]["full_print"]["reason_codes"] == ["PRINT_ASPECT_UNCERTAIN"]


@pytest.mark.parametrize("rotation", [[11., 0., 0.], [0., 7., 0.]])
def test_mild_single_axis_tilt_requires_metric_scale(rotation) -> None:
    from app.evaluation.crop_fixtures import _projected_corners

    page = np.full((640, 840, 3), 110, np.uint8)
    corners = _projected_corners(332, 228, (420.2, 312.7), tuple(rotation))
    mask = {"warp_quad": _vertices(corners, page), "source": "manual_quad"}
    aspect = 331 / 227

    unknown = _crop(page, {}, mask, dual_mode=True, trim_border=False)
    assert unknown.metadata["crop_modes"]["full_print"]["reason_codes"] == ["PRINT_ASPECT_UNCERTAIN"]
    assert abs((unknown.image.shape[1] - 1) / (unknown.image.shape[0] - 1) / aspect - 1) > .01

    measured = _crop(page, {}, mask, dual_mode=True, trim_border=False, print_aspect_ratio=aspect)
    assert measured.metadata["crop_modes"]["full_print"]["status"] == "ready"
    assert (measured.image.shape[1] - 1) / (measured.image.shape[0] - 1) == pytest.approx(aspect, rel=.004)


def test_clipped_or_occluded_proposals_remain_review_previews() -> None:
    page = np.random.default_rng(830).integers(0, 256, (140, 210, 3), dtype=np.uint8)
    mask = {
        "warp_quad": _vertices(_rect(0, 10, 175, 110), page),
        "visible_polygon": _vertices([(0, 10), (130, 10), (130, 45), (175, 45), (175, 110), (0, 110)], page),
        "source": "yolo_seg_mask",
    }
    neighbor = {"polygon": _vertices(_rect(132, 2, 190, 43), page)}
    result = _crop(page, {}, mask, dual_mode=True, neighbor_masks=[neighbor])
    assert set(result.metadata["crop_modes"]["full_print"]["reason_codes"]) == {"PRINT_AT_CAPTURE_EDGE", "PRINT_POSSIBLY_OCCLUDED"}
    assert result.image.size and result.image_only is None


def test_missing_corner_material_keeps_a_rectangular_warp_in_review() -> None:
    page = np.full((240, 320, 3), 91, np.uint8)
    page[30:200, 50:260] = 237
    cv2.fillConvexPoly(page, np.int32([[50, 30], [59, 30], [50, 39]]), (91, 91, 91))
    mask = {"warp_quad": _vertices(_rect(50, 30, 259, 199), page), "source": "manual_quad"}
    result = _crop(page, {}, mask, dual_mode=True, trim_border=False)
    assert result.metadata["crop_modes"]["full_print"]["reason_codes"] == ["PRINT_CORNERS_UNCERTAIN"]
