import cv2
import numpy as np
import pytest

from app.pipeline.crop_sampling import OWNED_BILINEAR
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


def test_ambiguous_paper_rim_requires_clear_backing_beyond_the_edge() -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages
    from app.pipeline.segmentation_filter import CONTRAST_COMPONENT_SOURCE

    dense = next(page for page in synthetic_crop_pages("development") if page.name == "dense_gutters")
    reference = dense.prints[9].reference_modes["full_print"]
    seed = reference.corners + np.float32([[.31, .16], [-.07, -.07], [-.02, 0], [.06, -.10]])
    mask = {"source": "classical_border_quad", "detector_source": CONTRAST_COMPONENT_SOURCE,
            "polygon": _vertices(seed, dense.image)}
    crop = _crop(dense.image, {}, mask, dual_mode=True, camera_matrix=dense.camera_matrix)
    assert crop.metadata["boundary_refinement"]["bottom"]["matched_stock_ring"]
    source = normalized_vertices_to_pixels([
        (point["x"], point["y"]) for point in crop.metadata["source_quad"]
    ], *dense.image.shape[1::-1])
    assert np.max(np.linalg.norm(source - reference.corners, axis=1)) < 1
    modes = crop.metadata["crop_modes"]
    assert crop.image_only is not None
    assert all(mode["status"] == "review_needed" and mode["reason_codes"] == ["PRINT_EDGE_CONFIRMATION_REQUIRED"]
               for mode in modes.values())

    approved = _crop(dense.image, {}, {"source": "manual_quad", "warp_quad": modes["full_print"]["warp_quad"],
                                      "image_quad": modes["image_only"]["warp_quad"]},
                     dual_mode=True, camera_matrix=dense.camera_matrix)
    assert all(mode["status"] == "ready" for mode in approved.metadata["crop_modes"].values())

    surround = np.zeros(dense.image.shape[:2], np.uint8)
    extended = reference.corners.copy()
    extended[2:, 1] += 5
    cv2.fillConvexPoly(surround, np.rint(extended * 16).astype(np.int32), 1, shift=4)
    pixels = (surround != 0) & (reference.mask == 0)
    for contrast in (-4, 4):
        altered = dense.image.copy()
        altered[pixels] = np.uint8(np.clip(altered[pixels].astype(int) + contrast, 0, 255))
        crop = _crop(altered, {}, mask, dual_mode=True, camera_matrix=dense.camera_matrix)
        assert not crop.metadata["boundary_refinement"]["bottom"].get("matched_stock_ring")
        assert all(mode["status"] == "review_needed" for mode in crop.metadata["crop_modes"].values())


@pytest.mark.parametrize("obstruction", [None, "corner", "neighbor"])
def test_ambiguous_stock_fit_improves_only_the_review_warp(obstruction) -> None:
    backing = (124, 136, 146)
    page = np.full((670, 820, 3), backing, np.uint8)
    page[80:620, 120:711] = (189, 212, 228)
    page[114:586, 154:694] = (109, 125, 143)
    page[89:611, 673:694] = (109, 125, 143)
    if obstruction == "corner":
        page[80:90, 700:711] = backing
    neighbors = [{"polygon": _vertices(_rect(709, 200, 800, 480), page)}] if obstruction == "neighbor" else []
    proposal = _rect(120, 80, 708, 619)
    mask = {"source": "yolo_seg_mask", "polygon": _vertices(proposal, page)}

    crop = _crop(page, {}, mask, dual_mode=True, neighbor_masks=neighbors)
    full, motif = (crop.metadata["crop_modes"][mode] for mode in ("full_print", "image_only"))
    assert full["status"] == motif["status"] == "review_needed"
    assert "PRINT_BOUNDARY_AMBIGUOUS" in full["reason_codes"]
    assert crop.image_only is None and crop.image_only_preview is None
    measured = normalized_vertices_to_pixels([(p["x"], p["y"]) for p in full["warp_quad"]], *page.shape[1::-1])
    boundary = crop.metadata["boundary_refinement"]["right"]
    if obstruction is None:
        assert "review_stock_fit" in boundary and full["sampling"] == OWNED_BILINEAR
        np.testing.assert_allclose(measured, _rect(120, 80, 710, 619), atol=.01)
    else:
        assert "review_stock_fit" not in boundary
        np.testing.assert_allclose(measured, proposal, atol=.01)


def test_faint_outer_edge_extends_review_preview_unless_a_neighbor_claims_the_band() -> None:
    from app.pipeline.segmentation_filter import CONTRAST_COMPONENT_SOURCE

    page = np.full((280, 360, 3), 150, np.uint8)
    page[60:171, 60:261] = 233
    page[70:161, 70:251] = (64, 109, 140)
    inner_stock = _rect(60, 60, 260, 170)
    mask = {"source": "classical_border_quad", "detector_source": CONTRAST_COMPONENT_SOURCE,
            "polygon": _vertices(inner_stock, page)}
    plain = _crop(page, {}, mask, dual_mode=True, print_aspect_ratio=200 / 110)
    assert all(mode["status"] == "ready" for mode in plain.metadata["crop_modes"].values())

    page[171:177, 60:261] = 152
    extended = _crop(page, {}, mask, dual_mode=True, print_aspect_ratio=200 / 116)
    assert extended.metadata["boundary_refinement"]["bottom"]["outer_stock_candidate"]
    corners = normalized_vertices_to_pixels([
        (point["x"], point["y"]) for point in extended.metadata["source_quad"]
    ], *page.shape[1::-1])
    assert np.allclose(corners, _rect(60, 60, 260, 176), atol=.5)
    assert extended.image_only is not None
    assert all(mode["reason_codes"] == ["PRINT_EDGE_CONFIRMATION_REQUIRED"] and mode["status"] == "review_needed"
               for mode in extended.metadata["crop_modes"].values())

    neighbor = {"polygon": _vertices(_rect(60, 171, 260, 176), page)}
    isolated = _crop(page, {}, mask, dual_mode=True, print_aspect_ratio=200 / 110, neighbor_masks=[neighbor])
    assert isolated.metadata["source_quad"] == plain.metadata["source_quad"]
    assert all(mode["status"] == "ready" for mode in isolated.metadata["crop_modes"].values())

    page[171:177, 60:69] = page[171:177, 252:261] = 150
    unresolved = _crop(page, {}, mask, dual_mode=True, print_aspect_ratio=200 / 110)
    assert unresolved.metadata["boundary_refinement"]["bottom"]["ambiguous_stock_step"]
    assert unresolved.metadata["source_quad"] == plain.metadata["source_quad"]
    assert all(mode["status"] == "review_needed" for mode in unresolved.metadata["crop_modes"].values())


def _pale_gutter_page(case):
    backing = (230, 233, 237)
    page = np.full((590, 670, 3), backing, np.uint8)
    outer = _rect(95, 196, 393, 517)
    motif = _rect(110, 211, 378, 502)
    neighbor = _rect(67, 65, 415, 174)
    if case == "tilted":
        outer[1, 1] += 15
    if case == "narrow_gutter":
        neighbor[2:, 1] += 18
    cv2.fillConvexPoly(page, neighbor.astype(np.int32), (38, 57, 73))
    fill = (190, 204, 219) if case == "pale_paper_rim" else (110, 143, 160)
    cv2.fillConvexPoly(page, outer.astype(np.int32), fill)
    cv2.fillConvexPoly(page, motif.astype(np.int32), (70, 112, 147))
    cv2.ellipse(page, (226, 332), (94, 78), 0, 0, 360, (45, 120, 189), -1)
    cv2.circle(page, (143, 476), 27, (187, 80, 130), -1)
    seed = outer.copy()
    seed[1, 1] += 38
    if case == "pale_cover":
        page[187:220, 38:450] = (225, 228, 232)
        seed[:2, 1] += 25
    if case == "pale_rim":
        page[196:221, 95:394] = (225, 228, 232)
        seed[:2, 1] += 25
    if case == "small_mark":
        page[196:220, 95:394] = backing
        page[201:203, 108:117] = (76, 65, 53)
        seed[:2, 1] = [192, 236]
    if case == "neighbor_claim":
        neighbor[2:, 1] += 33
    mask = {"source": "classical_border_quad", "polygon": _vertices(seed, page)}
    return page, outer, motif, mask, [{"polygon": _vertices(neighbor, page)}]


@pytest.mark.parametrize("case", ["colored_rim", "pale_paper_rim"])
def test_missing_side_recovers_visible_paper_before_a_stronger_neighbor(case):
    page, outer, motif, mask, neighbors = _pale_gutter_page(case)
    crop = _crop(page, {}, mask, dual_mode=True, enable_classical_retry=True, print_aspect_ratio=298 / 321,
                 neighbor_masks=neighbors)

    assert crop.metadata["boundary_refinement"]["top"]["review_pale_backing_fit"]["native_transition_support"] >= .90
    full, image = (crop.metadata["crop_modes"][mode] for mode in ("full_print", "image_only"))
    assert full["status"] == image["status"] == "review_needed"
    assert full["reason_codes"] == image["reason_codes"] == ["PRINT_EDGE_CONFIRMATION_REQUIRED"]
    assert full["sampling"] == OWNED_BILINEAR
    assert crop.image_only is not None
    for mode, expected in ((full, outer), (image, motif)):
        points = [(point["x"], point["y"]) for point in mode["warp_quad"]]
        measured = normalized_vertices_to_pixels(points, *page.shape[1::-1])
        np.testing.assert_allclose(measured, expected, atol=.4)


@pytest.mark.parametrize("case", ["pale_cover", "pale_rim", "small_mark", "neighbor_claim", "narrow_gutter", "tilted"])
def test_missing_pale_side_does_not_follow_an_unconfirmed_boundary(case):
    page, _, _, mask, neighbors = _pale_gutter_page(case)
    crop = _crop(page, {}, mask, dual_mode=True, enable_classical_retry=True, print_aspect_ratio=298 / 321,
                 neighbor_masks=neighbors)

    assert not any("review_pale_backing_fit" in side for side in crop.metadata["boundary_refinement"].values())
    assert all(mode["status"] == "review_needed" for mode in crop.metadata["crop_modes"].values())


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


@pytest.mark.parametrize("sigma", [0, 1.4, 2.4])
def test_soft_edge_keeps_white_on_white_motif_seed_from_ready_whole_crop(sigma) -> None:
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "white_on_white")
    page = cv2.GaussianBlur(fixture.image, (0, 0), sigma) if sigma else fixture.image
    motif = fixture.prints[0].reference_modes["image_only"]
    mask = {"source": "yolo_seg_mask", "polygon": _vertices(motif.corners, page)}

    crop = _crop(page, {}, mask, dual_mode=True)
    modes = crop.metadata["crop_modes"]
    if sigma == 2.4:
        assert modes["full_print"]["reason_codes"] == ["PRINT_BOUNDARY_BLURRED"]
        assert modes["full_print"]["status"] == modes["image_only"]["status"] == "review_needed"
        assert modes["full_print"]["sampling"] == OWNED_BILINEAR
        assert crop.image_only is None
    else:
        assert modes["full_print"]["status"] == "ready"


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


@pytest.mark.parametrize(("width", "bow", "needs_review"), [
    (680, -3, True), (680, 3, True), (1460, 5, True), (680, -2, False), (680, 0, False),
])
def test_coherent_motif_bow_routes_only_image_mode_to_review(width, bow, needs_review) -> None:
    from app.evaluation.cropping import evaluate_rendered_crop

    height, rim = round(.72 * width), round(.054 * width)
    x0, y0 = 85, 67
    x1, y1 = x0 + width - 1, y0 + height - 1
    y, x = np.mgrid[:y1 + 80, :x1 + 95].astype(np.float32)
    u = np.clip((x - x0 - rim) / (width - 2 * rim - 1), 0, 1)
    v = np.clip((y - y0 - rim) / (height - 2 * rim - 1), 0, 1)
    stock = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    depth = np.minimum.reduce((x - x0 - rim, x1 - rim - x, y - y0 - rim - 4 * u * (1 - u) * bow, y1 - rim - y))
    rng = np.random.default_rng(61)
    grain = cv2.resize(rng.normal(0, 9, (height // 9, width // 9)).astype(np.float32), x.shape[::-1])
    color = np.stack((160 + 23 * u - 31 * v, 149 + 25 * u - 28 * v, 151 + 19 * u - 19 * v), axis=2)
    alpha = np.clip(2 * depth + 1, 0, 1)[..., None]
    page = np.where(stock[..., None], alpha * (color + grain[..., None]) + (1 - alpha) * 235, (55, 61, 56))
    page = np.uint8(np.clip(np.rint(page + rng.normal(0, .6, x.shape)[..., None]), 0, 255))
    mask = {"source": "yolo_seg_mask", "polygon": _vertices(_rect(x0, y0, x1, y1), page)}
    crop = _crop(page, {}, mask, dual_mode=True, print_aspect_ratio=(width - 1) / (height - 1))
    full, image_only = (crop.metadata["crop_modes"][name] for name in ("full_print", "image_only"))
    assert full["status"] == "ready"
    if needs_review:
        assert image_only["status"] == "review_needed"
        assert image_only["reason_codes"] == ["IMAGE_BOUNDARY_CURVED"] and crop.image_only is None
    else:
        assert image_only["status"] == "ready" and crop.image_only is not None
    for mode, rendered, truth in ((full, crop.image, stock), (image_only, crop.image_only, depth >= 0)):
        if rendered is not None:
            quad = [(point["x"], point["y"]) for point in mode["warp_quad"]]
            metrics = evaluate_rendered_crop(truth, quad, np.asarray(mode["page_to_output"]), rendered.shape[:2],
                                             sampling=mode.get("sampling"))
            assert metrics["cutoff_beyond_1px"] == metrics["foreign_beyond_1px"] == 0


def test_native_motif_trace_follows_a_nonparabolic_boundary(curved_motif_page) -> None:
    page, outer = curved_motif_page
    mask = {"source": "manual_quad", "warp_quad": _vertices(outer, page)}
    crop = _crop(page, {}, mask, dual_mode=True)
    full, motif = (crop.metadata["crop_modes"][name] for name in ("full_print", "image_only"))
    assert full["status"] == "ready"
    assert motif["reason_codes"] == ["IMAGE_BOUNDARY_CURVED"]
    assert motif["status"] == "review_needed" and crop.image_only is None
    assert crop.image_only_preview is not None

    traces = motif["suggested_seam_traces"]
    top_trace = min(traces, key=lambda trace: np.median([point["y"] for point in trace]))
    points = normalized_vertices_to_pixels([(point["x"], point["y"]) for point in top_trace], *page.shape[1::-1])
    true_top = 56 + 3.5 * np.sin(2 * np.pi * (points[:, 0] - 65) / 370)
    assert np.max(np.abs(points[:, 1] - (true_top + .5))) < 1


@pytest.mark.parametrize("weak_feature", ["dark_edge", "bright_edge", "no_edge", "ruling", "gradient", "offset_step",
                                         "coherent_shadow", "short_hidden_curve"])
def test_native_motif_trace_recovers_only_supported_faint_edges(weak_feature) -> None:
    from app.pipeline.motif_boundary import close_motif_outline, motif_outline_corners_supported, trace_motif_boundary

    y, x = np.mgrid[:280, :590].astype(np.float32)
    along = (x - 40) / 500
    top = 44 + 3.3 * np.sin(np.pi * along) ** 2
    distance = np.minimum.reduce((x - 40, 540 - x, y - top, 239 - y))
    owned = np.clip(distance + .5, 0, 1)
    span = (.42, .48) if weak_feature == "short_hidden_curve" else (.38, .65)
    weak = (span[0] < along) & (along < span[1]) & (y < top + 24)
    tone = 232 - 62 * owned
    soft_step = owned * (-7 if weak_feature == "dark_edge" else 8)
    if weak_feature in {"dark_edge", "bright_edge"}:
        tone = np.where(weak, 232 + soft_step, tone)
    else:
        nuisance = {
            "no_edge": np.zeros_like(tone),
            "ruling": -13 * np.exp(-.5 * ((y - top + 2) / .6) ** 2),
            "gradient": (y - top) * .85,
            "offset_step": -7 * np.clip(y - top + 3.5, 0, 1),
            "coherent_shadow": -7 * np.clip(y - 46.8, 0, 1),
            "short_hidden_curve": np.zeros_like(tone),
        }[weak_feature]
        tone = np.where(weak, 232 + nuisance, tone)
    rng = np.random.default_rng(86)
    page = np.repeat(np.uint8(np.rint(tone + rng.normal(0, .28, tone.shape)))[..., None], 3, axis=2)
    seed = _rect(40, 44, 540, 239)
    edges = {side: {"rim_bgr": [232, 232, 232], "is_physical_border": True}
             for side in ("top", "right", "bottom", "left")}
    traces, curved, _, continuous = trace_motif_boundary(page, seed, edges)
    assert not continuous
    if weak_feature in {"dark_edge", "bright_edge"}:
        outline = close_motif_outline(seed, traces)
        assert outline is not None and motif_outline_corners_supported(page, outline, traces, edges)
        top_trace = traces[0]
        assert np.max(np.diff(top_trace[:, 0])) < 55
        xs = np.arange(231, 365)
        assert np.max(np.abs(np.interp(xs, *top_trace.T) - (top[0, xs] + .5))) < .35
    elif weak_feature in {"coherent_shadow", "short_hidden_curve"}:
        assert curved and close_motif_outline(seed, traces) is not None
    else:
        assert close_motif_outline(seed, traces) is None


def test_motif_preview_requires_source_evidence_at_an_inferred_corner(curved_motif_page) -> None:
    page, outer = curved_motif_page
    cv2.fillConvexPoly(page, np.int32([[64, 55], [75, 55], [64, 66]]), (236, 236, 236))
    mask = {"source": "manual_quad", "warp_quad": _vertices(outer, page)}
    crop = _crop(page, {}, mask, dual_mode=True)
    motif = crop.metadata["crop_modes"]["image_only"]
    assert motif["status"] == "review_needed" and motif["suggested_seam_traces"]
    assert "outline_preview" not in motif
    assert crop.image_only_preview is None


@pytest.mark.parametrize(("bow", "width", "offset", "shadow"), [
    (0, 21, -4, 0), (0, 91, -4, 0),
    (0, 29, -4, 22), (0, 29, 4, 22),
    (3.5, 29, -4, 22), (3.5, 29, 4, 22),
    (0, 29, -4, 70), (0, 29, 4, 70),
    (3.5, 29, -4, 70), (3.5, 29, 4, 70),
])
def test_motif_crop_reviews_displaced_weak_boundaries(bow, width, offset, shadow) -> None:
    page = np.full((345, 500, 3), (67, 78, 70), np.uint8)
    page[35:306, 45:456] = 236
    x = np.arange(65, 436)
    y = np.arange(46, 288)[:, None]
    chord = 56 + bow * np.sin(2 * np.pi * (x - 65) / 370)
    affected = np.abs(x - 250) <= width // 2
    boundary = chord + offset * affected
    color = np.where(affected[:, None], 229, (131, 108, 145))
    motif = np.clip(y - boundary + .5, 0, 1)[..., None]
    shadow_opacity = (affected * np.clip(y - chord + .5, 0, 1))[..., None]
    page[46:288, 65:436] = np.rint(motif * color + (1 - motif) * 236 - shadow * shadow_opacity)
    mask = {"source": "manual_quad", "warp_quad": _vertices(_rect(45, 35, 455, 305), page)}

    crop = _crop(page, {}, mask, dual_mode=True)
    full, image_only = (crop.metadata["crop_modes"][name] for name in ("full_print", "image_only"))
    assert full["status"] == "ready"
    assert image_only["status"] == "review_needed"
    assert image_only["reason_codes"] == ["IMAGE_BOUNDARY_CURVED" if bow else "IMAGE_BOUNDARY_UNCERTAIN"]
    assert crop.image_only is None and crop.image_only_preview is None
    if width == 91:
        top = [trace for trace in image_only["suggested_seam_traces"]
               if all(54 < point["y"] * 344 < 58 for point in trace)]
        assert len(top) == 2
        assert max(point["x"] * 499 for point in top[0]) < x[affected][0]
        assert min(point["x"] * 499 for point in top[1]) > x[affected][-1]


def test_outline_preview_has_no_foreign_color_and_preserves_the_projective_map() -> None:
    from app.pipeline.crop_sampling import warp_owned_outline

    contour = np.float32([[4.2, 3.4], [17.8, 2.8], [20.1, 13.3], [15.2, 15.6], [5.6, 15.2], [7.1, 9.4]])
    owned = np.array([[cv2.pointPolygonTest(contour, (x, y), False) >= 0 for x in range(25)] for y in range(21)])
    page = np.full((21, 25, 3), 230, np.uint8)
    page[owned] = (60, 109, 170)
    matrix = np.float64([[1.35, .12, .3], [-.09, 1.55, .7], [.003, -.004, 1.]])
    preview, projection = warp_owned_outline(page, contour, matrix)
    poison = page.copy()
    poison[~owned] = (11, 241, 7)
    assert np.array_equal(preview, warp_owned_outline(poison, contour, matrix)[0])
    alpha = preview[..., 3]
    assert np.any(alpha == 0) and np.any((alpha > 0) & (alpha < 255)) and np.any(alpha == 255)
    assert np.all(preview[alpha > 0, :3] == [60, 109, 170])
    assert np.all(preview[alpha == 0, :3] == 0)
    before = cv2.perspectiveTransform(contour[None], matrix)[0]
    after = cv2.perspectiveTransform(contour[None], projection)[0]
    assert np.allclose(before - after, np.floor(before.min(axis=0)), atol=1e-5)
    assert np.all(after >= 0) and np.all(after <= np.array(preview.shape[1::-1]) - 1)


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


@pytest.mark.parametrize("obstruction", ["neighbor", "capture_edge"])
def test_manually_added_outline_requires_a_visible_print(obstruction) -> None:
    page = np.full((220, 340, 3), 55, np.uint8)
    left = 0 if obstruction == "capture_edge" else 20
    page[23:194, left:214] = (109, 130, 169)
    outer = _vertices(_rect(left, 23, 213, 193), page)
    mask = {"source": "manual_added", "manual_added": True, "warp_quad": outer, "visible_polygon": outer}
    neighbors = []
    if obstruction == "neighbor":
        page[69:187, 164:318] = (159, 80, 52)
        neighbors.append({"polygon": _vertices(_rect(164, 69, 317, 186), page)})
    crop = _crop(page, {}, mask, dual_mode=True, trim_border=False, neighbor_masks=neighbors)
    assert crop.image_only is None
    reason = "PRINT_AT_CAPTURE_EDGE" if obstruction == "capture_edge" else "PRINT_POSSIBLY_OCCLUDED"
    assert all(mode["status"] == "review_needed" and reason in mode["reason_codes"]
               for mode in crop.metadata["crop_modes"].values())


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


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_selected_orientation_rotates_both_mode_pixels_and_transforms(degrees, monkeypatch) -> None:
    from app.pipeline import perspective

    def unexpected_inference(*args, **kwargs):
        pytest.fail("A reviewer-selected rotation must bypass orientation inference")

    monkeypatch.setattr(perspective, "correct_photo_orientation", unexpected_inference)
    page = np.random.default_rng(830).integers(0, 256, (120, 180, 3), dtype=np.uint8)
    outer, inner = _rect(10, 11, 149, 109), _rect(23, 21, 134, 97)
    result = correct_photo_crop(page, {}, {
        "warp_quad": _vertices(outer, page), "image_quad": _vertices(inner, page),
        "source": "manual_quad", "orientation_degrees": degrees,
    }, protect_regions=False, dual_mode=True)
    assert result.metadata["orientation"]["decision"] == "reviewer_selected"
    assert result.metadata["orientation"]["rotation_degrees"] == degrees
    for name, output, quad, expected in (
        ("full_print", result.image, outer, page[11:110, 10:150]),
        ("image_only", result.image_only, inner, page[21:98, 23:135]),
    ):
        assert np.array_equal(output, np.rot90(expected, k=-(degrees // 90)))
        assert output is not None
        height, width = output.shape[:2]
        mode = result.metadata["crop_modes"][name]
        assert mode["shape"] == {"width": width, "height": height}
        assert mode["status"] == "ready"
        corners = _rect(0, 0, width - 1, height - 1)
        mapped = cv2.perspectiveTransform(quad.reshape(1, 4, 2), np.asarray(mode["page_to_output"]))[0]
        np.testing.assert_allclose(mapped, np.roll(corners, -(degrees // 90), axis=0), atol=1e-5)


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


@pytest.mark.parametrize(("reach", "motif_covered"), [(0, False), (20, False), (27, True), (35, True)])
def test_pale_corner_holder_with_colored_outline_requires_review_only_where_it_covers(reach, motif_covered) -> None:
    from app.evaluation.crop_fixtures import _layer
    from app.evaluation.cropping import evaluate_rendered_crop

    print_image, references = _layer(
        "white_margin", np.random.default_rng(341), width=432, height=290,
        center=(420.5, 319.5), angles=(0., 0., 0.), margin=12,
    )
    outer, motif = (references[name] for name in ("full_print", "image_only"))
    page = np.full_like(print_image, 31)
    page[outer.mask != 0] = print_image[outer.mask != 0]
    if reach:
        holder = np.rint(outer.corners[0] + np.float32([[-5, -5], [reach, -5], [-5, reach]])).astype(np.int32)
        covered = cv2.fillConvexPoly(np.zeros(page.shape[:2], np.uint8), holder, 1)
        cv2.polylines(covered, [holder], True, 1, 2)
        assert bool(np.any(covered & motif.mask)) is motif_covered
        cv2.fillConvexPoly(page, holder, (229, 229, 229))
        cv2.polylines(page, [holder], True, (92, 155, 205), 2)

    # The detector mask can fill the corner even when the holder covers ink.
    mask = {
        "source": "yolo_seg_mask", "polygon": _vertices(motif.corners, page),
        "visible_polygon": _vertices(outer.corners, page),
    }
    result = _crop(page, {}, mask, dual_mode=True, enable_classical_retry=True)
    modes = result.metadata["crop_modes"]
    assert modes["full_print"]["status"] == ("review_needed" if reach else "ready")
    if reach:
        assert "PRINT_POSSIBLY_OCCLUDED" in modes["full_print"]["reason_codes"]
    mode = modes["image_only"]
    if motif_covered:
        assert mode["status"] == "review_needed" and result.image_only is None
        return
    assert mode["status"] == "ready" and result.image_only is not None
    assert mode.get("independent_of_full_print", False) is bool(reach)
    measured = evaluate_rendered_crop(
        motif.mask, [(p["x"], p["y"]) for p in mode["warp_quad"]], np.array(mode["page_to_output"]),
        result.image_only.shape[:2], expected_corners=motif.corners, expected_shape=motif.shape,
        sampling=mode["sampling"],
    )
    assert measured["foreign_source_samples"] == measured["cutoff_beyond_1px"] == 0


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


@pytest.mark.parametrize("coefficient", [.025, .05])
def test_radial_curvature_keeps_tilted_print_in_review(coefficient) -> None:
    from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages

    fixture = next(page for page in synthetic_crop_pages("development") if page.name == "camera_tilt")
    height, width = fixture.image.shape[:2]
    outer_scale = np.float32([width - 1, height - 1])
    center = np.float32([.2 * width, .38 * height])
    scale = max(width, height) / 2
    yy, xx = np.mgrid[:height, :width]
    rays = np.float32(np.stack([xx, yy], axis=-1)) - center
    distorted_radius = np.linalg.norm(rays, axis=2) / scale
    radius = distorted_radius.copy()
    for _ in range(7):
        radius -= (radius + coefficient * radius ** 3 - distorted_radius) / (1 + 3 * coefficient * radius ** 2)
    inverse = rays / (1 + coefficient * radius[..., None] ** 2) + center
    image = cv2.remap(fixture.image, inverse[..., 0].copy(), inverse[..., 1].copy(), cv2.INTER_LINEAR)
    box, mask = geometry_seed(fixture.prints[0], image.shape)
    seed = np.float32([(point["x"], point["y"]) for point in mask["polygon"]]) * outer_scale - center
    mask["polygon"] = _vertices(center + seed * (1 + coefficient * (seed ** 2).sum(axis=1, keepdims=True) / scale ** 2), image)

    crop = _crop(image, box, mask, dual_mode=True)
    full = crop.metadata["crop_modes"]["full_print"]
    assert full["status"] == "review_needed"
    assert "PRINT_BOUNDARY_CURVED" in full["reason_codes"]
    traces = full["suggested_edge_traces"]
    assert traces and all(len(trace) >= 2 for trace in traces)
    known_corners = np.float32(fixture.prints[0].reference_modes["full_print"].corners)
    for trace in traces:
        locations = np.float32([(point["x"], point["y"]) for point in trace]) * outer_scale
        x, y = locations.T
        observed = cv2.remap(inverse, x[:, None].copy(), y[:, None].copy(), cv2.INTER_LINEAR)[:, 0]
        assert max(abs(cv2.pointPolygonTest(known_corners, tuple(point), True)) for point in observed) < .9
    assert crop.metadata["crop_modes"]["image_only"]["status"] == "review_needed"
    assert crop.image_only is None and crop.full_print_preview is None


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
