import cv2
import numpy as np
import pytest

from app.evaluation.cropping import evaluate_rendered_crop
from app.pipeline import segmentation
from app.pipeline.geometry import normalized_vertices_to_pixels
from app.pipeline.perspective import correct_photo_crop
from app.pipeline.print_edges import (
    PRINT_EDGE_CONFIRMATION,
    PRINT_FAINT_RIM_REVIEW,
    PRINT_OUTER_EDGE_CONFLICT_REVIEW,
    PRINT_SHADOW_EDGE_REVIEW,
    line_fit_omits_print_stock,
    refine_print_boundary,
    unresolved_opposing_outer_edges,
)


_PAPER = np.float32([[221, 212], [1060, 212], [1060, 850], [221, 850]])
_MOTIF = _PAPER + np.float32([[48, 48], [-48, 48], [-48, -48], [48, -48]])


def _scan(
    shadow=None, *, warm_shadow=False, stock=None, ink=None, blur=.8,
    jpeg_quality=None, continued_guide=False,
):
    y, x = np.indices((1040, 1400), dtype=np.float32)
    distances = np.stack([y - 212, 1060 - x, 850 - y, x - 221])
    depth = distances.min(axis=0)
    paper = depth >= 0
    motif = depth >= 48
    lab = np.empty((*depth.shape, 3), np.float32)
    lab[:] = (251, 128, 128)
    if stock is None:
        stock = (219, 129, 135) if shadow else (184, 128, 128)
    lab[paper] = stock
    if continued_guide:
        lab[(y >= 212) & (y <= 250)] = stock
    texture = 17 * np.sin(x / 19) * np.cos(y / 24) * np.clip((depth - 48) / 20, 0, 1)
    if ink is None:
        ink = 180 if shadow else 226
    lab[motif, 0] = ink + texture[motif]
    if shadow:
        amount, width, center = shadow
        shade = np.where(distances < 0, amount * np.exp(-.5 * ((distances - center) / width)**2), 0).max(axis=0)
        lab[~paper, 0] -= shade[~paper]
        if warm_shadow:
            lab[~paper, 1:] += shade[~paper, None] * np.float32([.005, .09])
    lab[..., 0] += np.random.default_rng(8317).normal(0, .45, depth.shape)
    image = cv2.cvtColor(np.rint(lab).astype(np.uint8), cv2.COLOR_LAB2BGR)
    image = cv2.GaussianBlur(image, (0, 0), blur)
    if jpeg_quality:
        encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image, {"full_print": paper, "image_only": motif}


def _crop(image, seed):
    scale = np.float32(image.shape[1::-1]) - 1
    polygon = [{"x": float(x), "y": float(y)} for x, y in seed / scale]
    return correct_photo_crop(
        image, {}, {"polygon": polygon}, dual_mode=True, auto_orient=False, protect_regions=False,
        print_aspect_ratio=839 / 638,
    )


def _assert_review_geometry(image, truth, crop, reason):
    for mode, target, expected in (("full_print", crop.image, _PAPER), ("image_only", crop.image_only, _MOTIF)):
        evidence = crop.metadata["crop_modes"][mode]
        assert evidence["status"] == "review_needed" and reason in evidence["reason_codes"]
        vertices = [(point["x"], point["y"]) for point in evidence["warp_quad"]]
        corners = normalized_vertices_to_pixels(vertices, *image.shape[1::-1])
        assert np.max(np.linalg.norm(corners - expected, axis=1)) < 1
        metrics = evaluate_rendered_crop(truth[mode], vertices, evidence["page_to_output"], target.shape[:2],
                                         sampling=evidence["sampling"])
        assert metrics["foreign_source_samples"] == metrics["cutoff_beyond_1px"] == 0


@pytest.mark.parametrize(("shadow", "warm_shadow"), [((65, 7, -3), False), ((70, 8, -4), True)])
def test_broad_shadow_cannot_take_ownership_of_the_outer_print(shadow, warm_shadow):
    image, truth = _scan(shadow, warm_shadow=warm_shadow)
    crop = _crop(image, _PAPER)
    assert any(side["outward_search"].get("shadow_shoulder_support", 0) >= .70
               for side in crop.metadata["boundary_refinement"].values())
    _assert_review_geometry(image, truth, crop, PRINT_SHADOW_EDGE_REVIEW)


def test_a_dark_neutral_paper_rim_surrounding_a_bright_motif_still_recovers():
    image, _ = _scan()
    crop = _crop(image, _MOTIF)
    assert all(mode["status"] == "ready" for mode in crop.metadata["crop_modes"].values())
    assert all(side["outward_search"]["print_rim_supported"] for side in crop.metadata["boundary_refinement"].values())
    assert crop.image.shape[:2] == (639, 840)


def test_terminating_faint_rim_recovers_a_review_preview_from_a_motif_seed():
    image, truth = _scan(stock=(246, 128, 128), ink=155)
    crop = _crop(image, _MOTIF)
    assert all(3.4 <= side["weak_rim_transition_lab"] <= 9.5
               for side in crop.metadata["boundary_refinement"].values())
    _assert_review_geometry(image, truth, crop, PRINT_FAINT_RIM_REVIEW)


def test_competing_outer_edges_hold_both_modes_after_blur_and_recompression():
    image, _ = _scan(stock=(245, 129, 132), ink=150, blur=1.6, jpeg_quality=70)
    crop = _crop(image, _MOTIF)
    for mode in crop.metadata["crop_modes"].values():
        assert mode["status"] == "review_needed"
        assert PRINT_OUTER_EDGE_CONFLICT_REVIEW in mode["reason_codes"]


def test_faint_backing_guide_continued_past_corners_does_not_verify_a_rim():
    image, _ = _scan(stock=(246, 128, 128), ink=155, continued_guide=True)
    _, sides, reasons = refine_print_boundary(image, _MOTIF)
    for side in sides.values():
        assert side["outward_search"]["support"] >= .88
        assert side["outward_search"]["corner_support"] >= .78
    assert PRINT_FAINT_RIM_REVIEW not in reasons


_OPPOSING_SEED = np.float32([[210, 160], [550, 160], [550, 459], [210, 459]])


def _opposing_outer_line_scene(right_step: int = 4, *, competing_right: bool = False) -> np.ndarray:
    lab = np.full((640, 810, 3), (247, 128, 128), np.uint8)
    lab[160:460, 210:551] = (126, 132, 143)
    # The left pale strip terminates at a strong transition. The right
    # strip terminates at a weak but distributed illumination step.
    lab[160:460, 187:210] = (214, 129, 131)
    lab[160:460, 551:573] = (208, 129, 130)
    lab[160:460, 573:] = (208 + right_step, 129, 130)
    if competing_right:
        lab[160:460, 581:] = (208 + 2 * right_step, 129, 130)
    return cv2.GaussianBlur(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), (0, 0), .45)


@pytest.mark.parametrize("competing_right", [False, True], ids=["two-clear-lines", "opposite-side-ambiguous"])
def test_opposing_outer_lines_hold_both_modes_and_keep_the_owned_raster(competing_right):
    image = _opposing_outer_line_scene(competing_right=competing_right)
    polygon = segmentation._normalized_polygon(_OPPOSING_SEED, *image.shape[1::-1])
    mask = {
        "polygon": polygon,
        "visible_polygon": polygon,
        "source": "yolo_seg_mask",
        "detector_source": "yolo_onnx",
    }
    options = dict(dual_mode=True, auto_orient=False, protect_regions=False, print_aspect_ratio=340 / 299)
    baseline = correct_photo_crop(image, {}, mask, **options)
    reviewed = correct_photo_crop(
        image, {}, {**mask, "visible_polygon_source": "physical_print_boundary"}, **options,
    )

    original = baseline.metadata["crop_modes"]["full_print"]
    modes = reviewed.metadata["crop_modes"]
    assert original["status"] == "ready"
    for mode in modes.values():
        assert mode["status"] == "review_needed"
        assert mode["reason_codes"] == [PRINT_OUTER_EDGE_CONFLICT_REVIEW]
    for key in ("warp_quad", "page_to_output", "shape", "sampling"):
        assert modes["full_print"][key] == original[key]
    assert original["sampling"] == "owned_bilinear"
    np.testing.assert_array_equal(reviewed.image, baseline.image)

    traces = modes["full_print"]["suggested_edge_traces"]
    assert reviewed.metadata["boundary_refinement"]["right"]["outward_search"]["ambiguous"] == competing_right
    expected_columns = [[187, 187]] if competing_right else [[572, 572], [187, 187]]
    assert len(traces) == len(expected_columns) and all(len(trace) == 2 for trace in traces)
    native_traces = [
        normalized_vertices_to_pixels([(point["x"], point["y"]) for point in trace], *image.shape[1::-1])
        for trace in traces
    ]
    np.testing.assert_allclose([trace[:, 0] for trace in native_traces], expected_columns, atol=.6)


@pytest.mark.parametrize("new_step", [0, 12], ids=["missing-line", "consistent-pale-rim"])
def test_opposing_outer_line_evidence_is_remeasured_on_the_current_source(new_step):
    image = _opposing_outer_line_scene()
    fitted, sides, reasons = refine_print_boundary(image, _OPPOSING_SEED)
    assert fitted is not None and not reasons
    assert len(unresolved_opposing_outer_edges(image, fitted, sides)) == 2

    changed = _opposing_outer_line_scene(new_step)
    assert unresolved_opposing_outer_edges(changed, fitted, sides) == []


@pytest.mark.parametrize("layout", ["single_print", "matching_album", "continuing_line", "adjacent_print", "near_rail"])
def test_inward_line_fit_requires_stock_and_terminating_corners(layout):
    image, _ = _scan(stock=(225, 126, 122), ink=120, jpeg_quality=75)
    fitted = _PAPER.copy()
    fitted[1:3, 0] -= 48
    if layout == "matching_album":
        tone = image[400, 1040]
        image[:212] = image[851:] = tone
        image[:, :221] = image[:, 1061:] = tone
    elif layout == "continuing_line":
        image[:212, 1007:1061] = image[212, 1007:1061]
    elif layout == "adjacent_print":
        # A second, same-colour print sits within the larger contour. Its
        # narrow gap lies inside the stock corridor, away from either edge.
        image[212:851, 1053:1057] = image[400, 1080]
    elif layout == "near_rail":
        image[:, 1061:1069] = image[400, 1040]

    assert line_fit_omits_print_stock(image, _PAPER, fitted) is (layout == "single_print")


def test_retained_contour_is_the_crop_seed_and_holds_both_modes(monkeypatch, tmp_path):
    image, truth = _scan(stock=(225, 126, 122), ink=120, jpeg_quality=75)
    fitted = _PAPER.copy()
    fitted[1:3, 0] -= 48
    box = cv2.boundingRect(_PAPER)
    candidate = segmentation._ContourCandidate(box, _PAPER, float(truth["full_print"].mean()), "edge_contour")
    monkeypatch.setattr(segmentation, "_find_contour_candidates", lambda *_: [candidate])
    monkeypatch.setattr(segmentation, "_refine_quadrilateral_with_lines", lambda *_: (fitted, True))
    setting = segmentation.Settings(DATABASE_URL="sqlite://", REDIS_URL="redis://", YOLO_MODEL_PATH=tmp_path / "none.onnx")

    proposal, = segmentation.segment_album_page(image, setting, source_image=image, protected_regions=[]).detections

    assert proposal.review_reasons == [PRINT_EDGE_CONFIRMATION]
    assert proposal.mask["warp_quad"] == proposal.mask["polygon"]
    motif = segmentation._normalized_polygon(_MOTIF, *image.shape[1::-1])
    crop = correct_photo_crop(
        image, proposal.bounding_box, {**proposal.mask, "image_quad": motif},
        dual_mode=True, auto_orient=False, protect_regions=False,
        print_aspect_ratio=839 / 638,
    )
    _assert_review_geometry(image, truth, crop, PRINT_EDGE_CONFIRMATION)
