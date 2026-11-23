from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _detection(
    box: dict[str, float],
    confidence: float,
    *,
    family: str,
    polygon: list[tuple[float, float]] | None = None,
):
    from app.pipeline.segmentation import SegmentationDetection

    points = polygon or [
        (box["x1"], box["y1"]),
        (box["x2"], box["y1"]),
        (box["x2"], box["y2"]),
        (box["x1"], box["y2"]),
    ]
    return SegmentationDetection(
        bounding_box=box,
        mask={
            "polygon": [{"x": x, "y": y} for x, y in points],
            "source": "yolo_seg_mask" if family == "model" else "classical_border_quad",
            "detector_source": "yolo_onnx" if family == "model" else "edge_contour",
            "scores": {"area_ratio": (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])},
        },
        confidence=confidence,
        aspect_ratio=(box["x2"] - box["x1"]) / (box["y2"] - box["y1"]),
        geometry_valid=True,
        review_reasons=[],
    )


def test_quad_ordering_keeps_all_vertices_for_exact_diamond() -> None:
    from app.pipeline.geometry import order_quad_cyclic

    diamond = np.array([[100, 0], [200, 100], [100, 200], [0, 100]], dtype=np.float32)
    ordered = order_quad_cyclic(diamond)

    assert len(np.unique(ordered, axis=0)) == 4
    assert {tuple(point) for point in ordered} == {tuple(point) for point in diamond}
    assert cv2.isContourConvex(ordered.reshape(-1, 1, 2))


def test_normalized_vertex_round_trip_includes_last_pixel_centers() -> None:
    from app.pipeline.geometry import normalized_vertices_to_pixels, pixel_vertices_to_normalized

    normalized = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    pixels = normalized_vertices_to_pixels(normalized, width=101, height=51)

    assert np.array_equal(pixels, np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32))
    assert np.allclose(pixel_vertices_to_normalized(pixels, width=101, height=51), normalized)


def test_perspective_accepts_normalized_vertices_on_right_and_bottom_edges() -> None:
    from app.pipeline.perspective import crop_and_correct_photo_with_metadata

    image = np.full((80, 120, 3), 150, dtype=np.uint8)
    mask = {
        "source": "manual_quad",
        "polygon": [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
            {"x": 0.0, "y": 1.0},
        ],
    }

    crop, metadata = crop_and_correct_photo_with_metadata(
        image,
        {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        mask,
        trim_border=False,
        auto_orient=False,
        protected_regions=[],
        protect_regions=False,
    )

    assert crop.size > 0
    assert np.array_equal(crop, image)
    assert metadata["quad_validation"]["maximum_local_anisotropy"] < 1.01


@pytest.mark.parametrize("automatic", [True, False])
def test_refined_crop_samples_page_once_and_records_the_exact_corners(monkeypatch, automatic: bool) -> None:
    from types import SimpleNamespace

    from app.pipeline import perspective
    from app.pipeline.geometry import normalized_vertices_to_pixels

    rng = np.random.default_rng(283)
    page = rng.integers(0, 256, (215, 290, 3), dtype=np.uint8)
    original_quad = np.array([[50, 44], [244, 35], [229, 171], [39, 180]], dtype=np.float32)
    mask = {
        "source": "yolo_seg_mask" if automatic else "manual_quad",
        "polygon": [{"x": float(x / 289), "y": float(y / 214)} for x, y in original_quad],
    }

    def fixed_seams(analysis, *, reference_box, protected_regions):
        left, top, right, bottom = reference_box
        if automatic:
            assert min(left, top, analysis.shape[1] - right, analysis.shape[0] - bottom) >= 2
        else:
            assert reference_box == (0, 0, analysis.shape[1], analysis.shape[0])
        width, height = right - left, bottom - top
        quad = np.array([[0, 7], [width - 2, -2 if automatic else 0],
                         [width - 3, height - 4], [5, height - 2]], np.float32)
        return SimpleNamespace(crop_quad=quad, confidence=1.0, metadata=lambda: {"applied": True})

    monkeypatch.setattr(perspective, "find_physical_border", fixed_seams)
    monkeypatch.setattr(perspective, "detect_protected_regions", lambda image: [])
    result = perspective.correct_photo_crop(page, {}, mask, protected_regions=[], auto_orient=False)
    h, w = result.image.shape[:2]
    destination = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    quad = normalized_vertices_to_pixels(
        [[p["x"], p["y"]] for p in result.metadata["warp_quad"]], page.shape[1], page.shape[0]
    )
    expected = cv2.warpPerspective(page, cv2.getPerspectiveTransform(quad, destination), (w, h))

    # Rounding normalized vertices for JSON can change a few interpolation bins.
    assert np.abs(result.image.astype(float) - expected).mean() < 0.02
    assert result.metadata["warp_quad"] != result.metadata["source_quad"]
    assert result.metadata["target_height"] == h and result.metadata["target_width"] == w
    assert result.metadata["border_trim"]["before"] == {
        "height": result.rectified_untrimmed.shape[0], "width": result.rectified_untrimmed.shape[1],
    }


def test_border_analysis_halo_stops_at_page_edges() -> None:
    from app.pipeline.perspective import _analysis_padding

    destination = np.float32([[0, 0], [100, 0], [100, 80], [0, 80]])
    assert _analysis_padding(destination, np.eye(3), 101, 81) == (0, 0, 0, 0)
    assert _analysis_padding(destination, np.eye(3), 201, 181) == (0, 0, 3, 3)


def test_mask_aware_nms_suppresses_duplicates_but_preserves_overlapping_prints() -> None:
    from app.pipeline.segmentation import _nms_detections

    first = _detection({"x1": 0.10, "y1": 0.10, "x2": 0.60, "y2": 0.70}, 0.95, family="model")
    overlapping_print = _detection({"x1": 0.32, "y1": 0.12, "x2": 0.82, "y2": 0.72}, 0.90, family="model")
    duplicate = _detection({"x1": 0.105, "y1": 0.105, "x2": 0.605, "y2": 0.705}, 0.80, family="model")

    selected = _nms_detections([first, overlapping_print, duplicate])

    assert selected == [first, overlapping_print]


def test_cross_source_fusion_replaces_weaker_boundary_hypothesis() -> None:
    from app.pipeline.segmentation import fuse_candidate_evidence

    image = np.full((200, 240, 3), 242, dtype=np.uint8)
    cv2.rectangle(image, (48, 40), (191, 159), (35, 35, 35), 3)
    image[44:156, 52:188] = (55, 110, 175)
    model = _detection({"x1": 0.17, "y1": 0.17, "x2": 0.83, "y2": 0.83}, 0.70, family="model")
    classical = _detection({"x1": 0.20, "y1": 0.20, "x2": 0.80, "y2": 0.80}, 0.66, family="classical")

    selected, metadata = fuse_candidate_evidence(image, [model], [classical])

    assert len(selected) == 1
    assert selected[0].mask["fusion"]["selected_family"] == "classical"
    assert selected[0].mask["fusion"]["selection_reason"] == "higher_boundary_evidence"
    assert selected[0].mask["evidence_schema"] == "segmentation_detection_evidence_v1"
    assert 0.0 <= selected[0].mask["boundary_evidence"]["score"] <= 1.0
    assert 0.0 <= selected[0].mask["geometry_evidence"]["score"] <= 1.0
    assert metadata["model_hypotheses_replaced"] == 1
    assert metadata["clusters"][0]["matches"][0]["polygon_iou"] > 0.7


def test_mounted_print_refinement_replaces_fragments_and_rejects_empty_slots() -> None:
    from app.pipeline.segmentation import SegmentationResult, refine_album_page_result

    image = np.full((500, 600, 3), (45, 30, 120), dtype=np.uint8)
    image[110:360, 75:245] = (160, 160, 160)
    image[110:360, 275:445] = (212, 212, 212)
    split_print = _detection({"x1": 0.12, "y1": 0.23, "x2": 0.30, "y2": 0.58}, 0.82, family="model")
    blank_slot = _detection({"x1": 0.78, "y1": 0.25, "x2": 0.94, "y2": 0.73}, 0.97, family="model")

    result = refine_album_page_result(image, SegmentationResult([split_print, blank_slot], {}))

    assert len(result.detections) == 2
    assert all(d.mask["source"] == "chromatic_mount_boundary" for d in result.detections)
    assert all("chromatic_mount_boundary_needs_review" in d.review_reasons for d in result.detections)
    assert [d.bounding_box["x1"] for d in result.detections] == pytest.approx([75 / 599, 275 / 599], abs=0.01)
    assert result.metadata["page_proposal_rejections"] == {0: "replaced_by_mount_outline", 1: "album_mount_region"}


def test_page_container_guard_keeps_each_photo_and_preserves_overlap() -> None:
    from app.pipeline.segmentation import _page_containers

    page = _detection({"x1": 0.0, "y1": 0.0, "x2": 0.95, "y2": 0.89}, 0.93, family="model")
    left = _detection({"x1": 0.06, "y1": 0.17, "x2": 0.40, "y2": 0.72}, 0.82, family="classical")
    right = _detection({"x1": 0.45, "y1": 0.19, "x2": 0.89, "y2": 0.78}, 0.83, family="model")
    overlapping = _detection({"x1": 0.12, "y1": 0.22, "x2": 0.55, "y2": 0.62}, 0.85, family="model")

    image = np.full((160, 240, 3), 200, dtype=np.uint8)
    assert _page_containers([page, left, right], image) == {0}
    assert _page_containers([page, left, overlapping], image) == set()


def test_weak_container_requires_plain_paper_outside_a_well_bounded_print() -> None:
    from app.pipeline.segmentation import _page_containers

    page = _detection({"x1": 0.08, "y1": 0.06, "x2": 0.76, "y2": 0.58}, 0.88, family="model")
    page.mask["boundary_evidence"] = {"score": 0.32, "edge_support": 0.08}
    photo = _detection({"x1": 0.25, "y1": 0.18, "x2": 0.65, "y2": 0.51}, 0.82, family="classical")
    photo.mask["boundary_evidence"] = {"border_contrast": 0.75}
    other = _detection({"x1": 0.2, "y1": 0.68, "x2": 0.78, "y2": 0.94}, 0.90, family="model")
    image = np.full((180, 240, 3), 213, dtype=np.uint8)
    image[32:92, 60:156] = 60

    assert _page_containers([page, photo, other], image) == {0}
    image[10:65, 20:58] = 55
    assert _page_containers([page, photo, other], image) == set()


def test_classical_segmentation_records_measured_elapsed_time(monkeypatch) -> None:
    import app.pipeline.segmentation as segmentation

    ticks = iter((20.0, 20.01234))
    monkeypatch.setattr(segmentation, "perf_counter", lambda: next(ticks))

    result = segmentation.detect_photos_classical(np.full((80, 100, 3), 128, dtype=np.uint8))

    assert result.metadata["classical_time_ms"] == 12.34
    assert result.metadata["inference_time_ms"] == 12.34


def test_evaluator_runtime_reports_end_to_end_and_component_breakdown(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import evaluate_onnx_segmentation as evaluator

    first = evaluator._page_runtime(
        {"model": "yolov8_seg_onnx", "inference_time_ms": 25.0},
        32.5,
    )
    second = evaluator._page_runtime(
        {
            "model": "yolov8_seg_onnx",
            "inference_time_ms": 26.0,
            "classical_metadata": {"classical_time_ms": 14.0},
        },
        47.0,
    )
    pure_classical = evaluator._page_runtime(
        {
            "model": "classical_hybrid_quad",
            "inference_time_ms": 8.0,
            "classical_time_ms": 8.0,
        },
        11.0,
    )
    summary = evaluator._runtime_summary([first, second, pure_classical])

    assert first == {
        "segmentation_end_to_end_ms": 32.5,
        "model_ms": 25.0,
        "classical_ms": None,
        "orchestration_ms": 7.5,
        "classical_invoked": False,
    }
    assert second["classical_ms"] == 14.0
    assert pure_classical["model_ms"] is None
    assert pure_classical["classical_ms"] == 8.0
    assert summary["segmentation_end_to_end"]["samples"] == 3
    assert summary["mean_end_to_end_ms"] == pytest.approx((32.5 + 47.0 + 11.0) / 3.0)
    assert summary["model"]["samples"] == 2
    assert summary["classical"]["samples"] == 2
    assert summary["classical_page_count"] == 2


def test_page_31_risky_model_mask_does_not_become_a_duplicate() -> None:
    from app.config import Settings
    from app.evaluation.cropping import evaluate_page_crops
    from app.pipeline.enhancement import prepare_for_cutout
    from app.pipeline.segmentation import segment_album_page

    model_path = Path("models/yolov8-seg-album.onnx")
    manifest_path = Path("models/manifest.json")
    image_path = Path("data/golden_fixtures/page_31.jpg")
    if not model_path.is_file() or not image_path.is_file():
        pytest.skip("Registered model and locked page_31 fixture are required.")
    image, _ = prepare_for_cutout(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
    settings = Settings(
        DATABASE_URL="sqlite+aiosqlite:///./geometry-test.db",
        REDIS_URL="redis://localhost:6379/0",
        YOLO_MODEL_PATH=model_path,
        MODEL_MANIFEST_PATH=manifest_path,
        REQUIRE_REGISTERED_MODELS=True,
        ENABLE_CLASSICAL_SEGMENTATION_FALLBACK=True,
        YOLO_PROPOSAL_THRESHOLD=0.25,
        YOLO_CONFIDENCE_THRESHOLD=0.70,
    )

    result = segment_album_page(image, settings)

    assert result.metadata["model_set_risk"]["risky"] is True
    assert any(
        reason.startswith("MODEL_BOUNDARY_EVIDENCE_WEAK") for reason in result.metadata["classical_evidence_trigger"]
    )
    assert result.metadata["classical_time_ms"] > 0.0
    assert result.metadata["classical_time_ms"] == result.metadata["classical_metadata"]["classical_time_ms"]
    expected = np.loadtxt(image_path.parent / "labels/page_31.txt")[:, 1:].reshape(2, -1, 2)
    detected = [[(point["x"], point["y"]) for point in item.mask["polygon"]] for item in result.detections]
    height, width = image.shape[:2]
    metrics = evaluate_page_crops(expected, detected, width=width, height=height, iou_threshold=0.85)
    assert (metrics.true_positive, metrics.false_positive, metrics.false_negative) == (2, 0, 0)


def test_perspective_forwards_detection_evidence_for_confidence_aggregation() -> None:
    from app.pipeline.perspective import crop_and_correct_photo_with_metadata

    image = np.full((100, 140, 3), 128, dtype=np.uint8)
    boundary = {"score": 0.73, "edge_support": 0.8, "border_contrast": 0.66}
    geometry = {"score": 0.91, "valid": True, "rectangularity": 0.98}
    mask = {
        "source": "model_quad",
        "polygon": [
            {"x": 0.1, "y": 0.1},
            {"x": 0.9, "y": 0.1},
            {"x": 0.9, "y": 0.9},
            {"x": 0.1, "y": 0.9},
        ],
        "evidence_schema": "segmentation_detection_evidence_v1",
        "boundary_evidence": boundary,
        "geometry_evidence": geometry,
    }

    _, metadata = crop_and_correct_photo_with_metadata(
        image,
        {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9},
        mask,
        trim_border=False,
        auto_orient=False,
        protected_regions=[],
        protect_regions=False,
    )

    assert metadata["boundary_evidence"] == boundary
    assert metadata["geometry_evidence"] == geometry
    assert metadata["boundary_confidence"] == 0.73
    assert metadata["geometry_confidence"] == 0.91


def test_heuristic_orientation_is_advisory_even_with_strong_landscape_bias() -> None:
    from app.pipeline.orientation import correct_photo_orientation

    rng = np.random.default_rng(2)
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    image[:140] = (220, 180, 120)
    image[140:] = (50, 100, 60)
    image[145:] = np.clip(
        image[145:].astype(np.int16) + rng.integers(-35, 36, size=image[145:].shape),
        0,
        255,
    ).astype(np.uint8)
    cv2.line(image, (0, 140), (359, 140), (240, 240, 240), 2)

    corrected, metadata = correct_photo_orientation(image, model_path=None)

    assert np.array_equal(corrected, image)
    assert metadata["rotation_degrees"] == 0
    assert metadata["suggested_rotation_degrees"] == 180
    assert metadata["decision"] == "advisory_preserve"
    assert metadata["reason_code"] == "ORIENTATION_UNCERTAIN"
    assert 0.0 <= metadata["confidence"] <= 1.0


def test_high_confidence_orientation_model_still_rotates(monkeypatch) -> None:
    import app.pipeline.orientation as orientation

    monkeypatch.setattr(
        orientation,
        "_onnx_orientation_scores",
        lambda image, model_path: {0: 0.02, 90: 0.91, 180: 0.04, 270: 0.03},
    )
    image = np.full((40, 80, 3), 120, dtype=np.uint8)

    corrected, metadata = orientation.correct_photo_orientation(image, model_path=Path("model.onnx"))

    assert corrected.shape[:2] == (80, 40)
    assert metadata["rotation_degrees"] == 90
    assert metadata["decision"] == "auto_rotate"
    assert "reason_code" not in metadata
