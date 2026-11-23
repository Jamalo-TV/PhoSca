from __future__ import annotations

import pytest


def test_orientation_evaluation_applies_the_reference_correction_after_each_rotation(tmp_path, monkeypatch) -> None:
    from hashlib import sha256
    import json

    import cv2
    import numpy as np

    from app.pipeline import orientation
    from scripts.evaluate_photo_orientation import evaluate, read_manifest

    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    image_path = tmp_path / "sideways.png"
    assert cv2.imwrite(str(image_path), image)
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps({"schema_version": 1, "samples": [{
        "id": "sideways", "source_group": "Album A", "image": image_path.name,
        "image_sha256": sha256(image_path.read_bytes()).hexdigest(), "upright_clockwise_degrees": 90,
    }]}), encoding="utf-8")
    samples, _ = read_manifest(manifest)
    rotations = iter([(0, 90, "auto_rotate"), (90, 0, "preserve_and_review"),
                      (180, 180, "auto_rotate"), (270, 0, "confidently_preserve")])

    def decide(observed, *, model_path):
        applied, result, decision = next(rotations)
        assert np.array_equal(observed, orientation.rotate_image(image, applied))
        return orientation.rotate_image(observed, result), {
            "method": "onnx_orientation", "rotation_degrees": result, "confidence": .7, "decision": decision,
        }

    monkeypatch.setattr(orientation, "correct_photo_orientation", decide)
    result = evaluate(tmp_path / "test.onnx", samples)
    assert [case["expected_clockwise_degrees"] for case in result["cases"]] == [90, 0, 270, 180]
    assert result["by_source_group"]["album a"] == result["overall"] == {
        "cases": 4, "correct_final": 2, "accepted": 3, "accepted_wrong": 2,
        "wrong_rotations": 1, "wrongly_rotated_upright": 0, "review": 1,
    }


@pytest.mark.parametrize("key", ["source_group", "image_sha256"])
def test_orientation_evaluation_rejects_overlapping_training_sources(key) -> None:
    from scripts.evaluate_photo_orientation import check_disjoint

    test = {"source_group": "album a", "image_sha256": "first"}
    train = {"source_group": "album b", "image_sha256": "second", key: test[key]}
    with pytest.raises(ValueError, match=key):
        check_disjoint([test], [train])


def rectangle(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def test_crop_metrics_are_perfect_for_identical_instances() -> None:
    from app.evaluation.cropping import evaluate_cropping_dataset, evaluate_page_crops

    truth = [rectangle(0.1, 0.1, 0.4, 0.5), rectangle(0.55, 0.2, 0.9, 0.8)]
    page = evaluate_page_crops(truth, truth, width=800, height=600, accepted=[True, False])
    dataset = evaluate_cropping_dataset([page])

    assert page.true_positive == 2
    assert page.serious_failures == 0
    assert page.boundary_f_score == pytest.approx(1.0)
    assert dataset.detection_f1 == pytest.approx(1.0)
    assert dataset.polygon_iou == pytest.approx(1.0)
    assert dataset.automatic_acceptance_rate == pytest.approx(0.5)
    assert dataset.human_review_rate == pytest.approx(0.5)


def test_crop_metrics_measure_content_cutoff_and_background_independently() -> None:
    from app.evaluation.cropping import evaluate_page_crops

    truth = [rectangle(0.2, 0.2, 0.8, 0.8)]
    too_tight = [rectangle(0.3, 0.2, 0.8, 0.8)]
    too_wide = [rectangle(0.1, 0.2, 0.8, 0.8)]

    tight = evaluate_page_crops(truth, too_tight, width=600, height=400)
    wide = evaluate_page_crops(truth, too_wide, width=600, height=400)

    assert tight.content_cutoff > 0.15
    assert tight.background_retained == pytest.approx(0.0)
    assert wide.content_cutoff == pytest.approx(0.0)
    assert wide.background_retained > 0.1
    assert tight.serious_failures == 1


def test_crop_metrics_do_not_reuse_one_merged_detection_for_two_photos() -> None:
    from app.evaluation.cropping import evaluate_page_crops

    truth = [rectangle(0.1, 0.2, 0.4, 0.7), rectangle(0.5, 0.2, 0.8, 0.7)]
    merged = [rectangle(0.1, 0.2, 0.8, 0.7)]

    page = evaluate_page_crops(truth, merged, width=700, height=500, iou_threshold=0.4)

    assert page.true_positive == 1
    assert page.false_negative == 1
    assert page.serious_failures >= 1


def test_matching_maximizes_threshold_qualified_pairs(monkeypatch) -> None:
    from app.evaluation import cropping

    truth = [rectangle(0.1, 0.1, 0.5, 0.5), rectangle(0.2, 0.2, 0.6, 0.6)]
    predictions = [rectangle(0.3, 0.3, 0.7, 0.7), rectangle(0.4, 0.4, 0.8, 0.8)]
    scores = [[0.99, 0.87], [0.86, 0.80]]

    def overlap(expected, detected, width, height, *, truth_mask=None):
        iou = scores[truth.index(expected)][predictions.index(detected)]
        return iou, iou, 0.0, 0.0

    monkeypatch.setattr(cropping, "_mask_overlap", overlap)
    page = cropping.evaluate_page_crops(truth, predictions, width=100, height=90)

    assert page.true_positive == 2
    assert [match["prediction_index"] for match in page.matches] == [1, 0]


@pytest.mark.parametrize("lower_visible", [False, True])
def test_native_instance_truth_counts_all_visible_components(lower_visible) -> None:
    import numpy as np

    from app.evaluation.cropping import evaluate_page_crops

    scale = np.array([99, 99])
    upper = (np.array(rectangle(20, 10, 69, 19)) / scale).tolist()
    lower = (np.array(rectangle(20, 70, 69, 79)) / scale).tolist()
    mask = np.zeros((100, 100), np.uint8)
    mask[10:20, 20:70] = mask[70:80, 20:70] = 255

    page = evaluate_page_crops([upper], [lower if lower_visible else upper], width=100, height=100, truth_masks=[mask])

    match, = page.matches
    assert match["mask_iou"] == match["content_cutoff"] == .5
    assert match["background_retained"] == 0
    assert match["corner_error"] is None
    assert page.false_negative == 1


def test_rendered_quality_measures_sampler_footprint_and_landmarks() -> None:
    import numpy as np
    from app.evaluation.cropping import evaluate_rendered_crop

    mask = np.zeros((70, 90), dtype=np.uint8)
    mask[10:50, 20:70] = 1
    corners = np.float32([[20, 10], [69, 10], [69, 49], [20, 49]])
    landmarks = np.float32([[9, 7], [33, 14], [15, 29]])
    source_landmarks = landmarks + [20, 10]
    matrix = np.float64([[1, 0, -20], [0, 1, -10], [0, 0, 1]])
    evidence = evaluate_rendered_crop(
        mask, corners / [89, 69], matrix, (40, 50),
        expected_corners=corners, expected_shape=(40, 50),
        source_landmarks=source_landmarks, expected_landmarks=landmarks,
    )

    assert evidence["footprint_iou"] == 1.0
    assert evidence["foreign_beyond_1px"] == evidence["cutoff_beyond_1px"] == 0
    assert evidence["outside_source_samples"] == 0
    assert evidence["landmark_error_p95_px"] == pytest.approx(0.0)
    assert evidence["aspect_ratio_relative_error"] == 0.0
    assert evidence["corner_error_max_px"] == pytest.approx(0.0, abs=1e-5)
    assert evidence["orientation_matches_reference"] is True

    rotate_half_turn = np.float64([[-1, 0, 49], [0, -1, 39], [0, 0, 1]])
    rotated = evaluate_rendered_crop(
        mask, corners / [89, 69], rotate_half_turn @ matrix, (40, 50),
        expected_corners=corners, expected_shape=(40, 50),
    )
    assert rotated["footprint_iou"] == 1.0
    assert rotated["orientation_matches_reference"] is False

    shifted = matrix.copy()
    shifted[0, 2] += 3.5
    error = evaluate_rendered_crop(mask, corners / [89, 69], shifted, (40, 50))
    assert error["cutoff_beyond_1px"] > 0
    assert error["foreign_beyond_1px"] > 0


def test_synthetic_references_include_visible_ownership_and_review_cases() -> None:
    import numpy as np
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    pages = {page.name: page for page in synthetic_crop_pages("locked")}
    assert not pages["negative"].prints
    assert pages["touching"].prints[0].reference_modes["full_print"].recoverable
    assert not pages["touching_same_color"].prints[0].reference_modes["full_print"].recoverable
    top, bottom = [item.reference_modes["full_print"] for item in reversed(pages["overlap"].prints)]
    assert top.recoverable and not bottom.recoverable
    assert not np.any(top.mask & bottom.mask)
    mount = pages["photo_corners"].prints[0].reference_modes
    assert not mount["full_print"].recoverable and not mount["image_only"].recoverable
    paper_mount = pages["paper_corner_mount"].prints[0].reference_modes
    assert not paper_mount["full_print"].recoverable and paper_mount["image_only"].recoverable


def test_dense_development_prints_have_two_exact_nonoverlapping_references() -> None:
    import numpy as np
    from app.evaluation.crop_fixtures import synthetic_crop_pages

    page = next(page for page in synthetic_crop_pages("development") if page.name == "dense_gutters")
    assert len(page.prints) == 30
    full_masks = []
    for item in page.prints:
        full, motif = (item.reference_modes[name] for name in ("full_print", "image_only"))
        assert full.recoverable and motif.recoverable
        assert np.count_nonzero(motif.mask) > 0
        assert np.all(motif.mask <= full.mask)
        full_masks.append(full.mask)
    assert np.max(np.sum(full_masks, axis=0)) == 1
    assert any(np.array_equal(item.reference_modes["full_print"].mask, item.reference_modes["image_only"].mask)
               for item in page.prints)
    assert any(np.any(item.reference_modes["full_print"].mask > item.reference_modes["image_only"].mask)
               for item in page.prints)


def test_benchmark_rectifies_tilt_without_generator_intrinsics(tmp_path) -> None:
    from app.config import Settings
    from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages
    from scripts.evaluate_onnx_segmentation import _render_crops, _rendered_quality

    settings = Settings(DATABASE_URL="sqlite+aiosqlite://", REDIS_URL="redis://localhost:6379/15",
                        ENABLE_PRECISE_PHOTO_CROPS=True, ENABLE_PHOTO_ORIENTATION_CORRECTION=False)
    page = next(page for page in synthetic_crop_pages("development") if page.name == "camera_tilt")
    print_fixture = page.prints[0]
    proposals = [(*geometry_seed(print_fixture, page.image.shape), True)]
    rendered = _render_crops(page.image, proposals, settings, tmp_path)
    assert rendered[0]["metadata"]["aspect_ratio_estimation"]["method"] == "estimated_pinhole_camera"
    reference = {mode: {**vars(ref), "verified": True} for mode, ref in print_fixture.reference_modes.items()}
    matches = [{"truth_index": 0, "prediction_index": 0, "mask_iou": 1.0, "valid_detection": True}]
    rows = _rendered_quality([reference], matches, rendered, page.difficulty, [{"full_print", "image_only"}])
    assert {row["mode"] for row in rows} == {"full_print", "image_only"}
    assert all(row["passes"] and row["aspect_ratio_relative_error"] <= .01
               and row["corner_error_max_px"] <= 2 for row in rows)


def test_benchmark_records_review_for_unresolved_mask_ownership(monkeypatch, tmp_path) -> None:
    import json
    import sys
    from pathlib import Path

    from app.evaluation import crop_fixtures
    from app.pipeline import segmentation
    from app.pipeline.crop_validation import CROP_VALIDATION_UNAVAILABLE
    from scripts import evaluate_onnx_segmentation as evaluator

    page = next(page for page in crop_fixtures.synthetic_crop_pages("locked") if page.name == "borderless")
    box, mask = crop_fixtures.geometry_seed(page.prints[0], page.image.shape)
    detection = segmentation.SegmentationDetection(box, mask, .95, 1.2, True, [])
    result = segmentation.SegmentationResult([detection], {"visible_mask_refinement": {"ownership_unresolved": True}})
    monkeypatch.setattr(segmentation, "segment_album_page", lambda *args, **kwargs: result)
    monkeypatch.setattr(crop_fixtures, "synthetic_crop_pages", lambda _: [page])
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    monkeypatch.setenv("ENABLE_PHOTO_ORIENTATION_CORRECTION", "false")
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"detector-fixture")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--synthetic", "locked", "--model", str(model),
                                      "--visible-mask-model", str(tmp_path / "local.onnx"), "--output", str(output)])

    evaluator.main()

    rows = json.loads(output.read_text(encoding="utf-8"))["fixtures"][0]["crop_modes"]
    assert {row["mode"] for row in rows} == {"full_print", "image_only"}
    assert all(row["status"] == "review_needed" and row["reason_codes"] == [CROP_VALIDATION_UNAVAILABLE]
               and not row["automatic_release_passes"] for row in rows)


def test_benchmark_uses_native_instance_mask_for_detection_and_refined_scores(monkeypatch, tmp_path) -> None:
    import hashlib
    import json
    import sys
    from pathlib import Path

    import cv2
    import numpy as np

    from app.pipeline import segmentation
    from scripts import evaluate_onnx_segmentation as evaluator

    raster = np.zeros((100, 100), np.uint8)
    assert cv2.imwrite(str(tmp_path / "page.png"), raster)
    raster[10:20, 20:70] = raster[70:80, 20:70] = 255
    mask_path = tmp_path / "full.png"
    assert cv2.imwrite(str(mask_path), raster)
    upper = np.array(rectangle(20, 10, 69, 19)) / 99
    (tmp_path / "labels.txt").write_text("0 " + " ".join(map(str, upper.ravel())), encoding="utf-8")
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({"pages": [{
        "id": "occluded", "image": "page.png", "annotations": "labels.txt", "expected_count": 1,
        "expected_final_crops": [{"full_print": {
            "mask": mask_path.name, "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "recoverable": False,
        }}],
    }]}), encoding="utf-8")
    polygon = [{"x": float(x), "y": float(y)} for x, y in upper]
    item = segmentation.SegmentationDetection({}, {"polygon": polygon}, .95, 1.2, True, [])
    monkeypatch.setattr(segmentation, "segment_album_page", lambda *_, **__: segmentation.SegmentationResult([item], {}))
    monkeypatch.setattr(evaluator, "_render_crops", lambda *_, **__: [{
        "images": {}, "modes": {}, "metadata": {"refined_visible_polygon": polygon},
    }])
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"detector-fixture")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--dataset-manifest", str(manifest), "--model", str(model),
                                      "--visible-mask-model", str(tmp_path / "local.onnx"), "--visuals",
                                      "--output", str(output)])

    with pytest.raises(SystemExit) as rejected:
        evaluator.main()

    assert rejected.value.code == 2
    result = json.loads(output.read_text(encoding="utf-8"))["fixtures"][0]
    assert result["native_instance_mask_count"] == 1
    for measurement in ("metrics", "primary_detector_metrics", "final_instance_metrics"):
        assert result[measurement]["false_negative"] == 1
        assert result[measurement]["mask_iou"] == .5


def test_album_benchmark_counts_independent_motif_downloads(tmp_path) -> None:
    from app.config import Settings
    from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages
    from app.pipeline.crop_validation import geometrically_ready_crop_modes
    from scripts.evaluate_onnx_segmentation import _render_crops, _rendered_quality

    settings = Settings(DATABASE_URL="sqlite+aiosqlite://", REDIS_URL="redis://localhost:6379/15",
                        ENABLE_PRECISE_PHOTO_CROPS=True, ENABLE_PHOTO_ORIENTATION_CORRECTION=False)
    page = next(page for page in synthetic_crop_pages("development") if page.name == "paper_corner_mount")
    print_fixture = page.prints[0]
    rendered = _render_crops(page.image, [(*geometry_seed(print_fixture, page.image.shape), True)], settings, tmp_path)
    modes = rendered[0]["modes"]
    assert modes["full_print"]["status"] == "review_needed"
    assert geometrically_ready_crop_modes(modes, image_only_present=False, image_only_is_full_print=False) == set()
    assert geometrically_ready_crop_modes(modes, image_only_present=True, image_only_is_full_print=True) == set()
    ready = geometrically_ready_crop_modes(modes, image_only_present=True, image_only_is_full_print=False)
    assert ready == {"image_only"}

    reference = {mode: {**vars(ref), "verified": True} for mode, ref in print_fixture.reference_modes.items()}
    matches = [{"truth_index": 0, "prediction_index": 0, "mask_iou": 1.0, "valid_detection": True}]
    rows = _rendered_quality([reference], matches, rendered, "real", [ready])
    assert {row["mode"]: row["automatic_release_passes"] for row in rows} == {"full_print": False, "image_only": True}
    assert not any(row["automatic_release_passes"] for row in _rendered_quality([reference], matches, rendered, "real", [set()]))


def test_subsampled_crop_cannot_hide_lost_source_pixels() -> None:
    import cv2
    import numpy as np
    from scripts.evaluate_onnx_segmentation import _rendered_quality

    reference = np.zeros((100, 100), dtype=np.uint8)
    reference[20:80, 20:80] = 1
    truth = np.float32(rectangle(20, 20, 79, 79))
    corner_templates = (np.float32(rectangle(23, 23, 76, 76)), truth)
    for source_corners in corner_templates:
        output_quad = np.float32(rectangle(0, 0, 10, 10))
        matrix = cv2.getPerspectiveTransform(source_corners, output_quad)
        geometry = {"status": "ready", "warp_quad": (source_corners / 99).tolist(), "page_to_output": matrix.tolist()}
        rendered = [{"modes": {"full_print": geometry}, "images": {"full_print": np.zeros((11, 11, 3), np.uint8)}}]
        refs = [{"full_print": {"mask": reference, "corners": truth, "shape": (60, 60), "verified": True, "recoverable": True}}]
        matches = [{"truth_index": 0, "prediction_index": 0, "mask_iou": 1.0, "valid_detection": True}]
        evidence = _rendered_quality(refs, matches, rendered, "real", [{"full_print"}])[0]

        assert evidence["cutoff_beyond_1px"] == (0 if source_corners is truth else 448)
        assert evidence["sampling_shortfall_px"] >= 43
        assert evidence["passes"] is False


def test_real_release_manifest_rejects_one_album_and_duplicate_captures() -> None:
    from app.evaluation.cropping import is_locked_album_manifest

    first = {"source_group": "album-a", "image_sha256": "a" * 64, "annotation_provenance": "manually_verified"}
    second = {"source_group": "album-b", "image_sha256": "b" * 64, "annotation_provenance": "manually_verified"}
    split = {"album_independent": True}
    assert is_locked_album_manifest([first, second], split)
    assert not is_locked_album_manifest([first, second | {"source_group": " Album-A "}], split)
    assert not is_locked_album_manifest([first, second | {"image_sha256": "a" * 64}], split)
    assert not is_locked_album_manifest([first, second | {"image_sha256": ""}], split)
    assert not is_locked_album_manifest([first, second | {"annotation_provenance": "draft"}], split)


def test_real_mode_requires_reviewed_corners_and_dimensions(tmp_path) -> None:
    from scripts.evaluate_onnx_segmentation import _reference_masks

    polygon = rectangle(0.1, 0.1, 0.8, 0.7)
    reference = {"recoverable": True, "polygon": polygon}
    record = {
        "id": "source", "annotation_provenance": "manually_verified",
        "expected_final_crops": [{"full_print": reference}],
    }

    def verified():
        return _reference_masks(record, [polygon], (70, 90, 3), tmp_path / "split.json")[0]["full_print"]["verified"]

    assert not verified()
    reference["annotation_provenance"] = "manually_verified"
    assert not verified()
    reference.update(warp_quad=polygon, shape=[43, 63])
    assert verified()


def test_measured_physical_aspect_is_evaluated_without_a_reference_pixel_grid(tmp_path) -> None:
    import numpy as np
    from scripts.evaluate_onnx_segmentation import _reference_masks, _rendered_quality

    polygon = rectangle(3 / 29, 5 / 19, 19 / 29, 13 / 19)
    annotation = {
        "polygon": polygon, "warp_quad": polygon, "recoverable": True, "annotation_provenance": "manually_verified",
        "physical_size_mm": {"width": 16, "height": 8},
    }
    record = {
        "id": "measured-print", "annotation_provenance": "manually_verified",
        "expected_final_crops": [{"full_print": annotation}],
    }
    refs = _reference_masks(record, [polygon], (20, 30, 3), tmp_path / "album.json")
    assert refs[0]["full_print"]["verified"] is True
    assert refs[0]["full_print"]["shape"] is None
    matrix = [[1, 0, -3], [0, 1, -5], [0, 0, 1]]
    rendered = [{
        "modes": {"full_print": {"status": "ready", "warp_quad": polygon, "page_to_output": matrix}},
        "images": {"full_print": np.zeros((9, 17, 3), np.uint8)},
    }]
    matches = [{"truth_index": 0, "prediction_index": 0, "mask_iou": 1.0, "valid_detection": True}]
    row = _rendered_quality(refs, matches, rendered, "real", [{"full_print"}])[0]
    assert row["aspect_reference"] == "physical_size_mm"
    assert row["reference_aspect_ratio"] == 2
    assert row["aspect_ratio_relative_error"] == 0
    assert row["automatic_release_passes"] is True


def test_paired_reference_image_is_pinned_and_consistent_with_measurements(tmp_path) -> None:
    from hashlib import sha256

    import cv2
    import numpy as np
    from scripts.evaluate_onnx_segmentation import _reference_masks

    path = tmp_path / "reference.png"
    bitmap = np.full((9, 17, 3), 192, dtype=np.uint8)
    path.write_bytes(cv2.imencode(".png", bitmap)[1].tobytes())
    polygon = rectangle(.1, .1, .75, .6)
    annotation = {
        "recoverable": True, "polygon": polygon, "warp_quad": polygon, "reference_image": path.name,
        "physical_size_mm": {"width": 16, "height": 8}, "annotation_provenance": "manually_verified",
    }
    record = {
        "id": "paired-print", "annotation_provenance": "manually_verified",
        "expected_final_crops": [{"full_print": annotation}],
    }

    def reference():
        return _reference_masks(record, [polygon], (70, 90, 3), tmp_path / "album.json",
                                include_reference_images=True)[0]["full_print"]

    assert reference()["sidecar_pinned"] is False
    annotation["reference_image_sha256"] = sha256(path.read_bytes()).hexdigest()
    loaded = reference()
    assert loaded["sidecar_pinned"] is True and loaded["shape"] == (9, 17)
    assert np.array_equal(loaded["image"], bitmap)
    annotation["landmarks"] = [[4, 3]]
    with pytest.raises(ValueError, match="paired landmarks"):
        reference()
    annotation["source_landmarks"] = [[17, 13]]
    assert reference()["source_landmarks"].tolist() == [[17, 13]]
    annotation["physical_size_mm"] = {"width": 18, "height": 8}
    with pytest.raises(ValueError, match="contradicts its physical dimensions"):
        reference()
    bitmap[0, 0] = 50
    path.write_bytes(cv2.imencode(".png", bitmap)[1].tobytes())
    with pytest.raises(ValueError, match="reference hash mismatch"):
        reference()


def test_paired_native_scan_is_warped_in_its_own_coordinates(tmp_path) -> None:
    from hashlib import sha256

    import cv2
    import numpy as np
    from scripts.evaluate_onnx_segmentation import _reference_masks

    path = tmp_path / "master.tif"
    raster = np.full((21, 31, 3), 215, np.uint8)
    raster[4:17, 5:26] = (35, 90, 150)
    path.write_bytes(cv2.imencode(".tif", raster)[1].tobytes())
    digest = sha256(path.read_bytes()).hexdigest()
    polygon = rectangle(.1, .1, .75, .6)
    scan_quad = rectangle(5 / 30, 4 / 20, 25 / 30, 16 / 20)
    annotation = {
        "recoverable": True, "polygon": polygon, "warp_quad": polygon, "shape": [13, 21],
        "reference_image": path.name, "reference_image_sha256": digest, "reference_image_quad": scan_quad,
    }
    record = {"id": "native-reference", "image_sha256": "f" * 64, "expected_final_crops": [{"full_print": annotation}]}

    def reference(*, render=False):
        return _reference_masks(record, [polygon], (70, 90, 3), tmp_path / "album.json",
                                include_reference_images=render)[0]["full_print"]

    assert reference()["shape"] == (13, 21)
    assert reference()["image"] is None
    assert reference()["sidecar_pinned"] is True
    assert np.array_equal(reference(render=True)["image"], raster[4:17, 5:26])

    annotation["reference_image_quad"] = scan_quad[::-1]
    with pytest.raises(ValueError, match="clockwise"):
        reference()

    record["image_sha256"] = digest
    annotation["reference_image_quad"] = scan_quad
    with pytest.raises(ValueError, match="separate scan"):
        reference()


def test_real_reference_mask_pin_detects_changed_bytes(tmp_path) -> None:
    from hashlib import sha256

    import cv2
    import numpy as np
    from scripts.evaluate_onnx_segmentation import _reference_masks

    path = tmp_path / "paper-mask.png"
    image = np.full((50, 60), 255, dtype=np.uint8)
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    polygon = rectangle(0.0, 0.0, 1.0, 1.0)
    annotation = {"recoverable": True, "mask": path.name, "warp_quad": polygon, "shape": [50, 60]}
    record = {"id": "paper", "expected_final_crops": [{"full_print": annotation}]}

    def reference():
        return _reference_masks(record, [polygon], (50, 60, 3), tmp_path / "album.json")[0]["full_print"]

    assert reference()["sidecar_pinned"] is False
    annotation["mask_sha256"] = sha256(path.read_bytes()).hexdigest()
    assert reference()["sidecar_pinned"] is True
    image[0, 0] = 0
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    with pytest.raises(ValueError, match="reference hash mismatch"):
        reference()


def test_real_release_requires_disjoint_calibration_album_groups(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from app.pipeline.confidence import CALIBRATION_SCHEMA
    from scripts.evaluate_onnx_segmentation import _has_disjoint_calibration_holdout

    path = tmp_path / "calibration.json"
    settings = SimpleNamespace(segmentation_calibration_path=path, require_confidence_calibration=True)
    records = [{"source_group": "Album-A"}, {"source_group": "Album-B"}]
    detector = {"model_sha256": "a" * 64}
    assert not _has_disjoint_calibration_holdout(settings, records, detector)
    settings.require_confidence_calibration = False
    assert not _has_disjoint_calibration_holdout(settings, records, detector)
    settings.require_confidence_calibration = True
    artifact = {
        "schema_version": CALIBRATION_SCHEMA, "independent_holdout_id": "calibration-only", "sample_count": 20,
        "source_groups": [" album-b "], "input_sha256": "a" * 64, "dataset_manifest_sha256": "b" * 64,
        "source_images": {"c" * 64: "album-b"},
        "detector_profile": detector, "strata": {"default": {"raw": [0, 1], "calibrated": [0, 1]}},
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert not _has_disjoint_calibration_holdout(settings, records, detector)
    path.write_text(json.dumps(artifact | {
        "source_groups": ["album-calibration"], "source_images": {"c" * 64: "album-calibration"},
    }), encoding="utf-8")
    assert _has_disjoint_calibration_holdout(settings, records, detector)
    settings.require_confidence_calibration = False
    assert not _has_disjoint_calibration_holdout(settings, records, detector)
    settings.require_confidence_calibration = True
    duplicate_page = {"source_group": "album-with-a-different-id", "image_sha256": "c" * 64}
    assert not _has_disjoint_calibration_holdout(settings, [*records, duplicate_page], detector)
    assert not _has_disjoint_calibration_holdout(settings, records, {"model_sha256": "e" * 64})


def test_calibration_export_and_fit_use_reviewed_album_predictions(monkeypatch, tmp_path) -> None:
    import hashlib
    import json
    import sys
    from pathlib import Path

    import cv2
    import numpy as np

    from app.pipeline import segmentation
    from app.pipeline.confidence import calibrate_confidence
    from scripts import evaluate_onnx_segmentation as evaluator, fit_confidence_calibration as fitter

    model = tmp_path / "detector.onnx"
    model.write_bytes(b"calibration-detector")
    model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"models": {model.name: {"sha256": model_digest, "license": "test"}}}), encoding="utf-8")
    truth = rectangle(.05, .1, .45, .5)
    labels = tmp_path / "page.txt"
    labels.write_text("0 " + " ".join(str(value) for point in truth for value in point), encoding="utf-8")
    pages = []
    for index in range(2):
        image = tmp_path / f"page-{index}.png"
        assert cv2.imwrite(str(image), np.full((100, 100, 3), 120 + index, np.uint8))
        pages.append({
            "id": image.stem, "image": image.name, "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "annotations": labels.name, "annotations_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "annotation_provenance": "manually_verified", "source_group": f"album-{index}", "expected_count": 1,
        })
    inference_calls = []

    def detect(*_, **__):
        inference_calls.append(True)
        detections = []
        for index in range(10):
            points = truth if index == 0 else rectangle(.6, .65, .95, .96)
            mask = {"polygon": [{"x": x, "y": y} for x, y in points], "detector_source": "yolo_onnx"}
            detections.append(segmentation.SegmentationDetection({}, mask, .95 if index == 0 else .3, 1.2, True, []))
        return segmentation.SegmentationResult(detections, {
            "model_artifact": {"sha256": model_digest},
            "source_contour_refinement": {"revision": segmentation.SOURCE_CONTOUR_REVISION},
        })

    monkeypatch.setattr(segmentation, "segment_album_page", detect)
    monkeypatch.setattr(evaluator, "_render_crops", lambda _, proposals, *__, **___: [
        {"images": {}, "modes": {}, "metadata": {}} for _ in proposals
    ])
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    manifest = tmp_path / "dataset.json"
    export = tmp_path / "scores.json"
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--dataset-manifest", str(manifest), "--model", str(model),
                                      "--manifest", str(registry), "--calibration-output", str(export),
                                      "--output", str(report)])
    split = {"album_independent": True, "role": "test"}
    manifest.write_text(json.dumps({"split_policy": split, "pages": pages}), encoding="utf-8")
    with pytest.raises(SystemExit, match="album-independent calibration split"):
        evaluator.main()
    assert inference_calls == [] and not export.exists()

    manifest.write_text(json.dumps({"split_policy": split | {"role": "calibration"}, "pages": pages}), encoding="utf-8")
    evaluator.main()
    rows = json.loads(export.read_bytes())
    assert len(rows["rows"]) == 20
    assert {row["score"] for row in rows["rows"] if row["correct"]} == {.95}
    assert sum(row["correct"] for row in rows["rows"]) == 2
    assert not json.loads(report.read_bytes())["validation_gates"]["locked_real_album_manifest"]

    output = tmp_path / "calibration.json"
    monkeypatch.setattr(sys, "argv", ["fit", str(export), "--holdout-id", "fixture", "--output", str(output)])
    assert fitter.main() == 0
    value, evidence = calibrate_confidence(.95, "yolo_onnx", output, detector_profile=rows["detector_profile"])
    assert evidence["calibrated"] and value == 1
    assert calibrate_confidence(.3, "yolo_onnx", output, detector_profile=rows["detector_profile"])[0] == 0


def test_real_mode_coverage_counts_only_eligible_passing_crops() -> None:
    from scripts.evaluate_onnx_segmentation import _rendered_mode_passes

    mode = {
        "obstructed_routed_to_review": 1, "obstructed": 1, "verified_ready_failures": 0,
        "verified": 11, "real": {"expected": 10, "passes": 10, "automatic_release_passes": 8},
        "real_by_group": {"album-a": {"expected": 8, "automatic_release_passes": 8}},
    }
    assert not _rendered_mode_passes(mode, synthetic=False, real_manifest_locked=True, expected_photos=11)
    mode["real"]["automatic_release_passes"] = 9
    mode["real_by_group"]["album-b"] = {"expected": 2, "automatic_release_passes": 1}
    assert not _rendered_mode_passes(mode, synthetic=False, real_manifest_locked=True, expected_photos=11)
    mode["real"]["automatic_release_passes"] = 10
    mode["real_by_group"]["album-b"]["automatic_release_passes"] = 2
    assert _rendered_mode_passes(mode, synthetic=False, real_manifest_locked=True, expected_photos=11)


def test_review_receipt_is_bound_to_manifest_and_cannot_confirm_missing_evidence(tmp_path) -> None:
    import json
    from argparse import Namespace
    from hashlib import sha256

    import cv2
    import numpy as np
    from app.pipeline.image_ops import load_image, sha256_file
    from scripts.evaluate_onnx_segmentation import _check_references, _load_cases, _write_reviewed_reference_manifest
    from scripts.validate_segmentation import parse_yolo_segmentation

    folder = tmp_path / "draft"
    folder.mkdir()
    image, labels = folder / "page.png", folder / "labels.txt"
    assert cv2.imwrite(str(image), np.full((42, 62, 3), 100, dtype=np.uint8))
    polygon = rectangle(.1, .1, .8, .8)
    labels.write_text("0 " + " ".join(str(value) for point in polygon for value in point), encoding="utf-8")
    mode = {"polygon": polygon, "warp_quad": polygon, "recoverable": True}
    source = folder / "manifest.json"
    source.write_text(json.dumps({"split_policy": {"album_independent": False}, "pages": [{
        "id": "page-a", "source_group": "album-a", "expected_count": 1,
        "image": image.name, "image_sha256": sha256_file(image),
        "annotations": labels.name, "annotations_sha256": sha256_file(labels),
        "expected_final_crops": [{"full_print": mode | {"physical_size_mm": {"width": 150, "height": 100}},
                                  "image_only": mode}],
    }]}), encoding="utf-8")
    args = Namespace(dataset_manifest=source)
    records, split = _load_cases(args)
    preflight = _check_references(args, records, split, parse_yolo_segmentation, load_image, sha256_file)
    page_choice = {"page_id": "page-a", "mode": "page", "instance_index": None, "outcome": "verified"}
    crop_choice = {"page_id": "page-a", "mode": "full_print", "instance_index": 0, "outcome": "verified"}
    receipt = {"schema_version": 1, "manifest_sha256": "0" * 64, "reviewer": "Fixture reviewer",
               "reviewed_at": "2026-11-23T12:00:00Z", "decisions": [page_choice, crop_choice]}
    receipt_path, reviewed = tmp_path / "receipt.json", tmp_path / "result" / "reviewed.json"

    def apply():
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        _write_reviewed_reference_manifest(source, receipt_path, reviewed, preflight)

    with pytest.raises(ValueError, match="does not match"):
        apply()
    receipt["manifest_sha256"] = sha256_file(source)
    receipt["decisions"].append(crop_choice | {"mode": "image_only"})
    with pytest.raises(ValueError, match="INDEPENDENT_ASPECT_MISSING"):
        apply()
    assert not reviewed.exists()
    receipt["decisions"][-1]["outcome"] = "needs_revision"
    apply()

    assert sha256_file(source) == receipt["manifest_sha256"]
    args.dataset_manifest = reviewed
    records, split = _load_cases(args)
    assert records[0]["image"].resolve() == image.resolve()
    modes = records[0]["expected_final_crops"][0]
    assert records[0]["annotation_provenance"] == "manually_verified"
    assert modes["full_print"]["reference_review"]["receipt_sha256"] == sha256(receipt_path.read_bytes()).hexdigest()
    assert modes["image_only"]["annotation_provenance"] == "reference_review_needs_revision"
    result = _check_references(args, records, split, parse_yolo_segmentation, load_image, sha256_file)
    assert result["modes"]["full_print"]["verified"] == 1
    assert result["modes"]["image_only"]["verified"] == 0
    assert result["ready_for_locked_evaluation"] is False
    with pytest.raises(FileExistsError):
        apply()


def test_reference_check_requires_verified_positive_modes_from_both_albums(tmp_path) -> None:
    import json
    from argparse import Namespace
    from hashlib import sha256

    import cv2
    import numpy as np
    from scripts.evaluate_onnx_segmentation import _check_references, _load_cases
    from scripts.validate_segmentation import parse_yolo_segmentation

    polygon = rectangle(.1, .1, .8, .8)
    reference = {
        "polygon": polygon, "warp_quad": polygon, "recoverable": True,
        "physical_size_mm": {"width": 150, "height": 100}, "annotation_provenance": "manually_verified",
    }
    labels = tmp_path / "pixels.txt"
    labels.write_text("0 " + " ".join(str(n) for pair in polygon for n in pair), encoding="utf-8")

    def checksum(path):
        return sha256(path.read_bytes()).hexdigest()

    pages = []
    for name, intensity in (("album-a", 90), ("album-b", 120)):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), np.full((42, 62, 3), intensity, dtype=np.uint8))
        pages.append({
            "id": name, "source_group": name, "image": path.name, "image_sha256": checksum(path),
            "annotations": labels.name, "annotations_sha256": checksum(labels), "expected_count": 1,
            "annotation_provenance": "manually_verified",
            "expected_final_crops": [{"full_print": reference, "image_only": dict(reference)}],
        })
    source = tmp_path / "albums.json"
    args = Namespace(dataset_manifest=source)

    def check(review_directory=None):
        source.write_text(json.dumps({"split_policy": {"album_independent": True}, "pages": pages}), encoding="utf-8")
        records, split = _load_cases(args)
        return _check_references(args, records, split, parse_yolo_segmentation,
                                 lambda path: cv2.imread(str(path)), checksum, review_directory=review_directory)

    complete = check()
    assert complete["ready_for_locked_evaluation"] is True
    assert complete["shared_verified_recoverable_groups"] == ["album-a", "album-b"]
    assert complete["modes"]["full_print"]["physical_dimensions"] == 2
    assert not any(complete["reference_issues"].values())
    draft = pages[1]["expected_final_crops"][0]["image_only"]
    draft["annotation_provenance"] = "draft"
    draft.pop("physical_size_mm")
    partial = check()
    assert partial["ready_for_locked_evaluation"] is False
    assert partial["modes"]["image_only"]["verified_recoverable_groups"] == ["album-a"]
    assert partial["reference_issues"]["modes_with_fewer_than_two_verified_recoverable_groups"] == ["image_only"]
    assert partial["reference_issues"]["modes"] == [{
        "page_id": "album-b", "instance_index": 0, "mode": "image_only",
        "issues": ["MODE_ANNOTATION_UNVERIFIED", "INDEPENDENT_ASPECT_MISSING"],
    }]
    detail = tmp_path / "independent-detail.png"
    reference_pixels = np.full((61, 91, 3), 91, dtype=np.uint8)
    assert cv2.imwrite(str(detail), reference_pixels)
    reference.update(reference_image=detail.name, reference_image_sha256=checksum(detail))
    review_directory = tmp_path / "review"
    review = check(review_directory)
    assert review["reference_issues"] == partial["reference_issues"]
    assert review["ready_for_locked_evaluation"] is False
    sheet = (review_directory / "index.html").read_text(encoding="utf-8")
    assert review["reference_review_sheet"] == str(review_directory / "index.html")
    assert "Ganzer Abzug" in sheet and "Nur Bildmotiv" in sheet and "INDEPENDENT_ASPECT_MISSING" in sheet
    assert 'href="../album-b.png"' in sheet
    assert 'href="#page-001"' in sheet and 'id="page-001"' in sheet
    assert np.array_equal(
        cv2.imread(str(review_directory / "page-000-0-full_print-independent.png")), reference_pixels,
    )
    assert (review_directory / "page-000-0-full_print-draft.jpg").is_file()
    corner = cv2.imread(str(review_directory / "page-001-0-image_only-corner-0.png"))
    left, right = np.hsplit(corner, 2)
    assert np.all(left == 120)
    assert np.any(right != left)
    pages[1].pop("image_sha256")
    pages[1].pop("expected_final_crops")
    unpinned = check()
    assert unpinned["reference_files_pinned"] is False
    assert unpinned["reference_issues"]["pages"][0]["issues"] == ["PAGE_IMAGE_HASH_MISSING"]
    assert unpinned["modes"]["full_print"]["present"] == 1
    assert {entry["mode"] for entry in unpinned["reference_issues"]["modes"]} == {"full_print", "image_only"}

    pages[1].update(image_sha256=checksum(tmp_path / "album-b.png"), expected_final_crops=[{
        "full_print": reference, "image_only": dict(reference, recoverable=False),
    }])
    third_image = tmp_path / "album-c.png"
    assert cv2.imwrite(str(third_image), np.full((42, 62, 3), 130, dtype=np.uint8))
    pages.append(dict(pages[1], id="album-c", source_group="album-c", image=third_image.name,
                      image_sha256=checksum(third_image), expected_final_crops=[{
                          "full_print": dict(reference, recoverable=False), "image_only": reference,
                      }]))
    separate_groups = check()
    assert separate_groups["shared_verified_recoverable_groups"] == ["album-a"]
    assert separate_groups["reference_issues"]["modes_with_fewer_than_two_verified_recoverable_groups"] == []
    assert separate_groups["reference_issues"]["dataset"] == ["FEWER_THAN_TWO_GROUPS_WITH_BOTH_MODES"]
    assert not separate_groups["ready_for_locked_evaluation"]
