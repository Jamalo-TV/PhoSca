import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import piexif
from sqlalchemy import select


def synthetic_album_page() -> np.ndarray:
    image = np.full((420, 560, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (80, 100), (310, 280), (20, 20, 20), 4)
    cv2.rectangle(image, (88, 108), (302, 272), (70, 120, 180), -1)
    cv2.line(image, (180, 108), (210, 272), (255, 255, 255), 2)
    return image


def challenging_album_page() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    image = np.full((900, 1200, 3), 242, dtype=np.uint8)
    expected = [
        (70, 70, 450, 300),
        (680, 70, 450, 300),
        (70, 530, 450, 300),
        (680, 530, 450, 300),
    ]
    fills = [(70, 130, 185), (35, 90, 145), (205, 215, 220), (55, 85, 75)]
    for (x, y, w, h), color in zip(expected, fills, strict=True):
        cv2.rectangle(image, (x, y), (x + w, y + h), (250, 250, 250), -1)
        cv2.rectangle(image, (x + 8, y + 8), (x + w - 8, y + h - 8), color, -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (35, 35, 35), 3)

    # Strong internal geometry in the first print used to be split into
    # multiple "photos" by the contour fallback.
    cv2.rectangle(image, (95, 95), (265, 340), (25, 65, 120), -1)
    cv2.rectangle(image, (285, 95), (500, 340), (105, 160, 210), -1)

    # Handwritten captions should not be promoted into photo candidates.
    cv2.putText(image, "summer", (120, 455), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    cv2.putText(image, "family", (730, 455), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    return image, expected


def rotated_album_page() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((500, 700, 3), 242, dtype=np.uint8)
    center = (350, 250)
    size = (310, 190)
    angle = -17
    rect = (center, size, angle)
    points = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, points, (230, 230, 230))

    inner = (center, (size[0] - 20, size[1] - 20), angle)
    inner_points = cv2.boxPoints(inner).astype(np.int32)
    cv2.fillConvexPoly(image, inner_points, (50, 110, 175))
    cv2.polylines(image, [points], True, (35, 35, 35), 4)
    return image, points


def perspective_album_page() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((520, 720, 3), 242, dtype=np.uint8)
    points = np.array([[150, 100], [500, 70], [545, 340], [105, 365]], dtype=np.int32)
    inner = np.array([[167, 115], [486, 91], [525, 321], [125, 345]], dtype=np.int32)
    cv2.fillConvexPoly(image, points, (235, 235, 235))
    cv2.fillConvexPoly(image, inner, (45, 115, 180))
    cv2.line(image, tuple(inner[0]), tuple(inner[2]), (220, 235, 250), 3)
    cv2.polylines(image, [points], True, (35, 35, 35), 5)
    return image, points


def upright_portrait_photo() -> np.ndarray:
    image = np.full((220, 160, 3), 230, dtype=np.uint8)
    cv2.circle(image, (80, 55), 28, (70, 90, 130), -1)
    cv2.circle(image, (70, 48), 4, (250, 250, 250), -1)
    cv2.circle(image, (90, 48), 4, (250, 250, 250), -1)
    cv2.rectangle(image, (58, 88), (102, 170), (80, 120, 170), -1)
    cv2.line(image, (62, 118), (98, 118), (245, 245, 245), 2)
    return image


def low_contrast_album_page() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((620, 840, 3), 242, dtype=np.uint8)
    expected = (120, 150, 420, 280)
    x, y, w, h = expected
    cv2.rectangle(image, (x, y), (x + w, y + h), (218, 224, 226), -1)
    cv2.rectangle(image, (x + 30, y + 40), (x + 180, y + 230), (80, 140, 190), -1)
    cv2.circle(image, (x + 300, y + 150), 70, (45, 85, 70), -1)
    cv2.line(image, (x + 15, y + 260), (x + w - 20, y + 30), (230, 235, 240), 3)
    return image, expected


def _pixel_box(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(round(box["x1"] * width)),
        int(round(box["y1"] * height)),
        int(round(box["x2"] * width)),
        int(round(box["y2"] * height)),
    )


def _pixel_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def test_blur_score_distinguishes_sharp_from_blurry() -> None:
    from app.pipeline.image_ops import calculate_blur_score

    sharp = synthetic_album_page()
    blurry = cv2.GaussianBlur(sharp, (25, 25), 0)

    assert calculate_blur_score(sharp) > calculate_blur_score(blurry)


def test_classical_segmentation_detects_synthetic_photo() -> None:
    from app.pipeline.segmentation import detect_photos_classical

    result = detect_photos_classical(synthetic_album_page())

    assert result.metadata["model"] == "classical_hybrid_quad"
    assert len(result.detections) >= 1
    detection = result.detections[0]
    assert detection.confidence >= 0.7
    assert detection.geometry_valid is True


def test_classical_segmentation_consolidates_real_photo_regions() -> None:
    from app.pipeline.segmentation import detect_photos_classical

    image, expected_boxes = challenging_album_page()
    result = detect_photos_classical(image)
    height, width = image.shape[:2]
    detected_boxes = [_pixel_box(detection.bounding_box, width, height) for detection in result.detections]
    expected_xyxy = [(x, y, x + w, y + h) for x, y, w, h in expected_boxes]

    assert len(result.detections) == 4
    assert all(max(_pixel_iou(expected, detected) for detected in detected_boxes) > 0.65 for expected in expected_xyxy)


def test_classical_segmentation_detects_low_contrast_photo_region() -> None:
    from app.pipeline.segmentation import detect_photos_classical

    image, expected_box = low_contrast_album_page()
    result = detect_photos_classical(image)
    height, width = image.shape[:2]
    detected_boxes = [_pixel_box(detection.bounding_box, width, height) for detection in result.detections]
    expected_xyxy = (expected_box[0], expected_box[1], expected_box[0] + expected_box[2], expected_box[1] + expected_box[3])

    assert len(result.detections) == 1
    assert result.detections[0].mask["detector_source"] == "contrast_region"
    assert result.detections[0].mask["scores"]["area_ratio"] > 0.2
    assert _pixel_iou(expected_xyxy, detected_boxes[0]) > 0.85


def test_yolo_segmentation_parser_uses_proto_mask_polygon() -> None:
    from app.pipeline.segmentation import _parse_simple_box_outputs

    settings = SimpleNamespace(
        yolo_confidence_threshold=0.25,
        segmentation_min_aspect_ratio=0.2,
        segmentation_max_aspect_ratio=5.0,
    )
    prediction = np.array([[[8.0, 8.0, 8.0, 8.0, 0.91, 1.0]]], dtype=np.float32)
    proto = np.full((1, 1, 16, 16), -8.0, dtype=np.float32)
    proto[0, 0, 4:12, 5:11] = 8.0

    detections = _parse_simple_box_outputs(
        [prediction, proto],
        width=4000,
        height=3000,
        settings=settings,
        input_width=16,
        input_height=16,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.mask["source"] == "yolo_seg_mask"
    assert detection.mask["scores"]["mask_channels"] == 1.0
    assert 0.30 < detection.bounding_box["x1"] < 0.33
    assert 0.67 < detection.bounding_box["x2"] < 0.70


def test_yolo_box_parser_normalizes_coordinates_against_model_input() -> None:
    from app.pipeline.segmentation import _parse_simple_box_outputs

    settings = SimpleNamespace(
        yolo_confidence_threshold=0.25,
        segmentation_min_aspect_ratio=0.2,
        segmentation_max_aspect_ratio=5.0,
    )
    prediction = np.array([[[160.0, 96.0, 480.0, 384.0, 0.93]]], dtype=np.float32)

    detections = _parse_simple_box_outputs(
        [prediction],
        width=4000,
        height=3000,
        settings=settings,
        input_width=640,
        input_height=480,
    )

    assert len(detections) == 1
    box = detections[0].bounding_box
    assert 0.24 < box["x1"] < 0.26
    assert 0.79 < box["y2"] < 0.81




def test_yolo_box_parser_undoes_letterbox_padding() -> None:
    from app.pipeline.segmentation import _parse_simple_box_outputs

    settings = SimpleNamespace(
        yolo_confidence_threshold=0.25,
        segmentation_min_aspect_ratio=0.2,
        segmentation_max_aspect_ratio=5.0,
    )
    prediction = np.array([[[160.0, 240.0, 480.0, 400.0, 0.93]]], dtype=np.float32)

    detections = _parse_simple_box_outputs(
        [prediction],
        width=4000,
        height=2000,
        settings=settings,
        input_width=640,
        input_height=640,
    )

    assert len(detections) == 1
    box = detections[0].bounding_box
    assert 0.24 < box["x1"] < 0.26
    assert 0.24 < box["y1"] < 0.26
    assert 0.74 < box["y2"] < 0.76

def test_yolo_segmentation_parser_accepts_channels_last_proto() -> None:
    from app.pipeline.segmentation import _parse_simple_box_outputs

    settings = SimpleNamespace(
        yolo_confidence_threshold=0.25,
        segmentation_min_aspect_ratio=0.2,
        segmentation_max_aspect_ratio=5.0,
    )
    prediction = np.array([[[8.0, 8.0, 8.0, 8.0, 0.91, 1.0]]], dtype=np.float32)
    proto = np.full((1, 16, 16, 1), -8.0, dtype=np.float32)
    proto[0, 4:12, 5:11, 0] = 8.0

    detections = _parse_simple_box_outputs(
        [prediction, proto],
        width=4000,
        height=3000,
        settings=settings,
        input_width=16,
        input_height=16,
    )

    assert len(detections) == 1
    assert detections[0].mask["source"] == "yolo_seg_mask"


def test_classical_segmentation_rejects_smooth_saturated_sliver() -> None:
    from app.pipeline.segmentation import detect_photos_classical

    image = np.full((600, 900, 3), 242, dtype=np.uint8)
    cv2.rectangle(image, (360, 120), (560, 570), (210, 135, 40), -1)

    result = detect_photos_classical(image)

    assert result.detections == []
    assert result.metadata["rejected_candidates"]["smooth_contrast_false_positive"] == 1


def test_classical_segmentation_rejects_nested_partial_photo_candidate() -> None:
    from app.pipeline.segmentation import SegmentationDetection, _filter_nested_classical_detections

    full_photo = SegmentationDetection(
        bounding_box={"x1": 0.15, "y1": 0.18, "x2": 0.72, "y2": 0.76},
        mask={
            "polygon": [],
            "scores": {
                "area_ratio": 0.33,
                "outline_supported_sides": 4.0,
                "exterior_background_sides": 4.0,
            },
        },
        confidence=0.82,
        aspect_ratio=1.4,
        geometry_valid=True,
        review_reasons=[],
    )
    internal_crop = SegmentationDetection(
        bounding_box={"x1": 0.19, "y1": 0.25, "x2": 0.39, "y2": 0.69},
        mask={
            "polygon": [],
            "scores": {
                "area_ratio": 0.09,
                "outline_supported_sides": 3.0,
                "exterior_background_sides": 0.0,
            },
        },
        confidence=0.78,
        aspect_ratio=0.7,
        geometry_valid=True,
        review_reasons=[],
    )

    kept, rejected = _filter_nested_classical_detections([full_photo, internal_crop])

    assert kept == [full_photo]
    assert rejected == {"partial_photo_inside_larger_candidate": 1}


def test_classical_segmentation_preserves_rotated_photo_geometry() -> None:
    from app.pipeline.perspective import crop_and_correct_photo
    from app.pipeline.segmentation import detect_photos_classical

    image, _ = rotated_album_page()
    result = detect_photos_classical(image)

    assert len(result.detections) == 1
    polygon = result.detections[0].mask["polygon"]
    assert len(polygon) == 4
    assert result.detections[0].mask["source"] == "classical_border_quad"

    crop = crop_and_correct_photo(image, result.detections[0].bounding_box, result.detections[0].mask)
    height, width = crop.shape[:2]
    assert width > height
    assert 1.45 < width / height < 1.8


def test_classical_segmentation_detects_perspective_photo_corners() -> None:
    from app.pipeline.perspective import crop_and_correct_photo
    from app.pipeline.segmentation import detect_photos_classical

    image, _ = perspective_album_page()
    result = detect_photos_classical(image)

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.mask["source"] == "classical_border_quad"
    points = np.array([[point["x"] * image.shape[1], point["y"] * image.shape[0]] for point in detection.mask["polygon"]])
    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[2] - points[3])
    assert abs(top_width - bottom_width) > 45

    crop = crop_and_correct_photo(image, detection.bounding_box, detection.mask)
    height, width = crop.shape[:2]
    assert width > height
    assert 1.45 < width / height < 1.85


def test_photo_orientation_correction_rotates_confident_sideways_portrait() -> None:
    from app.pipeline.orientation import correct_photo_orientation, rotate_image

    sideways = rotate_image(upright_portrait_photo(), 90)
    corrected, metadata = correct_photo_orientation(sideways)

    assert metadata["rotation_degrees"] == 270
    assert corrected.shape[:2] == (220, 160)


def test_uniform_border_removal_trims_page_margin() -> None:
    from app.pipeline.border_removal import remove_uniform_border

    image = np.full((140, 220, 3), 245, dtype=np.uint8)
    image[20:120, 30:190] = (45, 95, 150)
    cv2.line(image, (55, 30), (160, 105), (230, 230, 230), 3)

    trimmed = remove_uniform_border(image, max_crop_ratio=0.2)

    assert trimmed.shape[0] < image.shape[0]
    assert trimmed.shape[1] < image.shape[1]
    assert trimmed.shape[0] >= 95
    assert trimmed.shape[1] >= 155


def test_cutout_prep_is_light_and_geometry_preserving() -> None:
    from app.pipeline.enhancement import prepare_for_cutout

    image, _ = low_contrast_album_page()

    prepared, metadata = prepare_for_cutout(image)

    assert prepared.shape == image.shape
    assert prepared.dtype == image.dtype
    assert metadata["order"] == "before_segmentation_and_crop"
    assert metadata["geometry_preserved"] is True
    assert "edge_preserving_bilateral" in metadata["steps"]


def test_segmentation_quality_gate_flags_distorted_geometry() -> None:
    from app.pipeline.quality import evaluate_segmentation_geometry, segmentation_review_reasons

    too_wide = evaluate_segmentation_geometry(400, 50)
    invalid = evaluate_segmentation_geometry(0, 50)

    assert too_wide.geometry_valid is False
    assert too_wide.review_reasons == ["aspect_ratio_too_wide"]
    assert invalid.review_reasons == ["invalid_detection_dimensions"]
    assert segmentation_review_reasons(confidence=0.65, confidence_threshold=0.7, quality=too_wide) == [
        "aspect_ratio_too_wide",
        "segmentation_confidence_below_threshold",
    ]


def test_ocr_spatial_classification_links_caption_to_photo() -> None:
    from app.pipeline.ocr import OCRBlock, classify_ocr_blocks

    blocks = [
        OCRBlock(
            text="Summer picnic",
            bounding_box={"x1": 0.18, "y1": 0.25, "x2": 0.45, "y2": 0.31},
            confidence=0.91,
            engine="sidecar",
        ),
        OCRBlock(
            text="1998",
            bounding_box={"x1": 0.1, "y1": 0.02, "x2": 0.25, "y2": 0.08},
            confidence=0.99,
            engine="sidecar",
        ),
    ]
    photos = [{"id": "11111111-1111-1111-1111-111111111111", "bounding_box": {"x1": 0.14, "y1": 0.23, "x2": 0.56, "y2": 0.67}}]

    classified = classify_ocr_blocks(blocks, photos)

    assert classified[0].text_type.value == "caption"
    assert classified[0].photo_id == "11111111-1111-1111-1111-111111111111"
    assert classified[1].text_type.value == "directory_name"


def test_enhancement_and_phash_primitives(tmp_path: Path) -> None:
    from app.pipeline.dedup import compute_phash, hamming_distance
    from app.pipeline.dl_enhancement import dl_enhance_photo
    from app.pipeline.image_ops import save_jpeg
    from unittest.mock import patch

    image = synthetic_album_page()
    image[130:170, 160:200] = (255, 255, 255)

    with patch("app.pipeline.dl_enhancement.get_dl_models") as mock_models, \
         patch("app.pipeline.inpainting.inpaint_scratches") as mock_inpaint:

        # Mock inpainting
        mock_inpaint.return_value = (image, {"scratch_removal": True})

        # Mock face enhancer
        class MockFaceEnhancer:
            def enhance(self, img, **kwargs):
                return None, None, img  # Just return the same image

        mock_models.return_value = (None, MockFaceEnhancer())

        enhanced, metadata = dl_enhance_photo(image)
        assert enhanced.shape == image.shape
        assert "LaMa" in metadata["models"]

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    save_jpeg(first, enhanced)
    save_jpeg(second, enhanced)
    assert hamming_distance(compute_phash(first), compute_phash(second)) == 0


def test_degradation_analysis_recommends_non_destructive_preset() -> None:
    from app.pipeline.degradation import analyze_degradation

    image = np.full((160, 220, 3), (185, 205, 235), dtype=np.uint8)
    noise = np.random.default_rng(42).normal(0, 12, image.shape).astype(np.int16)
    aged = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.line(aged, (20, 20), (190, 130), (250, 250, 250), 2)

    report = analyze_degradation(aged)

    assert report.overall_severity in {"moderate", "severe"}
    assert report.recommended_preset in {"balanced", "aggressive"}
    assert report.yellowing_score > 0
    assert report.dynamic_range >= 0
    assert report.should_denoise is True
    assert report.should_correct_color is True
    assert report.denoise_strength_recommended > 0


def test_variant_bundle_records_paths_metrics_and_selection(tmp_path: Path) -> None:
    from app.pipeline.image_ops import save_jpeg
    from app.pipeline.variants import build_variant_bundle

    original = synthetic_album_page()
    enhanced = cv2.resize(original, (original.shape[1] * 2, original.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    premium = enhanced.copy()
    original_path = tmp_path / "original.jpg"
    enhanced_path = tmp_path / "enhanced.jpg"
    premium_path = tmp_path / "premium.jpg"
    save_jpeg(original_path, original)
    save_jpeg(enhanced_path, enhanced)
    save_jpeg(premium_path, premium)

    bundle = build_variant_bundle(
        original_path,
        original,
        enhanced_path,
        enhanced,
        {"method": "deep_learning", "models": ["mock"], "config": {"preset": "balanced"}},
        premium_path,
        premium,
        {"method": "premium_preservation_restoration", "models": ["mock"]},
    )

    assert bundle["selected_variant"] == "enhanced"
    assert set(bundle["variants"]) == {"original", "enhanced", "premium"}
    assert bundle["variants"]["enhanced"]["selected"] is True
    assert bundle["variants"]["premium"]["metrics"]["shape"]["width"] == original.shape[1] * 2


def test_scratch_mask_does_not_target_normal_photo_edges() -> None:
    from app.pipeline.inpainting import generate_scratch_mask

    image = synthetic_album_page()
    mask = generate_scratch_mask(image)

    assert cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1]) < 0.01


def test_scratch_mask_detects_thin_line_defect() -> None:
    from app.pipeline.inpainting import generate_scratch_mask

    image = np.full((180, 240, 3), 150, dtype=np.uint8)
    cv2.line(image, (35, 25), (205, 155), (245, 245, 245), 1)

    mask = generate_scratch_mask(image, sensitivity=0.85)

    assert cv2.countNonZero(mask) > 40


def test_quality_assessor_rejects_obvious_overprocessing() -> None:
    from app.pipeline.quality_assessment import QualityAssessor

    image = synthetic_album_page()
    clipped = cv2.convertScaleAbs(image, alpha=3.0, beta=80)
    assessor = QualityAssessor()
    delta = assessor.compare(image, clipped)

    assert delta.clipping_delta > 0.05
    assert assessor.should_revert(delta) is True


def test_premium_enhancement_preserves_source_structure() -> None:
    from app.pipeline.diffusion_restoration import premium_enhance_photo
    from unittest.mock import patch

    image = synthetic_album_page()
    upscaled = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    with patch("app.pipeline.diffusion_restoration.dl_enhance_photo", return_value=(upscaled, {"models": ["mock-ai"]})):
        premium, metadata = premium_enhance_photo(image)

    assert premium.shape == upscaled.shape
    assert metadata["method"] == "premium_preservation_restoration"
    assert metadata["diffusion_used"] is False


def test_exif_caption_write_strips_sensitive_tags(tmp_path: Path) -> None:
    from app.pipeline.exif import write_caption_exif
    from app.pipeline.image_ops import save_jpeg

    path = tmp_path / "photo.jpg"
    save_jpeg(path, synthetic_album_page())
    exif = {
        "0th": {
            piexif.ImageIFD.Make: b"CameraCo",
            piexif.ImageIFD.Model: b"SecretModel",
        },
        "Exif": {},
        "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"N"},
        "1st": {},
        "thumbnail": None,
    }
    piexif.insert(piexif.dump(exif), str(path))

    metadata = write_caption_exif(path, "Summer picnic")
    loaded = piexif.load(str(path))

    assert metadata["gps_stripped"] is True
    assert loaded["GPS"] == {}
    assert piexif.ImageIFD.Make not in loaded["0th"]
    assert loaded["0th"][piexif.ImageIFD.ImageDescription] == b"Summer picnic"


async def test_full_synthetic_page_pipeline(client) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, AlbumStatus, ExtractedPhoto, OCRResult, Page
    from app.services.pipeline import process_page_pipeline

    settings = get_settings()
    album = Album(name="Synthetic Album", description=None)
    page_image = synthetic_album_page()

    async with get_session_factory()() as session:
        session.add(album)
        await session.flush()
        storage_dir = settings.storage_path / "uploads" / str(album.id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        page_path = storage_dir / "page.jpg"
        cv2.imwrite(str(page_path), page_image)
        page_path.with_suffix(".ocr.json").write_text(
            json.dumps(
                {
                    "blocks": [
                        {
                            "text": "Summer picnic",
                            "bounding_box": {"x1": 0.17, "y1": 0.28, "x2": 0.48, "y2": 0.35},
                            "confidence": 0.95,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        page = Page(
            album_id=album.id,
            original_filename="page.jpg",
            storage_path=str(page_path),
            file_size_bytes=page_path.stat().st_size,
            file_hash_sha256=hashlib.sha256(page_path.read_bytes()).hexdigest(),
        )
        session.add(page)
        album.total_pages = 1
        await session.commit()
        page_id = page.id

    result = await process_page_pipeline(page_id, settings)

    assert result["status"] in {"completed", "review_needed"}
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        album = await session.get(Album, page.album_id) if page is not None else None
        photos = (await session.scalars(select(ExtractedPhoto).where(ExtractedPhoto.page_id == page_id))).all()
        ocr_rows = (await session.scalars(select(OCRResult).where(OCRResult.page_id == page_id))).all()

    assert page is not None
    assert album is not None
    assert album.status == AlbumStatus.completed
    assert album.processed_pages == album.total_pages == 1
    assert page.blur_score is not None
    assert page.processing_metadata["steps"]["pre_cutout_enhancement"]["order"] == "before_segmentation_and_crop"
    assert photos
    assert Path(photos[0].storage_path).exists()
    assert photos[0].phash
    assert ocr_rows


async def test_photo_image_serves_variant_metadata_path(client) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, ExtractedPhoto, Page
    from app.pipeline.image_ops import save_jpeg

    settings = get_settings()
    image = synthetic_album_page()
    premium = np.full_like(image, (10, 80, 160))

    async with get_session_factory()() as session:
        album = Album(name="Variant Album", description=None)
        session.add(album)
        await session.flush()
        photo_dir = settings.storage_path / "processed" / str(album.id) / "photos"
        upload_dir = settings.storage_path / "uploads" / str(album.id)
        photo_dir.mkdir(parents=True, exist_ok=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        page_path = upload_dir / "page.jpg"
        original_path = photo_dir / "photo_original.jpg"
        enhanced_path = photo_dir / "photo_enhanced.jpg"
        premium_path = photo_dir / "non_conventional_premium_name.jpg"
        save_jpeg(page_path, image)
        save_jpeg(original_path, image)
        save_jpeg(enhanced_path, image)
        save_jpeg(premium_path, premium)
        page = Page(
            album_id=album.id,
            original_filename="page.jpg",
            storage_path=str(page_path),
            file_size_bytes=page_path.stat().st_size,
            file_hash_sha256=hashlib.sha256(page_path.read_bytes()).hexdigest(),
        )
        session.add(page)
        await session.flush()
        photo = ExtractedPhoto(
            page_id=page.id,
            album_id=album.id,
            storage_path=str(enhanced_path),
            original_storage_path=str(original_path),
            bounding_box={"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9},
            enhancement_applied={
                "selected_variant": "enhanced",
                "variants": {
                    "original": {"path": str(original_path)},
                    "enhanced": {"path": str(enhanced_path)},
                    "premium": {"path": str(premium_path)},
                },
            },
        )
        session.add(photo)
        await session.commit()
        photo_id = photo.id

    detail_response = await client.get(f"/api/v1/photos/{photo_id}")
    assert detail_response.status_code == 200, detail_response.text
    assert detail_response.json()["urls"]["premium"].endswith("variant=premium")

    image_response = await client.get(f"/api/v1/photos/{photo_id}/image?variant=premium")
    assert image_response.status_code == 200, image_response.text
    assert image_response.content.startswith(b"\xff\xd8")


async def test_failed_page_pipeline_writes_dlq_manifest(client) -> None:
    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Album, AuditLog, Page, PageStatus
    from app.services.dlq import failed_page_dlq_path
    from app.services.pipeline import process_pages_pipeline

    settings = get_settings()
    album = Album(name="Broken Album", description=None)
    page_image = synthetic_album_page()

    async with get_session_factory()() as session:
        session.add(album)
        await session.flush()
        storage_dir = settings.storage_path / "uploads" / str(album.id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        page_path = storage_dir / "page.jpg"
        cv2.imwrite(str(page_path), page_image)
        page = Page(
            album_id=album.id,
            original_filename="page.jpg",
            storage_path=str(page_path),
            file_size_bytes=page_path.stat().st_size,
            file_hash_sha256="0" * 64,
        )
        session.add(page)
        album.total_pages = 1
        await session.commit()
        page_id = page.id

    result = await process_pages_pipeline([page_id], settings)

    assert result["pages"][0]["status"] == "failed"
    dlq_dir = failed_page_dlq_path(settings, page_id)
    assert (dlq_dir / "manifest.json").exists()
    assert (dlq_dir / "page.jpg").exists()

    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        audit_rows = (await session.scalars(select(AuditLog).where(AuditLog.entity_id == page_id))).all()

    assert page is not None
    assert page.status == PageStatus.failed
    assert page.processing_metadata["steps"]["ingestion"]["dlq_manifest_path"] == str(dlq_dir / "manifest.json")
    assert any(row.action == "failed" and row.details.get("dlq_manifest_path") for row in audit_rows)
