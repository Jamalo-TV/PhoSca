import hashlib
import json
from pathlib import Path

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


def test_blur_score_distinguishes_sharp_from_blurry() -> None:
    from app.pipeline.image_ops import calculate_blur_score

    sharp = synthetic_album_page()
    blurry = cv2.GaussianBlur(sharp, (25, 25), 0)

    assert calculate_blur_score(sharp) > calculate_blur_score(blurry)


def test_classical_segmentation_detects_synthetic_photo() -> None:
    from app.pipeline.segmentation import detect_photos_classical

    result = detect_photos_classical(synthetic_album_page())

    assert result.metadata["model"] == "classical_contour"
    assert len(result.detections) >= 1
    detection = result.detections[0]
    assert detection.confidence >= 0.7
    assert detection.geometry_valid is True


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
    from app.pipeline.enhancement import enhance_photo
    from app.pipeline.image_ops import save_jpeg

    image = synthetic_album_page()
    image[130:170, 160:200] = (255, 255, 255)

    enhanced, metadata = enhance_photo(image)
    assert enhanced.shape == image.shape
    assert metadata["flash_recovery"] is True

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    save_jpeg(first, enhanced)
    save_jpeg(second, enhanced)
    assert hamming_distance(compute_phash(first), compute_phash(second)) == 0


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
    from app.models import Album, ExtractedPhoto, OCRResult, Page
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
        photos = (await session.scalars(select(ExtractedPhoto).where(ExtractedPhoto.page_id == page_id))).all()
        ocr_rows = (await session.scalars(select(OCRResult).where(OCRResult.page_id == page_id))).all()

    assert page is not None
    assert page.blur_score is not None
    assert photos
    assert Path(photos[0].storage_path).exists()
    assert photos[0].phash
    assert ocr_rows


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
