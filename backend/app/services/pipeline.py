import asyncio
from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.database import get_session_factory
from app.models import Album, AlbumStatus, AuditLog, ExtractedPhoto, OCRResult, OCRTextType, Page, PageStatus, PhotoStatus
from app.pipeline.dedup import compute_phash, hamming_distance
from app.pipeline.degradation import analyze_degradation
from app.pipeline.dl_enhancement import dl_enhance_photo
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.exif import write_caption_exif
from app.pipeline.image_ops import calculate_blur_score, load_image, save_jpeg, sha256_file
from app.pipeline.ocr import classify_ocr_blocks, run_paddle_ocr, run_sidecar_ocr
from app.pipeline.perspective import crop_and_correct_photo
from app.pipeline.quality import segmentation_review_reasons
from app.pipeline.segmentation import segment_album_page
from app.pipeline.variants import build_variant_bundle
from app.services.dlq import quarantine_page_failure


logger = structlog.get_logger(__name__)


def _append_metadata(entity: Page | ExtractedPhoto, step: str, payload: dict) -> None:
    if isinstance(entity, Page):
        metadata = deepcopy(entity.processing_metadata or {})
        steps = dict(metadata.get("steps") or {})
        steps[step] = payload
        metadata["steps"] = steps
        entity.processing_metadata = metadata
        return
    metadata = deepcopy(entity.enhancement_applied or {})
    metadata[step] = payload
    entity.enhancement_applied = metadata


async def ingestion_task_logic(session: AsyncSession, page_id: UUID, settings: Settings) -> dict:
    page = await session.get(Page, page_id)
    if page is None:
        raise ValueError(f"Page not found: {page_id}")

    path = Path(page.storage_path)
    actual_hash = sha256_file(path)
    if actual_hash != page.file_hash_sha256:
        page.status = PageStatus.failed
        page.error_message = "SHA-256 verification failed."
        dlq = quarantine_page_failure(page, settings, reason="sha256_verification_failed", details={"actual_sha256": actual_hash})
        _append_metadata(page, "ingestion", {"hash_verified": False, "actual_sha256": actual_hash, **dlq})
        session.add(AuditLog(entity_type="page", entity_id=page.id, action="failed", details={"reason": page.error_message}))
        await session.commit()
        logger.warning("page_ingestion_failed", page_id=str(page.id), reason="sha256_verification_failed", **dlq)
        raise ValueError(page.error_message)

    image = load_image(path)
    blur_score = calculate_blur_score(image)
    page.blur_score = blur_score
    page.status = PageStatus.review_needed if blur_score < settings.blur_threshold else PageStatus.queued
    _append_metadata(
        page,
        "ingestion",
        {
            "hash_verified": True,
            "blur_score": blur_score,
            "blur_threshold": settings.blur_threshold,
            "review_needed": blur_score < settings.blur_threshold,
        },
    )
    session.add(AuditLog(entity_type="page", entity_id=page.id, action="ingested", details={"blur_score": blur_score}))
    await session.commit()
    return {"page_id": str(page.id), "status": page.status.value, "blur_score": blur_score}


async def segmentation_task_logic(session: AsyncSession, page_id: UUID, settings: Settings) -> dict:
    page = await session.get(Page, page_id)
    if page is None:
        raise ValueError(f"Page not found: {page_id}")

    page.status = PageStatus.processing
    image = load_image(Path(page.storage_path))
    cutout_ready_image, cutout_prep_metadata = prepare_for_cutout(image)
    _append_metadata(page, "pre_cutout_enhancement", cutout_prep_metadata)
    result = segment_album_page(cutout_ready_image, settings)
    _append_metadata(page, "segmentation", result.metadata | {"detections": len(result.detections)})

    created_photo_ids: list[str] = []
    review_reasons_by_photo: dict[str, list[str]] = {}
    for detection in result.detections:
        review_reasons = segmentation_review_reasons(
            confidence=detection.confidence,
            confidence_threshold=settings.yolo_confidence_threshold,
            quality=detection,
        )
        photo_status = PhotoStatus.review_needed if review_reasons else PhotoStatus.pending
        photo = ExtractedPhoto(
            id=uuid4(),
            page_id=page.id,
            album_id=page.album_id,
            storage_path="",
            original_storage_path=None,
            bounding_box=detection.bounding_box,
            segmentation_mask=detection.mask,
            segmentation_confidence=detection.confidence,
            aspect_ratio=detection.aspect_ratio,
            geometry_valid=detection.geometry_valid,
            enhancement_applied={"review_reasons": review_reasons} if review_reasons else {},
            status=photo_status,
        )
        session.add(photo)
        created_photo_ids.append(str(photo.id))
        if review_reasons:
            review_reasons_by_photo[str(photo.id)] = review_reasons

    if not result.detections:
        page.status = PageStatus.review_needed
        page.error_message = "No photos detected; manual review required."
        session.add(AuditLog(entity_type="page", entity_id=page.id, action="review_flagged", details={"reason": page.error_message}))
    else:
        page.status = PageStatus.queued
        details = {"photo_ids": created_photo_ids, "review_reasons_by_photo": review_reasons_by_photo}
        session.add(AuditLog(entity_type="page", entity_id=page.id, action="segmented", details=details))
        logger.info(
            "page_segmented",
            page_id=str(page.id),
            detections=len(result.detections),
            review_photos=len(review_reasons_by_photo),
            model=result.metadata.get("model"),
        )

    await session.commit()
    return {"page_id": str(page.id), "photo_ids": created_photo_ids, "metadata": result.metadata}


async def perspective_correction_task_logic(session: AsyncSession, photo_id: UUID, settings: Settings) -> dict:
    photo = await session.scalar(
        select(ExtractedPhoto)
        .options(selectinload(ExtractedPhoto.page))
        .where(ExtractedPhoto.id == photo_id)
    )
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")

    page_image = load_image(Path(photo.page.storage_path))
    corrected = crop_and_correct_photo(page_image, photo.bounding_box, photo.segmentation_mask)
    original_path = settings.storage_path / "processed" / str(photo.album_id) / "photos" / f"{photo.id}_original.jpg"
    save_jpeg(original_path, corrected)
    photo.original_storage_path = str(original_path)
    photo.storage_path = str(original_path)
    session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="corrected", details={"path": str(original_path)}))
    await session.commit()
    return {"photo_id": str(photo.id), "storage_path": str(original_path)}


async def ocr_task_logic(session: AsyncSession, page_id: UUID, settings: Settings) -> dict:
    page = await session.scalar(select(Page).options(selectinload(Page.photos)).where(Page.id == page_id))
    if page is None:
        raise ValueError(f"Page not found: {page_id}")

    page_path = Path(page.storage_path)
    image = load_image(page_path)
    blocks, ocr_metadata = run_paddle_ocr(image, settings)
    if not blocks:
        blocks, sidecar_metadata = run_sidecar_ocr(page_path)
        ocr_metadata = {"primary": ocr_metadata, "fallback": sidecar_metadata}

    photos = [{"id": str(photo.id), "bounding_box": photo.bounding_box} for photo in page.photos]
    classified_blocks = classify_ocr_blocks(blocks, photos)
    for block in classified_blocks:
        session.add(
            OCRResult(
                page_id=page.id,
                photo_id=UUID(block.photo_id) if block.photo_id else None,
                text_content=block.text,
                text_type=block.text_type,
                bounding_box=block.bounding_box,
                ocr_engine=block.engine,
                confidence=block.confidence,
                spatial_classification_reason=block.reason,
            )
        )

    _append_metadata(page, "ocr", ocr_metadata | {"blocks": len(classified_blocks)})
    session.add(AuditLog(entity_type="page", entity_id=page.id, action="ocr_extracted", details={"blocks": len(classified_blocks)}))
    await session.commit()
    return {"page_id": str(page.id), "ocr_blocks": len(classified_blocks)}


async def enhancement_task_logic(session: AsyncSession, photo_id: UUID, settings: Settings, preset: str | None = "auto") -> dict:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")
    source_path = Path(photo.original_storage_path or photo.storage_path or "")
    if not source_path.exists():
        raise ValueError(f"Corrected source image missing for photo {photo_id}")

    from app.pipeline.diffusion_restoration import premium_enhance_photo

    source_img = load_image(source_path)
    degradation_report = analyze_degradation(source_img)
    effective_preset = degradation_report.recommended_preset if preset in {None, "", "auto"} else preset

    # 1. Standard Enhancement
    enhanced, metadata = dl_enhance_photo(source_img, preset=effective_preset, degradation_report=degradation_report)
    enhanced_path = settings.storage_path / "processed" / str(photo.album_id) / "photos" / f"{photo.id}_enhanced.jpg"
    save_jpeg(enhanced_path, enhanced)

    # 2. Premium Enhancement
    premium, premium_metadata = premium_enhance_photo(
        source_img,
        base_restored=enhanced,
        base_metadata=metadata,
        preset=effective_preset,
        degradation_report=degradation_report,
    )
    premium_path = settings.storage_path / "processed" / str(photo.album_id) / "photos" / f"{photo.id}_premium.jpg"
    save_jpeg(premium_path, premium)

    previous_metadata = dict(photo.enhancement_applied or {})
    variant_bundle = build_variant_bundle(
        original_path=Path(photo.original_storage_path or source_path),
        original=source_img,
        enhanced_path=enhanced_path,
        enhanced=enhanced,
        enhanced_metadata=metadata,
        premium_path=premium_path,
        premium=premium,
        premium_metadata=premium_metadata,
    )
    if previous_metadata.get("review_reasons"):
        variant_bundle["review_reasons"] = previous_metadata["review_reasons"]
    variant_bundle["degradation"] = degradation_report.as_dict()
    variant_bundle["standard"] = metadata
    variant_bundle["premium"] = premium_metadata

    photo.storage_path = str(enhanced_path)
    photo.enhancement_applied = variant_bundle
    session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="enhanced", details=photo.enhancement_applied | {"path": str(enhanced_path)}))
    await session.commit()
    return {"photo_id": str(photo.id), "storage_path": str(enhanced_path), "metadata": photo.enhancement_applied}


async def deduplication_task_logic(session: AsyncSession, photo_id: UUID) -> dict:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")
    phash = compute_phash(Path(photo.storage_path))
    photo.phash = phash

    existing = await session.scalars(
        select(ExtractedPhoto).where(
            ExtractedPhoto.id != photo.id,
            ExtractedPhoto.phash.is_not(None),
        )
    )
    duplicate_of = None
    distance = None
    for candidate in existing:
        candidate_distance = hamming_distance(phash, candidate.phash)
        if candidate_distance < 10:
            duplicate_of = candidate.id
            distance = candidate_distance
            break

    if duplicate_of is not None:
        photo.is_duplicate_of = duplicate_of
        photo.status = PhotoStatus.review_needed

    session.add(
        AuditLog(
            entity_type="photo",
            entity_id=photo.id,
            action="deduplicated",
            details={"phash": phash, "duplicate_of": str(duplicate_of) if duplicate_of else None, "distance": distance},
        )
    )
    await session.commit()
    return {"photo_id": str(photo.id), "phash": phash, "duplicate_of": str(duplicate_of) if duplicate_of else None}


async def persistence_task_logic(session: AsyncSession, photo_id: UUID) -> dict:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")

    caption = await session.scalar(
        select(OCRResult.text_content)
        .where(OCRResult.photo_id == photo.id, OCRResult.text_type == OCRTextType.caption)
        .order_by(OCRResult.is_verified.desc(), OCRResult.confidence.desc().nullslast())
        .limit(1)
    )
    final_path = Path(photo.storage_path)
    exif_metadata = write_caption_exif(final_path, caption)
    if photo.status != PhotoStatus.review_needed:
        photo.status = PhotoStatus.completed
    session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="persisted", details=exif_metadata | {"caption": caption}))
    await session.commit()
    return {"photo_id": str(photo.id), "status": photo.status.value, "exif": exif_metadata}


async def process_single_photo(session: AsyncSession, photo_id: UUID, settings: Settings) -> dict:
    await perspective_correction_task_logic(session, photo_id, settings)
    await enhancement_task_logic(session, photo_id, settings)
    await deduplication_task_logic(session, photo_id)
    return await persistence_task_logic(session, photo_id)


async def process_page_pipeline(page_id: UUID, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    async with get_session_factory()() as session:
        await ingestion_task_logic(session, page_id, settings)
        segmentation = await segmentation_task_logic(session, page_id, settings)
        await ocr_task_logic(session, page_id, settings)
        photo_results = []
        for photo_id in segmentation["photo_ids"]:
            photo_results.append(await process_single_photo(session, UUID(photo_id), settings))

        page = await session.scalar(select(Page).options(selectinload(Page.album)).where(Page.id == page_id))
        if page is None:
            raise ValueError(f"Page not found: {page_id}")
        failed_photos = [result for result in photo_results if result["status"] == PhotoStatus.failed.value]
        review_photos = [result for result in photo_results if result["status"] == PhotoStatus.review_needed.value]
        if failed_photos:
            page.status = PageStatus.failed
        elif review_photos or page.status == PageStatus.review_needed:
            page.status = PageStatus.review_needed
        else:
            page.status = PageStatus.completed
        if page.status in {PageStatus.completed, PageStatus.review_needed}:
            page.album.processed_pages += 1
        remaining = await session.scalar(
            select(Page.id).where(Page.album_id == page.album_id, Page.status.in_([PageStatus.uploaded, PageStatus.queued, PageStatus.processing])).limit(1)
        )
        if remaining is None:
            page.album.status = AlbumStatus.completed if page.status != PageStatus.failed else AlbumStatus.failed
        session.add(AuditLog(entity_type="page", entity_id=page.id, action="completed", details={"status": page.status.value}))
        await session.commit()
        return {"page_id": str(page_id), "status": page.status.value, "photos": photo_results}


async def process_pages_pipeline(page_ids: list[UUID], settings: Settings | None = None) -> dict:
    results = []
    for page_id in page_ids:
        try:
            results.append(await process_page_pipeline(page_id, settings))
        except Exception as exc:  # noqa: BLE001 - failed pages should be persisted and processing should continue.
            async with get_session_factory()() as session:
                page = await session.get(Page, page_id)
                if page is not None:
                    page.status = PageStatus.failed
                    page.error_message = str(exc)
                    dlq = quarantine_page_failure(page, settings or get_settings(), reason="pipeline_exception", details={"error": str(exc)})
                    _append_metadata(page, "dlq", dlq | {"reason": "pipeline_exception"})
                    session.add(AuditLog(entity_type="page", entity_id=page.id, action="failed", details={"error": str(exc), **dlq}))
                    await session.commit()
                    logger.exception("page_pipeline_failed", page_id=str(page.id), dlq_manifest_path=dlq["dlq_manifest_path"])
            results.append({"page_id": str(page_id), "status": "failed", "error": str(exc)})
    return {"pages": results}


def run_async(coro):
    return asyncio.run(coro)
