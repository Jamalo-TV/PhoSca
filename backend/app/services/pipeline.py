import asyncio
import hashlib
from time import perf_counter
from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid4

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.database import get_session_factory
from app.core.metrics import CANARY_PERCENTAGE, ENHANCEMENT_ROUTES, PROTECTED_REGIONS, SEGMENTATION_CONFIDENCE, SEGMENTATION_SECONDS
from app.models import Album, AlbumStatus, AuditLog, ExtractedPhoto, OCRResult, OCRTextType, Page, PageStatus, PhotoStatus, RunStatus
from app.pipeline.dedup import compute_phash, hamming_distance
from app.pipeline.degradation import analyze_degradation
from app.pipeline.confidence import calibrate_confidence
from app.pipeline.dl_enhancement import dl_enhance_photo
from app.pipeline.enhancement import prepare_for_cutout
from app.pipeline.exif import write_caption_exif
from app.pipeline.image_ops import calculate_blur_score, load_image, save_jpeg, sha256_file
from app.pipeline.ocr import classify_ocr_blocks, run_paddle_ocr, run_sidecar_ocr
from app.pipeline.perspective import crop_and_correct_photo_with_metadata
from app.pipeline.protected_regions import detect_protected_regions
from app.pipeline.presets import get_enhancement_config
from app.pipeline.quality import segmentation_review_reasons
from app.pipeline.segmentation import segment_album_page
from app.pipeline.variants import build_variant_bundle
from app.services.dlq import quarantine_page_failure
from app.services.provenance import asset_for_path, finish_run, finish_stage, get_or_create_run, record_asset, start_stage


logger = structlog.get_logger(__name__)
_local_pipeline_lock = asyncio.Lock()


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
    run = await get_or_create_run(session, page, settings)
    stage = await start_stage(session, run, "ingestion", input_path=Path(page.storage_path))

    path = Path(page.storage_path)
    actual_hash = sha256_file(path)
    if actual_hash != page.file_hash_sha256:
        page.status = PageStatus.failed
        page.error_message = "SHA-256 verification failed."
        dlq = quarantine_page_failure(page, settings, reason="sha256_verification_failed", details={"actual_sha256": actual_hash})
        _append_metadata(page, "ingestion", {"hash_verified": False, "actual_sha256": actual_hash, **dlq})
        session.add(AuditLog(entity_type="page", entity_id=page.id, action="failed", details={"reason": page.error_message}))
        finish_stage(stage, status=RunStatus.failed, details={"reason": "sha256_verification_failed"})
        finish_run(run, status=RunStatus.failed, details={"reason": "sha256_verification_failed"})
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
    input_asset = await record_asset(session, run, path=path, role="page_original", stage=stage)
    manifest = dict(run.manifest or {})
    manifest["input_asset_id"] = str(input_asset.id)
    run.manifest = manifest
    ingestion_details = page.processing_metadata.get("steps", {}).get("ingestion", {}) | {"asset_id": str(input_asset.id)}
    finish_stage(stage, status=RunStatus.review_needed if page.status == PageStatus.review_needed else RunStatus.completed, details=ingestion_details, output_path=path)
    await session.commit()
    return {"page_id": str(page.id), "status": page.status.value, "blur_score": blur_score}


async def segmentation_task_logic(session: AsyncSession, page_id: UUID, settings: Settings) -> dict:
    page = await session.get(Page, page_id)
    if page is None:
        raise ValueError(f"Page not found: {page_id}")
    run = await get_or_create_run(session, page, settings)
    existing_photo_ids = [str(value) for value in (await session.scalars(select(ExtractedPhoto.id).where(ExtractedPhoto.page_id == page.id))).all()]
    if existing_photo_ids:
        return {"page_id": str(page.id), "photo_ids": existing_photo_ids, "metadata": {"idempotent_replay": True, "run_id": str(run.id)}}
    stage = await start_stage(session, run, "segmentation", input_path=Path(page.storage_path))

    prior_review_required = page.status == PageStatus.review_needed
    page.status = PageStatus.processing
    image = load_image(Path(page.storage_path))
    cutout_ready_image, cutout_prep_metadata = prepare_for_cutout(image)
    _append_metadata(page, "pre_cutout_enhancement", cutout_prep_metadata)
    segmentation_started = perf_counter()
    result = segment_album_page(cutout_ready_image, settings)
    SEGMENTATION_SECONDS.labels(model=str(result.metadata.get("model", "unknown"))).observe(perf_counter() - segmentation_started)
    for detection in result.detections:
        SEGMENTATION_CONFIDENCE.labels(source=str((detection.mask or {}).get("detector_source", "unknown"))).observe(detection.confidence)
    protected_regions = detect_protected_regions(image) if settings.enable_protected_region_guard else []
    PROTECTED_REGIONS.observe(len(protected_regions))
    _append_metadata(page, "protected_regions", {"regions": protected_regions, "count": len(protected_regions)})
    _append_metadata(page, "segmentation", result.metadata | {"detections": len(result.detections)})

    created_photo_ids: list[str] = []
    review_reasons_by_photo: dict[str, list[str]] = {}
    for detection in result.detections:
        detector_source = str((detection.mask or {}).get("detector_source", "default"))
        calibrated_confidence, calibration = calibrate_confidence(detection.confidence, detector_source, settings.segmentation_calibration_path)
        review_reasons = segmentation_review_reasons(
            confidence=calibrated_confidence,
            confidence_threshold=settings.yolo_confidence_threshold,
            quality=detection,
        )
        if settings.require_confidence_calibration and not calibration["calibrated"]:
            review_reasons.append("segmentation_confidence_uncalibrated")
        photo_status = PhotoStatus.review_needed if review_reasons else PhotoStatus.pending
        photo = ExtractedPhoto(
            id=uuid4(),
            page_id=page.id,
            album_id=page.album_id,
            storage_path="",
            original_storage_path=None,
            bounding_box=detection.bounding_box,
            segmentation_mask=detection.mask,
            segmentation_confidence=calibrated_confidence,
            aspect_ratio=detection.aspect_ratio,
            geometry_valid=detection.geometry_valid,
            enhancement_applied={"review_reasons": review_reasons, "segmentation_confidence": calibration},
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
        # Ingestion and segmentation reasons are cumulative: a blur warning
        # must not be erased simply because later stages produced detections.
        page.status = PageStatus.review_needed if prior_review_required else PageStatus.queued
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
    finish_stage(stage, status=RunStatus.review_needed if page.status == PageStatus.review_needed else RunStatus.completed, details=result.metadata)
    await session.commit()
    return {"page_id": str(page.id), "photo_ids": created_photo_ids, "metadata": result.metadata}


async def perspective_correction_task_logic(
    session: AsyncSession,
    photo_id: UUID,
    settings: Settings,
    *,
    skip_if_exists: bool = False,
) -> dict:
    photo = await session.scalar(
        select(ExtractedPhoto)
        .options(selectinload(ExtractedPhoto.page))
        .where(ExtractedPhoto.id == photo_id)
    )
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")
    run = await get_or_create_run(session, photo.page, settings)
    if skip_if_exists and photo.original_storage_path and Path(photo.original_storage_path).exists():
        return {"photo_id": str(photo.id), "storage_path": photo.original_storage_path, "idempotent_replay": True, "run_id": str(run.id)}
    stage = await start_stage(session, run, "perspective_correction", input_path=Path(photo.page.storage_path))

    page_image = load_image(Path(photo.page.storage_path))
    protected_regions = list((photo.page.processing_metadata or {}).get("steps", {}).get("protected_regions", {}).get("regions") or [])
    try:
        corrected, correction_metadata = crop_and_correct_photo_with_metadata(
            page_image,
            photo.bounding_box,
            photo.segmentation_mask,
            trim_border=settings.enable_border_trim,
            auto_orient=settings.enable_photo_orientation_correction,
            orientation_model_path=settings.photo_orientation_model_path,
            protect_regions=settings.enable_protected_region_guard,
            protected_regions=protected_regions,
        )
    except ValueError as exc:
        reason = str(exc)
        photo.status = PhotoStatus.review_needed
        _append_metadata(photo, "correction", {"status": "rejected", "reason": reason})
        session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="correction_rejected", details={"reason": reason}))
        finish_stage(stage, status=RunStatus.review_needed, details={"reason": reason})
        await session.commit()
        return {"photo_id": str(photo.id), "storage_path": None, "review_required": True, "reason": reason}
    original_path = settings.storage_path / "processed" / str(photo.album_id) / "runs" / str(run.id) / "stages" / str(stage.id) / "photos" / f"{photo.id}_original.jpg"
    save_jpeg(original_path, corrected)
    photo.original_storage_path = str(original_path)
    photo.storage_path = str(original_path)
    parent_asset = await asset_for_path(session, Path(photo.page.storage_path))
    corrected_asset = await record_asset(
        session,
        run,
        path=original_path,
        role="corrected_print",
        stage=stage,
        parent_asset=parent_asset,
        metadata=correction_metadata,
    )
    correction_metadata = correction_metadata | {"asset_id": str(corrected_asset.id)}
    _append_metadata(photo, "correction", correction_metadata)
    session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="corrected", details={"path": str(original_path), **correction_metadata}))
    finish_stage(stage, status=RunStatus.completed, details=correction_metadata, output_path=original_path)
    await session.commit()
    return {"photo_id": str(photo.id), "storage_path": str(original_path)}


async def ocr_task_logic(session: AsyncSession, page_id: UUID, settings: Settings) -> dict:
    page = await session.scalar(select(Page).options(selectinload(Page.photos)).where(Page.id == page_id))
    if page is None:
        raise ValueError(f"Page not found: {page_id}")
    run = await get_or_create_run(session, page, settings)
    stage = await start_stage(session, run, "ocr", input_path=Path(page.storage_path))

    page_path = Path(page.storage_path)
    image = load_image(page_path)
    blocks, ocr_metadata = run_paddle_ocr(image, settings)
    if not blocks:
        blocks, sidecar_metadata = run_sidecar_ocr(page_path)
        ocr_metadata = {"primary": ocr_metadata, "fallback": sidecar_metadata}

    photos = [{"id": str(photo.id), "bounding_box": photo.bounding_box} for photo in page.photos]
    classified_blocks = classify_ocr_blocks(blocks, photos)
    await session.execute(delete(OCRResult).where(OCRResult.page_id == page.id, OCRResult.is_verified.is_(False)))
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

    _append_metadata(page, "ocr", ocr_metadata | {"blocks": len(classified_blocks), "replaced_unverified_rows": True})
    session.add(AuditLog(entity_type="page", entity_id=page.id, action="ocr_extracted", details={"blocks": len(classified_blocks)}))
    finish_stage(stage, status=RunStatus.completed, details=ocr_metadata | {"blocks": len(classified_blocks)})
    await session.commit()
    return {"page_id": str(page.id), "ocr_blocks": len(classified_blocks)}


async def enhancement_task_logic(session: AsyncSession, photo_id: UUID, settings: Settings, preset: str | None = "auto") -> dict:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")
    source_path = Path(photo.original_storage_path or photo.storage_path or "")
    if not source_path.exists():
        raise ValueError(f"Corrected source image missing for photo {photo_id}")
    page = await session.get(Page, photo.page_id)
    if page is None:
        raise ValueError(f"Page not found for photo {photo_id}")
    run = await get_or_create_run(session, page, settings)
    stage = await start_stage(session, run, "enhancement", input_path=source_path)

    from app.pipeline.diffusion_restoration import premium_enhance_photo

    source_img = load_image(source_path)
    degradation_report = analyze_degradation(source_img)
    effective_preset = degradation_report.recommended_preset if preset in {None, "", "auto"} else preset

    enhanced = None
    enhanced_path = None
    enhancement_enabled = settings.enable_conservative_enhancement or settings.enable_deep_enhancement
    metadata: dict = {"method": "disabled", "reason": "enhancement_feature_flags_disabled", "models": []}
    ENHANCEMENT_ROUTES.labels(route="conservative_enabled" if enhancement_enabled else "noop_disabled").inc()
    if enhancement_enabled:
        enhancement_config = get_enhancement_config(effective_preset, recommended_preset=degradation_report.recommended_preset)
        if settings.require_registered_models and enhancement_config.unsafe_models_enabled and enhancement_config.super_resolution_enabled:
            from app.pipeline.model_registry import verify_model_artifact

            verify_model_artifact(settings.realesrgan_model_path, settings.model_manifest_path)
            if enhancement_config.face_enhancement_enabled:
                verify_model_artifact(settings.gfpgan_model_path, settings.model_manifest_path)
        enhanced, metadata = dl_enhance_photo(
            source_img,
            preset=effective_preset,
            degradation_report=degradation_report,
            model_paths={"realesrgan": settings.realesrgan_model_path, "gfpgan": settings.gfpgan_model_path},
        )
        if metadata.get("planner", {}).get("no_op"):
            ENHANCEMENT_ROUTES.labels(route="noop_planner").inc()
            enhanced = None
            metadata = {**metadata, "method": "no_op", "reason": "planner_selected_source"}
        else:
            enhanced_path = settings.storage_path / "processed" / str(photo.album_id) / "runs" / str(run.id) / "stages" / str(stage.id) / "photos" / f"{photo.id}_enhanced.jpg"
            save_jpeg(enhanced_path, enhanced)

    premium = None
    premium_path = None
    premium_metadata: dict = {"method": "disabled", "reason": "premium_enhancement_feature_flag_disabled", "models": []}
    if settings.enable_premium_enhancement:
        premium, premium_metadata = premium_enhance_photo(
            source_img,
            base_restored=enhanced,
            base_metadata=metadata,
            preset=effective_preset,
            degradation_report=degradation_report,
        )
        premium_path = settings.storage_path / "processed" / str(photo.album_id) / "runs" / str(run.id) / "stages" / str(stage.id) / "photos" / f"{photo.id}_premium.jpg"
        save_jpeg(premium_path, premium)

    previous_metadata = dict(photo.enhancement_applied or {})
    canary_bucket = int(hashlib.sha256(str(photo.id).encode("ascii")).hexdigest()[:8], 16) % 100
    CANARY_PERCENTAGE.set(settings.canary_percentage)
    previous_metadata["release_policy"] = {
        "kill_switch": settings.release_kill_switch,
        "shadow_mode": settings.shadow_mode,
        "canary_percentage": settings.canary_percentage,
        "canary_bucket": canary_bucket,
        "canary_eligible": canary_bucket < settings.canary_percentage,
    }
    variant_bundle = build_variant_bundle(
        original_path=Path(photo.original_storage_path or source_path),
        original=source_img,
        enhanced_path=enhanced_path,
        enhanced=enhanced,
        enhanced_metadata=metadata,
        premium_path=premium_path,
        premium=premium,
        premium_metadata=premium_metadata,
        photo_status=photo.status,
        release_metadata=previous_metadata,
    )
    for preserved_key in ("review_reasons", "correction", "release_policy"):
        if previous_metadata.get(preserved_key):
            variant_bundle[preserved_key] = previous_metadata[preserved_key]
    variant_bundle["degradation"] = degradation_report.as_dict()
    variant_bundle["standard"] = metadata
    variant_bundle["premium"] = premium_metadata

    source_asset = await asset_for_path(session, source_path)
    produced_assets: dict[str, str] = {}
    for role, candidate_path, candidate_metadata in (
        ("enhanced_candidate", enhanced_path, metadata),
        ("premium_candidate", premium_path, premium_metadata),
    ):
        if candidate_path is not None:
            candidate_asset = await record_asset(
                session,
                run,
                path=candidate_path,
                role=role,
                stage=stage,
                parent_asset=source_asset,
                metadata=candidate_metadata,
            )
            produced_assets[role] = str(candidate_asset.id)

    photo.storage_path = str(source_path)
    photo.enhancement_applied = variant_bundle
    selected_path = photo.storage_path
    session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="enhanced", details=photo.enhancement_applied | {"path": selected_path}))
    finish_stage(
        stage,
        status=RunStatus.review_needed if photo.status == PhotoStatus.review_needed else RunStatus.completed,
        details=photo.enhancement_applied | {"assets": produced_assets},
        output_path=Path(selected_path),
    )
    await session.commit()
    return {"photo_id": str(photo.id), "storage_path": selected_path, "metadata": photo.enhancement_applied}


async def deduplication_task_logic(session: AsyncSession, photo_id: UUID, settings: Settings | None = None) -> dict:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")
    settings = settings or get_settings()
    page = await session.get(Page, photo.page_id)
    if page is None:
        raise ValueError(f"Page not found for photo {photo_id}")
    run = await get_or_create_run(session, page, settings)
    stage = await start_stage(session, run, "deduplication", input_path=Path(photo.storage_path))
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
    finish_stage(
        stage,
        status=RunStatus.review_needed if photo.status == PhotoStatus.review_needed else RunStatus.completed,
        details={"phash": phash, "duplicate_of": str(duplicate_of) if duplicate_of else None, "distance": distance},
        output_path=Path(photo.storage_path),
    )
    await session.commit()
    return {"photo_id": str(photo.id), "phash": phash, "duplicate_of": str(duplicate_of) if duplicate_of else None}


async def persistence_task_logic(session: AsyncSession, photo_id: UUID, settings: Settings | None = None) -> dict:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")
    settings = settings or get_settings()
    page = await session.get(Page, photo.page_id)
    if page is None:
        raise ValueError(f"Page not found for photo {photo_id}")
    run = await get_or_create_run(session, page, settings)
    stage = await start_stage(session, run, "persistence", input_path=Path(photo.storage_path))

    caption = await session.scalar(
        select(OCRResult.text_content)
        .where(OCRResult.photo_id == photo.id, OCRResult.text_type == OCRTextType.caption)
        .order_by(OCRResult.is_verified.desc(), OCRResult.confidence.desc().nullslast())
        .limit(1)
    )
    source_final_path = Path(photo.storage_path)
    final_path = source_final_path.with_name(f"{source_final_path.stem}_released{source_final_path.suffix}")
    exif_metadata = write_caption_exif(source_final_path, caption, output_path=final_path)
    source_asset = await asset_for_path(session, source_final_path)
    released_asset = await record_asset(
        session,
        run,
        path=final_path,
        role="released_variant",
        stage=stage,
        parent_asset=source_asset,
        metadata=exif_metadata,
    )
    metadata = dict(photo.enhancement_applied or {})
    variants = dict(metadata.get("variants") or {})
    original_variant = dict(variants.get("original") or {})
    if original_variant:
        original_variant["path"] = str(final_path)
        variants["original"] = original_variant
        metadata["variants"] = variants
    metadata["released_asset_path"] = str(final_path)
    metadata["released_asset_id"] = str(released_asset.id)
    photo.enhancement_applied = metadata
    photo.storage_path = str(final_path)
    if photo.status not in {PhotoStatus.review_needed, PhotoStatus.failed}:
        photo.status = PhotoStatus.completed
    session.add(AuditLog(entity_type="photo", entity_id=photo.id, action="persisted", details=exif_metadata | {"caption": caption}))
    finish_stage(
        stage,
        status=RunStatus.review_needed if photo.status == PhotoStatus.review_needed else RunStatus.completed,
        details=exif_metadata | {"caption": caption, "asset_id": str(released_asset.id)},
        output_path=final_path,
    )
    await session.commit()
    return {"photo_id": str(photo.id), "status": photo.status.value, "exif": exif_metadata}


async def process_single_photo(session: AsyncSession, photo_id: UUID, settings: Settings) -> dict:
    correction = await perspective_correction_task_logic(session, photo_id, settings, skip_if_exists=True)
    if correction.get("review_required"):
        return {"photo_id": str(photo_id), "status": PhotoStatus.review_needed.value, "reason": correction["reason"]}
    await enhancement_task_logic(session, photo_id, settings)
    await deduplication_task_logic(session, photo_id, settings)
    return await persistence_task_logic(session, photo_id, settings)


async def process_page_pipeline(page_id: UUID, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    async with get_session_factory()() as session:
        existing_page = await session.get(Page, page_id)
        if existing_page is None:
            raise ValueError(f"Page not found: {page_id}")
        if existing_page.status in {PageStatus.completed, PageStatus.review_needed}:
            return {"page_id": str(page_id), "status": existing_page.status.value, "photos": [], "idempotent_replay": True}
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
        await session.flush()
        remaining = await session.scalar(
            select(Page.id).where(Page.album_id == page.album_id, Page.status.in_([PageStatus.uploaded, PageStatus.queued, PageStatus.processing])).limit(1)
        )
        if remaining is None:
            failed_page = await session.scalar(select(Page.id).where(Page.album_id == page.album_id, Page.status == PageStatus.failed).limit(1))
            page.album.status = AlbumStatus.failed if failed_page is not None else AlbumStatus.completed
        session.add(AuditLog(entity_type="page", entity_id=page.id, action="completed", details={"status": page.status.value}))
        run = await get_or_create_run(session, page, settings)
        run_status = {
            PageStatus.completed: RunStatus.completed,
            PageStatus.review_needed: RunStatus.review_needed,
            PageStatus.failed: RunStatus.failed,
        }.get(page.status, RunStatus.running)
        finish_run(run, status=run_status, details={"page_status": page.status.value, "photo_count": len(photo_results)})
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
                    run = await get_or_create_run(session, page, settings or get_settings())
                    failed_stage = await start_stage(session, run, "pipeline_exception", input_path=Path(page.storage_path))
                    finish_stage(failed_stage, status=RunStatus.failed, details={"error": str(exc)})
                    finish_run(run, status=RunStatus.failed, details={"error": str(exc)})
                    dlq = quarantine_page_failure(page, settings or get_settings(), reason="pipeline_exception", details={"error": str(exc)})
                    _append_metadata(page, "dlq", dlq | {"reason": "pipeline_exception"})
                    session.add(AuditLog(entity_type="page", entity_id=page.id, action="failed", details={"error": str(exc), **dlq}))
                    await session.commit()
                    logger.exception("page_pipeline_failed", page_id=str(page.id), dlq_manifest_path=dlq["dlq_manifest_path"])
            results.append({"page_id": str(page_id), "status": "failed", "error": str(exc)})
    return {"pages": results}


async def process_pages_serialized(page_ids: list[UUID], settings: Settings | None = None) -> dict:
    """Serialize local background inference to protect SQLite and one GPU.

    Uploaded pages remain in ``uploaded`` state if the process restarts, so a
    later catch-up run can safely replay them.
    """
    async with _local_pipeline_lock:
        return await process_pages_pipeline(page_ids, settings)


def run_async(coro):
    return asyncio.run(coro)
