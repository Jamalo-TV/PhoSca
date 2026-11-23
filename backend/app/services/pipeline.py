import asyncio
import hashlib
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

import cv2
import numpy as np
import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.core.metrics import (
    CANARY_PERCENTAGE,
    ENHANCEMENT_ROUTES,
    PROTECTED_REGIONS,
    SEGMENTATION_CONFIDENCE,
    SEGMENTATION_SECONDS,
)
from app.database import get_session_factory
from app.models import (
    Album,
    AlbumStatus,
    AuditLog,
    ExtractedPhoto,
    OCRResult,
    OCRTextType,
    Page,
    PageStatus,
    PhotoStatus,
    RunStatus,
)
from app.pipeline.confidence import calibrate_confidence
from app.pipeline.crop_validation import (
    CROP_VALIDATION_UNAVAILABLE,
    automatic_crop_review_reasons,
    crop_validation_evidence,
    crop_validation_profile,
    geometrically_ready_crop_modes,
)
from app.pipeline.decisions import ReviewReason, component_review_reasons, confidence_components, is_reviewed_non_print
from app.pipeline.dedup import compute_phash, hamming_distance
from app.pipeline.degradation import analyze_degradation
from app.pipeline.dl_enhancement import dl_enhance_photo
from app.pipeline.enhancement import conservative_enhance_photo, prepare_for_cutout
from app.pipeline.exif import write_caption_exif
from app.pipeline.image_ops import calculate_blur_score, load_image, save_jpeg, sha256_file
from app.pipeline.ocr import classify_ocr_blocks, run_paddle_ocr, run_sidecar_ocr
from app.pipeline.perspective import apply_refined_visibility, correct_photo_crop
from app.pipeline.presets import get_enhancement_config
from app.pipeline.protected_regions import detect_protected_regions
from app.pipeline.quality import segmentation_review_reasons
from app.pipeline.segmentation import segment_album_page
from app.pipeline.variants import build_variant_bundle
from app.schemas.common import PrintSizeMM
from app.services.dlq import quarantine_page_failure
from app.services.provenance import (
    asset_for_path,
    finish_run,
    finish_stage,
    get_or_create_run,
    record_asset,
    start_stage,
)

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


def _update_photo_decision_metadata(
    photo: ExtractedPhoto,
    *,
    component_updates: dict[str, object] | None = None,
    reason_codes: list[str] | None = None,
) -> dict:
    """Merge independent confidence evidence without discarding earlier risk."""
    metadata = deepcopy(photo.enhancement_applied or {})
    components = confidence_components(metadata.get("confidence_components"), **(component_updates or {}))
    reasons = [str(reason) for reason in metadata.get("review_reasons") or []]
    reasons.extend(component_review_reasons(components))
    reasons.extend(str(reason) for reason in (reason_codes or []))
    metadata["confidence_components"] = components.as_dict()
    metadata["review_reasons"] = list(dict.fromkeys(reasons))
    photo.enhancement_applied = metadata
    if metadata["review_reasons"] and photo.status not in {PhotoStatus.failed}:
        photo.status = PhotoStatus.review_needed
    return metadata


def _source_variant_record(name: str, path: Path, *, selected: bool, metadata: dict | None = None) -> dict:
    return {
        "name": name,
        "kind": name,
        "path": str(path),
        "selected": selected,
        "model_chain": [],
        "params": {},
        "metrics": {},
        "warnings": [],
        "gates": [],
        "metadata": dict(metadata or {}),
    }


async def ingestion_task_logic(
    session: AsyncSession, page_id: UUID, settings: Settings, *, update_page_status: bool = True
) -> dict:
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
        dlq = quarantine_page_failure(
            page, settings, reason="sha256_verification_failed", details={"actual_sha256": actual_hash}
        )
        _append_metadata(page, "ingestion", {"hash_verified": False, "actual_sha256": actual_hash, **dlq})
        session.add(
            AuditLog(entity_type="page", entity_id=page.id, action="failed", details={"reason": page.error_message})
        )
        finish_stage(stage, status=RunStatus.failed, details={"reason": "sha256_verification_failed"})
        finish_run(run, status=RunStatus.failed, details={"reason": "sha256_verification_failed"})
        await session.commit()
        logger.warning("page_ingestion_failed", page_id=str(page.id), reason="sha256_verification_failed", **dlq)
        raise ValueError(page.error_message)

    image = await asyncio.to_thread(load_image, path)
    blur_score = await asyncio.to_thread(calculate_blur_score, image)
    page.blur_score = blur_score
    needs_review = blur_score < settings.blur_threshold
    if update_page_status:
        page.status = PageStatus.review_needed if needs_review else PageStatus.queued
    _append_metadata(
        page,
        "ingestion",
        {
            "hash_verified": True,
            "blur_score": blur_score,
            "blur_threshold": settings.blur_threshold,
            "review_needed": needs_review,
        },
    )
    session.add(AuditLog(entity_type="page", entity_id=page.id, action="ingested", details={"blur_score": blur_score}))
    input_asset = await record_asset(session, run, path=path, role="page_original", stage=stage)
    manifest = dict(run.manifest or {})
    manifest["input_asset_id"] = str(input_asset.id)
    run.manifest = manifest
    ingestion_details = page.processing_metadata.get("steps", {}).get("ingestion", {}) | {
        "asset_id": str(input_asset.id)
    }
    finish_stage(
        stage,
        status=RunStatus.review_needed if needs_review else RunStatus.completed,
        details=ingestion_details,
        output_path=path,
    )
    await session.commit()
    return {"page_id": str(page.id), "status": page.status.value, "blur_score": blur_score}


async def segmentation_task_logic(
    session: AsyncSession, page_id: UUID, settings: Settings, *, update_page_status: bool = True
) -> dict:
    page = await session.get(Page, page_id)
    if page is None:
        raise ValueError(f"Page not found: {page_id}")
    run = await get_or_create_run(session, page, settings)
    existing_photo_ids = [
        str(value)
        for value in (await session.scalars(select(ExtractedPhoto.id).where(ExtractedPhoto.page_id == page.id))).all()
    ]
    if existing_photo_ids:
        return {
            "page_id": str(page.id),
            "photo_ids": existing_photo_ids,
            "metadata": {"idempotent_replay": True, "run_id": str(run.id)},
        }
    stage = await start_stage(session, run, "segmentation", input_path=Path(page.storage_path))

    prior_review_required = page.status == PageStatus.review_needed
    page.status = PageStatus.processing
    image = await asyncio.to_thread(load_image, Path(page.storage_path))
    cutout_ready_image, cutout_prep_metadata = await asyncio.to_thread(prepare_for_cutout, image)
    _append_metadata(page, "pre_cutout_enhancement", cutout_prep_metadata)
    segmentation_started = perf_counter()
    result = await asyncio.to_thread(segment_album_page, cutout_ready_image, settings)
    SEGMENTATION_SECONDS.labels(model=str(result.metadata.get("model", "unknown"))).observe(
        perf_counter() - segmentation_started
    )
    for detection in result.detections:
        SEGMENTATION_CONFIDENCE.labels(source=str((detection.mask or {}).get("detector_source", "unknown"))).observe(
            detection.confidence
        )
    protected_regions = (
        await asyncio.to_thread(detect_protected_regions, image) if settings.enable_protected_region_guard else []
    )
    PROTECTED_REGIONS.observe(len(protected_regions))
    _append_metadata(page, "protected_regions", {
        "regions": protected_regions, "count": len(protected_regions),
        "enabled": settings.enable_protected_region_guard,
    })
    _append_metadata(page, "segmentation", result.metadata | {
        "detections": len(result.detections),
        "validation_profile": crop_validation_profile(settings),
    })

    created_photo_ids: list[str] = []
    review_reasons_by_photo: dict[str, list[str]] = {}
    for detection in result.detections:
        detector_source = str((detection.mask or {}).get("detector_source", "default"))
        calibrated_confidence, calibration = calibrate_confidence(
            detection.confidence, detector_source, settings.segmentation_calibration_path
        )
        review_reasons = segmentation_review_reasons(
            confidence=calibrated_confidence,
            confidence_threshold=settings.yolo_confidence_threshold,
            quality=detection,
        )
        mask: dict = detection.mask or {}
        boundary_value = mask.get("boundary_evidence")
        geometry_value = mask.get("geometry_evidence")
        boundary_evidence: dict = boundary_value if isinstance(boundary_value, dict) else {}
        geometry_evidence: dict = geometry_value if isinstance(geometry_value, dict) else {}
        components = confidence_components(
            detection_confidence=calibrated_confidence,
            boundary_confidence=boundary_evidence.get("score"),
            geometry_confidence=geometry_evidence.get("score"),
        )
        review_reasons.extend(component_review_reasons(components))
        if settings.require_confidence_calibration and not calibration["calibrated"]:
            review_reasons.append("segmentation_confidence_uncalibrated")
            review_reasons.append("SEGMENTATION_CONFIDENCE_UNCALIBRATED")
        review_reasons = list(dict.fromkeys(review_reasons))
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
            enhancement_applied={
                "review_reasons": review_reasons,
                "segmentation_review_reasons": detection.review_reasons,
                "segmentation_confidence": calibration,
                "confidence_components": components.as_dict(),
            },
            status=photo_status,
        )
        session.add(photo)
        created_photo_ids.append(str(photo.id))
        if review_reasons:
            review_reasons_by_photo[str(photo.id)] = review_reasons

    needs_review = (
        not result.detections
        or prior_review_required
        or (page.blur_score is not None and page.blur_score < settings.blur_threshold)
    )
    if update_page_status:
        page.status = PageStatus.review_needed if needs_review else PageStatus.queued
    if not result.detections:
        page.error_message = "No photos detected; manual review required."
        session.add(
            AuditLog(
                entity_type="page", entity_id=page.id, action="review_flagged", details={"reason": page.error_message}
            )
        )
    else:
        details = {"photo_ids": created_photo_ids, "review_reasons_by_photo": review_reasons_by_photo}
        session.add(AuditLog(entity_type="page", entity_id=page.id, action="segmented", details=details))
        logger.info(
            "page_segmented",
            page_id=str(page.id),
            detections=len(result.detections),
            review_photos=len(review_reasons_by_photo),
            model=result.metadata.get("model"),
        )

    finish_stage(
        stage,
        status=RunStatus.review_needed if needs_review else RunStatus.completed,
        details=result.metadata,
    )
    await session.commit()
    return {"page_id": str(page.id), "photo_ids": created_photo_ids, "metadata": result.metadata}


async def perspective_correction_task_logic(
    session: AsyncSession,
    photo_id: UUID,
    settings: Settings,
    *,
    skip_if_exists: bool = False,
    reject_invalid_edit: bool = False,
    page_image: np.ndarray | None = None,
) -> dict:
    photo = await session.scalar(
        select(ExtractedPhoto)
        .options(selectinload(ExtractedPhoto.page))
        .where(ExtractedPhoto.id == photo_id)
        .with_for_update()
    )
    if photo is None:
        raise ValueError(f"Photo not found: {photo_id}")
    run = await get_or_create_run(session, photo.page, settings)
    if skip_if_exists and photo.original_storage_path and Path(photo.original_storage_path).exists():
        previous = (photo.enhancement_applied or {}).get("correction") or {}
        modes = previous.get("crop_modes") or {}
        full_mode = modes.get("full_print") or {}
        image_mode = modes.get("image_only") or {}
        trimmed_path = previous.get("trimmed_path")
        separate_motif = image_mode.get("status") == "ready" and (trimmed_path or image_mode.get("independent_of_full_print"))
        if not separate_motif or (trimmed_path and Path(trimmed_path).is_file()):
            return {
                "photo_id": str(photo.id),
                "storage_path": photo.original_storage_path,
                "idempotent_replay": True,
                "run_id": str(run.id),
                "review_required": bool(full_mode and full_mode["status"] != "ready"),
                "reason": next(iter(full_mode.get("reason_codes", [])), "CROP_GEOMETRY_UNCERTAIN") if full_mode else None,
            }
    stage = await start_stage(session, run, "perspective_correction", input_path=Path(photo.page.storage_path))

    if page_image is None:
        page_image = await asyncio.to_thread(load_image, Path(photo.page.storage_path))
    automatic_detection = photo.segmentation_confidence is not None
    page_steps = (photo.page.processing_metadata or {}).get("steps") or {}
    protection_evidence = page_steps.get("protected_regions") or {}
    protected_regions = protection_evidence.get("regions") if protection_evidence.get("enabled", True) else None
    neighbor_masks = []
    if settings.enable_precise_photo_crops:
        rows = await session.execute(
            select(
                ExtractedPhoto.segmentation_mask,
                ExtractedPhoto.bounding_box,
                ExtractedPhoto.enhancement_applied["review_rejected_detection"],
            ).where(
                ExtractedPhoto.page_id == photo.page_id,
                ExtractedPhoto.id != photo.id,
                ExtractedPhoto.segmentation_mask.is_not(None),
            )
        )
        neighbor_masks = [mask for mask, box, rejection in rows if mask and not is_reviewed_non_print(rejection, box, mask)]
    try:
        size = (photo.segmentation_mask or {}).get("print_size_mm")
        manual_aspect = None
        if not automatic_detection and size is not None:
            manual_aspect = PrintSizeMM.model_validate(size).aspect_ratio
        crop_result = await asyncio.to_thread(
            correct_photo_crop,
            page_image,
            photo.bounding_box,
            photo.segmentation_mask,
            trim_border=settings.enable_border_trim,
            auto_orient=settings.enable_photo_orientation_correction,
            orientation_model_path=settings.photo_orientation_model_path,
            protect_regions=settings.enable_protected_region_guard,
            protected_regions=protected_regions,
            allow_page_face_overlap=automatic_detection and photo.geometry_valid,
            # Saving reviewer corners clears prediction confidence, even when the client retains its model tag.
            allow_border_expansion=automatic_detection,
            dual_mode=settings.enable_precise_photo_crops,
            enable_classical_retry=settings.enable_classical_segmentation_fallback,
            print_aspect_ratio=manual_aspect,
            neighbor_masks=[mask for mask in neighbor_masks if mask],
        )
        corrected = crop_result.image
        correction_metadata = crop_result.metadata
    except (ValueError, cv2.error) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "perspective_opencv_error"
        if reject_invalid_edit:
            raise ValueError(reason) from exc
        metadata = _update_photo_decision_metadata(
            photo,
            component_updates={"perspective_confidence": 0.0},
            reason_codes=[ReviewReason.QUADRILATERAL_UNSTABLE.value],
        )
        prior_correction = metadata.get("correction") or {}
        metadata["correction"] = {"status": "rejected", "reason": reason}
        if settings.enable_precise_photo_crops or prior_correction.get("crop_modes"):
            metadata["correction"]["crop_modes"] = {
                name: {"status": "review_needed", "reason_codes": [reason]}
                for name in ("full_print", "image_only")
            }
        photo.enhancement_applied = metadata
        session.add(
            AuditLog(entity_type="photo", entity_id=photo.id, action="correction_rejected", details={"reason": reason})
        )
        finish_stage(stage, status=RunStatus.review_needed, details={"reason": reason})
        await session.commit()
        return {"photo_id": str(photo.id), "storage_path": None, "review_required": True, "reason": reason}
    output_dir = (
        settings.storage_path
        / "processed"
        / str(photo.album_id)
        / "runs"
        / str(run.id)
        / "stages"
        / str(stage.id)
        / "photos"
    )
    untrimmed_path = output_dir / f"{photo.id}_source_corrected_untrimmed.jpg"
    crop_modes = correction_metadata.get("crop_modes") or {}
    orientation_reason = (correction_metadata.get("orientation") or {}).get("reason_code")
    if crop_modes and automatic_detection:
        prediction = photo.enhancement_applied or {}
        calibration = prediction.get("segmentation_confidence") or {}
        segmentation = page_steps.get("segmentation") or {}
        model = segmentation.get("model_artifact") or {}
        evidence = None
        if not segmentation.get("onnx_error") and segmentation.get("validation_profile") == crop_validation_profile(settings):
            evidence = crop_validation_evidence(settings, model.get("sha256"))
        auto_review = automatic_crop_review_reasons(
            settings, photo.segmentation_confidence, photo.geometry_valid, calibration.get("calibrated") is True,
            prediction.get("segmentation_review_reasons") or [], orientation_reason,
        )
        if evidence is None:
            auto_review.append(CROP_VALIDATION_UNAVAILABLE)
        if auto_review:
            for mode in crop_modes.values():
                if mode.get("status") == "ready":
                    mode["geometry_status"] = "ready"
                    mode["status"] = "review_needed"
                    mode["reason_codes"] = auto_review.copy()
        else:
            correction_metadata["crop_validation"] = evidence
    full_mode = crop_modes.get("full_print")
    image_mode = crop_modes.get("image_only") or {}
    ready = geometrically_ready_crop_modes(
        crop_modes, image_only_present=crop_result.image_only is not None,
        image_only_is_full_print=crop_result.image_only is corrected,
    )
    full_ready = "full_print" in ready
    image_ready = "image_only" in ready
    image_is_full = image_ready and crop_result.image_only is corrected
    trim_applied = bool(correction_metadata.get("border_trim", {}).get("applied"))
    trim_applied = (image_ready and not image_is_full) if crop_modes else trim_applied
    trimmed_path = output_dir / f"{photo.id}_source_corrected_trimmed.jpg" if trim_applied else None
    try:
        await asyncio.to_thread(save_jpeg, untrimmed_path, crop_result.rectified_untrimmed)
        if trimmed_path is not None:
            await asyncio.to_thread(save_jpeg, trimmed_path, crop_result.image_only if crop_modes else corrected)
    except (OSError, ValueError, cv2.error):
        untrimmed_path.unlink(missing_ok=True)
        if trimmed_path is not None:
            trimmed_path.unlink(missing_ok=True)
        raise
    source_path = untrimmed_path if crop_modes else trimmed_path or untrimmed_path
    photo.original_storage_path = str(source_path)
    photo.storage_path = str(source_path)
    correction_metadata = correction_metadata | {
        "status": "completed",
        "source_corrected_path": str(source_path),
        "untrimmed_path": str(untrimmed_path),
        "trimmed_path": str(trimmed_path) if trimmed_path else None,
    }
    parent_asset = await asset_for_path(session, Path(photo.page.storage_path))
    untrimmed_asset = await record_asset(
        session,
        run,
        path=untrimmed_path,
        role="corrected_print_untrimmed" if trimmed_path else "corrected_print",
        stage=stage,
        parent_asset=parent_asset,
        metadata=correction_metadata,
    )
    trimmed_asset = None
    if trimmed_path is not None:
        trimmed_asset = await record_asset(
            session,
            run,
            path=trimmed_path,
            role="corrected_print",
            stage=stage,
            parent_asset=parent_asset if crop_modes else untrimmed_asset,
            metadata=image_mode if crop_modes else correction_metadata.get("border_trim") or {},
        )
    correction_metadata = correction_metadata | {
        "asset_id": str((untrimmed_asset if crop_modes else trimmed_asset or untrimmed_asset).id),
        "untrimmed_asset_id": str(untrimmed_asset.id),
        "trimmed_asset_id": str(trimmed_asset.id) if trimmed_asset else None,
    }
    reason_codes: list[str] = []
    if orientation_reason:
        reason_codes.append(orientation_reason)
    if correction_metadata["protected_regions"].get("review_untrimmed"):
        reason_codes.append(ReviewReason.POSSIBLE_CONTENT_CUTOFF.value)
    prior_metadata = dict(photo.enhancement_applied or {})
    prior_modes = (prior_metadata.get("correction") or {}).get("crop_modes") or {}
    prior_reasons = set((prior_modes.get("full_print") or {}).get("reason_codes") or [])
    if prior_reasons:
        prior_metadata["review_reasons"] = [
            reason for reason in (prior_metadata.get("review_reasons") or []) if reason not in prior_reasons
        ]
        photo.enhancement_applied = prior_metadata
        if photo.status == PhotoStatus.review_needed and not prior_metadata["review_reasons"]:
            photo.status = PhotoStatus.pending
    if full_mode and not full_ready:
        reason_codes.extend(full_mode.get("reason_codes") or ["CROP_GEOMETRY_UNCERTAIN"])
    metadata = _update_photo_decision_metadata(
        photo,
        component_updates={
            "boundary_confidence": correction_metadata.get("boundary_confidence"),
            "geometry_confidence": correction_metadata.get("geometry_confidence"),
            "perspective_confidence": correction_metadata.get("perspective_confidence"),
            "orientation_confidence": correction_metadata.get("orientation_confidence"),
            "border_trim_confidence": correction_metadata.get("border_trim_confidence"),
        },
        reason_codes=reason_codes,
    )
    # A new correction invalidates every variant generated from the previous crop.
    metadata.pop("released_asset_id", None)
    metadata.pop("released_asset_path", None)
    variants = {
        "source_corrected": _source_variant_record("source_corrected", source_path, selected=True),
        "source_corrected_untrimmed": _source_variant_record(
            "source_corrected_untrimmed", untrimmed_path, selected=False
        ),
    }
    if trimmed_path is not None:
        variants["source_corrected_trimmed"] = _source_variant_record(
            "source_corrected_trimmed", trimmed_path, selected=False, metadata=correction_metadata.get("border_trim")
        )
    if full_ready:
        variants["full_print"] = _source_variant_record("full_print", source_path, selected=False, metadata=full_mode)
    if image_ready:
        variants["image_only"] = _source_variant_record(
            "image_only", trimmed_path or untrimmed_path, selected=False, metadata=image_mode
        )
    metadata["variants"] = variants
    metadata["selected_variant"] = "source_corrected"
    metadata["correction"] = correction_metadata
    photo.enhancement_applied = metadata
    photo.segmentation_mask = apply_refined_visibility(photo.segmentation_mask, correction_metadata)
    session.add(
        AuditLog(
            entity_type="photo",
            entity_id=photo.id,
            action="corrected",
            details={"path": str(source_path), **correction_metadata},
        )
    )
    stage_status = RunStatus.review_needed if photo.status == PhotoStatus.review_needed else RunStatus.completed
    finish_stage(stage, status=stage_status, details=correction_metadata, output_path=source_path)
    await session.commit()
    return {
        "photo_id": str(photo.id),
        "storage_path": str(source_path),
        "untrimmed_path": str(untrimmed_path),
        "trimmed_path": str(trimmed_path) if trimmed_path else None,
        "review_required": bool(full_mode and not full_ready),
        "reason": next(iter(full_mode.get("reason_codes", [])), "CROP_GEOMETRY_UNCERTAIN") if full_mode else None,
        "review_flagged": photo.status == PhotoStatus.review_needed,
    }


async def ocr_task_logic(session: AsyncSession, page_id: UUID, settings: Settings) -> dict:
    page = await session.scalar(select(Page).options(selectinload(Page.photos)).where(Page.id == page_id))
    if page is None:
        raise ValueError(f"Page not found: {page_id}")
    run = await get_or_create_run(session, page, settings)
    stage = await start_stage(session, run, "ocr", input_path=Path(page.storage_path))

    page_path = Path(page.storage_path)
    image = await asyncio.to_thread(load_image, page_path)
    try:
        blocks, ocr_metadata = await asyncio.to_thread(run_paddle_ocr, image, settings)
    except Exception as exc:  # noqa: BLE001 - caption recognition is optional for cutouts.
        blocks, ocr_metadata = [], {"ocr_engine": "paddleocr", "available": False, "error": str(exc)}
    if not blocks:
        try:
            blocks, sidecar_metadata = await asyncio.to_thread(run_sidecar_ocr, page_path)
        except Exception as exc:  # noqa: BLE001 - invalid sidecars must not block cutouts.
            sidecar_metadata = {"ocr_engine": "sidecar", "available": False, "error": str(exc)}
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
    session.add(
        AuditLog(
            entity_type="page", entity_id=page.id, action="ocr_extracted", details={"blocks": len(classified_blocks)}
        )
    )
    finish_stage(stage, status=RunStatus.completed, details=ocr_metadata | {"blocks": len(classified_blocks)})
    await session.commit()
    return {"page_id": str(page.id), "ocr_blocks": len(classified_blocks)}


async def enhancement_task_logic(
    session: AsyncSession, photo_id: UUID, settings: Settings, preset: str | None = "auto"
) -> dict:
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

    source_img = await asyncio.to_thread(load_image, source_path)
    degradation_report = await asyncio.to_thread(analyze_degradation, source_img)
    effective_preset = degradation_report.recommended_preset if preset in {None, "", "auto"} else preset

    conservative = None
    conservative_path = None
    conservative_metadata: dict = {"method": "disabled", "reason": "conservative_enhancement_disabled", "models": []}
    if settings.enable_conservative_enhancement:
        ENHANCEMENT_ROUTES.labels(route="conservative_deterministic").inc()
        conservative, conservative_metadata = await asyncio.to_thread(
            conservative_enhance_photo,
            source_img,
            preset=effective_preset,
            degradation_report=degradation_report,
        )
        if conservative_metadata.get("planner", {}).get("no_op"):
            ENHANCEMENT_ROUTES.labels(route="conservative_noop").inc()
            conservative = None
        else:
            conservative_path = (
                settings.storage_path
                / "processed"
                / str(photo.album_id)
                / "runs"
                / str(run.id)
                / "stages"
                / str(stage.id)
                / "photos"
                / f"{photo.id}_conservative.jpg"
            )
            save_jpeg(conservative_path, conservative)
    else:
        ENHANCEMENT_ROUTES.labels(route="conservative_disabled").inc()

    restoration = None
    restoration_path = None
    restoration_metadata: dict = {"method": "disabled", "reason": "deep_enhancement_disabled", "models": []}
    if settings.enable_deep_enhancement:
        ENHANCEMENT_ROUTES.labels(route="restoration_candidate").inc()
        enhancement_config = get_enhancement_config(
            effective_preset, recommended_preset=degradation_report.recommended_preset
        )
        model_artifacts: dict[str, dict] = {}
        registration_error: Exception | None = None
        if settings.require_registered_models:
            from app.pipeline.model_registry import verify_model_artifact

            try:
                model_artifacts["realesrgan"] = verify_model_artifact(
                    settings.realesrgan_model_path,
                    settings.model_manifest_path,
                    model_id=settings.realesrgan_model_path.name,
                )
                if enhancement_config.face_enhancement_enabled and degradation_report.should_enhance_faces:
                    model_artifacts["gfpgan"] = verify_model_artifact(
                        settings.gfpgan_model_path,
                        settings.model_manifest_path,
                        model_id=settings.gfpgan_model_path.name,
                    )
            except Exception as exc:  # noqa: BLE001 - optional model route must degrade to the faithful source.
                registration_error = exc
        if registration_error is not None:
            restoration_metadata = {
                "method": "failed",
                "variant_kind": "restoration_candidate",
                "error": str(registration_error),
                "reason": "model_registration_failed",
                "models": [],
                "model_artifacts": model_artifacts,
                "auto_release": False,
            }
            ENHANCEMENT_ROUTES.labels(route="restoration_registration_failed").inc()
        else:
            restoration, restoration_metadata = await asyncio.to_thread(
                dl_enhance_photo,
                source_img,
                preset=effective_preset,
                degradation_report=degradation_report,
                model_paths={"realesrgan": settings.realesrgan_model_path, "gfpgan": settings.gfpgan_model_path},
                allow_model_restoration=True,
                base_image=conservative,
                model_artifacts=model_artifacts,
            )
            if restoration_metadata.get("method") == "failed" or restoration_metadata.get("planner", {}).get("no_op"):
                restoration = None
                ENHANCEMENT_ROUTES.labels(route="restoration_noop_or_failed").inc()
            else:
                restoration_path = (
                    settings.storage_path
                    / "processed"
                    / str(photo.album_id)
                    / "runs"
                    / str(run.id)
                    / "stages"
                    / str(stage.id)
                    / "photos"
                    / f"{photo.id}_restoration.jpg"
                )
                save_jpeg(restoration_path, restoration)
    else:
        ENHANCEMENT_ROUTES.labels(route="restoration_disabled").inc()

    premium = None
    premium_path = None
    premium_metadata: dict = {"method": "disabled", "reason": "premium_enhancement_feature_flag_disabled", "models": []}
    if settings.enable_premium_enhancement:
        premium, premium_metadata = await asyncio.to_thread(
            premium_enhance_photo,
            source_img,
            base_restored=restoration if restoration is not None else conservative,
            base_metadata=restoration_metadata if restoration is not None else conservative_metadata,
            preset=effective_preset,
            degradation_report=degradation_report,
        )
        premium_path = (
            settings.storage_path
            / "processed"
            / str(photo.album_id)
            / "runs"
            / str(run.id)
            / "stages"
            / str(stage.id)
            / "photos"
            / f"{photo.id}_premium.jpg"
        )
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
    variant_bundle = await asyncio.to_thread(
        build_variant_bundle,
        original_path=Path(photo.original_storage_path or source_path),
        original=source_img,
        premium_path=premium_path,
        premium=premium,
        premium_metadata=premium_metadata,
        photo_status=photo.status,
        release_metadata=previous_metadata,
        canonical_kinds=True,
        conservative_path=conservative_path,
        conservative=conservative,
        conservative_metadata=conservative_metadata,
        restoration_path=restoration_path,
        restoration=restoration,
        restoration_metadata=restoration_metadata,
    )
    for preserved_key in ("review_reasons", "correction", "release_policy", "confidence_components"):
        if previous_metadata.get(preserved_key):
            variant_bundle[preserved_key] = previous_metadata[preserved_key]
    prior_variants = dict(previous_metadata.get("variants") or {})
    for name in ("source_corrected_untrimmed", "source_corrected_trimmed", "full_print", "image_only"):
        if name in prior_variants:
            variant_bundle["variants"][name] = prior_variants[name]
    variant_bundle["degradation"] = degradation_report.as_dict()
    variant_bundle["standard"] = conservative_metadata
    variant_bundle["restoration"] = restoration_metadata
    variant_bundle["premium"] = premium_metadata

    conservative_record = variant_bundle["variants"].get("conservative_enhanced") or {}
    failed_gates = [gate for gate in conservative_record.get("gates") or [] if gate.get("passed") is False]
    if conservative is None:
        enhancement_safety_confidence = 1.0
    elif failed_gates or conservative_metadata.get("final_quality_gate") == "reverted":
        enhancement_safety_confidence = 0.58
    elif any(step.get("quality_gate") == "reverted" for step in conservative_metadata.get("steps") or []):
        enhancement_safety_confidence = 0.82
    else:
        enhancement_safety_confidence = 0.92
    components = confidence_components(
        variant_bundle.get("confidence_components"),
        enhancement_safety_confidence=enhancement_safety_confidence,
    )
    enhancement_reasons = [str(reason) for reason in variant_bundle.get("review_reasons") or []]
    enhancement_reasons.extend(component_review_reasons(components))
    if restoration is not None:
        enhancement_reasons.append(ReviewReason.ENHANCEMENT_ARTIFACT_RISK.value)
        if any("gfpgan" in str(model).lower() for model in restoration_metadata.get("models") or []):
            enhancement_reasons.append(ReviewReason.FACE_RESTORATION_RISK.value)
    if premium is not None:
        enhancement_reasons.append(ReviewReason.ENHANCEMENT_ARTIFACT_RISK.value)
    variant_bundle["confidence_components"] = components.as_dict()
    variant_bundle["review_reasons"] = list(dict.fromkeys(enhancement_reasons))
    if variant_bundle["review_reasons"] and photo.status != PhotoStatus.failed:
        photo.status = PhotoStatus.review_needed

    source_asset = await asset_for_path(session, source_path)
    produced_assets: dict[str, str] = {}
    for role, candidate_path, candidate_metadata in (
        ("conservative_enhanced", conservative_path, conservative_metadata),
        ("restoration_candidate", restoration_path, restoration_metadata),
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

    photo.enhancement_applied = variant_bundle
    selected_variant = str(variant_bundle["selected_variant"])
    selected_record = variant_bundle["variants"].get(selected_variant) or variant_bundle["variants"]["source_corrected"]
    selected_path = str(selected_record["path"])
    photo.storage_path = selected_path
    session.add(
        AuditLog(
            entity_type="photo",
            entity_id=photo.id,
            action="enhanced",
            details=photo.enhancement_applied | {"path": selected_path},
        )
    )
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

    tenant_id = await session.scalar(select(Album.tenant_id).where(Album.id == photo.album_id))
    existing = await session.scalars(
        select(ExtractedPhoto)
        .join(Album, Album.id == ExtractedPhoto.album_id)
        .where(
            ExtractedPhoto.id != photo.id,
            ExtractedPhoto.phash.is_not(None),
            Album.tenant_id == tenant_id,
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
    selected_variant_name = str(metadata.get("selected_variant") or "source_corrected")
    selected_variant = dict(
        variants.get(selected_variant_name) or variants.get("source_corrected") or variants.get("original") or {}
    )
    if selected_variant:
        selected_variant["path"] = str(final_path)
        variants[
            selected_variant_name
            if selected_variant_name in variants
            else ("source_corrected" if "source_corrected" in variants else "original")
        ] = selected_variant
        metadata["variants"] = variants
    metadata["released_asset_path"] = str(final_path)
    metadata["released_asset_id"] = str(released_asset.id)
    photo.enhancement_applied = metadata
    photo.storage_path = str(final_path)
    if photo.status not in {PhotoStatus.review_needed, PhotoStatus.failed}:
        photo.status = PhotoStatus.completed
    session.add(
        AuditLog(
            entity_type="photo", entity_id=photo.id, action="persisted", details=exif_metadata | {"caption": caption}
        )
    )
    finish_stage(
        stage,
        status=RunStatus.review_needed if photo.status == PhotoStatus.review_needed else RunStatus.completed,
        details=exif_metadata | {"caption": caption, "asset_id": str(released_asset.id)},
        output_path=final_path,
    )
    await session.commit()
    return {"photo_id": str(photo.id), "status": photo.status.value, "exif": exif_metadata}


async def _update_album_progress(session: AsyncSession, album_id: UUID) -> None:
    await session.flush()
    album = await session.get(Album, album_id, with_for_update=True)
    counts = dict(
        (await session.execute(
            select(Page.status, func.count(Page.id)).where(Page.album_id == album_id).group_by(Page.status)
        )).all()
    )
    album.processed_pages = counts.get(PageStatus.completed, 0) + counts.get(PageStatus.review_needed, 0)
    if any(counts.get(state) for state in (PageStatus.uploaded, PageStatus.queued, PageStatus.processing)):
        album.status = AlbumStatus.processing
    else:
        album.status = AlbumStatus.failed if counts.get(PageStatus.failed) else AlbumStatus.completed


async def process_page_pipeline(page_id: UUID, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    async with get_session_factory()() as session:
        page = await session.get(Page, page_id)
        if page is None:
            raise ValueError(f"Page not found: {page_id}")
        ready_photos: dict[UUID, dict] = {}
        already_processed = page.status == PageStatus.completed
        if page.status not in {PageStatus.uploaded, PageStatus.completed}:
            photos = (await session.scalars(select(ExtractedPhoto).where(ExtractedPhoto.page_id == page_id))).all()
            for photo in photos:
                released_path = (photo.enhancement_applied or {}).get("released_asset_path")
                if not photo.storage_path or not Path(photo.storage_path).is_file():
                    continue
                if (
                    page.status == PageStatus.review_needed
                    or photo.status == PhotoStatus.completed
                    or released_path == photo.storage_path
                ):
                    ready_photos[photo.id] = {"photo_id": str(photo.id), "status": photo.status.value}
            already_processed = (
                page.status == PageStatus.review_needed and bool(photos) and len(photos) == len(ready_photos)
            )
        if already_processed:
            return {
                "page_id": str(page_id),
                "status": page.status.value,
                "photos": [],
                "idempotent_replay": True,
            }
        page.status = PageStatus.processing
        page.error_message = None
        await session.flush()
        await session.execute(update(Album).where(Album.id == page.album_id).values(status=AlbumStatus.processing))
        await session.commit()
        ingestion = await ingestion_task_logic(session, page_id, settings, update_page_status=False)
        segmentation = await segmentation_task_logic(session, page_id, settings, update_page_status=False)
        photo_ids = list(map(UUID, segmentation["photo_ids"]))
        photo_ids = [photo_id for photo_id in photo_ids if photo_id not in ready_photos]
        correction_source = await asyncio.to_thread(load_image, Path(page.storage_path)) if photo_ids else None
        corrections = [
            await perspective_correction_task_logic(
                session, photo_id, settings, skip_if_exists=True, page_image=correction_source
            )
            for photo_id in photo_ids
        ]
        del correction_source
        await ocr_task_logic(session, page_id, settings)
        photo_results = list(ready_photos.values())
        for photo_id, correction in zip(photo_ids, corrections):
            if correction.get("review_required"):
                photo_results.append({
                    "photo_id": str(photo_id),
                    "status": PhotoStatus.review_needed.value,
                    "reason": correction["reason"],
                })
                continue
            await enhancement_task_logic(session, photo_id, settings)
            await deduplication_task_logic(session, photo_id, settings)
            photo_results.append(await persistence_task_logic(session, photo_id, settings))

        failed_photos = [result for result in photo_results if result["status"] == PhotoStatus.failed.value]
        review_photos = [result for result in photo_results if result["status"] == PhotoStatus.review_needed.value]
        if failed_photos:
            page.status = PageStatus.failed
        elif review_photos or not photo_results or ingestion["blur_score"] < settings.blur_threshold:
            page.status = PageStatus.review_needed
        else:
            page.status = PageStatus.completed
        await _update_album_progress(session, page.album_id)
        session.add(
            AuditLog(entity_type="page", entity_id=page.id, action="completed", details={"status": page.status.value})
        )
        run = await get_or_create_run(session, page, settings)
        run_status = {
            PageStatus.completed: RunStatus.completed,
            PageStatus.review_needed: RunStatus.review_needed,
            PageStatus.failed: RunStatus.failed,
        }.get(page.status, RunStatus.running)
        finish_run(
            run, status=run_status, details={"page_status": page.status.value, "photo_count": len(photo_results)}
        )
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
                    failed_stage = await start_stage(
                        session, run, "pipeline_exception", input_path=Path(page.storage_path)
                    )
                    finish_stage(failed_stage, status=RunStatus.failed, details={"error": str(exc)})
                    finish_run(run, status=RunStatus.failed, details={"error": str(exc)})
                    dlq = quarantine_page_failure(
                        page, settings or get_settings(), reason="pipeline_exception", details={"error": str(exc)}
                    )
                    _append_metadata(page, "dlq", dlq | {"reason": "pipeline_exception"})
                    await _update_album_progress(session, page.album_id)
                    session.add(
                        AuditLog(
                            entity_type="page", entity_id=page.id, action="failed", details={"error": str(exc), **dlq}
                        )
                    )
                    await session.commit()
                    logger.exception(
                        "page_pipeline_failed", page_id=str(page.id), dlq_manifest_path=dlq["dlq_manifest_path"]
                    )
            results.append({"page_id": str(page_id), "status": "failed", "error": str(exc)})
    return {"pages": results}


async def process_pages_serialized(page_ids: list[UUID], settings: Settings | None = None) -> dict:
    """Serialize local background inference to protect SQLite and one GPU.

    Uploaded and partially processed pages can be replayed after a restart.
    """
    async with _local_pipeline_lock:
        return await process_pages_pipeline(page_ids, settings)


def run_async(coro):
    return asyncio.run(coro)
