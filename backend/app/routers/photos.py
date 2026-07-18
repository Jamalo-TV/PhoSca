from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.auth import Principal, get_principal, require_write_principal
from app.database import get_async_session
from app.models import AuditLog, ExtractedPhoto, OCRResult, Page, PhotoStatus, ProcessingRun, StageRun
from app.pipeline.presets import available_presets
from app.pipeline.variants import (
    released_variant_from_photo,
    variant_path_from_photo,
    variant_urls,
    variants_from_metadata,
)
from app.schemas import BoundingBoxUpdate, OCRRead, OCRUpdate, PhotoDetail, PhotoRead
from app.services.pipeline import (
    deduplication_task_logic,
    enhancement_task_logic,
    persistence_task_logic,
    perspective_correction_task_logic,
)

router = APIRouter(prefix="/api/v1", tags=["photos"])


async def _photo_for_principal(session: AsyncSession, photo_id: UUID, principal: Principal | None) -> ExtractedPhoto:
    query = select(ExtractedPhoto).where(ExtractedPhoto.id == photo_id)
    if principal is not None:
        query = query.where(ExtractedPhoto.album.has(tenant_id=principal.tenant_id))
    photo = await session.scalar(query)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")
    return photo


async def _page_for_principal(session: AsyncSession, page_id: UUID, principal: Principal | None) -> Page:
    query = select(Page).where(Page.id == page_id)
    if principal is not None:
        query = query.where(Page.album.has(tenant_id=principal.tenant_id))
    page = await session.scalar(query)
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found.")
    return page


def _photo_urls(photo: ExtractedPhoto) -> dict[str, str | None]:
    return variant_urls(photo)


def _rect_mask_from_box(box: dict[str, float]) -> dict:
    polygon = [
        {"x": box["x1"], "y": box["y1"]},
        {"x": box["x2"], "y": box["y1"]},
        {"x": box["x2"], "y": box["y2"]},
        {"x": box["x1"], "y": box["y2"]},
    ]
    return {
        "polygon": polygon,
        "visible_polygon": polygon,
        "warp_quad": polygon,
        "source": "manual_rect",
    }


async def _prediction_provenance(session: AsyncSession, photo: ExtractedPhoto) -> dict:
    page = await session.get(Page, photo.page_id)
    if page is None:
        return {}
    run = await session.scalar(
        select(ProcessingRun).where(ProcessingRun.page_id == page.id).order_by(ProcessingRun.created_at.desc()).limit(1)
    )
    stage = None
    if run is not None:
        stage = await session.scalar(
            select(StageRun)
            .where(StageRun.processing_run_id == run.id, StageRun.stage == "segmentation")
            .order_by(StageRun.attempt.desc())
            .limit(1)
        )
    return {
        "page_id": str(page.id),
        "page_sha256": page.file_hash_sha256,
        "run_id": str(run.id) if run else None,
        "pipeline_version": run.pipeline_version if run else None,
        "config_fingerprint": run.config_fingerprint if run else None,
        "segmentation_stage_id": str(stage.id) if stage else None,
        "segmentation_stage_attempt": stage.attempt if stage else None,
        "segmentation_stage_details": deepcopy(stage.details or {}) if stage else {},
        "segmentation": deepcopy((page.processing_metadata or {}).get("steps", {}).get("segmentation", {})),
    }


def _geometry_snapshot(photo: ExtractedPhoto) -> dict:
    return {
        "bounding_box": deepcopy(photo.bounding_box),
        "segmentation_mask": deepcopy(photo.segmentation_mask),
        "segmentation_confidence": photo.segmentation_confidence,
        "aspect_ratio": photo.aspect_ratio,
        "geometry_valid": photo.geometry_valid,
        "status": photo.status.value,
    }


async def _restore_review_provenance(session: AsyncSession, photo_id: UUID, review_provenance: dict) -> ExtractedPhoto:
    photo = await session.get(ExtractedPhoto, photo_id, populate_existing=True)
    if photo is None:
        raise ValueError(f"Photo not found after correction: {photo_id}")
    metadata = deepcopy(photo.enhancement_applied or {})
    metadata["review_provenance"] = deepcopy(review_provenance)
    photo.enhancement_applied = metadata
    await session.commit()
    return photo


@router.get("/presets")
async def get_presets() -> dict[str, dict]:
    return available_presets()


@router.get("/photos/{photo_id}", response_model=PhotoDetail)
async def get_photo(
    photo_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> PhotoDetail:
    photo = await _photo_for_principal(session, photo_id, principal)
    await session.refresh(photo, attribute_names=["ocr_results"])
    data = PhotoDetail.model_validate(photo)
    data.ocr_results = [OCRRead.model_validate(result) for result in photo.ocr_results]
    data.urls = _photo_urls(photo)
    return data


@router.get("/photos/{photo_id}/image")
async def get_photo_image(
    photo_id: UUID,
    variant: str | None = Query(default=None, pattern="^[a-zA-Z0-9_-]+$"),
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> FileResponse:
    photo = await _photo_for_principal(session, photo_id, principal)
    selected_variant = variant or released_variant_from_photo(photo)
    selected_path = variant_path_from_photo(photo, selected_variant)
    if not selected_path or not Path(selected_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    return FileResponse(
        selected_path,
        media_type="image/jpeg",
        filename=Path(selected_path).name,
        content_disposition_type="inline",
    )


@router.get("/photos/{photo_id}/variants")
async def get_photo_variants(
    photo_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> dict:
    photo = await _photo_for_principal(session, photo_id, principal)
    return {
        "selected_variant": released_variant_from_photo(photo),
        "variants": variants_from_metadata(photo.enhancement_applied),
        "urls": _photo_urls(photo),
    }


@router.get("/pages/{page_id}/image")
async def get_page_image(
    page_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> FileResponse:
    page = await _page_for_principal(session, page_id, principal)
    path = Path(page.storage_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        content_disposition_type="attachment",
    )


@router.get("/pages/{page_id}/ocr", response_model=list[OCRRead])
async def get_page_ocr(
    page_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[OCRResult]:
    await _page_for_principal(session, page_id, principal)
    return (
        await session.scalars(
            select(OCRResult).where(OCRResult.page_id == page_id).order_by(OCRResult.created_at.asc())
        )
    ).all()


@router.patch("/photos/{photo_id}/bounding-box", response_model=PhotoRead)
async def update_photo_bounding_box(
    photo_id: UUID,
    payload: BoundingBoxUpdate,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> ExtractedPhoto:
    photo = await _photo_for_principal(session, photo_id, principal)
    before = _geometry_snapshot(photo)
    prediction_provenance = await _prediction_provenance(session, photo)
    box = payload.bounding_box.as_dict()
    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bounding box must have positive area.")
    segmentation_mask = payload.segmentation_mask.as_dict() if payload.segmentation_mask else _rect_mask_from_box(box)
    if "visible_polygon" not in segmentation_mask:
        segmentation_mask["visible_polygon"] = deepcopy(segmentation_mask.get("polygon"))
    if "warp_quad" not in segmentation_mask and len(segmentation_mask.get("polygon") or []) == 4:
        segmentation_mask["warp_quad"] = deepcopy(segmentation_mask["polygon"])
    after = {
        "bounding_box": deepcopy(box),
        "segmentation_mask": deepcopy(segmentation_mask),
        "segmentation_confidence": None,
        "aspect_ratio": (box["x2"] - box["x1"]) / max(box["y2"] - box["y1"], 1e-9),
        "geometry_valid": True,
        "status": PhotoStatus.pending.value,
    }
    metadata = deepcopy(photo.enhancement_applied or {})
    review_provenance = deepcopy(metadata.get("review_provenance") or {})
    review_provenance.setdefault("original_prediction", before)
    review_provenance.setdefault("prediction_provenance", prediction_provenance)
    revisions = list(review_provenance.get("correction_revisions") or [])
    revisions.append(
        {
            "revision": len(revisions) + 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "before": before,
            "after": after,
            "source": segmentation_mask.get("source"),
        }
    )
    review_provenance["correction_revisions"] = revisions
    metadata["review_provenance"] = review_provenance
    photo.bounding_box = box
    photo.segmentation_mask = segmentation_mask
    photo.segmentation_confidence = None
    photo.aspect_ratio = after["aspect_ratio"]
    photo.geometry_valid = True
    photo.enhancement_applied = metadata
    photo.status = PhotoStatus.pending
    session.add(
        AuditLog(
            entity_type="photo",
            entity_id=photo.id,
            action="bounding_box_updated",
            details={
                "before": before,
                "after": after,
                "prediction_provenance": prediction_provenance,
                # Retain the legacy fields for existing audit consumers.
                "bounding_box": box,
                "segmentation_mask": segmentation_mask,
            },
        )
    )
    await session.commit()
    correction = await perspective_correction_task_logic(session, photo.id, settings)
    refreshed = await _restore_review_provenance(session, photo.id, review_provenance)
    if correction.get("review_required"):
        return refreshed
    await enhancement_task_logic(session, photo.id, settings)
    await _restore_review_provenance(session, photo.id, review_provenance)
    await deduplication_task_logic(session, photo.id, settings)
    await persistence_task_logic(session, photo.id, settings)
    return await _restore_review_provenance(session, photo.id, review_provenance)


@router.patch("/ocr/{ocr_id}", response_model=OCRRead)
async def update_ocr_result(
    ocr_id: UUID,
    payload: OCRUpdate,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(require_write_principal),
) -> OCRResult:
    query = select(OCRResult).join(Page, Page.id == OCRResult.page_id).where(OCRResult.id == ocr_id)
    if principal is not None:
        query = query.where(Page.album.has(tenant_id=principal.tenant_id))
    ocr = await session.scalar(query)
    if ocr is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OCR result not found.")
    ocr.text_content = payload.text_content
    ocr.text_type = payload.text_type
    ocr.is_verified = payload.is_verified
    session.add(
        AuditLog(entity_type="page", entity_id=ocr.page_id, action="ocr_updated", details={"ocr_id": str(ocr.id)})
    )
    await session.commit()
    if ocr.photo_id:
        await persistence_task_logic(session, ocr.photo_id)
    await session.refresh(ocr)
    return ocr


@router.post("/photos/{photo_id}/reprocess", response_model=PhotoRead)
async def reprocess_photo(
    photo_id: UUID,
    preset: str | None = Query(default="auto"),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> ExtractedPhoto:
    photo = await _photo_for_principal(session, photo_id, principal)
    correction = await perspective_correction_task_logic(session, photo.id, settings)
    if correction.get("review_required"):
        return await session.get(ExtractedPhoto, photo_id)
    await enhancement_task_logic(session, photo.id, settings, preset=preset)
    await deduplication_task_logic(session, photo.id, settings)
    await persistence_task_logic(session, photo.id, settings)
    refreshed = await session.get(ExtractedPhoto, photo_id)
    return refreshed
