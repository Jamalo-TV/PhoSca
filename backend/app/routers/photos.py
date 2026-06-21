from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.database import get_async_session
from app.models import AuditLog, ExtractedPhoto, OCRResult, Page, PhotoStatus
from app.pipeline.presets import available_presets
from app.pipeline.variants import variant_path_from_photo, variant_urls, variants_from_metadata
from app.schemas import BoundingBoxUpdate, OCRRead, OCRUpdate, PhotoDetail, PhotoRead
from app.services.pipeline import (
    deduplication_task_logic,
    enhancement_task_logic,
    persistence_task_logic,
    perspective_correction_task_logic,
)

router = APIRouter(prefix="/api/v1", tags=["photos"])


def _photo_urls(photo: ExtractedPhoto) -> dict[str, str | None]:
    return variant_urls(photo)


def _rect_mask_from_box(box: dict[str, float]) -> dict:
    return {
        "polygon": [
            {"x": box["x1"], "y": box["y1"]},
            {"x": box["x2"], "y": box["y1"]},
            {"x": box["x2"], "y": box["y2"]},
            {"x": box["x1"], "y": box["y2"]},
        ],
        "source": "manual_rect",
    }


@router.get("/presets")
async def get_presets() -> dict[str, dict]:
    return available_presets()


@router.get("/photos/{photo_id}", response_model=PhotoDetail)
async def get_photo(photo_id: UUID, session: AsyncSession = Depends(get_async_session)) -> PhotoDetail:
    photo = await session.get(ExtractedPhoto, photo_id, options=[selectinload(ExtractedPhoto.ocr_results)])
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")
    data = PhotoDetail.model_validate(photo)
    data.ocr_results = [OCRRead.model_validate(result) for result in photo.ocr_results]
    data.urls = _photo_urls(photo)
    return data


@router.get("/photos/{photo_id}/image")
async def get_photo_image(
    photo_id: UUID,
    variant: str = Query(default="enhanced", pattern="^[a-zA-Z0-9_-]+$"),
    session: AsyncSession = Depends(get_async_session),
) -> FileResponse:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")
    selected_path = variant_path_from_photo(photo, variant)
    if not selected_path or not Path(selected_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    return FileResponse(
        selected_path,
        media_type="image/jpeg",
        filename=Path(selected_path).name,
        content_disposition_type="inline",
    )


@router.get("/photos/{photo_id}/variants")
async def get_photo_variants(photo_id: UUID, session: AsyncSession = Depends(get_async_session)) -> dict:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")
    return {
        "selected_variant": (photo.enhancement_applied or {}).get("selected_variant", "enhanced"),
        "variants": variants_from_metadata(photo.enhancement_applied),
        "urls": _photo_urls(photo),
    }


@router.get("/pages/{page_id}/image")
async def get_page_image(page_id: UUID, session: AsyncSession = Depends(get_async_session)) -> FileResponse:
    page = await session.get(Page, page_id)
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found.")
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
async def get_page_ocr(page_id: UUID, session: AsyncSession = Depends(get_async_session)) -> list[OCRResult]:
    return (await session.scalars(select(OCRResult).where(OCRResult.page_id == page_id).order_by(OCRResult.created_at.asc()))).all()


@router.patch("/photos/{photo_id}/bounding-box", response_model=PhotoRead)
async def update_photo_bounding_box(
    photo_id: UUID,
    payload: BoundingBoxUpdate,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> ExtractedPhoto:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")
    box = payload.bounding_box.as_dict()
    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bounding box must have positive area.")
    segmentation_mask = payload.segmentation_mask.as_dict() if payload.segmentation_mask else _rect_mask_from_box(box)
    photo.bounding_box = box
    photo.segmentation_mask = segmentation_mask
    photo.status = PhotoStatus.pending
    session.add(
        AuditLog(
            entity_type="photo",
            entity_id=photo.id,
            action="bounding_box_updated",
            details={"bounding_box": box, "segmentation_mask": segmentation_mask},
        )
    )
    await session.commit()
    await perspective_correction_task_logic(session, photo.id, settings)
    await enhancement_task_logic(session, photo.id, settings)
    await deduplication_task_logic(session, photo.id)
    await persistence_task_logic(session, photo.id)
    return await session.get(ExtractedPhoto, photo_id)


@router.patch("/ocr/{ocr_id}", response_model=OCRRead)
async def update_ocr_result(
    ocr_id: UUID,
    payload: OCRUpdate,
    session: AsyncSession = Depends(get_async_session),
) -> OCRResult:
    ocr = await session.get(OCRResult, ocr_id)
    if ocr is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OCR result not found.")
    ocr.text_content = payload.text_content
    ocr.text_type = payload.text_type
    ocr.is_verified = payload.is_verified
    session.add(AuditLog(entity_type="page", entity_id=ocr.page_id, action="ocr_updated", details={"ocr_id": str(ocr.id)}))
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
) -> ExtractedPhoto:
    photo = await session.get(ExtractedPhoto, photo_id)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found.")
    await perspective_correction_task_logic(session, photo.id, settings)
    await enhancement_task_logic(session, photo.id, settings, preset=preset)
    await deduplication_task_logic(session, photo.id)
    await persistence_task_logic(session, photo.id)
    refreshed = await session.get(ExtractedPhoto, photo_id)
    return refreshed
