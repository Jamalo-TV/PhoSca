from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.core.auth import Principal, get_principal
from app.models import ExtractedPhoto, OCRResult, Page, PageStatus, PhotoStatus, ReviewEvent
from app.schemas import ReviewItem
from app.pipeline.variants import variants_from_metadata

router = APIRouter(prefix="/api/v1/review", tags=["review"])


class MobileReviewAnswer(BaseModel):
    task_type: str = Field(pattern="^(segmentation|geometry|ocr|enhancement)$")
    task_id: UUID
    answer: str = Field(min_length=1, max_length=120)
    reason_code: str | None = Field(default=None, max_length=120)
    idempotency_key: str = Field(min_length=8, max_length=64)
    reviewer: str = Field(default="local-mobile", max_length=120)
    metadata: dict = Field(default_factory=dict)


ALLOWED_ANSWERS = {
    "segmentation": {"correct", "missing_print", "too_many", "unsure"},
    "geometry": {"correct", "too_tight", "too_wide", "wrong_object", "split_print", "merged_prints", "protected_cut", "unsure"},
    "ocr": {"correct", "incorrect", "unsure"},
    "enhancement": {"left", "right", "equal", "both_bad", "unsure"},
}


@router.post("/events", status_code=status.HTTP_201_CREATED)
async def add_review_event(
    payload: MobileReviewAnswer,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
):
    if payload.answer not in ALLOWED_ANSWERS[payload.task_type]:
        from fastapi import HTTPException

        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unsupported answer for review task type.")
    existing = await session.scalar(select(ReviewEvent).where(ReviewEvent.idempotency_key == payload.idempotency_key))
    if existing is not None:
        return {"id": existing.id, "saved": True, "duplicate": True}
    event = ReviewEvent(
        task_type=payload.task_type,
        task_id=payload.task_id,
        answer=payload.answer,
        reason_code=payload.reason_code,
        idempotency_key=payload.idempotency_key,
        reviewer=payload.reviewer,
        event_metadata={**payload.metadata, "tenant_id": str(principal.tenant_id) if principal else None},
    )
    session.add(event)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await session.scalar(select(ReviewEvent).where(ReviewEvent.idempotency_key == payload.idempotency_key))
        return {"id": existing.id if existing else None, "saved": True, "duplicate": True}
    return {"id": event.id, "saved": True, "duplicate": False}


@router.get("/progress")
async def review_progress(session: AsyncSession = Depends(get_async_session)):
    total = await session.scalar(select(func.count()).select_from(ReviewEvent)) or 0
    rows = (await session.execute(select(ReviewEvent.task_type, func.count()).group_by(ReviewEvent.task_type))).all()
    return {"completed": total, "by_type": {task_type: count for task_type, count in rows}}


@router.get("/queue", response_model=list[ReviewItem])
async def get_review_queue(
    album_id: UUID | None = None,
    review_type: str = Query(default="segmentation", alias="type", pattern="^(segmentation|ocr|geometry|enhancement)$"),
    limit: int = Query(default=25, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    if review_type == "segmentation":
        reviewed = select(ReviewEvent.id).where(ReviewEvent.task_type == "segmentation", ReviewEvent.task_id == Page.id).exists()
        query = select(Page).where(Page.status.in_([PageStatus.review_needed, PageStatus.completed]), ~reviewed)
        if album_id:
            query = query.where(Page.album_id == album_id)
        if principal is not None:
            query = query.where(Page.album.has(tenant_id=principal.tenant_id))
        for page in (await session.scalars(query.order_by(case((Page.status == PageStatus.review_needed, 0), else_=1), Page.created_at.asc()).limit(limit))).all():
            photos = (await session.scalars(select(ExtractedPhoto).where(ExtractedPhoto.page_id == page.id).order_by(ExtractedPhoto.created_at.asc()))).all()
            items.append(
                ReviewItem(
                    item_type="segmentation",
                    id=page.id,
                    album_id=page.album_id,
                    page_id=page.id,
                    status=page.status.value,
                    reason=page.error_message,
                    preview_url=f"/api/v1/pages/{page.id}/image",
                    payload={
                        "detection_count": len(photos),
                        "detections": [
                            {
                                "photo_id": str(photo.id),
                                "bounding_box": photo.bounding_box,
                                "polygon": (photo.segmentation_mask or {}).get("polygon"),
                                "confidence": photo.segmentation_confidence,
                                "status": photo.status.value,
                            }
                            for photo in photos
                        ],
                    },
                )
            )
    elif review_type == "geometry":
        reviewed = select(ReviewEvent.id).where(ReviewEvent.task_type == "geometry", ReviewEvent.task_id == ExtractedPhoto.id).exists()
        query = select(ExtractedPhoto).where(~reviewed)
        if album_id:
            query = query.where(ExtractedPhoto.album_id == album_id)
        if principal is not None:
            query = query.where(ExtractedPhoto.album.has(tenant_id=principal.tenant_id))
        priority = case((ExtractedPhoto.status == PhotoStatus.review_needed, 0), (ExtractedPhoto.geometry_valid.is_(False), 0), else_=1)
        for photo in (await session.scalars(query.order_by(priority, ExtractedPhoto.segmentation_confidence.asc().nullsfirst(), ExtractedPhoto.created_at.asc()).limit(limit))).all():
            items.append(
                ReviewItem(
                    item_type="geometry",
                    id=photo.id,
                    album_id=photo.album_id,
                    page_id=photo.page_id,
                    status=photo.status.value,
                    reason="crop requires confirmation" if photo.status == PhotoStatus.review_needed else "quality-control sample",
                    preview_url=f"/api/v1/pages/{photo.page_id}/image",
                    payload={
                        "photo_id": str(photo.id),
                        "crop_url": f"/api/v1/photos/{photo.id}/image?variant=original" if photo.original_storage_path else None,
                        "bounding_box": photo.bounding_box,
                        "polygon": (photo.segmentation_mask or {}).get("polygon"),
                        "confidence": photo.segmentation_confidence,
                        "review_reasons": (photo.enhancement_applied or {}).get("review_reasons", []),
                    },
                )
            )
    elif review_type == "ocr":
        query = select(OCRResult, Page.album_id).join(Page, Page.id == OCRResult.page_id).where(OCRResult.is_verified.is_(False))
        if album_id:
            query = query.where(Page.album_id == album_id)
        if principal is not None:
            query = query.where(Page.album.has(tenant_id=principal.tenant_id))
        rows = (await session.execute(query.order_by(OCRResult.created_at.asc()).limit(limit))).all()
        for ocr, row_album_id in rows:
            items.append(
                ReviewItem(
                    item_type="ocr",
                    id=ocr.id,
                    album_id=row_album_id,
                    page_id=ocr.page_id,
                    status="unverified",
                    reason=ocr.spatial_classification_reason,
                    preview_url=f"/api/v1/pages/{ocr.page_id}/image",
                )
            )
    else:
        reviewed = select(ReviewEvent.id).where(ReviewEvent.task_type == "enhancement", ReviewEvent.task_id == ExtractedPhoto.id).exists()
        query = select(ExtractedPhoto).where(ExtractedPhoto.original_storage_path.is_not(None), ~reviewed)
        if album_id:
            query = query.where(ExtractedPhoto.album_id == album_id)
        if principal is not None:
            query = query.where(ExtractedPhoto.album.has(tenant_id=principal.tenant_id))
        photos = (await session.scalars(query.order_by(ExtractedPhoto.created_at.asc()))).all()
        for photo in photos:
            variants = variants_from_metadata(photo.enhancement_applied)
            candidate_name = next((name for name in ("conservative", "enhanced", "premium") if isinstance(variants.get(name), dict)), None)
            if candidate_name is None:
                continue
            items.append(
                ReviewItem(
                    item_type="enhancement",
                    id=photo.id,
                    album_id=photo.album_id,
                    page_id=photo.page_id,
                    status=photo.status.value,
                    reason="blind comparison",
                    preview_url=f"/api/v1/photos/{photo.id}/image?variant=original",
                    payload={
                        "photo_id": str(photo.id),
                        "original_url": f"/api/v1/photos/{photo.id}/image?variant=original",
                        "candidate_url": f"/api/v1/photos/{photo.id}/image?variant={candidate_name}",
                        "candidate_name": candidate_name,
                        "degradation": (photo.enhancement_applied or {}).get("degradation", {}),
                    },
                )
            )
            if len(items) >= limit:
                break
    return items
