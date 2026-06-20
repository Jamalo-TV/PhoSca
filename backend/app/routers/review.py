from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.models import ExtractedPhoto, OCRResult, Page, PageStatus, PhotoStatus
from app.schemas import ReviewItem

router = APIRouter(prefix="/api/v1/review", tags=["review"])


@router.get("/queue", response_model=list[ReviewItem])
async def get_review_queue(
    album_id: UUID | None = None,
    review_type: str = Query(default="segmentation", alias="type", pattern="^(segmentation|ocr|geometry)$"),
    session: AsyncSession = Depends(get_async_session),
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    if review_type == "segmentation":
        query = select(Page).where(Page.status == PageStatus.review_needed)
        if album_id:
            query = query.where(Page.album_id == album_id)
        for page in (await session.scalars(query.order_by(Page.created_at.asc()))).all():
            items.append(
                ReviewItem(
                    item_type="segmentation",
                    id=page.id,
                    album_id=page.album_id,
                    page_id=page.id,
                    status=page.status.value,
                    reason=page.error_message,
                    preview_url=f"/api/v1/pages/{page.id}/image",
                )
            )
    elif review_type == "geometry":
        query = select(ExtractedPhoto).where(ExtractedPhoto.geometry_valid.is_(False))
        if album_id:
            query = query.where(ExtractedPhoto.album_id == album_id)
        for photo in (await session.scalars(query.order_by(ExtractedPhoto.created_at.asc()))).all():
            items.append(
                ReviewItem(
                    item_type="geometry",
                    id=photo.id,
                    album_id=photo.album_id,
                    page_id=photo.page_id,
                    status=photo.status.value,
                    reason="photo aspect ratio failed geometry gate",
                    preview_url=f"/api/v1/photos/{photo.id}/image",
                )
            )
    else:
        query = select(OCRResult, Page.album_id).join(Page, Page.id == OCRResult.page_id).where(OCRResult.is_verified.is_(False))
        if album_id:
            query = query.where(Page.album_id == album_id)
        rows = (await session.execute(query.order_by(OCRResult.created_at.asc()))).all()
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
    return items
