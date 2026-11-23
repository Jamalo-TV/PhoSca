from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.auth import Principal, album_for_principal, get_principal, require_write_principal
from app.database import get_async_session
from app.models import Album, AuditLog, ExtractedPhoto, OCRResult, OCRTextType, Page, PageStatus, PhotoStatus
from app.schemas import AlbumCreate, AlbumRead, AlbumStats, AnalyzeRequest, JobQueued, PageRead, PhotoRead
from app.services.deletion import delete_album_with_assets
from app.services.pipeline import process_pages_serialized
from app.tasks.pipeline import process_pages_task

router = APIRouter(prefix="/api/v1/albums", tags=["albums"])
ANALYZABLE_PAGE_STATUSES = (PageStatus.uploaded, PageStatus.queued, PageStatus.processing, PageStatus.review_needed)


@router.post("", response_model=AlbumRead, status_code=status.HTTP_201_CREATED)
async def create_album(
    payload: AlbumCreate,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(require_write_principal),
) -> Album:
    album = Album(name=payload.name, description=payload.description, tenant_id=principal.tenant_id if principal else None)
    session.add(album)
    await session.flush()
    session.add(
        AuditLog(
            entity_type="album",
            entity_id=album.id,
            action="created",
            details={"name": album.name},
        )
    )
    await session.commit()
    await session.refresh(album)
    return album


@router.get("", response_model=list[AlbumStats])
async def list_albums(
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[AlbumStats]:
    query = select(Album)
    if principal is not None:
        query = query.where(Album.tenant_id == principal.tenant_id)
    albums = (await session.scalars(query.order_by(Album.created_at.desc()))).all()
    results: list[AlbumStats] = []
    for album in albums:
        photos_extracted = await session.scalar(select(func.count(ExtractedPhoto.id)).where(ExtractedPhoto.album_id == album.id))
        review_pages = await session.scalar(
            select(func.count(Page.id)).where(Page.album_id == album.id, Page.status == PageStatus.review_needed)
        )
        review_photos = await session.scalar(
            select(func.count(ExtractedPhoto.id)).where(
                ExtractedPhoto.album_id == album.id,
                ExtractedPhoto.status == PhotoStatus.review_needed,
            )
        )
        results.append(
            AlbumStats(
                id=album.id,
                name=album.name,
                description=album.description,
                directory_path=album.directory_path,
                status=album.status.value,
                total_pages=album.total_pages,
                processed_pages=album.processed_pages,
                photos_extracted=photos_extracted or 0,
                review_needed_count=(review_pages or 0) + (review_photos or 0),
            )
        )
    return results


@router.post("/{album_id}/analyze", response_model=JobQueued)
async def analyze_album(
    album_id: UUID,
    payload: AnalyzeRequest,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(require_write_principal),
) -> JobQueued:
    await album_for_principal(session, album_id, principal)

    if payload.page_ids is None:
        page_ids = (
            await session.scalars(
                select(Page.id).where(
                    Page.album_id == album_id,
                    Page.status.in_(ANALYZABLE_PAGE_STATUSES),
                )
            )
        ).all()
    else:
        page_ids = payload.page_ids
        existing_count = await session.scalar(select(func.count(Page.id)).where(Page.album_id == album_id, Page.id.in_(page_ids)))
        if existing_count != len(page_ids):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more pages do not belong to this album.")

    result = process_pages_task.delay([str(page_id) for page_id in page_ids])
    return JobQueued(job_id=result.id, status="queued", estimated_pages=len(page_ids))


@router.post("/{album_id}/analyze-now")
async def analyze_album_now(
    album_id: UUID,
    payload: AnalyzeRequest,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(require_write_principal),
) -> dict:
    await album_for_principal(session, album_id, principal)

    if payload.page_ids is None:
        page_ids = (
            await session.scalars(
                select(Page.id).where(
                    Page.album_id == album_id,
                    Page.status.in_(ANALYZABLE_PAGE_STATUSES),
                )
            )
        ).all()
    else:
        page_ids = payload.page_ids
        existing_count = await session.scalar(select(func.count(Page.id)).where(Page.album_id == album_id, Page.id.in_(page_ids)))
        if existing_count != len(page_ids):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more pages do not belong to this album.")

    await session.commit()
    return await process_pages_serialized(page_ids)


@router.get("/{album_id}", response_model=AlbumStats)
async def get_album(
    album_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> AlbumStats:
    album = await album_for_principal(session, album_id, principal)
    photos_extracted = await session.scalar(select(func.count(ExtractedPhoto.id)).where(ExtractedPhoto.album_id == album.id))
    review_pages = await session.scalar(select(func.count(Page.id)).where(Page.album_id == album.id, Page.status == PageStatus.review_needed))
    review_photos = await session.scalar(
        select(func.count(ExtractedPhoto.id)).where(ExtractedPhoto.album_id == album.id, ExtractedPhoto.status == PhotoStatus.review_needed)
    )
    return AlbumStats(
        id=album.id,
        name=album.name,
        description=album.description,
        directory_path=album.directory_path,
        status=album.status.value,
        total_pages=album.total_pages,
        processed_pages=album.processed_pages,
        photos_extracted=photos_extracted or 0,
        review_needed_count=(review_pages or 0) + (review_photos or 0),
    )


@router.delete("/{album_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_album(
    album_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> Response:
    album = await album_for_principal(session, album_id, principal)
    await delete_album_with_assets(session, album, settings)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{album_id}/pages", response_model=list[PageRead])
async def list_album_pages(
    album_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[Page]:
    await album_for_principal(session, album_id, principal)
    return (
        await session.scalars(select(Page).where(Page.album_id == album_id).order_by(Page.created_at.asc()))
    ).all()


@router.get("/{album_id}/photos", response_model=list[PhotoRead])
async def list_album_photos(
    album_id: UUID,
    status_filter: PhotoStatus | None = Query(default=None, alias="status"),
    needs_review: bool | None = None,
    has_caption: bool | None = None,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[ExtractedPhoto]:
    await album_for_principal(session, album_id, principal)
    query = select(ExtractedPhoto).where(ExtractedPhoto.album_id == album_id)
    if status_filter is not None:
        query = query.where(ExtractedPhoto.status == status_filter)
    if needs_review is True:
        query = query.where(ExtractedPhoto.status == PhotoStatus.review_needed)
    if has_caption is not None:
        caption_exists = select(OCRResult.id).where(OCRResult.photo_id == ExtractedPhoto.id, OCRResult.text_type == OCRTextType.caption).exists()
        query = query.where(caption_exists if has_caption else ~caption_exists)
    return (await session.scalars(query.order_by(ExtractedPhoto.created_at.asc()))).all()
