from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.auth import Principal, album_for_principal, require_write_principal
from app.core.rate_limit import limiter
from app.database import get_async_session
from app.models import PageStatus
from app.schemas import PageUploadResponse
from app.services.chunked_uploads import receive_upload_chunk
from app.services.pipeline import process_pages_serialized
from app.services.uploads import save_page_uploads

router = APIRouter(prefix="/api/v1/albums/{album_id}/pages", tags=["pages"])


def _schedule_analysis(response: PageUploadResponse, background_tasks: BackgroundTasks, settings: Settings) -> None:
    if not settings.auto_analyze_uploads:
        return
    page_ids = [
        item.page_id
        for item in response.pages
        if item.status in {PageStatus.uploaded, PageStatus.queued, PageStatus.processing}
    ]
    if page_ids:
        background_tasks.add_task(process_pages_serialized, page_ids, settings)
        response.analysis_queued = True


@router.post("/upload", response_model=PageUploadResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def upload_pages(
    request: Request,
    background_tasks: BackgroundTasks,
    album_id: UUID,
    files: list[UploadFile] | None = File(default=None),
    file: list[UploadFile] | None = File(default=None),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
) -> PageUploadResponse:
    uploaded_files = (files or []) + (file or [])
    if not uploaded_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required.")
    await album_for_principal(session, album_id, principal)
    response = await save_page_uploads(
        session=session,
        settings=settings,
        album_id=album_id,
        files=uploaded_files,
        idempotent_retry=request.headers.get("X-Upload-Retry") == "1",
    )
    _schedule_analysis(response, background_tasks, settings)
    return response


@router.post("/upload/chunk")
@limiter.limit("30/minute")
async def upload_page_chunk(
    request: Request,
    background_tasks: BackgroundTasks,
    album_id: UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
    principal: Principal | None = Depends(require_write_principal),
):
    await album_for_principal(session, album_id, principal)
    response = await receive_upload_chunk(
        session=session,
        settings=settings,
        album_id=album_id,
        upload_id=request.headers.get("X-Upload-ID"),
        original_filename=request.headers.get("X-Filename"),
        content_range=request.headers.get("Content-Range"),
        chunk=file,
        idempotent_retry=request.headers.get("X-Upload-Retry") == "1",
    )
    if isinstance(response, PageUploadResponse):
        _schedule_analysis(response, background_tasks, settings)
    return response
