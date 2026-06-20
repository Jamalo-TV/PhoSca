from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.rate_limit import limiter
from app.database import get_async_session
from app.schemas import PageUploadResponse
from app.services.chunked_uploads import receive_upload_chunk
from app.services.uploads import save_page_uploads

router = APIRouter(prefix="/api/v1/albums/{album_id}/pages", tags=["pages"])


@router.post("/upload", response_model=PageUploadResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def upload_pages(
    request: Request,
    album_id: UUID,
    files: list[UploadFile] | None = File(default=None),
    file: list[UploadFile] | None = File(default=None),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> PageUploadResponse:
    uploaded_files = (files or []) + (file or [])
    if not uploaded_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required.")
    return await save_page_uploads(session=session, settings=settings, album_id=album_id, files=uploaded_files)


@router.post("/upload/chunk")
@limiter.limit("30/minute")
async def upload_page_chunk(
    request: Request,
    album_id: UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
):
    return await receive_upload_chunk(
        session=session,
        settings=settings,
        album_id=album_id,
        upload_id=request.headers.get("X-Upload-ID"),
        original_filename=request.headers.get("X-Filename"),
        content_range=request.headers.get("Content-Range"),
        chunk=file,
    )
