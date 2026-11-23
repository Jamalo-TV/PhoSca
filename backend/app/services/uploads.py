import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Album, AlbumStatus, AuditLog, Page, PageStatus
from app.schemas.pages import PageUploadItem, PageUploadResponse
from app.utils.file_security import ValidatedUpload, read_and_validate_upload


async def save_page_uploads(
    *,
    session: AsyncSession,
    settings: Settings,
    album_id: UUID,
    files: list[UploadFile],
    idempotent_retry: bool = False,
) -> PageUploadResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required.")

    album = await session.get(Album, album_id)
    if album is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Album not found.")

    request_size = 0
    validated_uploads: list[ValidatedUpload] = []
    for upload in files:
        validated_upload, request_size = await read_and_validate_upload(
            upload,
            max_file_size=settings.max_upload_size,
            max_request_size=settings.max_request_size,
            current_request_size=request_size,
            max_image_pixels=settings.max_image_pixels,
            max_image_frames=settings.max_image_frames,
        )
        validated_uploads.append(validated_upload)

    return await save_validated_page_uploads(
        session=session,
        settings=settings,
        album_id=album_id,
        validated_uploads=validated_uploads,
        idempotent_retry=idempotent_retry,
    )


async def save_validated_page_uploads(
    *,
    session: AsyncSession,
    settings: Settings,
    album_id: UUID,
    validated_uploads: list[ValidatedUpload],
    idempotent_retry: bool = False,
) -> PageUploadResponse:
    album = await session.get(Album, album_id)
    if album is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Album not found.")

    storage_dir = settings.storage_path / "uploads" / str(album_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    created_pages: list[Page] = []
    saved_paths: list[Path] = []

    try:
        for validated_upload in validated_uploads:
            file_hash = hashlib.sha256(validated_upload.content).hexdigest()
            if file_hash in seen_hashes:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Duplicate file in upload request.")
            seen_hashes.add(file_hash)

            existing_page = await session.scalar(
                select(Page).where(Page.album_id == album.id, Page.file_hash_sha256 == file_hash)
            )
            # Mobile retries are intentionally idempotent. Returning the original,
            # hash-verified page lets the client safely remove its offline copy.
            if existing_page is not None:
                if idempotent_retry:
                    created_pages.append(existing_page)
                    continue
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="File was already uploaded.")

            page_id = uuid4()
            storage_path = storage_dir / f"{page_id}{validated_upload.extension}"
            storage_path.write_bytes(validated_upload.content)
            saved_paths.append(storage_path)

            page = Page(
                id=page_id,
                album_id=album.id,
                original_filename=validated_upload.original_filename,
                storage_path=str(storage_path),
                file_size_bytes=len(validated_upload.content),
                file_hash_sha256=file_hash,
                status=PageStatus.uploaded,
                processing_metadata={
                    "upload": {
                        "mime_type": validated_upload.mime_type,
                        "stored_extension": validated_upload.extension,
                        "decode": validated_upload.decode_metadata,
                    }
                },
            )
            session.add(page)
            session.add(
                AuditLog(
                    entity_type="page",
                    entity_id=page.id,
                    action="uploaded",
                    details={
                        "album_id": str(album.id),
                        "original_filename": validated_upload.original_filename,
                        "storage_path": str(storage_path),
                        "file_size_bytes": len(validated_upload.content),
                        "file_hash_sha256": file_hash,
                    },
                )
            )
            created_pages.append(page)

        album.total_pages += len(saved_paths)
        if saved_paths and settings.auto_analyze_uploads:
            album.status = AlbumStatus.processing
        await session.commit()
    except Exception:
        await session.rollback()
        for saved_path in saved_paths:
            if saved_path.exists():
                saved_path.unlink()
        raise

    return PageUploadResponse(
        pages=[
            PageUploadItem(
                page_id=page.id,
                filename=page.original_filename,
                status=page.status,
                sha256=page.file_hash_sha256,
                verified=True,
            )
            for page in created_pages
        ]
    )
