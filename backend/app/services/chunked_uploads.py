import re
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.schemas.pages import PageUploadResponse
from app.services.uploads import save_validated_page_uploads
from app.utils.file_security import detect_mime_type, validate_original_filename, ALLOWED_MIME_TYPES, ValidatedUpload


CONTENT_RANGE_PATTERN = re.compile(r"^bytes (?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+)$")


async def receive_upload_chunk(
    *,
    session: AsyncSession,
    settings: Settings,
    album_id: UUID,
    upload_id: str | None,
    original_filename: str | None,
    content_range: str | None,
    chunk: UploadFile,
) -> dict | PageUploadResponse:
    if not content_range:
        raise HTTPException(status_code=status.HTTP_411_LENGTH_REQUIRED, detail="Content-Range header is required.")
    match = CONTENT_RANGE_PATTERN.match(content_range)
    if match is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Range header.")

    start = int(match.group("start"))
    end = int(match.group("end"))
    total = int(match.group("total"))
    if end < start or total <= 0 or end >= total:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chunk range.")
    if total > settings.max_upload_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds upload limit.")

    safe_filename = validate_original_filename(original_filename or chunk.filename)
    chunk_bytes = await chunk.read()
    expected_size = end - start + 1
    if len(chunk_bytes) != expected_size:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chunk size does not match Content-Range.")

    upload_token = upload_id or str(uuid4())
    if any(marker in upload_token for marker in ("..", "/", "\\", "\x00")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload id.")

    chunk_dir = settings.storage_path / "uploads" / ".chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"{upload_token}.part"

    mode = "r+b" if chunk_path.exists() else "w+b"
    with chunk_path.open(mode) as handle:
        handle.seek(start)
        handle.write(chunk_bytes)

    if end + 1 < total:
        return {
            "upload_id": upload_token,
            "status": "partial",
            "received_bytes": end + 1,
            "total_bytes": total,
        }

    content = chunk_path.read_bytes()
    chunk_path.unlink(missing_ok=True)
    if len(content) != total:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Assembled upload size mismatch.")

    detected_mime_type = detect_mime_type(content)
    if detected_mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only JPEG and PNG uploads are allowed.")

    return await save_validated_page_uploads(
        session=session,
        settings=settings,
        album_id=album_id,
        validated_uploads=[
            ValidatedUpload(
                original_filename=safe_filename,
                content=content,
                mime_type=detected_mime_type,
                extension=ALLOWED_MIME_TYPES[detected_mime_type],
            )
        ],
    )
