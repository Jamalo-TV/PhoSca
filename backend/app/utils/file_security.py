from dataclasses import dataclass
from io import BytesIO

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import magic
except ImportError:  # pragma: no cover - production images install python-magic.
    magic = None


ALLOWED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


@dataclass(frozen=True)
class ValidatedUpload:
    original_filename: str
    content: bytes
    mime_type: str
    extension: str
    decode_metadata: dict


def inspect_image_content(content: bytes, *, max_image_pixels: int, max_image_frames: int) -> dict:
    """Decode-verify an upload before persisting it.

    MIME detection alone accepts decompression bombs and multi-frame files whose
    semantics the processing pipeline cannot preserve.  We intentionally
    reject those inputs instead of silently selecting an arbitrary frame.
    """
    try:
        with Image.open(BytesIO(content)) as probe:
            image_format = probe.format
            width, height = probe.size
            frames = int(getattr(probe, "n_frames", 1))
            orientation = int(probe.getexif().get(274, 1))
            if width < 1 or height < 1 or width * height > max_image_pixels:
                raise ValueError("image dimensions exceed the configured pixel limit")
            if frames > max_image_frames:
                raise ValueError("multi-frame images are not supported")
            # Force a real decode, rather than only checking container headers.
            probe.seek(0)
            decoded = ImageOps.exif_transpose(probe)
            decoded.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image upload: {exc}") from exc
    return {
        "format": image_format,
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "frames": frames,
        "exif_orientation": orientation,
        "icc_profile_present": bool(getattr(decoded, "info", {}).get("icc_profile")),
    }


def validate_original_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required.")
    stripped = filename.strip()
    if not stripped:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required.")
    if any(marker in stripped for marker in ("..", "/", "\\", "\x00")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")
    return stripped[:255]


def detect_mime_type(content: bytes) -> str:
    if magic is None:
        raise RuntimeError("python-magic is required for upload MIME validation.")
    return magic.from_buffer(content[:4096], mime=True)


async def read_and_validate_upload(
    upload: UploadFile,
    *,
    max_file_size: int,
    max_request_size: int,
    current_request_size: int,
    max_image_pixels: int = 40_000_000,
    max_image_frames: int = 1,
) -> tuple[ValidatedUpload, int]:
    original_filename = validate_original_filename(upload.filename)
    chunks: list[bytes] = []
    file_size = 0

    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        file_size += len(chunk)
        current_request_size += len(chunk)
        if file_size > max_file_size:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds upload limit.")
        if current_request_size > max_request_size:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Request exceeds upload limit.")
        chunks.append(chunk)

    content = b"".join(chunks)
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    detected_mime_type = detect_mime_type(content)
    if detected_mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only JPEG and PNG uploads are allowed.")

    claimed_mime_type = upload.content_type
    if claimed_mime_type and claimed_mime_type != detected_mime_type:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File MIME type does not match its content.")

    decode_metadata = inspect_image_content(
        content,
        max_image_pixels=max_image_pixels,
        max_image_frames=max_image_frames,
    )
    return (
        ValidatedUpload(
            original_filename=original_filename,
            content=content,
            mime_type=detected_mime_type,
            extension=ALLOWED_MIME_TYPES[detected_mime_type],
            decode_metadata=decode_metadata,
        ),
        current_request_size,
    )
