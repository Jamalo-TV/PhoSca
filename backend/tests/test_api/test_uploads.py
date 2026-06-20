from pathlib import Path
from uuid import UUID

from httpx import AsyncClient
from sqlalchemy import select


JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"jpeg-data" + b"\xff\xd9"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"png-data"


async def create_album(client: AsyncClient) -> str:
    response = await client.post("/api/v1/albums", json={"name": "Family Archive", "description": None})
    assert response.status_code == 201, response.text
    album_id = response.json()["id"]
    UUID(album_id)
    return album_id


async def test_upload_accepts_valid_jpeg_and_stores_uuid_filename(client: AsyncClient) -> None:
    album_id = await create_album(client)

    response = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("album-page.jpg", JPEG_BYTES, "image/jpeg"))],
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["pages"]) == 1
    assert body["pages"][0]["filename"] == "album-page.jpg"
    assert body["pages"][0]["status"] == "uploaded"

    from app.database import get_session_factory
    from app.models import Album, Page

    async with get_session_factory()() as session:
        page = await session.scalar(select(Page).where(Page.id == UUID(body["pages"][0]["page_id"])))
        album = await session.get(Album, UUID(album_id))

    assert page is not None
    assert album is not None
    assert album.total_pages == 1
    stored_path = Path(page.storage_path)
    assert stored_path.exists()
    assert stored_path.name == f"{page.id}.jpg"
    assert "album-page" not in stored_path.name


async def test_upload_rejects_invalid_magic_mime(client: AsyncClient) -> None:
    album_id = await create_album(client)

    response = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("not-an-image.jpg", b"plain text", "image/jpeg"))],
    )

    assert response.status_code == 400
    assert response.json()["error"] == "http_error"


async def test_upload_rejects_path_traversal_filename(client: AsyncClient) -> None:
    album_id = await create_album(client)

    response = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("../evil.jpg", JPEG_BYTES, "image/jpeg"))],
    )

    assert response.status_code == 400
    assert "request_id" in response.json()


async def test_upload_rejects_oversized_file(client: AsyncClient) -> None:
    album_id = await create_album(client)
    oversized = b"\xff\xd8\xff" + (b"x" * 5000)

    response = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("big.jpg", oversized, "image/jpeg"))],
    )

    assert response.status_code == 413


async def test_upload_rejects_claimed_mime_mismatch(client: AsyncClient) -> None:
    album_id = await create_album(client)

    response = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("page.png", PNG_BYTES, "image/jpeg"))],
    )

    assert response.status_code == 400


async def test_chunked_upload_assembles_and_validates_image(client: AsyncClient) -> None:
    album_id = await create_album(client)
    first = JPEG_BYTES[:8]
    second = JPEG_BYTES[8:]
    upload_id = "chunk-test-upload"

    partial = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload/chunk",
        headers={
            "Content-Range": f"bytes 0-{len(first) - 1}/{len(JPEG_BYTES)}",
            "X-Upload-ID": upload_id,
            "X-Filename": "chunked.jpg",
        },
        files={"file": ("chunk", first, "application/octet-stream")},
    )
    assert partial.status_code == 200, partial.text
    assert partial.json()["status"] == "partial"

    complete = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload/chunk",
        headers={
            "Content-Range": f"bytes {len(first)}-{len(JPEG_BYTES) - 1}/{len(JPEG_BYTES)}",
            "X-Upload-ID": upload_id,
            "X-Filename": "chunked.jpg",
        },
        files={"file": ("chunk", second, "application/octet-stream")},
    )
    assert complete.status_code == 200, complete.text
    assert complete.json()["pages"][0]["filename"] == "chunked.jpg"
