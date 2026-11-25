from functools import lru_cache
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
from httpx import AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy import select


def _valid_jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), (120, 130, 140)).save(buffer, format="JPEG")
    return buffer.getvalue()


JPEG_BYTES = _valid_jpeg_bytes()
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
    assert body["analysis_queued"] is False
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


async def test_analyze_accepts_uploaded_page_ids_as_json_strings(client: AsyncClient, monkeypatch) -> None:
    album_id = await create_album(client)
    upload_response = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("album-page.jpg", JPEG_BYTES, "image/jpeg"))],
    )
    assert upload_response.status_code == 201, upload_response.text
    page_id = upload_response.json()["pages"][0]["page_id"]

    from app.routers import albums as albums_router

    async def fake_process_pages_serialized(page_ids):
        assert page_ids == [UUID(page_id)]
        return {"pages": [{"page_id": str(page_ids[0]), "status": "completed"}]}

    monkeypatch.setattr(albums_router, "process_pages_serialized", fake_process_pages_serialized)

    response = await client.post(f"/api/v1/albums/{album_id}/analyze-now", json={"page_ids": [page_id]})

    assert response.status_code == 200, response.text
    assert response.json()["pages"][0]["page_id"] == page_id


async def test_uploaded_page_is_cropped_automatically_even_if_ocr_fails(
    client: AsyncClient, monkeypatch, tmp_path
) -> None:
    from app.config import get_settings
    from app.services import pipeline

    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "true")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "32768")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "65536")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "unavailable.onnx"))
    get_settings.cache_clear()

    def unavailable_ocr(*args):
        raise RuntimeError("OCR inference unavailable")

    monkeypatch.setattr(pipeline, "run_paddle_ocr", unavailable_ocr)
    album_id = await create_album(client)
    page = Image.new("RGB", (560, 420), (245, 245, 245))
    draw = ImageDraw.Draw(page)
    draw.rectangle((80, 100, 310, 280), fill=(20, 20, 20))
    draw.rectangle((88, 108, 302, 272), fill=(180, 120, 70))
    draw.line((180, 108, 210, 272), fill=(255, 255, 255), width=2)
    buffer = BytesIO()
    page.save(buffer, format="JPEG")

    upload = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("album-page.jpg", buffer.getvalue(), "image/jpeg"))],
    )
    assert upload.status_code == 201, upload.text
    assert upload.json()["analysis_queued"] is True

    pages = (await client.get(f"/api/v1/albums/{album_id}/pages")).json()
    assert pages[0]["status"] in {"completed", "review_needed"}
    assert pages[0]["processing_metadata"]["steps"]["ocr"]["primary"]["error"] == "OCR inference unavailable"
    photos = (await client.get(f"/api/v1/albums/{album_id}/photos")).json()
    assert len(photos) == 1
    crop = await client.get(f"/api/v1/photos/{photos[0]['id']}/image")
    assert crop.status_code == 200, crop.text
    with Image.open(BytesIO(crop.content)) as image:
        assert 180 < image.width < 260
        assert 140 < image.height < 210


async def test_upload_keeps_both_crop_modes_when_model_also_returns_a_collapsed_box(client, monkeypatch, tmp_path) -> None:
    from app.config import get_settings
    from app.pipeline import segmentation
    from app.services import pipeline

    model = tmp_path / "test.onnx"
    model.touch()
    for name, value in {
        "AUTO_ANALYZE_UPLOADS": "true", "MAX_UPLOAD_SIZE": "65536", "MAX_REQUEST_SIZE": "131072",
        "YOLO_MODEL_PATH": str(model), "REQUIRE_REGISTERED_MODELS": "false", "YOLO_PROPOSAL_THRESHOLD": "0.025",
        "ENABLE_CLASSICAL_SEGMENTATION_FALLBACK": "false", "ENABLE_PRECISE_PHOTO_CROPS": "true",
        "ENABLE_PHOTO_ORIENTATION_CORRECTION": "false", "ENABLE_PROTECTED_REGION_GUARD": "false",
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()

    prediction = np.zeros((2, 37), dtype=np.float32)
    prediction[0, :5] = [385.15134, 29.117088, 253.19464, 58.15648, 0.04268247]
    prediction[1, :5] = [229, 256.4, 305.5, 244.4, 0.88]
    prediction[1, 5] = 1
    proto = np.ones((1, 32, 128, 128), dtype=np.float32)
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input", shape=[1, 3, 512, 512])],
        get_providers=lambda: ["CPUExecutionProvider"],
        run=lambda *_: [prediction, proto],
    )

    @lru_cache
    def load_session(*_):
        return session

    monkeypatch.setattr(segmentation, "_cached_onnx_session", load_session)
    monkeypatch.setattr(pipeline, "run_paddle_ocr", lambda *_: None)
    album_id = await create_album(client)
    page = Image.new("RGB", (672, 520), (232, 232, 232))
    draw = ImageDraw.Draw(page)
    draw.rectangle((100, 100, 500, 420), fill=(246, 246, 246))
    draw.rectangle((124, 124, 476, 396), fill=(175, 116, 82))
    buffer = BytesIO()
    page.save(buffer, format="PNG")
    upload = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("album-page.png", buffer.getvalue(), "image/png"))],
    )
    assert upload.status_code == 201 and upload.json()["analysis_queued"], upload.text
    processed, = (await client.get(f"/api/v1/albums/{album_id}/pages")).json()
    metadata = processed["processing_metadata"]["steps"]["segmentation"]
    assert metadata["rejected_model_proposals"] == {"collapsed_source_polygon": 1}
    assert not metadata["fallback_used"] and "onnx_error" not in metadata
    photo, = (await client.get(f"/api/v1/albums/{album_id}/photos")).json()
    assert photo["crop_download_urls"] == {"full_print": None, "image_only": None}

    def vertices(left, top, right, bottom):
        return [{"x": x / 671, "y": y / 519} for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]

    stock, motif = vertices(100, 100, 500, 420), vertices(124, 124, 476, 396)
    url = f"/api/v1/photos/{photo['id']}"
    saved = await client.patch(f"{url}/bounding-box", json={
        "bounding_box": {"x1": 100 / 671, "y1": 100 / 519, "x2": 500 / 671, "y2": 420 / 519},
        "segmentation_mask": {
            "source": "manual_quad", "polygon": stock, "warp_quad": stock, "image_quad": motif,
            "print_size_mm": {"width": 400, "height": 320},
        },
    })
    assert saved.status_code == 200, saved.text
    for mode, expected_size in (("full_print", (401, 321)), ("image_only", (353, 273))):
        download = await client.get(saved.json()["crop_download_urls"][mode])
        assert download.status_code == 200, download.text
        with Image.open(BytesIO(download.content)) as crop:
            assert crop.size == expected_size


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


async def test_upload_rejects_duplicate_file_within_same_album(client: AsyncClient) -> None:
    album_id = await create_album(client)

    first = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("album-page.jpg", JPEG_BYTES, "image/jpeg"))],
    )
    assert first.status_code == 201, first.text

    duplicate = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files=[("files", ("album-page-copy.jpg", JPEG_BYTES, "image/jpeg"))],
    )

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "File was already uploaded."


async def test_upload_allows_same_file_in_different_albums(client: AsyncClient) -> None:
    first_album_id = await create_album(client)
    second_album_id = await create_album(client)

    first = await client.post(
        f"/api/v1/albums/{first_album_id}/pages/upload",
        files=[("files", ("album-page.jpg", JPEG_BYTES, "image/jpeg"))],
    )
    assert first.status_code == 201, first.text

    second = await client.post(
        f"/api/v1/albums/{second_album_id}/pages/upload",
        files=[("files", ("album-page-again.jpg", JPEG_BYTES, "image/jpeg"))],
    )

    assert second.status_code == 201, second.text
    assert second.json()["pages"][0]["filename"] == "album-page-again.jpg"


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
