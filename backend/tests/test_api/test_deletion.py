from io import BytesIO

from PIL import Image


def _jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), (90, 120, 160)).save(buffer, format="JPEG")
    return buffer.getvalue()


async def test_delete_album_removes_owned_upload_and_processed_trees(client) -> None:
    from app.config import get_settings

    created = await client.post("/api/v1/albums", json={"name": "Delete me", "description": None})
    assert created.status_code == 201
    album_id = created.json()["id"]
    uploaded = await client.post(
        f"/api/v1/albums/{album_id}/pages/upload",
        files={"files": ("page.jpg", _jpeg(), "image/jpeg")},
    )
    assert uploaded.status_code == 201, uploaded.text
    settings = get_settings()
    upload_dir = settings.storage_path / "uploads" / album_id
    processed_dir = settings.storage_path / "processed" / album_id
    processed_dir.mkdir(parents=True)
    (processed_dir / "derived.jpg").write_bytes(b"not-an-image")
    assert upload_dir.exists() and processed_dir.exists()

    deleted = await client.delete(f"/api/v1/albums/{album_id}")
    assert deleted.status_code == 204
    assert not upload_dir.exists()
    assert not processed_dir.exists()
    missing = await client.get(f"/api/v1/albums/{album_id}")
    assert missing.status_code == 404
