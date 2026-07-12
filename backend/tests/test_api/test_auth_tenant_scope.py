from sqlalchemy import select


async def test_api_key_authentication_roles_and_album_tenant_scope(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.core.auth import hash_api_key
    from app.database import get_session_factory
    from app.models import Album, ApiKey, ApiRole, Tenant

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    get_settings.cache_clear()
    async with get_session_factory()() as session:
        tenant_a = Tenant(name="tenant-a")
        tenant_b = Tenant(name="tenant-b")
        session.add_all([tenant_a, tenant_b])
        await session.flush()
        session.add_all(
            [
                ApiKey(tenant_id=tenant_a.id, label="viewer", key_hash=hash_api_key("viewer-key"), role=ApiRole.viewer),
                ApiKey(tenant_id=tenant_a.id, label="editor", key_hash=hash_api_key("editor-key"), role=ApiRole.editor),
                ApiKey(tenant_id=tenant_b.id, label="other", key_hash=hash_api_key("other-key"), role=ApiRole.editor),
            ]
        )
        await session.commit()

    assert (await client.get("/api/v1/albums")).status_code == 401
    assert (await client.post("/api/v1/albums", headers={"X-API-Key": "viewer-key"}, json={"name": "Denied", "description": None})).status_code == 403
    assert (await client.get("/api/v1/jobs/not-a-visible-job", headers={"X-API-Key": "viewer-key"})).status_code == 403
    created = await client.post("/api/v1/albums", headers={"X-API-Key": "editor-key"}, json={"name": "Scoped", "description": None})
    assert created.status_code == 201, created.text
    album_id = created.json()["id"]
    assert (await client.get("/api/v1/albums", headers={"X-API-Key": "editor-key"})).json()[0]["id"] == album_id
    assert (await client.get(f"/api/v1/albums/{album_id}", headers={"X-API-Key": "other-key"})).status_code == 404

    async with get_session_factory()() as session:
        album = await session.scalar(select(Album).where(Album.id == album_id))
        assert album is not None and album.tenant_id is not None
