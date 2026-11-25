from sqlalchemy import select


async def test_api_key_authentication_roles_and_album_tenant_scope(client, monkeypatch, tmp_path) -> None:
    from uuid import uuid4

    from PIL import Image

    from app.config import get_settings
    from app.core.auth import hash_api_key
    from app.database import get_session_factory
    from app.models import Album, ApiKey, ApiRole, Page, PageStatus, Tenant

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

    source = tmp_path / "private.jpg"
    Image.new("RGB", (300, 300), "#eeeeee").save(source)
    async with get_session_factory()() as session:
        album = await session.scalar(select(Album).where(Album.id == album_id))
        assert album is not None and album.tenant_id is not None
        page = Page(
            album_id=album.id, original_filename=source.name, storage_path=str(source),
            file_hash_sha256="a" * 64, status=PageStatus.completed,
        )
        session.add(page)
        await session.commit()
        url = f"/api/v1/pages/{page.id}/manual-photos"
        trace_url = f"/api/v1/pages/{page.id}/edge-trace"
    draft = {"request_id": str(uuid4()), "warp_quad": [
        {"x": x, "y": y} for x, y in ((.1, .1), (.7, .1), (.7, .6), (.1, .6))
    ]}
    assert (await client.post(url, json=draft)).status_code == 401
    assert (await client.post(url, json=draft, headers={"X-API-Key": "viewer-key"})).status_code == 403
    assert (await client.post(url, json=draft, headers={"X-API-Key": "other-key"})).status_code == 404
    coords = dict(start_x=.15, start_y=.25, middle_x=.5, middle_y=.25, end_x=.85, end_y=.25)
    assert (await client.get(trace_url, params=coords)).status_code == 401
    assert (await client.get(trace_url, params=coords, headers={"X-API-Key": "other-key"})).status_code == 404
    measured = await client.get(trace_url, params=coords, headers={"X-API-Key": "viewer-key"})
    assert measured.status_code == 200
    assert measured.json() == {"traces": []}


async def test_browser_session_scopes_requests_and_revokes_the_cookie(client, monkeypatch) -> None:
    from app.config import get_settings
    from app.core import auth
    from app.database import get_session_factory
    from app.models import ApiKey, ApiRole, Tenant

    sessions = {}

    class MemoryRedis:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def set(self, key, value, *, ex):
            assert ex == auth.SESSION_SECONDS
            sessions[key] = value

        async def get(self, key):
            return sessions.get(key)

        async def delete(self, key):
            sessions.pop(key, None)

    monkeypatch.setattr(auth.Redis, "from_url", lambda *_args, **_kwargs: MemoryRedis())
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    get_settings.cache_clear()
    async with get_session_factory()() as session:
        tenant = Tenant(name="operator")
        session.add(tenant)
        await session.flush()
        session.add(ApiKey(
            tenant_id=tenant.id, label="editor", key_hash=auth.hash_api_key("browser-key"), role=ApiRole.editor,
        ))
        await session.commit()

    base = "https://testserver/api/v1"
    auth_url = f"{base}/auth/session"
    headers = {"X-CSRF-Token": "1"}
    assert (await client.get(auth_url)).json() == {"required": True, "tenant_id": None}
    assert (await client.get(f"{base}/albums")).status_code == 401
    sign_in = await client.post(auth_url, headers={**headers, "X-API-Key": "browser-key"})
    assert sign_in.json() == {"required": True, "tenant_id": str(tenant.id)}
    attributes = sign_in.headers["set-cookie"].lower()
    assert all(attribute in attributes for attribute in ("secure", "httponly", "samesite=strict"))
    assert "browser-key" not in attributes
    assert sign_in.headers["cache-control"] == "private, no-store"

    albums = f"{base}/albums"
    assert (await client.post(albums, json={"name": "missing token"})).status_code == 403
    cross_site = {**headers, "Sec-Fetch-Site": "cross-site"}
    assert (await client.post(albums, headers=cross_site, json={"name": "wrong site"})).status_code == 403
    created = await client.post(albums, headers=headers, json={"name": "Private album"})
    assert created.status_code == 201
    assert [item["name"] for item in (await client.get(albums)).json()] == ["Private album"]
    assert (await client.delete(auth_url, headers=headers)).status_code == 204
    assert sessions == {}
    assert (await client.get(albums)).status_code == 401
