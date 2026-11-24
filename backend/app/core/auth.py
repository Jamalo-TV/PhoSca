"""Database-backed tenant API-key authentication for the control plane."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import get_async_session
from app.models import Album, ApiKey, ApiRole, Tenant


SESSION_COOKIE = "phosca_session"
SESSION_SECONDS = 8 * 60 * 60
SAME_ORIGIN_HEADER = "X-CSRF-Token"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@dataclass(frozen=True)
class Principal:
    key_id: UUID
    tenant_id: UUID
    role: ApiRole


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _active_keys():
    return select(ApiKey).join(Tenant).where(ApiKey.is_active.is_(True), Tenant.is_active.is_(True))


def _principal(record: ApiKey) -> Principal:
    return Principal(key_id=record.id, tenant_id=record.tenant_id, role=record.role)


async def principal_for_key(session: AsyncSession, x_api_key: str | None) -> Principal:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key is required.")
    key_hash = hash_api_key(x_api_key)
    record = await session.scalar(_active_keys().where(ApiKey.key_hash == key_hash))
    # Constant-time comparison prevents different handling for malformed keys.
    if record is None or not hmac.compare_digest(record.key_hash, key_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return _principal(record)


def _session_key(token: str) -> str:
    return f"phosca:session:{hash_api_key(token)}"


async def set_browser_session(settings: Settings, token: str, principal: Principal) -> None:
    try:
        async with Redis.from_url(settings.redis_url, socket_timeout=3, socket_connect_timeout=3) as redis:
            await redis.set(_session_key(token), str(principal.key_id), ex=SESSION_SECONDS)
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Browser session storage is unavailable.") from exc


async def revoke_browser_session(settings: Settings, token: str | None) -> None:
    if not token:
        return
    try:
        async with Redis.from_url(settings.redis_url, socket_timeout=3, socket_connect_timeout=3) as redis:
            await redis.delete(_session_key(token))
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Browser session storage is unavailable.") from exc


async def browser_principal(session: AsyncSession, settings: Settings, token: str | None) -> Principal | None:
    if not token:
        return None
    try:
        async with Redis.from_url(
            settings.redis_url, decode_responses=True, socket_timeout=3, socket_connect_timeout=3,
        ) as redis:
            key_id = await redis.get(_session_key(token))
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Browser session storage is unavailable.") from exc
    if key_id is None:
        return None
    record = await session.scalar(_active_keys().where(ApiKey.id == UUID(key_id)))
    return _principal(record) if record is not None else None


def require_same_origin_header(request: Request) -> None:
    # A cross-origin page cannot send this non-safelisted header with cookies.
    # The session cookie is Strict and CORS never allows credentials.
    if request.headers.get(SAME_ORIGIN_HEADER) != "1" or request.headers.get("Sec-Fetch-Site") == "cross-site":
        raise HTTPException(status_code=403, detail="Browser session mutations require a same-origin request.")


async def get_principal(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> Principal | None:
    if not settings.auth_required:
        return None
    if x_api_key is not None:
        return await principal_for_key(session, x_api_key)
    principal = await browser_principal(session, settings, request.cookies.get(SESSION_COOKIE))
    if principal is None:
        raise HTTPException(status_code=401, detail="A browser session or X-API-Key is required.")
    if request.method not in SAFE_METHODS:
        require_same_origin_header(request)
    return principal


async def require_write_principal(principal: Principal | None = Depends(get_principal)) -> Principal | None:
    if principal is not None and principal.role == ApiRole.viewer:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This API key is read-only.")
    return principal


async def require_admin_principal(principal: Principal | None = Depends(get_principal)) -> Principal | None:
    if principal is not None and principal.role != ApiRole.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This endpoint requires an administrator API key.")
    return principal


async def album_for_principal(session: AsyncSession, album_id: UUID, principal: Principal | None) -> Album:
    query = select(Album).where(Album.id == album_id)
    if principal is not None:
        query = query.where(Album.tenant_id == principal.tenant_id)
    album = await session.scalar(query)
    if album is None:
        # Deliberately do not disclose whether a foreign album exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Album not found.")
    return album
