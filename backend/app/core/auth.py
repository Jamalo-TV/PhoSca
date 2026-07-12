"""Database-backed tenant API-key authentication for the control plane."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import get_async_session
from app.models import Album, ApiKey, ApiRole, Tenant


@dataclass(frozen=True)
class Principal:
    key_id: UUID
    tenant_id: UUID
    role: ApiRole


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def get_principal(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> Principal | None:
    if not settings.auth_required:
        return None
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key is required.")
    key_hash = hash_api_key(x_api_key)
    record = await session.scalar(
        select(ApiKey).join(Tenant).where(ApiKey.is_active.is_(True), Tenant.is_active.is_(True), ApiKey.key_hash == key_hash)
    )
    # Constant-time comparison prevents different handling for malformed keys.
    if record is None or not hmac.compare_digest(record.key_hash, key_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return Principal(key_id=record.id, tenant_id=record.tenant_id, role=record.role)


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
