"""Same-origin browser sign-in for protected JSON, previews and downloads."""

from secrets import token_urlsafe

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.auth import (
    SESSION_COOKIE, SESSION_SECONDS, Principal, browser_principal, principal_for_key,
    require_same_origin_header, revoke_browser_session, set_browser_session,
)
from app.core.rate_limit import limiter
from app.database import get_async_session


router = APIRouter(prefix="/api/v1/auth/session", tags=["authentication"])
_COOKIE_OPTIONS = {"path": "/api/v1", "secure": True, "httponly": True, "samesite": "strict"}


def _session_status(settings: Settings, principal: Principal | None) -> dict:
    return {
        "required": settings.auth_required,
        "tenant_id": str(principal.tenant_id) if principal else None,
    }


@router.get("")
async def session_status(
    request: Request, session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    principal = None
    if settings.auth_required:
        principal = await browser_principal(session, settings, request.cookies.get(SESSION_COOKIE))
    return _session_status(settings, principal)


@router.post("")
@limiter.limit("10/minute")
async def sign_in(
    request: Request, response: Response, x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_async_session), settings: Settings = Depends(get_settings),
) -> dict:
    if not settings.auth_required:
        raise HTTPException(status_code=409, detail="Browser sign-in is disabled.")
    require_same_origin_header(request)
    principal = await principal_for_key(session, x_api_key)
    await revoke_browser_session(settings, request.cookies.get(SESSION_COOKIE))
    token = token_urlsafe(32)
    await set_browser_session(settings, token, principal)
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_SECONDS, **_COOKIE_OPTIONS)
    return _session_status(settings, principal)


@router.delete("", status_code=204)
async def sign_out(request: Request, settings: Settings = Depends(get_settings)) -> Response:
    require_same_origin_header(request)
    await revoke_browser_session(settings, request.cookies.get(SESSION_COOKIE))
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, **_COOKIE_OPTIONS)
    return response
