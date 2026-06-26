from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.core.logging import configure_logging
from app.core.rate_limit import limiter
from app.database import get_session_factory
from app.models import AuditLog
from app.routers import albums, jobs, pages, photos, review, search, training


def error_response(request: Request, status_code: int, error: str, detail: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "detail": detail, "request_id": request_id},
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="Photo Album Digitization Platform", version="0.1.0")
    app.state.limiter = limiter

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"],
    )

    @app.middleware("http")
    async def request_context_and_security_headers(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = None
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()

        if request.method in {"POST", "PATCH", "DELETE"} and request.url.path.startswith("/api/v1") and response.status_code < 500:
            try:
                async with get_session_factory()() as session:
                    session.add(
                        AuditLog(
                            entity_type="api_request",
                            entity_id=uuid4(),
                            action=f"{request.method} {request.url.path}",
                            details={
                                "actor": request.headers.get("X-Actor") or request.client.host if request.client else "unknown",
                                "request_id": request_id,
                                "status_code": response.status_code,
                            },
                        )
                    )
                    await session.commit()
            except Exception as exc:  # noqa: BLE001 - audit failure must not break the user request.
                structlog.get_logger(__name__).warning("mutation_audit_failed", error=str(exc))

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; "
            "connect-src 'self' http://localhost:* http://127.0.0.1:*; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return error_response(
            request,
            exc.status_code,
            "http_error",
            str(exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_error",
            str(exc),
        )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return error_response(
            request,
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate_limited",
            str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        structlog.get_logger(__name__).exception("unhandled_exception", exc_info=exc)
        return error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_server_error",
            "Unexpected server error.",
        )

    app.include_router(albums.router)
    app.include_router(pages.router)
    app.include_router(jobs.router)
    app.include_router(photos.router)
    app.include_router(review.router)
    app.include_router(search.router)
    app.include_router(training.router)
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception as exc:  # noqa: BLE001 - metrics should not block app startup in tests.
        structlog.get_logger(__name__).warning("metrics_instrumentation_failed", error=str(exc))
    return app


app = create_app()
