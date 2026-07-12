from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import get_async_session
from app.pipeline.model_registry import ModelArtifactError, verify_model_artifact


router = APIRouter(tags=["health"])


@router.get("/local-ca.crt", include_in_schema=False)
async def local_ca_certificate() -> FileResponse:
    """Download only the public local CA used by the mobile development server."""
    certificate = Path(".cert/phosca-local-ca.crt").resolve()
    if not certificate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Local CA certificate is not configured.")
    return FileResponse(certificate, media_type="application/x-x509-ca-cert", filename="phosca-local-ca.crt")


@router.get("/health/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", include_in_schema=False)
async def readiness(
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    checks: dict[str, object] = {"database": "ok", "segmentation_model": "not_configured"}
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - readiness needs concise failure evidence.
        checks["database"] = f"error:{type(exc).__name__}"
    if settings.yolo_model_path.exists():
        try:
            checks["segmentation_model"] = verify_model_artifact(
                settings.yolo_model_path,
                settings.model_manifest_path,
            )
        except ModelArtifactError as exc:
            checks["segmentation_model"] = f"error:{exc}"
    elif not settings.enable_classical_segmentation_fallback:
        checks["segmentation_model"] = "error:missing_and_fallback_disabled"
    ready = checks["database"] == "ok" and not str(checks["segmentation_model"]).startswith("error:")
    return JSONResponse(status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "ok" if ready else "not_ready", "checks": checks})
