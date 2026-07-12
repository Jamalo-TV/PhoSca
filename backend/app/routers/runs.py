from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.core.auth import Principal, get_principal
from app.models import Asset, Page, ProcessingRun, StageRun


router = APIRouter(prefix="/api/v1", tags=["runs"])


def _asset_payload(asset: Asset) -> dict:
    return {
        "id": str(asset.id),
        "stage_run_id": str(asset.stage_run_id) if asset.stage_run_id else None,
        "parent_asset_id": str(asset.parent_asset_id) if asset.parent_asset_id else None,
        "role": asset.role,
        "storage_path": asset.storage_path,
        "content_sha256": asset.content_sha256,
        "media_type": asset.media_type,
        "width": asset.width,
        "height": asset.height,
        "metadata": asset.asset_metadata,
    }


def _stage_payload(stage: StageRun) -> dict:
    return {
        "id": str(stage.id),
        "stage": stage.stage,
        "attempt": stage.attempt,
        "status": stage.status.value,
        "input_asset_sha256": stage.input_asset_sha256,
        "output_asset_sha256": stage.output_asset_sha256,
        "details": stage.details,
        "created_at": stage.created_at,
        "updated_at": stage.updated_at,
    }


def _run_payload(run: ProcessingRun, stages: list[StageRun], assets: list[Asset]) -> dict:
    return {
        "id": str(run.id),
        "page_id": str(run.page_id),
        "input_sha256": run.input_sha256,
        "pipeline_version": run.pipeline_version,
        "config_fingerprint": run.config_fingerprint,
        "status": run.status.value,
        "manifest": run.manifest,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "stages": [_stage_payload(stage) for stage in stages],
        "assets": [_asset_payload(asset) for asset in assets],
    }


@router.get("/pages/{page_id}/runs")
async def list_page_runs(
    page_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> list[dict]:
    page_query = select(Page).where(Page.id == page_id)
    if principal is not None:
        page_query = page_query.where(Page.album.has(tenant_id=principal.tenant_id))
    if await session.scalar(page_query) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found.")
    runs = (await session.scalars(select(ProcessingRun).where(ProcessingRun.page_id == page_id).order_by(ProcessingRun.created_at.desc()))).all()
    result: list[dict] = []
    for run in runs:
        stages = (await session.scalars(select(StageRun).where(StageRun.processing_run_id == run.id).order_by(StageRun.created_at.asc()))).all()
        assets = (await session.scalars(select(Asset).where(Asset.processing_run_id == run.id).order_by(Asset.created_at.asc()))).all()
        result.append(_run_payload(run, stages, assets))
    return result


@router.get("/runs/{run_id}")
async def get_run(
    run_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    principal: Principal | None = Depends(get_principal),
) -> dict:
    query = select(ProcessingRun).join(Page).where(ProcessingRun.id == run_id)
    if principal is not None:
        query = query.where(Page.album.has(tenant_id=principal.tenant_id))
    run = await session.scalar(query)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing run not found.")
    stages = (await session.scalars(select(StageRun).where(StageRun.processing_run_id == run.id).order_by(StageRun.created_at.asc()))).all()
    assets = (await session.scalars(select(Asset).where(Asset.processing_run_id == run.id).order_by(Asset.created_at.asc()))).all()
    return _run_payload(run, stages, assets)
