"""Run and stage provenance for reproducible, idempotent processing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Asset, Page, ProcessingRun, RunStatus, StageRun
from app.pipeline.image_ops import sha256_file


def _fingerprint_settings(settings: Settings) -> tuple[str, dict[str, Any]]:
    """Hash all non-secret pipeline settings used for deterministic replay."""
    raw = settings.model_dump()
    # URLs can contain secrets and do not alter image semantics.
    for name in ("database_url", "redis_url", "cors_origins"):
        raw.pop(name, None)
    normalized = json.loads(json.dumps(raw, default=str, sort_keys=True))
    fingerprint = hashlib.sha256(json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    return fingerprint, normalized


async def get_or_create_run(session: AsyncSession, page: Page, settings: Settings) -> ProcessingRun:
    fingerprint, snapshot = _fingerprint_settings(settings)
    input_hash = page.file_hash_sha256 or sha256_file(Path(page.storage_path))
    query = select(ProcessingRun).where(
        ProcessingRun.page_id == page.id,
        ProcessingRun.input_sha256 == input_hash,
        ProcessingRun.pipeline_version == settings.pipeline_version,
        ProcessingRun.config_fingerprint == fingerprint,
    )
    run = await session.scalar(query)
    if run is not None:
        return run
    run = ProcessingRun(
        page_id=page.id,
        input_sha256=input_hash,
        pipeline_version=settings.pipeline_version,
        config_fingerprint=fingerprint,
        status=RunStatus.running,
        manifest={
            "schema_version": 1,
            "input": {"storage_path": page.storage_path, "sha256": input_hash},
            "pipeline_version": settings.pipeline_version,
            "settings": snapshot,
        },
    )
    session.add(run)
    await session.flush()
    return run


async def start_stage(session: AsyncSession, run: ProcessingRun, stage: str, *, input_path: Path | None = None) -> StageRun:
    latest_attempt = await session.scalar(
        select(func.max(StageRun.attempt)).where(StageRun.processing_run_id == run.id, StageRun.stage == stage)
    )
    record = StageRun(
        processing_run_id=run.id,
        stage=stage,
        attempt=int(latest_attempt or 0) + 1,
        status=RunStatus.running,
        input_asset_sha256=sha256_file(input_path) if input_path and input_path.exists() else None,
    )
    session.add(record)
    await session.flush()
    return record


def finish_stage(
    record: StageRun,
    *,
    status: RunStatus,
    details: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> None:
    record.status = status
    record.details = dict(details or {})
    if output_path and output_path.exists():
        record.output_asset_sha256 = sha256_file(output_path)


def finish_run(run: ProcessingRun, *, status: RunStatus, details: dict[str, Any] | None = None) -> None:
    run.status = status
    manifest = dict(run.manifest or {})
    manifest["result"] = dict(details or {}) | {"status": status.value}
    run.manifest = manifest


async def record_asset(
    session: AsyncSession,
    run: ProcessingRun,
    *,
    path: Path,
    role: str,
    stage: StageRun | None = None,
    parent_asset: Asset | None = None,
    metadata: dict[str, Any] | None = None,
) -> Asset:
    """Register a completed immutable asset exactly once by storage path."""
    if not path.is_file():
        raise ValueError(f"Cannot register missing asset: {path}")
    existing = await session.scalar(select(Asset).where(Asset.storage_path == str(path)))
    if existing is not None:
        return existing
    width: int | None = None
    height: int | None = None
    media_type = "application/octet-stream"
    try:
        with Image.open(path) as image:
            width, height = image.size
            media_type = Image.MIME.get(image.format or "", media_type)
    except OSError:
        pass
    asset = Asset(
        processing_run_id=run.id,
        stage_run_id=stage.id if stage else None,
        parent_asset_id=parent_asset.id if parent_asset else None,
        role=role,
        storage_path=str(path),
        content_sha256=sha256_file(path),
        media_type=media_type,
        width=width,
        height=height,
        asset_metadata=dict(metadata or {}),
    )
    session.add(asset)
    await session.flush()
    return asset


async def asset_for_path(session: AsyncSession, path: Path) -> Asset | None:
    return await session.scalar(select(Asset).where(Asset.storage_path == str(path)))
