"""Conservative derivative/DLQ retention with machine-readable evidence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.metrics import RETENTION_DELETIONS
from app.models import Asset, AuditLog


PRESERVED_ASSET_ROLES = {"page_original", "corrected_print", "released_variant"}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


async def enforce_retention(session: AsyncSession, settings: Settings, *, execute: bool = False, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=settings.retention_derivative_days)
    processed_root = settings.storage_path / "processed"
    candidates = (await session.scalars(select(Asset).where(Asset.created_at < cutoff, Asset.role.not_in(PRESERVED_ASSET_ROLES)))).all()
    actions: list[dict[str, Any]] = []
    for asset in candidates:
        path = Path(asset.storage_path)
        safe = _inside(path, processed_root)
        exists = path.is_file()
        action = {"kind": "asset", "asset_id": str(asset.id), "role": asset.role, "path": str(path), "safe": safe, "exists": exists, "deleted": False}
        if execute and safe:
            if exists:
                path.unlink()
            await session.execute(delete(Asset).where(Asset.id == asset.id))
            session.add(AuditLog(entity_type="asset", entity_id=asset.id, action="retention_deleted", details={"role": asset.role, "path": str(path), "cutoff": cutoff.isoformat()}))
            action["deleted"] = True
            RETENTION_DELETIONS.labels(kind="asset", result="deleted").inc()
        elif not safe:
            RETENTION_DELETIONS.labels(kind="asset", result="unsafe_path").inc()
        actions.append(action)

    dlq_cutoff = now - timedelta(days=settings.retention_dlq_days)
    dlq_root = settings.storage_path / "failed"
    if dlq_root.exists():
        for path in sorted(item for item in dlq_root.rglob("*") if item.is_file()):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified >= dlq_cutoff:
                continue
            safe = _inside(path, dlq_root)
            action = {"kind": "dlq", "path": str(path), "safe": safe, "exists": True, "deleted": False}
            if execute and safe:
                path.unlink()
                action["deleted"] = True
                RETENTION_DELETIONS.labels(kind="dlq", result="deleted").inc()
            actions.append(action)

    if execute:
        await session.commit()
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "execute" if execute else "dry-run",
        "derivative_cutoff": cutoff.isoformat(),
        "dlq_cutoff": dlq_cutoff.isoformat(),
        "preserved_roles": sorted(PRESERVED_ASSET_ROLES),
        "candidate_count": len(actions),
        "deleted_count": sum(bool(action["deleted"]) for action in actions),
        "unsafe_count": sum(not bool(action["safe"]) for action in actions),
        "actions": actions,
    }
