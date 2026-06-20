import json
import shutil
from pathlib import Path
from uuid import UUID

from app.config import Settings
from app.models import Page


def quarantine_page_failure(page: Page, settings: Settings, *, reason: str, details: dict | None = None) -> dict:
    source_path = Path(page.storage_path)
    dlq_dir = settings.storage_path / "failed" / "pages" / str(page.id)
    dlq_dir.mkdir(parents=True, exist_ok=True)

    copied_path = None
    if source_path.exists():
        copied_path = dlq_dir / source_path.name
        shutil.copy2(source_path, copied_path)

    manifest = {
        "page_id": str(page.id),
        "album_id": str(page.album_id),
        "original_filename": page.original_filename,
        "source_path": str(source_path),
        "quarantined_path": str(copied_path) if copied_path else None,
        "reason": reason,
        "details": details or {},
    }
    manifest_path = dlq_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"dlq_manifest_path": str(manifest_path), "dlq_image_path": str(copied_path) if copied_path else None}


def failed_page_dlq_path(settings: Settings, page_id: UUID) -> Path:
    return settings.storage_path / "failed" / "pages" / str(page_id)
