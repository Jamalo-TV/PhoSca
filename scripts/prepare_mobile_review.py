"""Process newly captured pages and prepare conservative mobile review tasks."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.database import get_session_factory  # noqa: E402
from app.models import ExtractedPhoto, Page, PageStatus  # noqa: E402
from app.pipeline.variants import variants_from_metadata  # noqa: E402
from app.services.pipeline import process_pages_pipeline  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("album_id", type=UUID)
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "mobile_review_preparation.json")
    parser.add_argument("--no-enhancement-candidates", action="store_true")
    args = parser.parse_args()

    settings = get_settings().model_copy(update={
        "enable_conservative_enhancement": not args.no_enhancement_candidates,
        "enable_deep_enhancement": not args.no_enhancement_candidates,
        "enable_premium_enhancement": False,
        "enable_classical_segmentation_fallback": False,
        "shadow_mode": True,
        "release_kill_switch": True,
    })
    async with get_session_factory()() as session:
        pages = (await session.scalars(
            select(Page).where(Page.album_id == args.album_id).order_by(Page.created_at.asc())
        )).all()
        pending_ids = [page.id for page in pages if page.status in {PageStatus.uploaded, PageStatus.queued, PageStatus.processing}]

    results = await process_pages_pipeline(pending_ids, settings) if pending_ids else {"pages": []}

    async with get_session_factory()() as session:
        pages = (await session.scalars(select(Page).where(Page.album_id == args.album_id))).all()
        photos = (await session.scalars(select(ExtractedPhoto).where(ExtractedPhoto.album_id == args.album_id))).all()
    variants = Counter()
    review_photos = 0
    for photo in photos:
        variants.update(variants_from_metadata(photo.enhancement_applied).keys())
        review_photos += photo.status.value == "review_needed"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "album_id": str(args.album_id),
        "settings": {
            "pipeline_version": settings.pipeline_version,
            "model_path": str(settings.yolo_model_path),
            "enhancement_candidates": settings.enable_deep_enhancement,
            "shadow_mode": settings.shadow_mode,
            "release_kill_switch": settings.release_kill_switch,
        },
        "page_statuses": dict(Counter(page.status.value for page in pages)),
        "page_count": len(pages),
        "photo_count": len(photos),
        "review_photo_count": review_photos,
        "variant_counts": dict(variants),
        "processed_this_run": len(results.get("pages", [])),
        "failures": [item for item in results.get("pages", []) if item.get("status") == "failed"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
