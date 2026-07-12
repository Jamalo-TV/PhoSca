"""Run retention safely; default is a non-mutating evidence preview."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.database import get_session_factory  # noqa: E402
from app.services.retention import enforce_retention  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Delete eligible derivatives and DLQ files.")
    parser.add_argument("--report", type=Path, default=Path("storage/evidence/retention-report.json"))
    args = parser.parse_args()
    async with get_session_factory()() as session:
        report = await enforce_retention(session, get_settings(), execute=args.execute)
    payload = json.dumps(report, indent=2, sort_keys=True).encode("utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    args.report.with_suffix(args.report.suffix + ".sha256").write_text(f"{digest}  {args.report.name}\n", encoding="ascii")
    print(json.dumps({"report": str(args.report), "sha256": digest, "mode": report["mode"], "deleted_count": report["deleted_count"], "unsafe_count": report["unsafe_count"]}))
    return 1 if report["unsafe_count"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
