import json
import sys
from collections.abc import AsyncGenerator
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeMagic:
    @staticmethod
    def from_buffer(content: bytes, mime: bool = True) -> str:
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        return "application/octet-stream"


@pytest.fixture
def training_inventory(tmp_path):
    def write(model: Path, checkpoint_digest: str, *, sources=None, **fields):
        inventory = tmp_path / f"{model.name}.training-provenance.json"
        payload = {
            "schema_version": 1,
            "model_sha256": checkpoint_digest,
            "model_origin": "test checkpoint",
            "review": {"reviewer": "test-review", "evidence": "generated source for this test"},
            "sources": sources if sources is not None else [{
                "image_sha256": "c" * 64, "source_group": "test-training-album",
                "source": "generated test fixture", "split": "train",
            }],
            **fields,
        }
        inventory.write_text(json.dumps(payload), encoding="utf-8")
        return {"path": inventory.name, "sha256": sha256(inventory.read_bytes()).hexdigest()}

    return write


@pytest.fixture
def curved_motif_page():
    page = np.full((345, 500, 3), (67, 78, 70), np.uint8)
    page[35:306, 45:456] = 236
    x = np.arange(65, 436)
    y = np.arange(50, 288)
    top = 56 + 3.5 * np.sin(2 * np.pi * (x - 65) / 370)
    foreground = np.clip(y[:, None] - top + .5, 0, 1)[..., None]
    page[50:288, 65:436] = np.rint(foreground * np.array([131, 108, 145]) + (1 - foreground) * 236)
    return page, np.float32([[45, 35], [455, 35], [455, 305], [45, 305]])


@pytest.fixture
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[AsyncClient, None]:
    db_path = tmp_path / "test.db"
    storage_path = tmp_path / "storage"

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("STORAGE_PATH", str(storage_path))
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "4096")
    monkeypatch.setenv("MAX_REQUEST_SIZE", "8192")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SEGMENTATION_TRAINING_WORKSPACE", str(tmp_path / "segmentation_training"))
    monkeypatch.setenv("AUTO_ANALYZE_UPLOADS", "false")

    from app.config import get_settings
    from app.database import Base, dispose_engine, get_engine
    from app.models import Album  # noqa: F401
    from app.utils import file_security

    await dispose_engine()
    get_settings.cache_clear()
    monkeypatch.setattr(file_security, "magic", FakeMagic)

    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client

    await dispose_engine()
    get_settings.cache_clear()


@pytest.fixture
def rounded_corner_page():
    import cv2

    height = width = 485
    rows, columns = np.indices((height, width), dtype=np.float32)
    backing = np.full((height, width, 3), (185, 128, 144), np.uint8)
    backing[..., 0] = np.rint(185 + 22 * np.cos(rows * np.pi / 6))
    quad = np.float32([[75, 75], [403, 75], [403, 405], [75, 405]])
    rectangle = (columns >= 75) & (columns <= 403) & (rows >= 75) & (rows <= 405)
    wedges = np.zeros_like(rectangle)
    radius = 16
    for x, y in quad:
        u, v = np.abs(columns - x), np.abs(rows - y)
        wedges |= (u < radius) & (v < radius) & (np.hypot(u - radius, v - radius) > radius)
    paper = rectangle & ~wedges
    raster = backing.copy()
    raster[paper] = (135, 144, 123)
    return cv2.cvtColor(raster, cv2.COLOR_LAB2BGR), quad, paper, rectangle, backing
