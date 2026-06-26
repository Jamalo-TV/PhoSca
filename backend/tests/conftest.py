import sys
from collections.abc import AsyncGenerator
from pathlib import Path

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
