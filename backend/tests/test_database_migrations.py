from io import StringIO
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.config import get_settings


def test_postgresql_migrations_create_enums_once_and_use_matching_uuid_types(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    output = StringIO()
    config = Config(str(root / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(root / "alembic"))
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://example.invalid/phosca")
    monkeypatch.setenv("REDIS_URL", "redis://example.invalid/0")
    get_settings.cache_clear()
    try:
        command.upgrade(config, "head", sql=True)
    finally:
        get_settings.cache_clear()

    sql = output.getvalue()
    for enum in ("run_status", "stage_run_status", "api_role"):
        assert sql.count(f"CREATE TYPE {enum} AS ENUM") == 1
    assert "page_id UUID NOT NULL" in sql.split("CREATE TABLE processing_runs (", 1)[1].split(");", 1)[0]
    assert "tenant_id UUID NOT NULL" in sql.split("CREATE TABLE api_keys (", 1)[1].split(");", 1)[0]
    assert "ALTER TABLE review_events ALTER COLUMN task_id TYPE UUID USING task_id::uuid" in sql
