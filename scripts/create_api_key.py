"""Create or rotate a tenant-scoped API key; prints the secret exactly once."""
from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.auth import hash_api_key
from app.database import get_session_factory
from app.models import ApiKey, ApiRole, Tenant


async def create_key(tenant_name: str, label: str, role: ApiRole, secret: str | None = None) -> tuple[ApiKey, str]:
    api_secret = secret or secrets.token_urlsafe(32)
    async with get_session_factory()() as session:
        tenant = await session.scalar(select(Tenant).where(Tenant.name == tenant_name))
        if tenant is None:
            tenant = Tenant(name=tenant_name)
            session.add(tenant)
            await session.flush()
        record = ApiKey(tenant_id=tenant.id, label=label, key_hash=hash_api_key(api_secret), role=role)
        session.add(record)
        await session.commit()
        return record, api_secret


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tenant-scoped PhoSca API key.")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--role", choices=[role.value for role in ApiRole], default=ApiRole.editor.value)
    parser.add_argument("--secret", help="Optional externally generated secret; otherwise a secure random value is created.")
    args = parser.parse_args()
    record, secret = asyncio.run(create_key(args.tenant, args.label, ApiRole(args.role), args.secret))
    print(f"id={record.id}")
    print(f"tenant={args.tenant}")
    print(f"role={record.role.value}")
    print(f"api_key={secret}")


if __name__ == "__main__":
    main()
