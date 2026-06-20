import asyncio
import socket

from sqlalchemy.ext.asyncio import create_async_engine


async def check(url: str) -> None:
    try:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            result = await conn.exec_driver_sql("select 1")
            print(url, result.scalar())
        await engine.dispose()
    except Exception as exc:
        print(url, type(exc).__name__, exc)


print("resolve postgres", end=" ")
try:
    print(socket.gethostbyname("postgres"))
except Exception as exc:
    print(type(exc).__name__, exc)

asyncio.run(check("postgresql+asyncpg://digitizer:digitizer_password@postgres:5432/digitizer"))
asyncio.run(check("postgresql+asyncpg://digitizer:digitizer_password@172.18.0.3:5432/digitizer"))

