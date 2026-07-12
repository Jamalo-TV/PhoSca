async def test_liveness_and_readiness(client) -> None:
    live = await client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    ready = await client.get("/health/ready")
    assert ready.status_code == 200, ready.text
    assert ready.json()["checks"]["database"] == "ok"
