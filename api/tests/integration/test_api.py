from fastapi.testclient import TestClient

import main


async def _noop_async():
    return None


class _LangfuseStub:
    def flush(self):
        return None


def _stub_lifespan_dependencies(monkeypatch):
    monkeypatch.setattr(main.embedding_model, "warmup", lambda: None)
    monkeypatch.setattr(main.embedding_store, "open", lambda: None)
    monkeypatch.setattr(main.embedding_store, "close", lambda: None)
    monkeypatch.setattr(main, "init_http_client", _noop_async)
    monkeypatch.setattr(main, "close_http_client", _noop_async)
    monkeypatch.setattr(main, "get_client", lambda: _LangfuseStub())


def test_health_endpoint(monkeypatch):
    _stub_lifespan_dependencies(monkeypatch)

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_internal_embedding_requires_secret(monkeypatch):
    _stub_lifespan_dependencies(monkeypatch)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/ai/internal/embedding",
            json={"user_id": "user-1", "biography": "Bio"},
        )

    assert response.status_code == 401
