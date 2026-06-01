import httpx

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    return _client


async def init_http_client() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=30.0)


async def close_http_client() -> None:
    if _client:
        await _client.aclose()
