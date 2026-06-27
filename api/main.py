import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.http_client import close_http_client, init_http_client
from src.embeddings.model import embedding_model
from src.embeddings.store import embedding_store
from src.middlewares.error_handler import (
    AppError,
    MissingBiographyError,
    app_error_handler,
    missing_biography_handler,
)
from src.observability import configure_logging
from langfuse import get_client
from src.modules.recommendations.recommendations_router import router

logger = configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    embedding_model.warmup()
    embedding_store.open()
    await init_http_client()
    yield
    await close_http_client()
    embedding_store.close()
    get_client().flush()




app = FastAPI(title="UdeSA-Migos AI Service", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request, call_next):
    if request.url.path in {"/health", "/favicon.ico"}:
        return await call_next(request)

    import time

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000)
    level = logging.ERROR if response.status_code >= 500 else logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        level,
        f"{response.status_code} {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "request_id": request.headers.get("x-request-id"),
        },
    )
    return response

app.add_exception_handler(MissingBiographyError, missing_biography_handler)
app.add_exception_handler(AppError, app_error_handler)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
