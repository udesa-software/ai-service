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
from langfuse import get_client
from src.modules.recommendations.recommendations_router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


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

app.add_exception_handler(MissingBiographyError, missing_biography_handler)
app.add_exception_handler(AppError, app_error_handler)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
