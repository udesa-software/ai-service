import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from langfuse import observe

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


class EmbeddingModel:
    @observe(as_type="generation")
    def encode(self, text: str) -> np.ndarray:
        return _get_model().encode(text, normalize_embeddings=True)

    def warmup(self) -> None:
        _get_model()


embedding_model = EmbeddingModel()
