import logging
import time
from typing import Optional

import numpy as np
from huggingface_hub import InferenceClient, InferenceTimeoutError
from langfuse import observe

from src.config.settings import settings

logger = logging.getLogger(__name__)

_client: InferenceClient | None = None

# Configuración de reintentos
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.5  # segundos
BACKOFF_MULTIPLIER = 2.0


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(token=settings.hf_api_token)
    return _client


def _to_sentence_embedding(raw: np.ndarray) -> np.ndarray:
    """Convierte la respuesta de HF a un vector de oración L2-normalizado."""
    vector = np.asarray(raw, dtype=np.float32)
    if vector.ndim == 2:
        vector = vector.mean(axis=0)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _feature_extraction_with_retry(
    client: InferenceClient,
    text: str,
    model: str,
) -> Optional[np.ndarray]:
    """
    Extrae embeddings de HF con reintentos exponenciales.
    
    Args:
        client: InferenceClient configurado
        text: Texto a embeber
        model: Nombre del modelo en HF
        
    Returns:
        Array de embeddings o None si falla después de reintentos
        
    Raises:
        RuntimeError: Si se agotan los reintentos
    """
    backoff = INITIAL_BACKOFF
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Intento {attempt + 1}/{MAX_RETRIES} para extraer embedding")
            raw = client.feature_extraction(
                text,
                model=model,
                normalize=True,
            )
            logger.debug(f"Embedding extraído exitosamente en intento {attempt + 1}")
            return raw
            
        except InferenceTimeoutError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"Timeout en intento {attempt + 1}/{MAX_RETRIES}. "
                    f"Reintentando en {backoff}s: {str(e)}"
                )
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
            else:
                logger.error(
                    f"Timeout después de {MAX_RETRIES} intentos: {str(e)}"
                )
                
        except Exception as e:
            last_error = e
            logger.error(
                f"Error al extraer embedding en intento {attempt + 1}/{MAX_RETRIES}: {str(e)}"
            )
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Reintentando en {backoff}s")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
    
    raise RuntimeError(
        f"No se pudo extraer embedding después de {MAX_RETRIES} intentos. "
        f"Último error: {str(last_error)}"
    )


class EmbeddingModel:
    @observe(as_type="generation")
    def encode(self, text: str) -> np.ndarray:
        """
        Codifica un texto a embedding usando Hugging Face Inference API.
        
        Args:
            text: Texto a embeber
            
        Returns:
            Vector de embedding L2-normalizado
            
        Raises:
            RuntimeError: Si falla la extracción después de reintentos
        """
        raw = _feature_extraction_with_retry(
            _get_client(),
            text,
            settings.embedding_model,
        )
        return _to_sentence_embedding(raw)

    def warmup(self) -> None:
        """Realiza un warmup del servicio para inicializar la conexión."""
        try:
            self.encode("warmup")
            logger.info("Warmup del modelo de embeddings completado exitosamente")
        except Exception as e:
            logger.warning(f"Warmup falló: {str(e)}. El servicio seguirá funcionando.")


embedding_model = EmbeddingModel()
