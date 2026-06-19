import logging

import numpy as np

from src.embeddings.model import embedding_model
from src.embeddings.store import embedding_store
from src.middlewares.error_handler import MissingBiographyError
from src.modules.recommendations.recommendations_repository import recommendations_repository

from langfuse import observe, propagate_attributes

logger = logging.getLogger(__name__)

TOP_N = 10


class RecommendationsService:
    def __init__(self, repository, model, cache):
        self.repository = repository
        self.model = model
        self.cache = cache

    @observe()
    def _get_or_compute_embedding(self, user_id: str, biography: str) -> np.ndarray:
        logger.info(f"[Embedding] Buscando cache para user_id={user_id}")
        cached = self.cache.get(user_id, biography)
        if cached is not None:
            logger.info(f"[Embedding] Cache HIT para user_id={user_id}")
            return cached
        logger.info(f"[Embedding] Cache MISS para user_id={user_id} — computando con HF")
        vector = self.model.encode(biography)
        logger.info(f"[Embedding] Embedding computado para user_id={user_id}, shape={vector.shape}")
        self.cache.save(user_id, biography, vector)
        logger.info(f"[Embedding] Embedding guardado en Supabase para user_id={user_id}")
        return vector

    @observe()
    def update_biography_embedding(self, user_id: str, biography: str) -> None:
        logger.info(f"[Embedding] Actualizando embedding para user_id={user_id}")
        if biography and biography.strip():
            try:
                vector = self.model.encode(biography)
                logger.info(f"[Embedding] Embedding calculado para user_id={user_id}, shape={vector.shape}")
                self.cache.save(user_id, biography, vector)
                logger.info(f"[Embedding] Embedding guardado en Supabase para user_id={user_id}")
            except Exception as e:
                logger.error(f"[Embedding] ERROR actualizando embedding para user_id={user_id}: {e}")
                raise
        else:
            logger.warning(f"[Embedding] Biografía vacía para user_id={user_id}, se omite el update")

    @observe()
    async def get_recommendations(self, requester_id: str) -> list[dict]:
        logger.info(f"[Recommendations] Iniciando para requester_id={requester_id}")
        propagate_attributes(user_id=requester_id)

        try:
            biography = await self.repository.get_user_biography(requester_id)
        except Exception as e:
            logger.error(f"[Recommendations] ERROR obteniendo biografía para {requester_id}: {e}")
            raise

        logger.info(f"[Recommendations] Biografía obtenida para {requester_id}: '{biography[:50] if biography else None}...'")
        if not biography or not biography.strip():
            logger.warning(f"[Recommendations] Biografía vacía para {requester_id} — lanzando MissingBiographyError")
            raise MissingBiographyError()

        try:
            excluded_ids = await self.repository.get_excluded_user_ids(requester_id)
            logger.info(f"[Recommendations] {len(excluded_ids)} IDs excluidos para {requester_id}")
        except Exception as e:
            logger.error(f"[Recommendations] ERROR obteniendo exclusiones para {requester_id}: {e}")
            raise

        try:
            requester_vector = self._get_or_compute_embedding(requester_id, biography)
        except Exception as e:
            logger.error(f"[Recommendations] ERROR computando embedding del requester {requester_id}: {e}")
            raise

        all_excluded_ids = excluded_ids | {requester_id}
        vector_candidates = self.cache.find_nearest_candidates(
            requester_vector,
            all_excluded_ids,
            TOP_N,
        )
        if vector_candidates:
            logger.info(
                f"[Recommendations] Retornando {len(vector_candidates)} recomendaciones vectoriales para {requester_id}"
            )
            return vector_candidates

        try:
            logger.warning(
                f"[Recommendations] Sin resultados vectoriales para {requester_id}; usando fallback completo"
            )
            candidates = await self.repository.get_all_candidates(excluded_ids, requester_id)
            logger.info(f"[Recommendations] {len(candidates)} candidatos encontrados para {requester_id}")
        except Exception as e:
            logger.error(f"[Recommendations] ERROR obteniendo candidatos para {requester_id}: {e}")
            raise

        if not candidates:
            logger.info(f"[Recommendations] Sin candidatos para {requester_id}, retornando lista vacía")
            return []

        try:
            candidate_vectors = np.array(
                [self._get_or_compute_embedding(c["id"], c["biography"]) for c in candidates]
            )
        except Exception as e:
            logger.error(f"[Recommendations] ERROR computando embeddings de candidatos: {e}")
            raise

        # Cosine similarity: dot product (vectors are L2-normalized)
        similarities = candidate_vectors @ requester_vector
        top_indices = np.argsort(similarities)[::-1][:TOP_N]

        results = [
            {
                "id": candidates[i]["id"],
                "username": candidates[i]["username"],
                "biography": candidates[i]["biography"],
            }
            for i in top_indices
        ]
        logger.info(f"[Recommendations] Retornando {len(results)} recomendaciones para {requester_id}")
        return results


recommendations_service = RecommendationsService(
    repository=recommendations_repository,
    model=embedding_model,
    cache=embedding_store,
)
