import numpy as np

from src.embeddings.model import embedding_model
from src.embeddings.store import embedding_store
from src.middlewares.error_handler import MissingBiographyError
from src.modules.recommendations.recommendations_repository import recommendations_repository

from langfuse import observe, propagate_attributes

TOP_N = 10


class RecommendationsService:
    def __init__(self, repository, model, cache):
        self.repository = repository
        self.model = model
        self.cache = cache

    @observe()
    def _get_or_compute_embedding(self, user_id: str, biography: str) -> np.ndarray:
        cached = self.cache.get(user_id, biography)
        if cached is not None:
            return cached
        vector = self.model.encode(biography)
        self.cache.save(user_id, biography, vector)
        return vector

    @observe()
    async def get_recommendations(self, requester_id: str) -> list[dict]:
        propagate_attributes(user_id=requester_id)
        biography = await self.repository.get_user_biography(requester_id)
        if not biography or not biography.strip():
            raise MissingBiographyError()

        excluded_ids = await self.repository.get_excluded_user_ids(requester_id)
        candidates = await self.repository.get_all_candidates(excluded_ids, requester_id)
        if not candidates:
            return []

        requester_vector = self.model.encode(biography)

        candidate_vectors = np.array(
            [self._get_or_compute_embedding(c["id"], c["biography"]) for c in candidates]
        )

        # Cosine similarity: dot product (vectors are L2-normalized)
        similarities = candidate_vectors @ requester_vector

        top_indices = np.argsort(similarities)[::-1][:TOP_N]

        return [
            {
                "id": candidates[i]["id"],
                "username": candidates[i]["username"],
                "biography": candidates[i]["biography"],
            }
            for i in top_indices
        ]


recommendations_service = RecommendationsService(
    repository=recommendations_repository,
    model=embedding_model,
    cache=embedding_store,
)
