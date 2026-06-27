from unittest.mock import ANY, AsyncMock, MagicMock

import numpy as np
import pytest

from src.middlewares.error_handler import MissingBiographyError
from src.modules.recommendations.recommendations_service import RecommendationsService

# ─── Fixtures ──────────────────────────────────────────────────────────────────

REQUESTER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

CANDIDATE_A = {
    "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "username": "alice",
    "biography": "Estudiante de Ingeniería apasionada por la robótica y el código.",
    "profile_photo_url": "https://example.com/alice.jpg",
}
CANDIDATE_B = {
    "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
    "username": "bob",
    "biography": "Me gusta el cine independiente y la filosofía continental.",
    "profile_photo_url": "https://example.com/bob.jpg",
}
CANDIDATE_C = {
    "id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
    "username": "carol",
    "biography": "Apasionada por la IA y el machine learning en español.",
    "profile_photo_url": "https://example.com/carol.jpg",
}

BIOGRAPHY_REQUESTER = "Hago investigación en NLP y me interesa la IA aplicada."


def make_service(
    biography=BIOGRAPHY_REQUESTER,
    excluded_ids=None,
    candidates=None,
    cached_embeddings=None,
    vector_candidates=None,
) -> RecommendationsService:
    """Build a RecommendationsService with fully controlled mock dependencies."""
    repository = MagicMock()
    repository.get_user_biography = AsyncMock(return_value=biography)
    repository.get_excluded_user_ids = AsyncMock(return_value=excluded_ids or set())
    repository.get_all_candidates = AsyncMock(
        return_value=candidates if candidates is not None else [CANDIDATE_A, CANDIDATE_B]
    )

    cached_embeddings = cached_embeddings or {}

    model = MagicMock()
    # encode always returns a normalized unit vector (dot product = cosine similarity)
    model.encode.side_effect = lambda text: _deterministic_vector(text)

    cache = MagicMock()
    cache.get.side_effect = lambda user_id, bio: cached_embeddings.get(user_id)
    cache.save.return_value = None
    cache.find_nearest_candidates.return_value = vector_candidates or []

    return RecommendationsService(repository=repository, model=model, cache=cache)


def _deterministic_vector(text: str) -> np.ndarray:
    """Returns a stable unit vector derived from the text hash (for deterministic tests)."""
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    v = rng.standard_normal(768)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ─── MissingBiographyError ─────────────────────────────────────────────────────

class TestMissingBiography:
    async def test_raises_when_biography_is_none(self):
        service = make_service(biography=None)
        with pytest.raises(MissingBiographyError):
            await service.get_recommendations(REQUESTER_ID)

    async def test_raises_when_biography_is_empty_string(self):
        service = make_service(biography="")
        with pytest.raises(MissingBiographyError):
            await service.get_recommendations(REQUESTER_ID)

    async def test_raises_when_biography_is_only_whitespace(self):
        service = make_service(biography="   ")
        with pytest.raises(MissingBiographyError):
            await service.get_recommendations(REQUESTER_ID)

    async def test_raises_correct_exception_type(self):
        """MissingBiographyError debe ser instancia de AppError para que el handler lo capture."""
        from src.middlewares.error_handler import AppError
        service = make_service(biography=None)
        with pytest.raises(AppError):
            await service.get_recommendations(REQUESTER_ID)

    async def test_does_not_query_candidates_when_no_biography(self):
        service = make_service(biography=None)
        with pytest.raises(MissingBiographyError):
            await service.get_recommendations(REQUESTER_ID)
        service.repository.get_all_candidates.assert_not_called()

    async def test_does_not_call_model_when_no_biography(self):
        service = make_service(biography="")
        with pytest.raises(MissingBiographyError):
            await service.get_recommendations(REQUESTER_ID)
        service.model.encode.assert_not_called()


# ─── Sin candidatos ────────────────────────────────────────────────────────────

class TestNoCandidates:
    async def test_returns_empty_list_when_no_candidates(self):
        service = make_service(candidates=[])
        result = await service.get_recommendations(REQUESTER_ID)
        assert result == []

    async def test_only_computes_requester_embedding_when_no_candidates(self):
        service = make_service(candidates=[])
        await service.get_recommendations(REQUESTER_ID)
        encode_calls = [call[0][0] for call in service.model.encode.call_args_list]
        assert encode_calls == [BIOGRAPHY_REQUESTER]


# ─── Consultas al repositorio ─────────────────────────────────────────────────

class TestRepositoryCalls:
    async def test_queries_biography_with_correct_requester_id(self):
        service = make_service()
        await service.get_recommendations(REQUESTER_ID)
        service.repository.get_user_biography.assert_called_once_with(REQUESTER_ID)

    async def test_queries_excluded_ids_with_correct_requester_id(self):
        service = make_service()
        await service.get_recommendations(REQUESTER_ID)
        service.repository.get_excluded_user_ids.assert_called_once_with(REQUESTER_ID)

    async def test_passes_excluded_ids_to_get_all_candidates(self):
        excluded = {"eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"}
        service = make_service(excluded_ids=excluded)
        await service.get_recommendations(REQUESTER_ID)
        service.repository.get_all_candidates.assert_called_once_with(excluded, REQUESTER_ID)

    async def test_passes_requester_id_to_get_all_candidates(self):
        service = make_service()
        await service.get_recommendations(REQUESTER_ID)
        _, called_requester_id = service.repository.get_all_candidates.call_args[0]
        assert called_requester_id == REQUESTER_ID


# ─── Formato de respuesta ─────────────────────────────────────────────────────

class TestResponseFormat:
    async def test_returns_list_of_dicts(self):
        service = make_service()
        result = await service.get_recommendations(REQUESTER_ID)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    async def test_each_result_has_id_username_biography(self):
        service = make_service()
        result = await service.get_recommendations(REQUESTER_ID)
        for item in result:
            assert "id" in item
            assert "username" in item
            assert "biography" in item
            assert "profile_photo_url" in item

    async def test_each_result_has_exactly_four_fields(self):
        service = make_service()
        result = await service.get_recommendations(REQUESTER_ID)
        for item in result:
            assert set(item.keys()) == {"id", "username", "biography", "profile_photo_url"}

    async def test_ids_come_from_candidates(self):
        service = make_service(candidates=[CANDIDATE_A, CANDIDATE_B])
        result = await service.get_recommendations(REQUESTER_ID)
        returned_ids = {item["id"] for item in result}
        expected_ids = {CANDIDATE_A["id"], CANDIDATE_B["id"]}
        assert returned_ids == expected_ids


# ─── Búsqueda vectorial ───────────────────────────────────────────────────────

class TestVectorSearch:
    async def test_uses_vector_candidates_without_fetching_all_candidates(self):
        service = make_service(vector_candidates=[CANDIDATE_C])

        result = await service.get_recommendations(REQUESTER_ID)

        assert result == [CANDIDATE_C]
        service.repository.get_all_candidates.assert_not_called()
        service.cache.find_nearest_candidates.assert_called_once_with(
            ANY,
            {REQUESTER_ID},
            10,
        )


# ─── Ordenamiento por similitud ───────────────────────────────────────────────

class TestSimilarityRanking:
    async def test_returns_results_sorted_by_similarity_descending(self):
        """El candidato más similar debe aparecer primero."""
        requester_bio = "Me apasiona la inteligencia artificial y el NLP."
        # bio_similar es semánticamente cercana al requester
        bio_similar = "Investigo machine learning y procesamiento de lenguaje natural."
        bio_different = "Me encanta el surf y los deportes acuáticos en la playa."

        close_candidate = {
            "id": "eeee-1",
            "username": "near",
            "biography": bio_similar,
            "profile_photo_url": "https://example.com/near.jpg",
        }
        far_candidate = {
            "id": "ffff-2",
            "username": "far",
            "biography": bio_different,
            "profile_photo_url": "https://example.com/far.jpg",
        }

        # Usamos vectores controlados: el close_candidate tiene alta similitud
        v_requester = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v_close = np.array([0.99, 0.14, 0.0], dtype=np.float32)   # ~cos 0.99
        v_far = np.array([0.0, 0.0, 1.0], dtype=np.float32)        # cos 0.0

        vector_map = {
            requester_bio: v_requester,
            bio_similar: v_close,
            bio_different: v_far,
        }

        service = make_service(
            biography=requester_bio,
            candidates=[far_candidate, close_candidate],  # orden invertido
        )
        service.model.encode.side_effect = lambda text: vector_map[text]
        service.cache.get.return_value = None

        result = await service.get_recommendations(REQUESTER_ID)

        assert result[0]["id"] == close_candidate["id"]
        assert result[1]["id"] == far_candidate["id"]

    async def test_ties_in_similarity_do_not_crash(self):
        """Dos candidatos con similitud idéntica no deben generar un error."""
        v = np.array([1.0, 0.0], dtype=np.float32)
        service = make_service(candidates=[CANDIDATE_A, CANDIDATE_B])
        service.model.encode.return_value = v
        service.cache.get.return_value = None

        result = await service.get_recommendations(REQUESTER_ID)
        assert len(result) == 2


# ─── Límite de resultados ─────────────────────────────────────────────────────

class TestTopNLimit:
    async def test_returns_at_most_top_n_candidates(self):
        """Nunca devuelve más de TOP_N (10) resultados."""
        many_candidates = [
            {
                "id": f"id-{i}",
                "username": f"user{i}",
                "biography": f"Bio número {i}",
                "profile_photo_url": f"https://example.com/user{i}.jpg",
            }
            for i in range(25)
        ]
        service = make_service(candidates=many_candidates)
        service.cache.get.return_value = None

        result = await service.get_recommendations(REQUESTER_ID)
        assert len(result) <= 10

    async def test_returns_all_candidates_when_fewer_than_top_n(self):
        """Si hay menos de 10 candidatos, devuelve todos."""
        few_candidates = [CANDIDATE_A, CANDIDATE_B, CANDIDATE_C]
        service = make_service(candidates=few_candidates)
        service.cache.get.return_value = None

        result = await service.get_recommendations(REQUESTER_ID)
        assert len(result) == 3


# ─── Cache de embeddings ──────────────────────────────────────────────────────

class TestEmbeddingCache:
    async def test_uses_cached_embedding_when_available(self):
        """Si el embedding está en cache, no debe llamar a model.encode para ese candidato."""
        cached_vector = _deterministic_vector("cached")
        cached = {CANDIDATE_A["id"]: cached_vector}

        service = make_service(candidates=[CANDIDATE_A], cached_embeddings=cached)

        await service.get_recommendations(REQUESTER_ID)

        # model.encode solo debería llamarse para el requester (no para CANDIDATE_A)
        encode_calls = [call[0][0] for call in service.model.encode.call_args_list]
        assert CANDIDATE_A["biography"] not in encode_calls

    async def test_computes_embedding_when_not_cached(self):
        """Si el embedding no está en cache, debe llamar a model.encode y guardarlo."""
        service = make_service(candidates=[CANDIDATE_A])
        service.cache.get.return_value = None

        await service.get_recommendations(REQUESTER_ID)

        encode_calls = [call[0][0] for call in service.model.encode.call_args_list]
        assert CANDIDATE_A["biography"] in encode_calls

    async def test_saves_embedding_after_computing(self):
        """Después de calcular un embedding nuevo, debe guardarlo en cache."""
        service = make_service(candidates=[CANDIDATE_A])
        service.cache.get.return_value = None

        await service.get_recommendations(REQUESTER_ID)

        service.cache.save.assert_any_call(
            CANDIDATE_A["id"],
            CANDIDATE_A["biography"],
            ANY,
        )

    async def test_saves_requester_embedding_to_cache(self):
        """El embedding del requester también se cachea para futuras consultas."""
        service = make_service(candidates=[CANDIDATE_A])
        service.cache.get.return_value = None

        await service.get_recommendations(REQUESTER_ID)

        saved_ids = [call[0][0] for call in service.cache.save.call_args_list]
        assert REQUESTER_ID in saved_ids
