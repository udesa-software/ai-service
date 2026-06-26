from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from src.embeddings.model import EmbeddingModel
from src.modules.recommendations.recommendations_service import RecommendationsService

from tests.benchmark.benchmark_dataset import BENCHMARK_USERS, BenchmarkUser

TOP_K = 3
REAL_BENCHMARK_ENV = "RUN_REAL_EMBEDDINGS_BENCHMARK"
PLACEHOLDER_TOKENS = {"", "hf_test_token"}


@dataclass
class QueryBenchmarkResult:
    user_id: str
    recommended_ids: list[str]
    expected_ids: tuple[str, ...]
    latency_ms: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float

    def pretty_report(self) -> str:
        return (
            f"{self.user_id}: recommended={self.recommended_ids} "
            f"expected={list(self.expected_ids)} "
            f"precision@{TOP_K}={self.precision_at_k:.3f} "
            f"recall@{TOP_K}={self.recall_at_k:.3f} "
            f"rr={self.reciprocal_rank:.3f} "
            f"latency_ms={self.latency_ms:.3f}"
        )


@dataclass
class BenchmarkSummary:
    queries: list[QueryBenchmarkResult]
    avg_precision_at_k: float
    avg_recall_at_k: float
    mean_reciprocal_rank: float
    hit_rate_at_k: float
    avg_latency_ms: float
    p95_latency_ms: float

    def pretty_report(self) -> str:
        lines = [
            "AI recommendations benchmark with real embeddings",
            f"queries={len(self.queries)} top_k={TOP_K}",
            f"avg_precision@{TOP_K}={self.avg_precision_at_k:.3f}",
            f"avg_recall@{TOP_K}={self.avg_recall_at_k:.3f}",
            f"mrr={self.mean_reciprocal_rank:.3f}",
            f"hit_rate@{TOP_K}={self.hit_rate_at_k:.3f}",
            f"avg_latency_ms={self.avg_latency_ms:.3f}",
            f"p95_latency_ms={self.p95_latency_ms:.3f}",
            "",
            "Per query:",
        ]
        lines.extend(query.pretty_report() for query in self.queries)
        return "\n".join(lines)


class InMemoryRealEmbeddingCache:
    def __init__(self, users: tuple[BenchmarkUser, ...], model: EmbeddingModel):
        self._users = {user.id: user for user in users}
        self._model = model
        self._requester_cache: dict[str, np.ndarray] = {}
        self._candidate_vectors = {
            user.id: self._model.encode(user.biography)
            for user in users
        }

    def get(self, user_id: str, biography: str) -> np.ndarray | None:
        if user_id in self._candidate_vectors:
            return self._candidate_vectors[user_id]
        return self._requester_cache.get(user_id)

    def save(self, user_id: str, biography: str, vector: np.ndarray) -> None:
        self._requester_cache[user_id] = vector

    def find_nearest_candidates(self, vector: np.ndarray, exclude_ids: set[str], limit: int) -> list[dict]:
        scored = []
        for user_id, user in self._users.items():
            if user_id in exclude_ids:
                continue
            similarity = float(np.dot(self._candidate_vectors[user_id], vector))
            scored.append((similarity, user))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"id": user.id, "username": user.username, "biography": user.biography}
            for _, user in scored[:limit]
        ]

    @property
    def candidate_vectors(self) -> dict[str, np.ndarray]:
        return self._candidate_vectors


class BenchmarkRepository:
    def __init__(self, users: tuple[BenchmarkUser, ...]):
        self._users = {user.id: user for user in users}

    async def get_user_biography(self, user_id: str) -> str | None:
        return self._users[user_id].biography

    async def get_excluded_user_ids(self, user_id: str) -> set[str]:
        return set()

    async def get_all_candidates(self, exclude_ids: set[str], requester_id: str) -> list[dict]:
        return [
            {"id": user.id, "username": user.username, "biography": user.biography}
            for user in self._users.values()
            if user.id not in exclude_ids and user.id != requester_id
        ]


def real_benchmark_enabled() -> bool:
    return os.getenv(REAL_BENCHMARK_ENV, "").lower() in {"1", "true", "yes"}


def validate_real_benchmark_env() -> None:
    token = os.getenv("HF_API_TOKEN", "").strip()
    if token in PLACEHOLDER_TOKENS:
        raise RuntimeError(
            "HF_API_TOKEN no esta configurado para el benchmark real. "
            "Exportalo en la shell actual o agregalo a ai-service/api/.env antes de correr pytest."
        )


def _precision_at_k(recommended_ids: list[str], expected_ids: tuple[str, ...], k: int) -> float:
    recommended_top_k = recommended_ids[:k]
    hits = sum(1 for item in recommended_top_k if item in expected_ids)
    return hits / k


def _recall_at_k(recommended_ids: list[str], expected_ids: tuple[str, ...], k: int) -> float:
    recommended_top_k = recommended_ids[:k]
    hits = sum(1 for item in recommended_top_k if item in expected_ids)
    return hits / len(expected_ids)


def _reciprocal_rank(recommended_ids: list[str], expected_ids: tuple[str, ...]) -> float:
    for index, user_id in enumerate(recommended_ids, start=1):
        if user_id in expected_ids:
            return 1.0 / index
    return 0.0


async def run_recommendations_benchmark() -> BenchmarkSummary:
    model = EmbeddingModel()
    repository = BenchmarkRepository(BENCHMARK_USERS)
    cache = InMemoryRealEmbeddingCache(BENCHMARK_USERS, model)
    service = RecommendationsService(repository=repository, model=model, cache=cache)

    query_results: list[QueryBenchmarkResult] = []

    for user in BENCHMARK_USERS:
        start = time.perf_counter()
        recommendations = await service.get_recommendations(user.id)
        elapsed_ms = (time.perf_counter() - start) * 1000
        recommended_ids = [item["id"] for item in recommendations[:TOP_K]]
        expected_ids = user.expected_top_ids

        query_results.append(
            QueryBenchmarkResult(
                user_id=user.id,
                recommended_ids=recommended_ids,
                expected_ids=expected_ids,
                latency_ms=elapsed_ms,
                precision_at_k=_precision_at_k(recommended_ids, expected_ids, TOP_K),
                recall_at_k=_recall_at_k(recommended_ids, expected_ids, TOP_K),
                reciprocal_rank=_reciprocal_rank(recommended_ids, expected_ids),
            )
        )

    latencies = sorted(item.latency_ms for item in query_results)
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)

    return BenchmarkSummary(
        queries=query_results,
        avg_precision_at_k=sum(item.precision_at_k for item in query_results) / len(query_results),
        avg_recall_at_k=sum(item.recall_at_k for item in query_results) / len(query_results),
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in query_results) / len(query_results),
        hit_rate_at_k=sum(1 for item in query_results if item.reciprocal_rank > 0) / len(query_results),
        avg_latency_ms=sum(item.latency_ms for item in query_results) / len(query_results),
        p95_latency_ms=latencies[p95_index],
    )


def build_real_embedding_service() -> tuple[RecommendationsService, InMemoryRealEmbeddingCache]:
    model = EmbeddingModel()
    repository = BenchmarkRepository(BENCHMARK_USERS)
    cache = InMemoryRealEmbeddingCache(BENCHMARK_USERS, model)
    service = RecommendationsService(repository=repository, model=model, cache=cache)
    return service, cache
