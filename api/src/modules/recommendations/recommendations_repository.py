from src.config.http_client import get_http_client
from src.config.settings import settings


class RecommendationsRepository:
    async def get_user_biography(self, user_id: str) -> str | None:
        client = get_http_client()
        r = await client.get(
            f"{settings.users_internal_url}/internal/users/{user_id}",
            headers={"x-internal-secret": settings.internal_secret},
        )
        if r.status_code == 410:
            return None
        r.raise_for_status()
        return r.json().get("biography")

    async def get_excluded_user_ids(self, user_id: str) -> set[str]:
        client = get_http_client()
        r = await client.get(
            f"{settings.friends_internal_url}/internal/friends/user/{user_id}/exclusions",
            headers={"x-internal-secret": settings.internal_secret},
        )
        r.raise_for_status()
        return set(r.json()["excludedIds"])

    async def get_all_candidates(self, exclude_ids: set[str], requester_id: str) -> list[dict]:
        client = get_http_client()
        all_excluded = list(exclude_ids | {requester_id})
        r = await client.post(
            f"{settings.users_internal_url}/internal/users/candidates",
            json={"excludeIds": all_excluded},
            headers={"x-internal-secret": settings.internal_secret},
        )
        r.raise_for_status()
        return r.json()


recommendations_repository = RecommendationsRepository()
