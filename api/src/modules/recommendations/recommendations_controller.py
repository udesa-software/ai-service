from src.modules.recommendations.recommendations_service import recommendations_service


class RecommendationsController:
    async def get_recommendations(self, user_id: str) -> list[dict]:
        return await recommendations_service.get_recommendations(user_id)


recommendations_controller = RecommendationsController()
