from fastapi import APIRouter, Depends

from src.middlewares.authenticate import get_current_user_id
from src.modules.recommendations.recommendations_controller import recommendations_controller

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/recommendations")
async def get_recommendations(user_id: str = Depends(get_current_user_id)) -> list[dict]:
    return await recommendations_controller.get_recommendations(user_id)
