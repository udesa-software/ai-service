from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from src.config.settings import settings
from src.middlewares.authenticate import get_current_user_id
from src.modules.recommendations.recommendations_controller import recommendations_controller
from src.modules.recommendations.recommendations_service import recommendations_service

router = APIRouter(prefix="/api/ai", tags=["ai"])


class UpdateEmbeddingRequest(BaseModel):
    user_id: str
    biography: str


@router.get("/recommendations")
async def get_recommendations(user_id: str = Depends(get_current_user_id)) -> list[dict]:
    return await recommendations_controller.get_recommendations(user_id)


@router.post("/internal/embedding")
async def update_embedding(
    payload: UpdateEmbeddingRequest,
    x_internal_secret: str = Header(None, alias="x-internal-secret"),
):
    if not x_internal_secret or x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    recommendations_service.update_biography_embedding(payload.user_id, payload.biography)
    return {"status": "ok"}
