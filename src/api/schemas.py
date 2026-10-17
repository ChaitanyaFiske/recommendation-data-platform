
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str


class RecommendationItem(BaseModel):
    movieId: int
    title: str
    genres: str
    hybrid_score: float


class RecommendationsResponse(BaseModel):
    user_id: int
    count: int = Field(ge=0)
    recommendations: list[RecommendationItem]


class MovieResponse(BaseModel):
    movieId: int
    title: str
    genres: str
