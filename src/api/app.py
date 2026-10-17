
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.api.recommendation_service import RecommendationService

from src.api.schemas import (
    HealthResponse,
    RecommendationsResponse,
    MovieResponse,
)


logging.basicConfig(level=logging.INFO)
service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service

    service = RecommendationService()
    try:
        yield
    finally:
        if service is not None:
            service.close()
            service = None


app = FastAPI(
    title="Movie Recommendation API",
    description="Personalized movie recommendations using a hybrid model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not ready.",
        )

    return {"status": "ok", "model": "hybrid"}


@app.get("/recommendations/{user_id}", response_model=RecommendationsResponse)
def get_recommendations(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=100),
):
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not ready.",
        )

    try:
        recommendations = service.recommend(user_id, top_n)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "user_id": user_id,
        "count": len(recommendations),
        "recommendations": recommendations,
    }


@app.get("/movies/{movie_id}", response_model=MovieResponse)
def get_movie(movie_id: int):
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not ready.",
        )

    movie = service.get_movie(movie_id)

    if movie is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie {movie_id} was not found.",
        )

    return movie
