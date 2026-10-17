
import pytest
from fastapi.testclient import TestClient

import src.api.app as api_module


class FakeRecommendationService:
    """Lightweight test double for the real recommendation service."""

    def recommend(self, user_id: int, top_n: int = 10):
        if user_id < 1:
            raise ValueError("user_id must be a positive integer.")

        if user_id == 999:
            raise LookupError(
                f"No rating history found for user {user_id}."
            )

        movies = [
            {
                "movieId": 1,
                "title": "Toy Story (1995)",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
                "hybrid_score": 0.95,
            },
            {
                "movieId": 2,
                "title": "Jumanji (1995)",
                "genres": "Adventure|Children|Fantasy",
                "hybrid_score": 0.85,
            },
        ]

        return movies[:top_n]

    def get_movie(self, movie_id: int):
        if movie_id == 2959:
            return {
                "movieId": 2959,
                "title": "Fight Club (1999)",
                "genres": "Action|Crime|Drama|Thriller",
            }

        return None

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "RecommendationService",
        FakeRecommendationService,
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "hybrid",
    }


def test_recommendations_endpoint(client):
    response = client.get("/recommendations/1?top_n=2")

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 1
    assert data["count"] == 2
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["title"] == "Toy Story (1995)"


def test_unknown_user_returns_404(client):
    response = client.get("/recommendations/999")

    assert response.status_code == 404
    assert "No rating history" in response.json()["detail"]


def test_invalid_user_returns_400(client):
    response = client.get("/recommendations/0")

    assert response.status_code == 400
    assert "positive integer" in response.json()["detail"]


@pytest.mark.parametrize("top_n", [0, 101])
def test_invalid_top_n_returns_422(client, top_n):
    response = client.get(f"/recommendations/1?top_n={top_n}")

    assert response.status_code == 422


def test_movie_details_endpoint(client):
    response = client.get("/movies/2959")

    assert response.status_code == 200
    assert response.json()["title"] == "Fight Club (1999)"


def test_unknown_movie_returns_404(client):
    response = client.get("/movies/999999999")

    assert response.status_code == 404
    assert "was not found" in response.json()["detail"]
