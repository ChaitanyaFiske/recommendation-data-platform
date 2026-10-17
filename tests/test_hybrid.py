
from pathlib import Path

import duckdb
import numpy as np
import pytest

from src.recommendation.hybrid import (
    combine_scores,
    percentile_scale,
    recommend,
)


ROOT = Path(__file__).resolve().parents[1]
RATINGS_FILE = ROOT / "data" / "processed" / "ratings.parquet"
MOVIES_FILE = ROOT / "data" / "processed" / "movies.parquet"


def test_percentile_scale_returns_values_between_zero_and_one():
    scores = percentile_scale([10, 20, 30])

    assert np.all(scores >= 0)
    assert np.all(scores <= 1)
    assert scores[0] < scores[1] < scores[2]


def test_combine_scores_uses_configured_weights():
    result = combine_scores(
        collaborative=np.array([1.0, 0.0]),
        content=np.array([0.0, 1.0]),
        popularity=np.array([0.5, 0.5]),
        collab_weight=0.9,
        content_weight=0.0,
        popularity_weight=0.1,
    )

    np.testing.assert_allclose(result, [0.95, 0.05])


def test_combine_scores_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        combine_scores(
            [0.5], [0.5], [0.5],
            collab_weight=0.8,
            content_weight=0.1,
            popularity_weight=0.2,
        )


def test_combine_scores_rejects_negative_weights():
    with pytest.raises(ValueError, match="negative"):
        combine_scores(
            [0.5], [0.5], [0.5],
            collab_weight=1.1,
            content_weight=-0.1,
            popularity_weight=0.0,
        )


def test_combine_scores_rejects_different_array_shapes():
    with pytest.raises(ValueError, match="same shape"):
        combine_scores(
            [0.5, 0.6],
            [0.5],
            [0.5, 0.6],
        )


def test_recommend_rejects_invalid_user_id():
    with pytest.raises(ValueError, match="positive integer"):
        recommend(user_id=0, top_n=10)


def test_recommend_rejects_invalid_top_n():
    with pytest.raises(ValueError, match="at least 1"):
        recommend(user_id=1, top_n=0)


@pytest.mark.skipif(
    not RATINGS_FILE.exists() or not MOVIES_FILE.exists(),
    reason="Processed MovieLens Parquet files are required.",
)
def test_recommendations_exclude_movies_already_rated():
    user_id = 1
    recommendations = recommend(user_id=user_id, top_n=10)

    assert len(recommendations) <= 10
    assert recommendations["movieId"].is_unique

    movie_ids = recommendations["movieId"].tolist()

    with duckdb.connect() as con:
        already_rated = con.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?) r
            WHERE r.userId = ?
              AND r.movieId IN (SELECT unnest(?))
            """,
            [str(RATINGS_FILE), user_id, movie_ids],
        ).fetchone()[0]

    assert already_rated == 0
