import argparse
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "recommendations"
LOG_DIR = ROOT / "logs"

RATINGS_FILE = DATA_DIR / "ratings.parquet"
MOVIES_FILE = DATA_DIR / "movies.parquet"

CONFIG = load_config()

CATALOG_SIZE = CONFIG["catalog_size"]
MIN_RATINGS = CONFIG["minimum_movie_ratings"]
GLOBAL_MEAN = CONFIG["global_mean_rating"]

COLLAB_WEIGHT = CONFIG["weights"]["collaborative"]
CONTENT_WEIGHT = CONFIG["weights"]["content"]
POPULARITY_WEIGHT = CONFIG["weights"]["popularity"]


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "hybrid.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_catalog(con):
    """Create one consistent catalog and its item-index mapping."""
    catalog = con.execute(
        """
        SELECT
            m.movieId,
            m.title,
            m.genres,
            COUNT(r.movieId) AS rating_count,
            AVG(r.rating) AS average_rating
        FROM read_parquet(?) m
        JOIN read_parquet(?) r USING (movieId)
        GROUP BY m.movieId, m.title, m.genres
        HAVING COUNT(r.movieId) >= ?
        ORDER BY rating_count DESC, m.movieId
        LIMIT ?
        """,
        [str(MOVIES_FILE), str(RATINGS_FILE),
         MIN_RATINGS, CATALOG_SIZE],
    ).fetchdf()

    if catalog.empty:
        raise ValueError("The candidate catalog is empty.")

    catalog["item_idx"] = np.arange(len(catalog), dtype=np.int32)

    # Use this exact mapping for all interaction joins.
    con.register("catalog_mapping", catalog[["movieId", "item_idx"]])

    logging.info("Loaded %s candidate movies.", len(catalog))
    return catalog


def collaborative_scores(con, catalog, user_id):
    """Score catalog movies using item-to-item rating similarity."""
    interactions = con.execute(
        """
        SELECT c.item_idx, r.userId, r.rating
        FROM read_parquet(?) r
        JOIN catalog_mapping c USING (movieId)
        """,
        [str(RATINGS_FILE)],
    ).fetchdf()

    if interactions.empty:
        return np.zeros(len(catalog))

    # User IDs may be sparse, so use their actual numeric indices.
    matrix = csr_matrix(
        (
            interactions["rating"].to_numpy(dtype=np.float32),
            (
                interactions["item_idx"].to_numpy(dtype=np.int32),
                interactions["userId"].to_numpy(dtype=np.int32),
            ),
        ),
        shape=(
            len(catalog),
            int(interactions["userId"].max()) + 1,
        ),
        dtype=np.float32,
    )

    # Cosine similarity between movie rating vectors.
    normalized_matrix = normalize(matrix, norm="l2", axis=1)
    del matrix, interactions

    similarities = normalized_matrix @ normalized_matrix.T
    del normalized_matrix

    history = con.execute(
        """
        SELECT c.item_idx, r.rating
        FROM read_parquet(?) r
        JOIN catalog_mapping c USING (movieId)
        WHERE r.userId = ?
        """,
        [str(RATINGS_FILE), user_id],
    ).fetchdf()

    positive = history[history["rating"] >= 4.0]

    if positive.empty:
        logging.info(
            "User %s has no positively rated catalog movies; "
            "collaborative scores will be zero.",
            user_id,
        )
        return np.zeros(len(catalog))

    indices = positive["item_idx"].to_numpy(dtype=np.int32)
    weights = (
        positive["rating"].to_numpy(dtype=np.float32) - 3.5
    )

    # Similarity-weighted positive ratings.
    scores = np.asarray(similarities[:, indices] @ weights).ravel()
    denominator = np.asarray(similarities[:, indices] @ np.ones(
        len(indices), dtype=np.float32
    )).ravel()

    scores = np.divide(
        scores,
        denominator,
        out=np.zeros_like(scores, dtype=np.float32),
        where=denominator > 0,
    )

    return scores


def content_scores(con, catalog, user_id):
    """Score movies according to the user's genre preferences."""
    
    genre_preferences = con.execute(
        """
        SELECT
            genre,
            AVG(r.rating) AS mean_rating
        FROM read_parquet(?) r
        JOIN read_parquet(?) m USING (movieId),
        UNNEST(string_split(m.genres, '|')) AS genre_table(genre)
        WHERE r.userId = ?
          AND m.genres <> '(no genres listed)'
        GROUP BY genre
        """,
        [str(RATINGS_FILE), str(MOVIES_FILE), user_id],
    ).fetchdf()


    if genre_preferences.empty:
        return np.zeros(len(catalog))

    preferences = dict(
        zip(genre_preferences["genre"], genre_preferences["mean_rating"])
    )
    user_mean = float(genre_preferences["mean_rating"].mean())

    scores = []
    for genres in catalog["genres"].fillna(""):
        movie_genres = [
            genre for genre in genres.split("|")
            if genre and genre != "(no genres listed)"
        ]
        matching = [
            preferences[genre]
            for genre in movie_genres
            if genre in preferences
        ]
        scores.append(
            float(np.mean(matching)) if matching else user_mean
        )

    return np.asarray(scores, dtype=np.float32)


def popularity_scores(catalog):
    """Calculate Bayesian-weighted ratings."""
    counts = catalog["rating_count"].to_numpy(dtype=np.float64)
    averages = catalog["average_rating"].to_numpy(dtype=np.float64)

    weighted = (
        counts / (counts + 1000) * averages
        + 1000 / (counts + 1000) * GLOBAL_MEAN
    )
    return weighted


def percentile_scale(scores):
    """Put differently scaled signals on a comparable 0–1 scale."""
    return pd.Series(scores).rank(pct=True, method="average").to_numpy()


def combine_scores(
    collaborative,
    content,
    popularity,
    collab_weight=COLLAB_WEIGHT,
    content_weight=CONTENT_WEIGHT,
    popularity_weight=POPULARITY_WEIGHT,
):
    """Combine normalized recommendation signals using configured weights."""
    weights = [collab_weight, content_weight, popularity_weight]

    if any(weight < 0 for weight in weights):
        raise ValueError("Weights cannot be negative.")

    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Recommendation weights must sum to 1.")

    collaborative = np.asarray(collaborative, dtype=float)
    content = np.asarray(content, dtype=float)
    popularity = np.asarray(popularity, dtype=float)

    if not (
        collaborative.shape == content.shape == popularity.shape
    ):
        raise ValueError("All score arrays must have the same shape.")

    return (
        collab_weight * collaborative
        + content_weight * content
        + popularity_weight * popularity
    )



def recommend(user_id, top_n=10):
    setup_logging()

    if user_id < 1:
        raise ValueError("user_id must be a positive integer.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if not RATINGS_FILE.exists() or not MOVIES_FILE.exists():
        raise FileNotFoundError(
            "Processed Parquet files are missing. Run the processing "
            "scripts before using the recommender."
        )

    con = duckdb.connect()
    try:
        catalog = load_catalog(con)

        # Confirm the user exists in the ratings data.
        user_exists = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?) WHERE userId = ?",
            [str(RATINGS_FILE), user_id],
        ).fetchone()[0]

        if user_exists == 0:
            raise ValueError(f"User {user_id} has no rating history.")

        collab = collaborative_scores(con, catalog, user_id)
        content = content_scores(con, catalog, user_id)
        popularity = popularity_scores(catalog)

        # Normalize signals before combining them.
        catalog["collaborative_score"] = percentile_scale(collab)
        catalog["content_score"] = percentile_scale(content)
        catalog["popularity_score"] = percentile_scale(popularity)

        catalog["hybrid_score"] = combine_scores(
            catalog["collaborative_score"],
            catalog["content_score"],
            catalog["popularity_score"],
        )

        # Exclude every movie this user has already rated, including
        # movies outside the candidate catalog.
        seen = con.execute(
            """
            SELECT DISTINCT movieId
            FROM read_parquet(?)
            WHERE userId = ?
            """,
            [str(RATINGS_FILE), user_id],
        ).fetchdf()["movieId"]

        recommendations = (
            catalog[~catalog["movieId"].isin(seen)]
            .sort_values(
                ["hybrid_score", "rating_count"],
                ascending=[False, False],
            )
            .head(top_n)
            [
                [
                    "movieId",
                    "title",
                    "genres",
                    "rating_count",
                    "collaborative_score",
                    "content_score",
                    "popularity_score",
                    "hybrid_score",
                ]
            ]
            .copy()
        )

        output_file = OUTPUT_DIR / f"user_{user_id}_hybrid.csv"
        recommendations.to_csv(output_file, index=False)

        print(recommendations.to_string(index=False))
        print(f"\nSaved to: {output_file}")
        logging.info(
            "Generated %s hybrid recommendations for user %s.",
            len(recommendations),
            user_id,
        )

        return recommendations
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hybrid movie recommendations."
    )
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    recommend(args.user_id, args.top_n)
