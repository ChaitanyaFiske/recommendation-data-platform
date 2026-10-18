
import logging
import threading

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.recommendation.hybrid import (
    CATALOG_SIZE,
    GLOBAL_MEAN,
    MIN_RATINGS,
    COLLAB_WEIGHT,
    CONTENT_WEIGHT,
    POPULARITY_WEIGHT,
    percentile_scale,
    content_scores,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RATINGS_FILE = DATA_DIR / "ratings.parquet"
MOVIES_FILE = DATA_DIR / "movies.parquet"


class RecommendationService:
    """Prepare recommendation data once and reuse it for API requests."""

    def __init__(self):
        if not RATINGS_FILE.exists() or not MOVIES_FILE.exists():
            raise FileNotFoundError(
                "Processed Parquet files are missing. "
                "Run the data-processing scripts first."
            )

        self._lock = threading.Lock()
        self.con = duckdb.connect()
        self.catalog = None
        self.similarity = None
        self.popularity_scores = None

        try:
            self._prepare_model()
        except Exception:
            self.con.close()
            raise

    def _prepare_model(self):
        logging.info("Preparing recommendation catalog...")

        self.catalog = self.con.execute(
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
            [
                str(MOVIES_FILE),
                str(RATINGS_FILE),
                MIN_RATINGS,
                CATALOG_SIZE,
            ],
        ).fetchdf()

        if self.catalog.empty:
            raise ValueError("No movies available in the candidate catalog.")

        self.catalog["item_idx"] = np.arange(
            len(self.catalog), dtype=np.int32
        )

        self.con.register(
            "catalog_mapping",
            self.catalog[["movieId", "item_idx"]],
        )

        logging.info("Building collaborative similarity matrix...")

        interaction_cursor = self.con.execute(
            """
            SELECT c.item_idx, r.userId, r.rating
            FROM read_parquet(?) r
            JOIN catalog_mapping c USING (movieId)
            """,
            [str(RATINGS_FILE)],
        )

        item_chunks = []
        user_chunks = []
        rating_chunks = []
        max_user_id = 0

        while True:
            batch = interaction_cursor.fetch_df_chunk(100)

            if batch.empty:
                break

            item_chunks.append(
                batch["item_idx"].to_numpy(dtype=np.int32)
            )
            user_chunks.append(
                batch["userId"].to_numpy(dtype=np.int32)
            )
            rating_chunks.append(
                batch["rating"].to_numpy(dtype=np.float32)
            )

            max_user_id = max(
                max_user_id,
                int(batch["userId"].max()),
            )

            del batch

        logging.info("Converting interaction batches to sparse matrix...")

        item_indices = np.concatenate(item_chunks)
        user_indices = np.concatenate(user_chunks)
        rating_values = np.concatenate(rating_chunks)

        del item_chunks, user_chunks, rating_chunks

        item_user = csr_matrix(
            (
                rating_values,
                (item_indices, user_indices),
            ),
            shape=(len(self.catalog), max_user_id + 1),
            dtype=np.float32,
        )

        del item_indices, user_indices, rating_values

        normalized = normalize(item_user, norm="l2", axis=1)
        self.similarity = normalized @ normalized.T

        self.popularity_scores = (
            self.catalog["rating_count"].to_numpy(dtype=float)
            / (
                self.catalog["rating_count"].to_numpy(dtype=float)
                + MIN_RATINGS
            )
            * self.catalog["average_rating"].to_numpy(dtype=float)
            + MIN_RATINGS
            / (
                self.catalog["rating_count"].to_numpy(dtype=float)
                + MIN_RATINGS
            )
            * GLOBAL_MEAN
        )

        del item_user, normalized

        logging.info(
            "Recommendation service ready: %s candidate movies.",
            len(self.catalog),
        )

    def recommend(self, user_id: int, top_n: int = 10):
        if user_id < 1:
            raise ValueError("user_id must be a positive integer.")
        if top_n < 1 or top_n > 100:
            raise ValueError("top_n must be between 1 and 100.")

        with self._lock:
            user_exists = self.con.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE userId = ?
                """,
                [str(RATINGS_FILE), user_id],
            ).fetchone()[0]

            if not user_exists:
                raise LookupError(f"No rating history found for user {user_id}.")

            history = self.con.execute(
                """
                SELECT movieId, rating
                FROM read_parquet(?)
                WHERE userId = ?
                """,
                [str(RATINGS_FILE), user_id],
            ).fetchdf()

            positive = history[
                (history["rating"] >= 4.0)
                & history["movieId"].isin(self.catalog["movieId"])
            ].merge(
                self.catalog[["movieId", "item_idx"]],
                on="movieId",
                how="inner",
            )

            if positive.empty:
                collaborative = np.zeros(
                    len(self.catalog), dtype=np.float32
                )
            else:
                indices = positive["item_idx"].to_numpy(dtype=np.int32)
                weights = (
                    positive["rating"].to_numpy(dtype=np.float32) - 3.5
                )
                similarities = self.similarity[:, indices]

                numerator = np.asarray(similarities @ weights).ravel()
                denominator = np.asarray(
                    similarities
                    @ np.ones(len(indices), dtype=np.float32)
                ).ravel()

                collaborative = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator, dtype=np.float32),
                    where=denominator > 0,
                )

            content = content_scores(
                self.con, self.catalog, user_id
            )

            seen_ids = set(history["movieId"].tolist())
            candidates = self.catalog[
                ~self.catalog["movieId"].isin(seen_ids)
            ].copy()

            if candidates.empty:
                return []

            indices = candidates["item_idx"].to_numpy(dtype=np.int32)

            candidates["collaborative_score"] = percentile_scale(
                collaborative[indices]
            )
            candidates["content_score"] = percentile_scale(
                content[indices]
            )
            candidates["popularity_score"] = percentile_scale(
                self.popularity_scores[indices]
            )

            candidates["hybrid_score"] = (
                COLLAB_WEIGHT * candidates["collaborative_score"]
                + CONTENT_WEIGHT * candidates["content_score"]
                + POPULARITY_WEIGHT * candidates["popularity_score"]
            )

            ranked = candidates.sort_values(
                ["hybrid_score", "rating_count", "movieId"],
                ascending=[False, False, True],
            ).head(top_n)

            columns = [
                "movieId",
                "title",
                "genres",
                "hybrid_score",
            ]
            return ranked[columns].to_dict(orient="records")

    def get_movie(self, movie_id: int):
        with self._lock:
            movie = self.con.execute(
                """
                SELECT movieId, title, genres
                FROM read_parquet(?)
                WHERE movieId = ?
                LIMIT 1
                """,
                [str(MOVIES_FILE), movie_id],
            ).fetchdf()

        if movie.empty:
            return None

        return movie.iloc[0].to_dict()

    def close(self):
        with self._lock:
            self.con.close()
