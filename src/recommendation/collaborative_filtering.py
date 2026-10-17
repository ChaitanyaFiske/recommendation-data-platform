
from pathlib import Path
import argparse

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = DATA / "recommendations"
OUT.mkdir(parents=True, exist_ok=True)

CATALOG_SIZE = 1500


def main(user_id: int, top_n: int):
    con = duckdb.connect()

    try:
        movies_path = (DATA / "movies.parquet").as_posix()
        ratings_path = (DATA / "ratings.parquet").as_posix()

        con.execute(f"""
            CREATE VIEW movies AS
            SELECT * FROM read_parquet('{movies_path}')
        """)
        con.execute(f"""
            CREATE VIEW ratings AS
            SELECT * FROM read_parquet('{ratings_path}')
        """)

        catalog = con.execute(f"""
            SELECT
                m.movieId,
                m.title,
                m.genres,
                COUNT(*) AS rating_count,
                AVG(r.rating) AS average_rating
            FROM movies m
            JOIN ratings r USING (movieId)
            GROUP BY m.movieId, m.title, m.genres
            ORDER BY rating_count DESC, m.movieId
            LIMIT {CATALOG_SIZE}
        """).fetchdf()

        catalog = catalog.reset_index(drop=True)
        catalog["item_idx"] = np.arange(len(catalog))

        user_ratings = con.execute("""
            SELECT movieId, rating
            FROM ratings
            WHERE userId = ?
        """, [user_id]).fetchdf()

        if user_ratings.empty:
            raise ValueError(f"Unknown user ID: {user_id}")

        interactions = con.execute(f"""
            SELECT
                r.userId,
                c.item_idx,
                r.rating
            FROM ratings r
            JOIN (
                SELECT movieId,
                       ROW_NUMBER() OVER (
                           ORDER BY COUNT(*) DESC, movieId
                       ) - 1 AS item_idx
                FROM ratings
                GROUP BY movieId
                ORDER BY COUNT(*) DESC, movieId
                LIMIT {CATALOG_SIZE}
            ) c USING (movieId)
        """).fetchdf()

        # The same deterministic ordering is used to create the
        # catalog and interaction indices.
        item_user = csr_matrix(
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

        similarity = cosine_similarity(
            item_user,
            dense_output=True,
        ).astype(np.float32)

        global_mean = float(interactions["rating"].mean())

        positive = user_ratings[
            user_ratings["rating"] >= 4.0
        ].merge(
            catalog[["movieId", "item_idx"]],
            on="movieId",
        )

        if not positive.empty:
            indices = positive["item_idx"].to_numpy(dtype=int)
            weights = np.maximum(
                positive["rating"].to_numpy() - global_mean,
                0.0,
            )
            scores = (
                similarity[:, indices] @ weights
            ) / max(float(weights.sum()), 1e-8)
        else:
            scores = catalog["average_rating"].to_numpy()

        catalog["collaborative_score"] = np.asarray(
            scores
        ).reshape(-1)

        # Never recommend a movie already rated by this user.
        seen = set(user_ratings["movieId"])
        recommendations = catalog[
            ~catalog["movieId"].isin(seen)
        ].copy()

        recommendations = recommendations.sort_values(
            ["collaborative_score", "rating_count"],
            ascending=[False, False],
        ).head(top_n)

        output_path = OUT / f"user_{user_id}_collaborative.csv"
        recommendations.to_csv(output_path, index=False)

        print(recommendations[
            [
                "movieId",
                "title",
                "genres",
                "rating_count",
                "collaborative_score",
            ]
        ].to_string(index=False))
        print(f"\nSaved to: {output_path}")

    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    if args.user_id <= 0:
        parser.error("--user-id must be positive")
    if not 1 <= args.top_n <= 100:
        parser.error("--top-n must be between 1 and 100")

    main(args.user_id, args.top_n)
