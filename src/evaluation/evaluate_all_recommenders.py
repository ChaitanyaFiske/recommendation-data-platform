
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RATINGS_FILE = DATA_DIR / "ratings.parquet"
MOVIES_FILE = DATA_DIR / "movies.parquet"
OUTPUT_DIR = DATA_DIR / "evaluation"

CATALOG_SIZE = 1500
MIN_RATINGS = 1000
MIN_TRAIN_RATINGS = 5
N_EVALUATION_USERS = 200
TOP_K = 10
RELEVANT_RATING = 4.0

COLLAB_WEIGHT = 0.50
CONTENT_WEIGHT = 0.30
POPULARITY_WEIGHT = 0.20


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def percentile_scale(scores):
    return pd.Series(scores).rank(pct=True, method="average").to_numpy()


def get_genre_preferences(history):
    preferences = {}
    counts = {}

    for row in history.itertuples(index=False):
        genres = [
            g for g in str(row.genres).split("|")
            if g and g != "(no genres listed)"
        ]

        for genre in genres:
            preferences.setdefault(genre, []).append(float(row.rating))

    for genre, ratings in preferences.items():
        counts[genre] = float(np.mean(ratings))

    return counts


def score_content(catalog, genre_preferences, user_mean):
    scores = []

    for genres in catalog["genres"].fillna(""):
        movie_genres = [
            g for g in genres.split("|")
            if g and g != "(no genres listed)"
        ]
        matching = [
            genre_preferences[g]
            for g in movie_genres
            if g in genre_preferences
        ]
        scores.append(
            float(np.mean(matching)) if matching else user_mean
        )

    return np.asarray(scores, dtype=np.float32)


def ranking_metrics(recommended_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    hits = [int(movie_id in relevant_ids) for movie_id in recommended_ids]

    precision = sum(hits) / TOP_K
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )
    ideal_hits = min(len(relevant_ids), TOP_K)
    idcg = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )
    ndcg = dcg / idcg if idcg else 0.0

    return precision, recall, ndcg


def main():
    setup_logging()

    if not RATINGS_FILE.exists() or not MOVIES_FILE.exists():
        raise FileNotFoundError(
            "Processed Parquet files are missing. Run the processing "
            "scripts first."
        )

    con = duckdb.connect()

    try:
        ratings_path = str(RATINGS_FILE).replace("'", "''")
        movies_path = str(MOVIES_FILE).replace("'", "''")

        con.execute(
            f"CREATE TEMP VIEW all_ratings AS "
            f"SELECT * FROM read_parquet('{ratings_path}')"
        )
        con.execute(
            f"CREATE TEMP VIEW all_movies AS "
            f"SELECT * FROM read_parquet('{movies_path}')"
        )

        cutoff = con.execute(
            """
            SELECT quantile_cont(timestamp, 0.8)
            FROM all_ratings
            """
        ).fetchone()[0]

        con.execute(
            f"""
            CREATE TEMP VIEW train_ratings AS
            SELECT * FROM all_ratings
            WHERE timestamp <= {cutoff}
            """
        )
        con.execute(
            f"""
            CREATE TEMP VIEW test_ratings AS
            SELECT * FROM all_ratings
            WHERE timestamp > {cutoff}
            """
        )

        global_mean = con.execute(
            "SELECT AVG(rating) FROM train_ratings"
        ).fetchone()[0]

        logging.info("Temporal cutoff: %s", cutoff)
        logging.info("Training mean rating: %.4f", global_mean)

        # Construct the catalog using training interactions only.
        catalog = con.execute(
            """
            SELECT
                m.movieId,
                m.title,
                m.genres,
                COUNT(*) AS rating_count,
                AVG(r.rating) AS average_rating
            FROM all_movies m
            JOIN train_ratings r USING (movieId)
            GROUP BY m.movieId, m.title, m.genres
            HAVING COUNT(*) >= ?
            ORDER BY rating_count DESC, m.movieId
            LIMIT ?
            """,
            [MIN_RATINGS, CATALOG_SIZE],
        ).fetchdf()

        if catalog.empty:
            raise ValueError("The training candidate catalog is empty.")

        catalog["item_idx"] = np.arange(len(catalog), dtype=np.int32)
        con.register("catalog_mapping", catalog[["movieId", "item_idx"]])

        # Bayesian popularity baseline from training data only.
        counts = catalog["rating_count"].to_numpy(dtype=float)
        averages = catalog["average_rating"].to_numpy(dtype=float)
        catalog["popularity_score"] = (
            counts / (counts + MIN_RATINGS) * averages
            + MIN_RATINGS / (counts + MIN_RATINGS) * global_mean
        )

        # Build item-user matrix from training interactions only.
        interactions = con.execute(
            """
            SELECT c.item_idx, r.userId, r.rating
            FROM train_ratings r
            JOIN catalog_mapping c USING (movieId)
            """
        ).fetchdf()

        max_user_id = int(interactions["userId"].max())
        item_user = csr_matrix(
            (
                interactions["rating"].to_numpy(dtype=np.float32),
                (
                    interactions["item_idx"].to_numpy(dtype=np.int32),
                    interactions["userId"].to_numpy(dtype=np.int32),
                ),
            ),
            shape=(len(catalog), max_user_id + 1),
            dtype=np.float32,
        )

        normalized = normalize(item_user, norm="l2", axis=1)
        similarity = normalized @ normalized.T
        del item_user, normalized, interactions

        # Select users who have enough training history and at least
        # one relevant, unseen test movie inside the shared catalog.
        eligible = con.execute(
            """
            WITH train_counts AS (
                SELECT userId, COUNT(*) AS train_count
                FROM train_ratings
                GROUP BY userId
                HAVING COUNT(*) >= ?
            ),
            relevant_users AS (
                SELECT DISTINCT t.userId
                FROM test_ratings t
                JOIN catalog_mapping c USING (movieId)
                WHERE t.rating >= ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM train_ratings tr
                      WHERE tr.userId = t.userId
                        AND tr.movieId = t.movieId
                  )
            )
            SELECT tc.userId
            FROM train_counts tc
            JOIN relevant_users ru USING (userId)
            ORDER BY tc.userId
            """,
            [MIN_TRAIN_RATINGS, RELEVANT_RATING],
        ).fetchdf()

        if eligible.empty:
            raise ValueError("No eligible evaluation users were found.")

        users = (
            eligible.sample(
                n=min(N_EVALUATION_USERS, len(eligible)),
                random_state=42,
            )["userId"]
            .astype(int)
            .tolist()
        )
        user_sql = ",".join(map(str, users))

        logging.info(
            "Evaluating %s users against %s candidate movies.",
            len(users),
            len(catalog),
        )

        # All user histories used for scoring come from training data.
        histories = con.execute(
            f"""
            SELECT r.userId, r.movieId, r.rating, m.genres
            FROM train_ratings r
            JOIN all_movies m USING (movieId)
            WHERE r.userId IN ({user_sql})
            """
        ).fetchdf()

        # Test labels are only used for scoring the recommendations.
        test_data = con.execute(
            f"""
            SELECT userId, movieId, rating
            FROM test_ratings
            WHERE userId IN ({user_sql})
              AND rating >= {RELEVANT_RATING}
              AND movieId IN (SELECT movieId FROM catalog_mapping)
            """
        ).fetchdf()

        relevant_by_user = (
            test_data.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        # Training history excludes previously seen candidates.
        seen_by_user = (
            histories.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        history_by_user = {
            int(user_id): group
            for user_id, group in histories.groupby("userId")
        }

        metric_rows = []
        per_user_rows = []

        for user_id in users:
            history = history_by_user[user_id]
            seen = seen_by_user[user_id]
            relevant = relevant_by_user.get(user_id, set())

            candidates = catalog[
                ~catalog["movieId"].isin(seen)
            ].copy()

            if candidates.empty or not relevant:
                continue

            # Align the model scores with the exact catalog row indices.
            candidate_indices = candidates["item_idx"].to_numpy(
                dtype=np.int32
            )

            positive = history[
                (history["rating"] >= RELEVANT_RATING)
                & history["movieId"].isin(catalog["movieId"])
            ].merge(
                catalog[["movieId", "item_idx"]],
                on="movieId",
                how="inner",
            )

            if positive.empty:
                collab_all = np.zeros(len(catalog), dtype=np.float32)
            else:
                item_indices = positive["item_idx"].to_numpy(
                    dtype=np.int32
                )
                weights = (
                    positive["rating"].to_numpy(dtype=np.float32) - 3.5
                )
                similarities = similarity[:, item_indices]

                numerator = np.asarray(similarities @ weights).ravel()
                denominator = np.asarray(
                    similarities @ np.ones(
                        len(item_indices), dtype=np.float32
                    )
                ).ravel()

                collab_all = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator, dtype=np.float32),
                    where=denominator > 0,
                )

            genre_preferences = get_genre_preferences(history)
            user_mean = float(history["rating"].mean())

            content_all = score_content(
                catalog, genre_preferences, user_mean
            )

            # Score only unseen candidates when normalizing ranks.
            collab = percentile_scale(collab_all[candidate_indices])
            content = percentile_scale(content_all[candidate_indices])
            popularity = percentile_scale(
                candidates["popularity_score"].to_numpy()
            )

            candidates["collaborative_score"] = collab
            candidates["content_score"] = content
            candidates["popularity_rank"] = popularity
            candidates["hybrid_score"] = (
                COLLAB_WEIGHT * collab
                + CONTENT_WEIGHT * content
                + POPULARITY_WEIGHT * popularity
            )

            # Deterministic tie-breaking makes results reproducible.
            ranking_scores = {
                "popularity": "popularity_score",
                "content_based": "content_score",
                "collaborative_filtering": "collaborative_score",
                "hybrid": "hybrid_score",
            }

            for model_name, score_col in ranking_scores.items():
                ranked = candidates.sort_values(
                    [score_col, "popularity_score", "movieId"],
                    ascending=[False, False, True],
                )
                recommended_ids = ranked.head(TOP_K)["movieId"].tolist()

                precision, recall, ndcg = ranking_metrics(
                    recommended_ids, relevant
                )

                per_user_rows.append(
                    {
                        "userId": user_id,
                        "model": model_name,
                        "precision_at_10": precision,
                        "recall_at_10": recall,
                        "ndcg_at_10": ndcg,
                    }
                )

        if not per_user_rows:
            raise ValueError("No users could be evaluated.")

        per_user = pd.DataFrame(per_user_rows)
        summary = (
            per_user.groupby("model")[
                ["precision_at_10", "recall_at_10", "ndcg_at_10"]
            ]
            .mean()
            .reset_index()
            .sort_values("ndcg_at_10", ascending=False)
        )

        summary.to_csv(
            OUTPUT_DIR / "recommender_comparison.csv",
            index=False,
        )
        per_user.to_csv(
            OUTPUT_DIR / "recommender_comparison_by_user.csv",
            index=False,
        )

        print("\nEvaluation summary:")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"\nUsers evaluated: {per_user['userId'].nunique()}")
        print(f"Candidate catalog size: {len(catalog)}")
        print(f"Temporal cutoff: {cutoff}")
        print(f"Results saved to: {OUTPUT_DIR}")

    finally:
        con.close()


if __name__ == "__main__":
    setup_logging()
    main()
