
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
N_USERS = 200
TOP_K = 10
RELEVANT_RATING = 4.0
RANDOM_SEED = 42

# Candidate weight combinations: collaborative, content, popularity.
WEIGHT_CONFIGS = [
    (1.0, 0.0, 0.0),
    (0.9, 0.1, 0.0),
    (0.9, 0.0, 0.1),
    (0.8, 0.1, 0.1),
    (0.8, 0.2, 0.0),
    (0.8, 0.0, 0.2),
    (0.7, 0.2, 0.1),
    (0.7, 0.1, 0.2),
    (0.7, 0.3, 0.0),
    (0.6, 0.2, 0.2),
    (0.5, 0.3, 0.2),
    (0.5, 0.2, 0.3),
]


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

    for row in history.itertuples(index=False):
        genres = [
            g for g in str(row.genres).split("|")
            if g and g != "(no genres listed)"
        ]
        for genre in genres:
            preferences.setdefault(genre, []).append(float(row.rating))

    return {
        genre: float(np.mean(ratings))
        for genre, ratings in preferences.items()
    }


def score_content(catalog, preferences, fallback):
    scores = []

    for genres in catalog["genres"].fillna(""):
        movie_genres = [
            g for g in genres.split("|")
            if g and g != "(no genres listed)"
        ]
        matching = [
            preferences[g] for g in movie_genres if g in preferences
        ]
        scores.append(
            float(np.mean(matching)) if matching else fallback
        )

    return np.asarray(scores, dtype=np.float32)


def get_metrics(recommended_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    hits = [int(movie_id in relevant_ids) for movie_id in recommended_ids]

    precision = sum(hits) / TOP_K
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )
    ideal_count = min(len(relevant_ids), TOP_K)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_count)
    )
    ndcg = dcg / idcg if idcg else 0.0

    return precision, recall, ndcg


def main():
    setup_logging()

    if not RATINGS_FILE.exists() or not MOVIES_FILE.exists():
        raise FileNotFoundError(
            "Processed Parquet files are missing. Run the processing scripts first."
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

        # Chronological 70% / 15% / 15% split.
        q70, q85 = con.execute(
            """
            SELECT
                quantile_cont(timestamp, 0.70),
                quantile_cont(timestamp, 0.85)
            FROM all_ratings
            """
        ).fetchone()

        con.execute(
            f"CREATE TEMP VIEW train_ratings AS "
            f"SELECT * FROM all_ratings WHERE timestamp <= {q70}"
        )
        con.execute(
            f"""
            CREATE TEMP VIEW validation_ratings AS
            SELECT * FROM all_ratings
            WHERE timestamp > {q70} AND timestamp <= {q85}
            """
        )
        con.execute(
            f"CREATE TEMP VIEW test_ratings AS "
            f"SELECT * FROM all_ratings WHERE timestamp > {q85}"
        )

        global_mean = con.execute(
            "SELECT AVG(rating) FROM train_ratings"
        ).fetchone()[0]

        logging.info(
            "Temporal cutoffs: training <= %.1f; validation <= %.1f",
            q70, q85,
        )

        # Catalog is built only from training interactions.
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
            raise ValueError("No candidate movies found in the training data.")

        catalog["item_idx"] = np.arange(len(catalog), dtype=np.int32)
        con.register("catalog_mapping", catalog[["movieId", "item_idx"]])

        # Popularity score from training data.
        counts = catalog["rating_count"].to_numpy(dtype=float)
        averages = catalog["average_rating"].to_numpy(dtype=float)
        catalog["popularity_score"] = (
            counts / (counts + MIN_RATINGS) * averages
            + MIN_RATINGS / (counts + MIN_RATINGS) * global_mean
        )

        # Collaborative model: training data only.
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

        # Choose validation users using validation labels only.
        def eligible_users(split_name):
            # split_name is a fixed internal constant, not user input.
            query = f"""
                WITH train_counts AS (
                    SELECT userId, COUNT(*) AS n
                    FROM train_ratings
                    GROUP BY userId
                    HAVING COUNT(*) >= {MIN_TRAIN_RATINGS}
                ),
                relevant AS (
                    SELECT DISTINCT s.userId
                    FROM {split_name} s
                    JOIN catalog_mapping c USING (movieId)
                    WHERE s.rating >= {RELEVANT_RATING}
                      AND NOT EXISTS (
                          SELECT 1
                          FROM train_ratings t
                          WHERE t.userId = s.userId
                            AND t.movieId = s.movieId
                      )
                )
                SELECT tc.userId
                FROM train_counts tc
                JOIN relevant r USING (userId)
                ORDER BY tc.userId
            """
            return con.execute(query).fetchdf()["userId"].astype(int).tolist()

        validation_eligible = eligible_users("validation_ratings")
        test_eligible = eligible_users("test_ratings")

        if not validation_eligible or not test_eligible:
            raise ValueError(
                "No eligible validation or test users found. "
                "Check the split and minimum rating thresholds."
            )

        rng = np.random.default_rng(RANDOM_SEED)

        validation_users = sorted(
            rng.choice(
                validation_eligible,
                size=min(N_USERS, len(validation_eligible)),
                replace=False,
            ).astype(int).tolist()
        )
        test_users = sorted(
            rng.choice(
                test_eligible,
                size=min(N_USERS, len(test_eligible)),
                replace=False,
            ).astype(int).tolist()
        )

        logging.info(
            "Validation users: %d; test users: %d; catalog: %d",
            len(validation_users), len(test_users), len(catalog),
        )

        # Load histories for the selected users from training data only.
        all_users = sorted(set(validation_users + test_users))
        user_sql = ",".join(map(str, all_users))

        histories = con.execute(
            f"""
            SELECT r.userId, r.movieId, r.rating, m.genres
            FROM train_ratings r
            JOIN all_movies m USING (movieId)
            WHERE r.userId IN ({user_sql})
            """
        ).fetchdf()

        history_by_user = {
            int(uid): group.copy()
            for uid, group in histories.groupby("userId")
        }
        seen_by_user = {
            uid: set(group["movieId"].tolist())
            for uid, group in history_by_user.items()
        }

        def relevant_items(split_name, users):
            ids = ",".join(map(str, users))
            frame = con.execute(
                f"""
                SELECT s.userId, s.movieId
                FROM {split_name} s
                JOIN catalog_mapping c USING (movieId)
                WHERE s.userId IN ({ids})
                  AND s.rating >= {RELEVANT_RATING}
                  AND NOT EXISTS (
                      SELECT 1
                      FROM train_ratings t
                      WHERE t.userId = s.userId
                        AND t.movieId = s.movieId
                  )
                """
            ).fetchdf()

            return {
                int(uid): set(group["movieId"].tolist())
                for uid, group in frame.groupby("userId")
            }

        validation_relevant = relevant_items(
            "validation_ratings", validation_users
        )
        test_relevant = relevant_items("test_ratings", test_users)

        # Precompute each user's component scores against the shared catalog.
        def score_user(user_id):
            history = history_by_user[user_id]
            seen = seen_by_user[user_id]

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
                indices = positive["item_idx"].to_numpy(dtype=np.int32)
                weights = (
                    positive["rating"].to_numpy(dtype=np.float32) - 3.5
                )
                sims = similarity[:, indices]
                numerator = np.asarray(sims @ weights).ravel()
                denominator = np.asarray(
                    sims @ np.ones(len(indices), dtype=np.float32)
                ).ravel()
                collab_all = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator, dtype=np.float32),
                    where=denominator > 0,
                )

            preferences = get_genre_preferences(history)
            content_all = score_content(
                catalog, preferences, float(history["rating"].mean())
            )

            candidates = catalog[
                ~catalog["movieId"].isin(seen)
            ].copy()

            if candidates.empty:
                return candidates

            idx = candidates["item_idx"].to_numpy(dtype=np.int32)
            candidates["collab_rank"] = percentile_scale(collab_all[idx])
            candidates["content_rank"] = percentile_scale(content_all[idx])
            candidates["popularity_rank"] = percentile_scale(
                candidates["popularity_score"].to_numpy()
            )
            return candidates

        def evaluate_users(users, relevant_map, configs):
            rows = []

            for user_id in users:
                relevant = relevant_map.get(user_id, set())
                if not relevant:
                    continue

                candidates = score_user(user_id)
                if candidates.empty:
                    continue

                # Pure collaborative baseline and each candidate hybrid.
                for collab_w, content_w, popularity_w in configs:
                    scores = (
                        collab_w * candidates["collab_rank"].to_numpy()
                        + content_w * candidates["content_rank"].to_numpy()
                        + popularity_w * candidates["popularity_rank"].to_numpy()
                    )

                    ranked = candidates.assign(combined_score=scores).sort_values(
                        ["combined_score", "popularity_score", "movieId"],
                        ascending=[False, False, True],
                    )
                    recommended = ranked.head(TOP_K)["movieId"].tolist()
                    precision, recall, ndcg = get_metrics(recommended, relevant)

                    rows.append(
                        {
                            "userId": user_id,
                            "collaborative_weight": collab_w,
                            "content_weight": content_w,
                            "popularity_weight": popularity_w,
                            "precision_at_10": precision,
                            "recall_at_10": recall,
                            "ndcg_at_10": ndcg,
                        }
                    )

            return pd.DataFrame(rows)

        logging.info("Tuning weights on validation users...")
        validation_results = evaluate_users(
            validation_users, validation_relevant, WEIGHT_CONFIGS
        )

        if validation_results.empty:
            raise ValueError("No validation metrics were produced.")

        validation_summary = (
            validation_results.groupby(
                [
                    "collaborative_weight",
                    "content_weight",
                    "popularity_weight",
                ]
            )[
                ["precision_at_10", "recall_at_10", "ndcg_at_10"]
            ]
            .mean()
            .reset_index()
            .sort_values(
                ["ndcg_at_10", "recall_at_10", "precision_at_10"],
                ascending=False,
            )
        )

        best = validation_summary.iloc[0]
        best_weights = (
            float(best["collaborative_weight"]),
            float(best["content_weight"]),
            float(best["popularity_weight"]),
        )

        logging.info("Best validation weights: %s", best_weights)
        logging.info(
            "Best validation NDCG@10: %.6f",
            best["ndcg_at_10"],
        )

        # Final test: compare the selected hybrid with collaborative-only.
        final_configs = [(1.0, 0.0, 0.0), best_weights]
        test_results = evaluate_users(
            test_users, test_relevant, final_configs
        )

        test_results["model"] = test_results.apply(
            lambda row: (
                "collaborative_filtering"
                if (
                    row["collaborative_weight"] == 1.0
                    and row["content_weight"] == 0.0
                    and row["popularity_weight"] == 0.0
                )
                else "tuned_hybrid"
            ),
            axis=1,
        )

        test_summary = (
            test_results.groupby("model")[
                ["precision_at_10", "recall_at_10", "ndcg_at_10"]
            ]
            .mean()
            .reset_index()
            .sort_values("ndcg_at_10", ascending=False)
        )

        validation_summary.to_csv(
            OUTPUT_DIR / "hybrid_weight_tuning.csv", index=False
        )
        test_summary.to_csv(
            OUTPUT_DIR / "tuned_hybrid_test_results.csv", index=False
        )
        test_results.to_csv(
            OUTPUT_DIR / "tuned_hybrid_test_by_user.csv", index=False
        )

        print("\nValidation weight search (top 10):")
        print(
            validation_summary.head(10).to_string(
                index=False, float_format=lambda x: f"{x:.6f}"
            )
        )
        print("\nSelected weights:")
        print(
            f"Collaborative={best_weights[0]:.2f}, "
            f"Content={best_weights[1]:.2f}, "
            f"Popularity={best_weights[2]:.2f}"
        )
        print("\nFinal test comparison:")
        print(
            test_summary.to_string(
                index=False, float_format=lambda x: f"{x:.6f}"
            )
        )
        print(
            f"\nValidation users evaluated: "
            f"{validation_results['userId'].nunique()}"
        )
        print(f"Final test users evaluated: {test_results['userId'].nunique()}")
        print(f"Candidate catalog size: {len(catalog)}")
        print(f"Results saved to: {OUTPUT_DIR}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
