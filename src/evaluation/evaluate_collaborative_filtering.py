
from pathlib import Path
import logging
import math

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUTPUT = DATA / "evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CATALOG_SIZE = 1500
SAMPLE_USERS = 200
TOP_K = 10
RELEVANT_RATING = 4.0
MIN_TRAIN_RATINGS = 5


def main():
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

        cutoff = con.execute("""
            SELECT quantile_cont(timestamp, 0.8)
            FROM ratings
        """).fetchone()[0]

        logger.info("Temporal cutoff: %s", cutoff)

        con.execute(f"""
            CREATE TEMP VIEW train AS
            SELECT *
            FROM ratings
            WHERE timestamp <= {cutoff}
        """)

        con.execute(f"""
            CREATE TEMP VIEW test AS
            SELECT *
            FROM ratings
            WHERE timestamp > {cutoff}
        """)

        # Select the most-rated movies using training data only.
        catalog = con.execute(f"""
            SELECT
                m.movieId,
                m.title,
                COUNT(r.movieId) AS rating_count,
                AVG(r.rating) AS average_rating
            FROM movies AS m
            JOIN train AS r USING (movieId)
            GROUP BY m.movieId, m.title
            ORDER BY rating_count DESC, m.movieId
            LIMIT {CATALOG_SIZE}
        """).fetchdf()

        catalog = catalog.reset_index(drop=True)
        catalog["item_idx"] = np.arange(len(catalog))

        logger.info("Candidate catalog: %s movies", len(catalog))

        con.register("catalog_df", catalog[["movieId", "item_idx"]])
        con.execute("""
            CREATE TEMP VIEW catalog AS
            SELECT * FROM catalog_df
        """)

        # Users must have training history and relevant test
        # ratings among the candidate movies.
        users = con.execute(f"""
            SELECT tr.userId
            FROM train AS tr
            JOIN catalog AS c USING (movieId)
            GROUP BY tr.userId
            HAVING COUNT(*) >= {MIN_TRAIN_RATINGS}
               AND EXISTS (
                   SELECT 1
                   FROM test AS te
                   JOIN catalog AS cc USING (movieId)
                   WHERE te.userId = tr.userId
                     AND te.rating >= {RELEVANT_RATING}
                     AND NOT EXISTS (
                         SELECT 1
                         FROM train AS old
                         WHERE old.userId = te.userId
                           AND old.movieId = te.movieId
                     )
               )
            ORDER BY hash(tr.userId)
            LIMIT {SAMPLE_USERS}
        """).fetchdf()

        if users.empty:
            raise RuntimeError("No eligible users found.")

        logger.info("Evaluation users: %s", len(users))

        con.register("eval_users_df", users)
        con.execute("""
            CREATE TEMP VIEW eval_users AS
            SELECT * FROM eval_users_df
        """)

        # Fetch only ratings for the selected movie catalog.
        train_df = con.execute("""
            SELECT
                r.userId,
                c.item_idx,
                r.rating
            FROM train AS r
            JOIN catalog AS c USING (movieId)
        """).fetchdf()

        logger.info(
            "Training interactions loaded: %s",
            f"{len(train_df):,}",
        )

        # Item-by-user sparse matrix.
        # Values are explicit ratings, not binary interactions.
        item_user = csr_matrix(
            (
                train_df["rating"].to_numpy(dtype=np.float32),
                (
                    train_df["item_idx"].to_numpy(dtype=np.int32),
                    train_df["userId"].to_numpy(dtype=np.int32),
                ),
            ),
            shape=(
                len(catalog),
                int(train_df["userId"].max()) + 1,
            ),
            dtype=np.float32,
        )

        logger.info("Computing item-item cosine similarities.")

        similarities = cosine_similarity(
            item_user,
            dense_output=True,
        ).astype(np.float32)

        # Popularity baseline, calculated from training data.
        global_average = float(train_df["rating"].mean())

        counts = train_df.groupby("item_idx").size()
        averages = train_df.groupby("item_idx")["rating"].mean()

        catalog["train_count"] = (
            catalog["item_idx"].map(counts).fillna(0)
        )
        catalog["train_average"] = (
            catalog["item_idx"].map(averages).fillna(global_average)
        )

        m = 1000.0
        catalog["popularity_score"] = (
            catalog["train_count"] * catalog["train_average"]
            + m * global_average
        ) / (catalog["train_count"] + m)

        # Retain only relevant, previously unseen test items
        # within the shared candidate catalog.
        relevant = con.execute(f"""
            SELECT te.userId, te.movieId
            FROM test AS te
            JOIN eval_users AS u USING (userId)
            JOIN catalog AS c USING (movieId)
            WHERE te.rating >= {RELEVANT_RATING}
              AND NOT EXISTS (
                  SELECT 1
                  FROM train AS tr
                  WHERE tr.userId = te.userId
                    AND tr.movieId = te.movieId
              )
        """).fetchdf()

        relevant_by_user = (
            relevant.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        # Collect training histories for evaluated users.
        histories = {
            int(user_id): group
            for user_id, group in train_df[
                train_df["userId"].isin(users["userId"])
            ].groupby("userId")
        }

        item_ids = catalog["movieId"].to_numpy()
        item_indices = catalog["item_idx"].to_numpy()
        popularity_scores = catalog["popularity_score"].to_numpy()

        rows = []

        for user_id in users["userId"].astype(int):
            history = histories.get(user_id)

            if history is None or history.empty:
                continue

            rated_indices = history["item_idx"].to_numpy(dtype=int)
            rated_values = history["rating"].to_numpy()

            # Recommend from the same catalog, excluding
            # every movie the user rated in the training set.
            unseen = np.ones(len(catalog), dtype=bool)
            unseen[rated_indices] = False

            # Similarity-weighted preference:
            # higher-rated movies contribute more positive evidence.
            liked = rated_values >= RELEVANT_RATING
            liked_indices = rated_indices[liked]
            liked_values = rated_values[liked]

            if len(liked_indices):
                weights = liked_values - global_average
                weights = np.maximum(weights, 0.0)

                cf_scores = (
                    similarities[:, liked_indices] @ weights
                ) / max(float(weights.sum()), 1e-8)
            else:
                # Cold-start fallback for users without high ratings.
                cf_scores = popularity_scores.copy()

            cf_scores = np.asarray(cf_scores).reshape(-1)

            for model, scores in [
                ("popularity", popularity_scores),
                ("collaborative_filtering", cf_scores),
            ]:
                candidate_indices = np.flatnonzero(unseen)

                order = candidate_indices[
                    np.lexsort((
                        item_ids[candidate_indices],
                        -scores[candidate_indices],
                    ))
                ][:TOP_K]

                for rank, idx in enumerate(order, start=1):
                    rows.append({
                        "userId": user_id,
                        "movieId": int(item_ids[idx]),
                        "model": model,
                        "rank": rank,
                    })

        ranked = pd.DataFrame(rows)
        results = []

        for model, group in ranked.groupby("model"):
            metrics = []

            for user_id, user_group in group.groupby("userId"):
                relevant_items = relevant_by_user.get(
                    int(user_id), set()
                )

                if not relevant_items:
                    continue

                hits = 0
                dcg = 0.0

                for row in user_group.sort_values("rank").itertuples():
                    if row.movieId in relevant_items:
                        hits += 1
                        dcg += 1 / math.log2(row.rank + 1)

                ideal_hits = min(len(relevant_items), TOP_K)
                idcg = sum(
                    1 / math.log2(rank + 1)
                    for rank in range(1, ideal_hits + 1)
                )

                metrics.append({
                    "precision_at_10": hits / TOP_K,
                    "recall_at_10": hits / len(relevant_items),
                    "ndcg_at_10": dcg / idcg if idcg else 0.0,
                })

            metric_df = pd.DataFrame(metrics)

            results.append({
                "model": model,
                "evaluated_users": len(metric_df),
                "precision_at_10": metric_df[
                    "precision_at_10"
                ].mean(),
                "recall_at_10": metric_df[
                    "recall_at_10"
                ].mean(),
                "ndcg_at_10": metric_df["ndcg_at_10"].mean(),
            })

        report = pd.DataFrame(results)
        report_path = OUTPUT / "collaborative_filtering_comparison.csv"
        report.to_csv(report_path, index=False)

        print("\nCOLLABORATIVE FILTERING EVALUATION")
        print(report.to_string(index=False))
        print(f"\nSaved to: {report_path}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
