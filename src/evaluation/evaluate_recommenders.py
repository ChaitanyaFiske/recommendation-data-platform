
from pathlib import Path
import logging
import math

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUTPUT = DATA / "evaluation"
LOGS = ROOT / "logs"

OUTPUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_USERS = 200
TOP_K = 10
RELEVANT_RATING = 4.0


def main():
    con = duckdb.connect()

    try:
        movies_path = (DATA / "movies.parquet").as_posix()
        ratings_path = (DATA / "ratings.parquet").as_posix()

        con.execute(
            f"""
            CREATE VIEW movies AS
            SELECT * FROM read_parquet('{movies_path}')
            """
        )
        con.execute(
            f"""
            CREATE VIEW ratings AS
            SELECT * FROM read_parquet('{ratings_path}')
            """
        )

        # Split globally by timestamp to avoid training on
        # future ratings.
        cutoff = con.execute(
            """
            SELECT quantile_cont(timestamp, 0.8)
            FROM ratings
            """
        ).fetchone()[0]

        logger.info("Temporal split timestamp: %s", cutoff)

        con.execute(
            f"""
            CREATE TEMP VIEW train AS
            SELECT *
            FROM ratings
            WHERE timestamp <= {cutoff}
            """
        )

        con.execute(
            f"""
            CREATE TEMP VIEW test AS
            SELECT *
            FROM ratings
            WHERE timestamp > {cutoff}
            """
        )

        # Choose a deterministic, distributed sample of users
        # who have enough training history and relevant test items.
        con.execute(
            f"""
            CREATE TEMP TABLE eval_users AS
            SELECT tr.userId
            FROM train AS tr
            WHERE EXISTS (
                SELECT 1
                FROM test AS te
                WHERE te.userId = tr.userId
                  AND te.rating >= {RELEVANT_RATING}
                  AND NOT EXISTS (
                      SELECT 1
                      FROM train AS previous
                      WHERE previous.userId = te.userId
                        AND previous.movieId = te.movieId
                  )
            )
            GROUP BY tr.userId
            HAVING COUNT(*) >= 5
            ORDER BY hash(tr.userId)
            LIMIT {SAMPLE_USERS}
            """
        )

        user_count = con.execute(
            "SELECT COUNT(*) FROM eval_users"
        ).fetchone()[0]

        if user_count == 0:
            raise RuntimeError("No eligible evaluation users found.")

        logger.info("Evaluating %s users.", user_count)

        # Only training ratings are used to estimate preferences.
        con.execute(
            """
            CREATE TEMP VIEW genre_preferences AS
            SELECT
                r.userId,
                g.genre,
                AVG(r.rating) AS preference
            FROM train AS r
            JOIN eval_users AS u
                ON r.userId = u.userId
            JOIN movies AS m
                ON r.movieId = m.movieId
            CROSS JOIN UNNEST(
                string_split(m.genres, '|')
            ) AS g(genre)
            WHERE g.genre <> '(no genres listed)'
            GROUP BY r.userId, g.genre
            """
        )

        # Popularity statistics use training data only.
        con.execute(
            """
            CREATE TEMP VIEW global_stats AS
            SELECT AVG(rating) AS global_average
            FROM train
            """
        )

        con.execute(
            """
            CREATE TEMP VIEW movie_scores AS
            WITH stats AS (
                SELECT
                    movieId,
                    COUNT(*) AS rating_count,
                    AVG(rating) AS average_rating
                FROM train
                GROUP BY movieId
            )
            SELECT
                m.movieId,
                COALESCE(s.rating_count, 0) AS rating_count,
                COALESCE(
                    (
                        s.rating_count * s.average_rating
                        + 1000 * g.global_average
                    ) / (s.rating_count + 1000),
                    g.global_average
                ) AS score
            FROM movies AS m
            LEFT JOIN stats AS s USING (movieId)
            CROSS JOIN global_stats AS g
            """
        )

        # Exclude all movies present in a user's training history.
        con.execute(
            """
            CREATE TEMP VIEW candidates AS
            SELECT
                u.userId,
                m.movieId,
                m.title,
                m.genres
            FROM eval_users AS u
            CROSS JOIN movies AS m
            WHERE NOT EXISTS (
                SELECT 1
                FROM train AS r
                WHERE r.userId = u.userId
                  AND r.movieId = m.movieId
            )
            """
        )

        con.execute(
            """
            CREATE TEMP VIEW content_scores AS
            SELECT
                c.userId,
                c.movieId,
                AVG(p.preference) AS preference_score
            FROM candidates AS c
            CROSS JOIN UNNEST(
                string_split(c.genres, '|')
            ) AS g(genre)
            JOIN genre_preferences AS p
                ON p.userId = c.userId
               AND p.genre = g.genre
            WHERE g.genre <> '(no genres listed)'
            GROUP BY c.userId, c.movieId
            """
        )

        con.execute(
            """
            CREATE TEMP VIEW scored AS
            SELECT
                c.userId,
                c.movieId,
                c.title,
                'popularity' AS model,
                s.score AS score,
                s.rating_count
            FROM candidates AS c
            JOIN movie_scores AS s USING (movieId)

            UNION ALL

            SELECT
                c.userId,
                c.movieId,
                c.title,
                'content_based' AS model,
                COALESCE(cs.preference_score, s.score) AS score,
                s.rating_count
            FROM candidates AS c
            JOIN movie_scores AS s USING (movieId)
            LEFT JOIN content_scores AS cs
                ON c.userId = cs.userId
               AND c.movieId = cs.movieId
            """
        )

        recommendations = con.execute(
            f"""
            SELECT userId, movieId, model, rank
            FROM (
                SELECT
                    userId,
                    movieId,
                    model,
                    ROW_NUMBER() OVER (
                        PARTITION BY userId, model
                        ORDER BY
                            score DESC,
                            rating_count DESC,
                            movieId
                    ) AS rank
                FROM scored
            )
            WHERE rank <= {TOP_K}
            ORDER BY userId, model, rank
            """
        ).fetchdf()

        relevant = con.execute(
            f"""
            SELECT te.userId, te.movieId
            FROM test AS te
            JOIN eval_users AS u USING (userId)
            WHERE te.rating >= {RELEVANT_RATING}
              AND NOT EXISTS (
                  SELECT 1
                  FROM train AS tr
                  WHERE tr.userId = te.userId
                    AND tr.movieId = te.movieId
              )
            """
        ).fetchdf()

        relevant_by_user = (
            relevant.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        ideal_dcg = {}
        for user_id, items in relevant_by_user.items():
            ideal_dcg[user_id] = sum(
                1 / math.log2(rank + 1)
                for rank in range(
                    1, min(len(items), TOP_K) + 1
                )
            )

        results = []

        for model, group in recommendations.groupby("model"):
            user_metrics = []

            for user_id, user_group in group.groupby("userId"):
                relevant_items = relevant_by_user.get(
                    user_id, set()
                )
                if not relevant_items:
                    continue

                user_group = user_group.sort_values("rank")
                hits = 0
                dcg = 0.0

                for row in user_group.itertuples():
                    if row.movieId in relevant_items:
                        hits += 1
                        dcg += 1 / math.log2(row.rank + 1)

                user_metrics.append({
                    "precision_at_10": hits / TOP_K,
                    "recall_at_10": hits / len(relevant_items),
                    "ndcg_at_10": (
                        dcg / ideal_dcg[user_id]
                        if ideal_dcg[user_id] else 0.0
                    ),
                })

            metrics = pd.DataFrame(user_metrics)

            results.append({
                "model": model,
                "evaluated_users": len(metrics),
                "precision_at_10": metrics[
                    "precision_at_10"
                ].mean(),
                "recall_at_10": metrics[
                    "recall_at_10"
                ].mean(),
                "ndcg_at_10": metrics["ndcg_at_10"].mean(),
                "relevant_rating_threshold": RELEVANT_RATING,
                "top_k": TOP_K,
            })

        report = pd.DataFrame(results)
        report_path = OUTPUT / "model_comparison.csv"
        report.to_csv(report_path, index=False)

        print("\nMODEL EVALUATION RESULTS")
        print(report.to_string(index=False))
        print(f"\nSaved to: {report_path}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
