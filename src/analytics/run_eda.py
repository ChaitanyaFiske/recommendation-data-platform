
from pathlib import Path
import logging

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "analytics"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / "eda.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


def parquet_source(filename: str) -> str:
    path = (DATA_DIR / filename).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")

    escaped_path = path.as_posix().replace("'", "''")
    return f"read_parquet('{escaped_path}')"


def export_query(connection, name: str, query: str):
    """Execute an analytical query and export its result to CSV."""

    output_path = (OUTPUT_DIR / f"{name}.csv").resolve()
    escaped_output = output_path.as_posix().replace("'", "''")

    logger.info("Running analysis: %s", name)

    connection.execute(
        f"""
        COPY ({query})
        TO '{escaped_output}'
        (HEADER, DELIMITER ',')
        """
    )

    row_count = connection.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{escaped_output}')"
    ).fetchone()[0]

    logger.info(
        "Saved %s with %s rows",
        output_path.name,
        row_count,
    )


def main():
    connection = duckdb.connect()

    try:
        movies = parquet_source("movies.parquet")
        ratings = parquet_source("ratings.parquet")

        # Register reusable views for SQL queries.
        connection.execute(
            f"CREATE VIEW movies AS SELECT * FROM {movies}"
        )
        connection.execute(
            f"CREATE VIEW ratings AS SELECT * FROM {ratings}"
        )

        # 1. Rating distribution
        export_query(
            connection,
            "rating_distribution",
            """
            SELECT
                rating,
                COUNT(*) AS rating_count,
                ROUND(
                    100.0 * COUNT(*) /
                    SUM(COUNT(*)) OVER (),
                    2
                ) AS percentage
            FROM ratings
            GROUP BY rating
            ORDER BY rating
            """,
        )

        # 2. Most popular movies, adjusted for small samples.
        # Require at least 1,000 ratings.
        # Weighted rating = v/(v+m)*R + m/(v+m)*C
        # v = movie rating count
        # R = movie average rating
        # m = minimum rating count
        # C = overall average rating
        export_query(
            connection,
            "top_movies",
            """
            WITH movie_stats AS (
                SELECT
                    m.movieId,
                    m.title,
                    m.genres,
                    COUNT(*) AS rating_count,
                    AVG(r.rating) AS average_rating
                FROM movies AS m
                JOIN ratings AS r
                    ON m.movieId = r.movieId
                GROUP BY
                    m.movieId,
                    m.title,
                    m.genres
            ),
            global_stats AS (
                SELECT
                    AVG(rating) AS global_average
                FROM ratings
            )
            SELECT
                s.movieId,
                s.title,
                s.genres,
                s.rating_count,
                ROUND(s.average_rating, 4)
                    AS average_rating,
                ROUND(
                    (
                        s.rating_count * s.average_rating
                        + 1000 * g.global_average
                    ) / (s.rating_count + 1000),
                    4
                ) AS weighted_rating
            FROM movie_stats AS s
            CROSS JOIN global_stats AS g
            WHERE s.rating_count >= 1000
            ORDER BY weighted_rating DESC, rating_count DESC
            LIMIT 100
            """,
        )

        # 3. Genre performance
        export_query(
            connection,
            "genre_performance",
            """
            WITH movie_genres AS (
                SELECT
                    m.movieId,
                    unnest(string_split(m.genres, '|')) AS genre
                FROM movies AS m
            )
            SELECT
                mg.genre,
                COUNT(*) AS rating_count,
                COUNT(DISTINCT r.userId) AS unique_users,
                COUNT(DISTINCT r.movieId) AS movies_rated,
                ROUND(AVG(r.rating), 4) AS average_rating
            FROM movie_genres AS mg
            JOIN ratings AS r
                ON mg.movieId = r.movieId
            WHERE mg.genre <> '(no genres listed)'
            GROUP BY mg.genre
            HAVING COUNT(*) >= 1000
            ORDER BY average_rating DESC, rating_count DESC
            """,
        )

        # 4. User activity distribution
        export_query(
            connection,
            "user_activity",
            """
            WITH user_stats AS (
                SELECT
                    userId,
                    COUNT(*) AS ratings_given,
                    AVG(rating) AS average_rating_given
                FROM ratings
                GROUP BY userId
            )
            SELECT
                CASE
                    WHEN ratings_given = 1 THEN '1 rating'
                    WHEN ratings_given BETWEEN 2 AND 10
                        THEN '2-10 ratings'
                    WHEN ratings_given BETWEEN 11 AND 50
                        THEN '11-50 ratings'
                    WHEN ratings_given BETWEEN 51 AND 100
                        THEN '51-100 ratings'
                    WHEN ratings_given BETWEEN 101 AND 500
                        THEN '101-500 ratings'
                    ELSE '501+ ratings'
                END AS activity_group,
                COUNT(*) AS user_count,
                ROUND(AVG(ratings_given), 2)
                    AS average_ratings_per_user,
                ROUND(AVG(average_rating_given), 4)
                    AS average_personal_rating
            FROM user_stats
            GROUP BY activity_group
            ORDER BY MIN(ratings_given)
            """,
        )

        # 5. Overall dataset summary
        export_query(
            connection,
            "dataset_summary",
            """
            SELECT
                (SELECT COUNT(*) FROM movies)
                    AS movie_count,
                (SELECT COUNT(*) FROM ratings)
                    AS rating_count,
                (SELECT COUNT(DISTINCT userId) FROM ratings)
                    AS user_count,
                (SELECT COUNT(DISTINCT movieId) FROM ratings)
                    AS rated_movie_count,
                (SELECT ROUND(AVG(rating), 4) FROM ratings)
                    AS global_average_rating,
                (SELECT ROUND(MIN(rating), 2) FROM ratings)
                    AS minimum_rating,
                (SELECT ROUND(MAX(rating), 2) FROM ratings)
                    AS maximum_rating
            """,
        )

        logger.info("All EDA analyses completed successfully.")
        logger.info("Output directory: %s", OUTPUT_DIR)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
