
from pathlib import Path
import argparse

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "recommendations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main(user_id: int, top_n: int):
    movies_path = (
        DATA_DIR / "movies.parquet"
    ).resolve().as_posix().replace("'", "''")

    ratings_path = (
        DATA_DIR / "ratings.parquet"
    ).resolve().as_posix().replace("'", "''")

    output_path = (
        OUTPUT_DIR / f"user_{user_id}_recommendations.csv"
    ).resolve().as_posix().replace("'", "''")

    connection = duckdb.connect()

    try:
        connection.execute(
            f"CREATE VIEW movies AS SELECT * FROM read_parquet('{movies_path}')"
        )
        connection.execute(
            f"CREATE VIEW ratings AS SELECT * FROM read_parquet('{ratings_path}')"
        )

        # Load only the selected user's ratings into a small table.
        connection.execute(
            """
            CREATE TEMP TABLE selected_user_ratings AS
            SELECT userId, movieId, rating
            FROM ratings
            WHERE userId = ?
            """,
            [user_id],
        )

        user_rating_count = connection.execute(
            "SELECT COUNT(*) FROM selected_user_ratings"
        ).fetchone()[0]

        if user_rating_count == 0:
            raise ValueError(
                f"User {user_id} has no ratings in the dataset."
            )

        print(
            f"Building recommendations for user {user_id} "
            f"({user_rating_count:,} ratings)."
        )

        # Estimate the user's preference for each genre.
        connection.execute(
            """
            CREATE TEMP TABLE user_genre_preferences AS
            SELECT
                genre,
                AVG(r.rating) AS user_avg_rating,
                COUNT(*) AS observations
            FROM selected_user_ratings AS r
            JOIN movies AS m
                ON r.movieId = m.movieId
            CROSS JOIN UNNEST(
                string_split(m.genres, '|')
            ) AS g(genre)
            WHERE genre <> '(no genres listed)'
            GROUP BY genre
            """
        )

        # Global weighted scores provide a fallback for genres
        # the user has never rated.
        connection.execute(
            """
            CREATE TEMP TABLE global_movie_scores AS
            WITH movie_stats AS (
                SELECT
                    movieId,
                    COUNT(*) AS rating_count,
                    AVG(rating) AS average_rating
                FROM ratings
                GROUP BY movieId
            ),
            global_stats AS (
                SELECT AVG(rating) AS global_average
                FROM ratings
            )
            SELECT
                s.movieId,
                s.rating_count,
                (
                    s.rating_count * s.average_rating
                    + 1000 * g.global_average
                ) / (s.rating_count + 1000)
                    AS weighted_rating
            FROM movie_stats AS s
            CROSS JOIN global_stats AS g
            """
        )

        query = """
            WITH candidate_genres AS (
                SELECT
                    m.movieId,
                    m.title,
                    m.genres,
                    genre
                FROM movies AS m
                CROSS JOIN UNNEST(
                    string_split(m.genres, '|')
                ) AS g(genre)
                WHERE genre <> '(no genres listed)'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM selected_user_ratings AS ur
                      WHERE ur.movieId = m.movieId
                  )
            ),
            candidate_scores AS (
                SELECT
                    c.movieId,
                    c.title,
                    c.genres,
                    AVG(p.user_avg_rating)
                        AS genre_preference_score,
                    COUNT(p.genre)
                        AS matched_genres,
                    MAX(s.rating_count)
                        AS rating_count,
                    MAX(s.weighted_rating)
                        AS global_score
                FROM candidate_genres AS c
                LEFT JOIN user_genre_preferences AS p
                    ON c.genre = p.genre
                LEFT JOIN global_movie_scores AS s
                    ON c.movieId = s.movieId
                GROUP BY
                    c.movieId,
                    c.title,
                    c.genres
            )
            SELECT
                movieId,
                title,
                genres,
                rating_count,
                ROUND(
                    genre_preference_score, 4
                ) AS genre_preference_score,
                ROUND(
                    COALESCE(
                        genre_preference_score,
                        global_score
                    ), 4
                ) AS personalized_score,
                matched_genres
            FROM candidate_scores
            ORDER BY
                personalized_score DESC,
                rating_count DESC NULLS LAST,
                movieId
            LIMIT {top_n}
        """.format(top_n=int(top_n))

        connection.execute(
            f"""
            COPY ({query})
            TO '{output_path}'
            (HEADER, DELIMITER ',')
            """
        )

        recommendations = connection.execute(
            f"""
            SELECT *
            FROM read_csv_auto('{output_path}')
            """
        ).df()

        print("\nPersonalized recommendations:")
        print(recommendations.to_string(index=False))
        print(f"\nSaved to: {output_path}")

    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate genre-based movie recommendations."
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="MovieLens user ID (default: 1)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations (default: 10)",
    )

    args = parser.parse_args()

    if args.user_id <= 0:
        parser.error("--user-id must be positive")

    if not 1 <= args.top_n <= 100:
        parser.error("--top-n must be between 1 and 100")

    main(args.user_id, args.top_n)
