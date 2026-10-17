from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "recommendations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    movies_path = (
        DATA_DIR / "movies.parquet"
    ).resolve().as_posix()

    ratings_path = (
        DATA_DIR / "ratings.parquet"
    ).resolve().as_posix()

    output_path = (
        OUTPUT_DIR / "popular_movies.csv"
    ).resolve().as_posix()

    # Escape paths for SQL string literals.
    movies_path = movies_path.replace("'", "''")
    ratings_path = ratings_path.replace("'", "''")
    output_path = output_path.replace("'", "''")

    query = f"""
        WITH movie_stats AS (
            SELECT
                m.movieId,
                m.title,
                m.genres,
                COUNT(*) AS rating_count,
                AVG(r.rating) AS average_rating
            FROM read_parquet('{movies_path}') AS m
            JOIN read_parquet('{ratings_path}') AS r
                ON m.movieId = r.movieId
            GROUP BY
                m.movieId,
                m.title,
                m.genres
        ),
        global_stats AS (
            SELECT AVG(rating) AS global_average
            FROM read_parquet('{ratings_path}')
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
        WHERE s.rating_count >= 100
        ORDER BY
            weighted_rating DESC,
            rating_count DESC
        LIMIT 100
    """

    connection = duckdb.connect()

    try:
        connection.execute(
            f"""
            COPY ({query})
            TO '{output_path}'
            (HEADER, DELIMITER ',')
            """
        )

        print("Popularity-based recommendations created.")
        print(f"Saved to: {output_path}")

        result = connection.execute(
            f"""
            SELECT *
            FROM read_csv_auto('{output_path}')
            LIMIT 10
            """
        ).df()

        print("\nTop 10 recommended movies:")
        print(result.to_string(index=False))

    finally:
        connection.close()


if __name__ == "__main__":
    main()