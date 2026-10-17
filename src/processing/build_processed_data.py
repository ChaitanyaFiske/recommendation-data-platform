
from pathlib import Path
import logging

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / "data_processing.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


def csv_source(filename: str) -> str:
    path = (RAW_DIR / filename).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")

    escaped_path = path.as_posix().replace("'", "''")
    return f"read_csv_auto('{escaped_path}', header=true)"


def build_parquet(
    connection,
    name: str,
    query: str,
) -> int:
    """Write one processed dataset and return its row count."""

    output_path = (PROCESSED_DIR / f"{name}.parquet").resolve()
    escaped_output = output_path.as_posix().replace("'", "''")

    logger.info("Building %s.parquet", name)

    connection.execute(
        f"""
        COPY ({query})
        TO '{escaped_output}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{escaped_output}')
        """
    ).fetchone()[0]

    logger.info(
        "Created %s.parquet with %s rows",
        name,
        row_count,
    )

    return row_count


def main():
    connection = duckdb.connect()

    try:
        movies = csv_source("movies.csv")
        ratings = csv_source("ratings.csv")
        tags = csv_source("tags.csv")
        links = csv_source("links.csv")

        # Investigate the earlier tag null-count discrepancy.
        tag_counts = connection.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (
                    WHERE tag IS NULL
                ) AS null_tags,
                COUNT(*) FILTER (
                    WHERE tag IS NOT NULL
                      AND TRIM(tag) = ''
                ) AS blank_tags
            FROM {tags}
            """
        ).fetchone()

        logger.info(
            "Tag audit: total=%s, null=%s, blank_strings=%s",
            *tag_counts,
        )

        # Keep valid records and preserve original column names.
        build_parquet(
            connection,
            "movies",
            f"""
            SELECT movieId, title, genres
            FROM {movies}
            """,
        )

        build_parquet(
            connection,
            "ratings",
            f"""
            SELECT userId, movieId, rating, timestamp
            FROM {ratings}
            WHERE userId IS NOT NULL
              AND movieId IS NOT NULL
              AND rating BETWEEN 0.5 AND 5.0
              AND rating * 2 = ROUND(rating * 2)
              AND timestamp IS NOT NULL
            """,
        )

        # Exclude missing or whitespace-only tags from this
        # processed version; retain the original CSV unchanged.
        build_parquet(
            connection,
            "tags",
            f"""
            SELECT userId, movieId, tag, timestamp
            FROM {tags}
            WHERE userId IS NOT NULL
              AND movieId IS NOT NULL
              AND tag IS NOT NULL
              AND TRIM(tag) <> ''
              AND timestamp IS NOT NULL
            """,
        )

        # Missing external IDs are allowed and retained.
        build_parquet(
            connection,
            "links",
            f"""
            SELECT movieId, imdbId, tmdbId
            FROM {links}
            """,
        )

        logger.info("Processed datasets created successfully.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
