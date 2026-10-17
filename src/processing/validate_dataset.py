
from pathlib import Path
import logging

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"
REPORT_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / "data_quality.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


def csv_source(filename: str) -> str:
    """Return a DuckDB CSV source using an absolute path."""

    path = (DATA_DIR / filename).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Escape single quotes in the path for SQL.
    escaped_path = path.as_posix().replace("'", "''")

    return f"read_csv_auto('{escaped_path}', header=true)"


def main():
    movies = csv_source("movies.csv")
    ratings = csv_source("ratings.csv")
    tags = csv_source("tags.csv")
    links = csv_source("links.csv")

    checks = []

    def run_check(name, sql, severity="error"):
        value = int(duckdb.sql(sql).fetchone()[0])

        checks.append({
            "check": name,
            "value": value,
            "severity": severity,
            "status": (
                "PASS"
                if value == 0 or severity == "info"
                else "FAIL"
            ),
        })

        logger.info("%s: %s", name, value)

    logger.info("Starting data-quality validation.")

    # 1. Duplicate identifiers
    run_check(
        "Duplicate movie ID groups",
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT movieId
            FROM {movies}
            GROUP BY movieId
            HAVING COUNT(*) > 1
        )
        """,
    )

    run_check(
        "Duplicate user-movie rating pairs",
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT userId, movieId
            FROM {ratings}
            GROUP BY userId, movieId
            HAVING COUNT(*) > 1
        )
        """,
    )

    # 2. Invalid rating values
    run_check(
        "Ratings outside 0.5 to 5.0",
        f"""
        SELECT COUNT(*)
        FROM {ratings}
        WHERE rating < 0.5 OR rating > 5.0
        """,
    )

    run_check(
        "Ratings not on half-star increments",
        f"""
        SELECT COUNT(*)
        FROM {ratings}
        WHERE rating * 2 != ROUND(rating * 2)
        """,
    )

    # 3. Missing required values
    run_check(
        "Movies missing title or movie ID",
        f"""
        SELECT COUNT(*)
        FROM {movies}
        WHERE movieId IS NULL
           OR title IS NULL
           OR TRIM(title) = ''
        """,
    )

    run_check(
        "Ratings with missing required fields",
        f"""
        SELECT COUNT(*)
        FROM {ratings}
        WHERE userId IS NULL
           OR movieId IS NULL
           OR rating IS NULL
           OR timestamp IS NULL
        """,
    )

    run_check(
        "Tags with missing text",
        f"""
        SELECT COUNT(*)
        FROM {tags}
        WHERE tag IS NULL OR TRIM(tag) = ''
        """,
        severity="info",
    )

    run_check(
        "Links with missing TMDB ID",
        f"""
        SELECT COUNT(*)
        FROM {links}
        WHERE tmdbId IS NULL
        """,
        severity="info",
    )

    # 4. Referential integrity
    run_check(
        "Ratings referencing unknown movies",
        f"""
        SELECT COUNT(*)
        FROM {ratings} AS r
        WHERE NOT EXISTS (
            SELECT 1
            FROM {movies} AS m
            WHERE m.movieId = r.movieId
        )
        """,
    )

    run_check(
        "Tags referencing unknown movies",
        f"""
        SELECT COUNT(*)
        FROM {tags} AS t
        WHERE NOT EXISTS (
            SELECT 1
            FROM {movies} AS m
            WHERE m.movieId = t.movieId
        )
        """,
    )

    # 5. Report generation
    report = pd.DataFrame(checks)
    report_path = REPORT_DIR / "data_quality_report.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("DATA QUALITY REPORT")
    print("=" * 65)
    print(report.to_string(index=False))
    print(f"\nReport saved to: {report_path}")

    failures = report[
        (report["severity"] == "error")
        & (report["status"] == "FAIL")
    ]

    if not failures.empty:
        logger.error(
            "%s critical data-quality checks failed.",
            len(failures),
        )
        raise SystemExit(1)

    logger.info("All critical data-quality checks passed.")


if __name__ == "__main__":
    main()
