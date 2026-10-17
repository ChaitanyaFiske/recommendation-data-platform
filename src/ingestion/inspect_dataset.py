
from pathlib import Path
import logging

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "ingestion.log"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# Expected columns for each dataset
EXPECTED_COLUMNS = {
    "movies.csv": {
        "movieId", "title", "genres"
    },
    "ratings.csv": {
        "userId", "movieId", "rating", "timestamp"
    },
    "tags.csv": {
        "userId", "movieId", "tag", "timestamp"
    },
    "links.csv": {
        "movieId", "imdbId", "tmdbId"
    },
}


def inspect_csv(filename: str, expected_columns: set) -> bool:
    """Validate a CSV file and report its basic data quality."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return False

    logger.info("=" * 60)
    logger.info("Inspecting %s", filename)

    try:
        # Read only the header first
        header = pd.read_csv(file_path, nrows=0)
        actual_columns = set(header.columns)

        missing_columns = expected_columns - actual_columns

        if missing_columns:
            logger.error(
                "Missing expected columns: %s",
                sorted(missing_columns),
            )
            return False

        logger.info("Columns: %s", list(header.columns))

        # Process in chunks to limit memory usage
        total_rows = 0
        missing_values = pd.Series(dtype="int64")
        sample_dtypes = None
        sample_rows = None

        for chunk in pd.read_csv(
            file_path,
            chunksize=500_000,
        ):
            total_rows += len(chunk)

            if sample_dtypes is None:
                sample_dtypes = chunk.dtypes.astype(str)
                sample_rows = chunk.head(5)

            chunk_missing = chunk.isna().sum()
            missing_values = missing_values.add(
                chunk_missing,
                fill_value=0,
            )

        logger.info("Total data rows: %s", total_rows)
        logger.info("Sample data types:\n%s", sample_dtypes)
        logger.info("First five rows:\n%s", sample_rows)

        nonzero_missing = missing_values[
            missing_values > 0
        ]

        if nonzero_missing.empty:
            logger.info("Missing values: none detected")
        else:
            logger.warning(
                "Missing values by column:\n%s",
                nonzero_missing,
            )

        logger.info("Inspection completed for %s", filename)
        return True

    except Exception:
        logger.exception("Failed to inspect %s", filename)
        return False


def main():
    """Inspect every required dataset."""

    if not DATA_DIR.exists():
        logger.error(
            "Dataset directory does not exist: %s",
            DATA_DIR,
        )
        raise SystemExit(1)

    results = {}

    for filename, columns in EXPECTED_COLUMNS.items():
        results[filename] = inspect_csv(filename, columns)

    logger.info("=" * 60)
    logger.info("FINAL INSPECTION SUMMARY")

    for filename, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info("%s: %s", filename, status)

    if not all(results.values()):
        logger.error("Some dataset checks failed.")
        raise SystemExit(1)

    logger.info("All required datasets passed inspection.")


if __name__ == "__main__":
    main()
