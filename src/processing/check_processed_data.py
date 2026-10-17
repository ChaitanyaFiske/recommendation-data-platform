
from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

connection = duckdb.connect()

try:
    for name in ["movies", "ratings", "tags", "links"]:
        path = (DATA_DIR / f"{name}.parquet").resolve()
        escaped_path = path.as_posix().replace("'", "''")

        rows = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet('{escaped_path}')"
        ).fetchone()[0]

        print(f"{name:10} {rows:>12,} rows")

finally:
    connection.close()
