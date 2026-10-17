
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config" / "model_config.json"


def load_config(config_path=CONFIG_FILE):
    """Load and validate the recommendation model configuration."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {
        "model_name",
        "model_version",
        "catalog_size",
        "minimum_movie_ratings",
        "bayesian_rating_prior",
        "global_mean_rating",
        "top_n",
        "weights",
    }

    missing = required - config.keys()
    if missing:
        raise ValueError(
            f"Missing configuration keys: {sorted(missing)}"
        )

    weights = config["weights"]
    expected_weights = {"collaborative", "content", "popularity"}

    if set(weights) != expected_weights:
        raise ValueError(
            "Weights must contain collaborative, content, and popularity."
        )

    values = list(weights.values())

    if any(
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or weight < 0
        for weight in values
    ):
        raise ValueError("Weights must be non-negative numbers.")

    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError("Model weights must sum to 1.")

    for key in (
        "catalog_size",
        "minimum_movie_ratings",
        "bayesian_rating_prior",
        "top_n",
    ):
        value = config[key]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{key} must be a positive integer.")

    mean = config["global_mean_rating"]
    if (
        not isinstance(mean, (int, float))
        or isinstance(mean, bool)
        or not 0 <= mean <= 5
    ):
        raise ValueError("global_mean_rating must be between 0 and 5.")

    return config
