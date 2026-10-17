
import json

import pytest

from src.recommendation.config import load_config


def test_project_config_loads():
    config = load_config()

    assert config["model_name"] == "hybrid_collaborative_popularity"
    assert config["catalog_size"] == 1500
    assert sum(config["weights"].values()) == pytest.approx(1.0)


def test_missing_config_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def test_weights_must_sum_to_one(tmp_path):
    config = {
        "model_name": "test",
        "model_version": "1.0.0",
        "catalog_size": 100,
        "minimum_movie_ratings": 10,
        "bayesian_rating_prior": 10,
        "global_mean_rating": 3.5,
        "top_n": 10,
        "weights": {
            "collaborative": 0.8,
            "content": 0.1,
            "popularity": 0.3,
        },
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="sum to 1"):
        load_config(config_path)
