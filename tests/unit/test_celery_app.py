import importlib
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def redis_url():
    return "redis://store-listing-store-listing-redis:6379/0"


def test_tasks_celery_app_uses_redis_url_env(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)
        assert module.celery_app.conf.broker_url == redis_url
        assert module.celery_app.conf.result_backend == redis_url


def test_tasks_celery_app_defaults_to_localhost() -> None:
    with patch.dict(os.environ, {"REDIS_URL": ""}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)
        assert module.celery_app.conf.broker_url == "redis://localhost:6379/0"
        assert module.celery_app.conf.result_backend == "redis://localhost:6379/0"


def test_fetch_trends_manual_marks_partial_when_some_seeds_fail(redis_url: str) -> None:
    """Black-box: with any failed seed, status must be partial, not success."""
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)
    with (
        patch.object(module.GoogleTrendsClient, "harvest_and_store") as mock_harvest,
    ):
        mock_harvest.return_value = (
            [object()],
            ["hoodie"],
        )
        result = module.fetch_trends_manual.run(["funny t-shirt", "hoodie"])
    assert result["status"] == "partial"
    assert result["failed_seeds"] == ["hoodie"]
    assert result["results_count"] == 1


def test_fetch_trends_manual_marks_failed_when_all_seeds_fail(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)
    with patch.object(module.GoogleTrendsClient, "harvest_and_store") as mock_harvest:
        mock_harvest.return_value = ([], ["funny t-shirt", "hoodie"])
        result = module.fetch_trends_manual.run(["funny t-shirt", "hoodie"])
    assert result["status"] == "failed"
    assert result["failed_seeds"] == ["funny t-shirt", "hoodie"]
    assert result["results_count"] == 0
