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
