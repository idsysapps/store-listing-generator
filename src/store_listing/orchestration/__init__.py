import os

from celery import Celery

DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def celery_redis_url() -> str:
    return os.environ.get("REDIS_URL") or DEFAULT_REDIS_URL


celery_app = Celery(
    "store_listing",
    broker=celery_redis_url(),
    backend=celery_redis_url(),
)

__all__ = ["celery_app", "celery_redis_url"]
