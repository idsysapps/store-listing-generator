from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
    "store_listing",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

__all__ = ["celery_app"]
