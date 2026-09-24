from celery import Celery
from celery.schedules import crontab

from store_listing.ingest.trends import DatabaseClient, GoogleTrendsClient, TrendHarvestRequest
from store_listing.orchestration import celery_redis_url

celery_app = Celery(
    "store_listing",
    broker=celery_redis_url(),
    backend=celery_redis_url(),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

celery_app.conf.beat_schedule = {
    "daily-trend-harvest": {
        "task": "store_listing.orchestration.tasks.fetch_daily_trends",
        "schedule": crontab(hour=6, minute=0),
    },
}


@celery_app.task
def fetch_daily_trends() -> dict:
    db_client = DatabaseClient()
    trends_client = GoogleTrendsClient(db_client=db_client)

    request = TrendHarvestRequest()
    results, failed_seeds = trends_client.harvest_and_store(request)

    return {
        "status": "failed"
        if len(failed_seeds) == len(request.seed_keywords)
        else ("partial" if failed_seeds else "success"),
        "results_count": len(results),
        "failed_seeds": failed_seeds,
        "seeds": request.seed_keywords,
    }


@celery_app.task
def fetch_trends_manual(seeds: list[str] | None = None) -> dict:
    db_client = DatabaseClient()
    trends_client = GoogleTrendsClient(db_client=db_client)

    request = TrendHarvestRequest(seed_keywords=seeds or ["funny t-shirt", "hoodie", "gift"])
    results, failed_seeds = trends_client.harvest_and_store(request)

    return {
        "status": "failed"
        if len(failed_seeds) == len(request.seed_keywords)
        else ("partial" if failed_seeds else "success"),
        "results_count": len(results),
        "failed_seeds": failed_seeds,
        "seeds": request.seed_keywords,
    }
