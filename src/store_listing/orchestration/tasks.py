from celery import Celery
from celery.schedules import crontab

from store_listing.ingest.trends import DatabaseClient, GoogleTrendsClient, TrendHarvestRequest
from store_listing.orchestration import celery_redis_url
from store_listing.orchestration.promotion import (
    STARTER_SEEDS,
    candidates_to_promote,
    compute_promotion_score,
    enforce_cap,
)

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
    "seed-promotion": {
        "task": "store_listing.orchestration.tasks.promote_seeds",
        "schedule": crontab(hour=6, minute=5),
    },
}


@celery_app.task
def fetch_daily_trends() -> dict:
    db_client = DatabaseClient()
    trends_client = GoogleTrendsClient(db_client=db_client)

    active_seeds = [seed.query for seed in db_client.list_active_seeds()]
    request = TrendHarvestRequest(seed_keywords=active_seeds or STARTER_SEEDS)
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


@celery_app.task
def promote_seeds() -> dict:
    """Auto-promote pending seed candidates that meet their source's rules."""
    db_client = DatabaseClient()

    pending = db_client.list_pending_candidates()
    cross_seed_counts = db_client.cross_seed_counts()
    to_promote = candidates_to_promote(pending, cross_seed_counts)
    if not to_promote:
        return {"status": "success", "promoted": 0, "archived": 0}

    active = db_client.list_active_seeds()
    archives = enforce_cap(active, len(to_promote))

    for query in archives:
        db_client.archive_active_seed(query)

    by_id = {candidate.id: candidate for candidate in pending}
    for candidate_id in to_promote:
        candidate = by_id[candidate_id]
        db_client.insert_active_seed(
            query=candidate.query,
            promotion_score=candidate.promotion_score,
            candidate_id=candidate.id,
        )
        db_client.promote_candidate(candidate.id)

    return {
        "status": "success",
        "promoted": len(to_promote),
        "archived": len(archives),
    }


@celery_app.task
def backfill_seeds_from_candidates() -> dict:
    """One-time: pre-populate seed_candidates from existing trend_scores."""
    db_client = DatabaseClient()
    processed = 0
    for (
        source_seed,
        query,
        query_type,
        source,
        score,
        delta,
    ) in db_client.list_historic_rising_top():
        promotion_score = compute_promotion_score(source, query_type, score, delta)
        db_client.upsert_seed_candidate(
            source_seed=source_seed,
            query=query,
            source=source,
            query_type=query_type,
            score=score,
            delta=delta,
            promotion_score=promotion_score,
        )
        processed += 1
    return {"status": "success", "candidates_processed": processed}
