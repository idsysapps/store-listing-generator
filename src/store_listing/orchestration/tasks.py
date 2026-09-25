import os

from celery import Celery
from celery.schedules import crontab

from store_listing.ingest.trends import (
    AutocompleteHarvester,
    DatabaseClient,
    GoogleTrendsClient,
    InstagramClient,
    PinterestClient,
    RedditClient,
    TikTokClient,
    TrendHarvestRequest,
    XClient,
    YouTubeClient,
)
from store_listing.orchestration import celery_redis_url
from store_listing.orchestration.promotion import (
    STARTER_SEEDS,
    candidates_to_promote,
    compute_promotion_score,
    enforce_cap,
)
from store_listing.orchestration.source_health import with_source_health

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


def _tiktok_enabled() -> bool:
    return os.environ.get("TIKTOK_ENABLED", "false").lower() in ("1", "true", "yes")


def _instagram_enabled() -> bool:
    return os.environ.get("INSTAGRAM_ENABLED", "false").lower() in ("1", "true", "yes")


celery_app.conf.beat_schedule = {
    "daily-trend-harvest": {
        "task": "store_listing.orchestration.tasks.fetch_daily_trends",
        "schedule": crontab(hour=6, minute=0),
    },
    "seed-promotion": {
        "task": "store_listing.orchestration.tasks.promote_seeds",
        "schedule": crontab(hour=6, minute=5),
    },
    "pinterest-micro-trends": {
        "task": "store_listing.orchestration.tasks.fetch_pinterest_trends",
        "schedule": crontab(hour=6, minute=15, day_of_week="0,3"),
    },
    "marketplace-suggestions": {
        "task": "store_listing.orchestration.tasks.fetch_marketplace_suggestions",
        "schedule": crontab(hour=6, minute=20),
    },
    "x-micro-trends": {
        "task": "store_listing.orchestration.tasks.fetch_x_trends",
        "schedule": crontab(hour=6, minute=25, day_of_week="1,4"),
    },
    "reddit-micro-trends": {
        "task": "store_listing.orchestration.tasks.fetch_reddit_trends",
        "schedule": crontab(hour=6, minute=30),
    },
    "youtube-micro-trends": {
        "task": "store_listing.orchestration.tasks.fetch_youtube_trends",
        "schedule": crontab(hour=6, minute=35),
    },
}

if _tiktok_enabled():
    # TikTok is disabled by default: the Apify actor bills per returned video and
    # the free tag-page path is TLS-blocked from our egress. Re-enable by setting
    # TIKTOK_ENABLED=true with cost caps (see #86); the harvest runs weekly
    # (Sunday) so spend stays under Apify's $5/month free tier.
    celery_app.conf.beat_schedule["tiktok-micro-trends"] = {
        "task": "store_listing.orchestration.tasks.fetch_tiktok_trends",
        "schedule": crontab(hour=6, minute=10, day_of_week=0),
    }

if _instagram_enabled():
    # Instagram is disabled by default: fully login-walled, Apify required
    # (~$0.40-2.50/1K posts). Re-enable by setting INSTAGRAM_ENABLED=true;
    # the harvest runs weekly (Saturday) to cap Apify spend.
    celery_app.conf.beat_schedule["instagram-micro-trends"] = {
        "task": "store_listing.orchestration.tasks.fetch_instagram_trends",
        "schedule": crontab(hour=6, minute=40, day_of_week=5),
    }


def _active_seed_keywords(db_client: DatabaseClient) -> list[str]:
    return [seed.query for seed in db_client.list_active_seeds()]


def _harvest_status(results_count: int, failed_seeds: list[str], seeds: list[str]) -> dict:
    return {
        "status": "failed"
        if len(failed_seeds) == len(seeds)
        else ("partial" if failed_seeds else "success"),
        "results_count": results_count,
        "failed_seeds": failed_seeds,
        "seeds": seeds,
    }


def _bulk_status(results_count: int, failed_targets: list[str], seeds: list[str]) -> dict:
    """Status for harvesters that expand each seed into many targets (tiktok/pinterest/etc)."""
    if results_count == 0 and failed_targets:
        status: str = "failed"
    elif failed_targets:
        status = "partial"
    else:
        status = "success"
    return {
        "status": status,
        "results_count": results_count,
        "failed_targets": failed_targets,
        "seeds": seeds,
    }


@celery_app.task
@with_source_health("google")
def fetch_daily_trends() -> dict:
    db_client = DatabaseClient()
    trends_client = GoogleTrendsClient(db_client=db_client)

    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed_seeds = trends_client.harvest_and_store(request)

    return _harvest_status(len(results), failed_seeds, request.seed_keywords)


@celery_app.task
@with_source_health("google")
def fetch_trends_manual(seeds: list[str] | None = None) -> dict:
    db_client = DatabaseClient()
    trends_client = GoogleTrendsClient(db_client=db_client)

    request = TrendHarvestRequest(seed_keywords=seeds or ["funny t-shirt", "hoodie", "gift"])
    results, failed_seeds = trends_client.harvest_and_store(request)

    return _harvest_status(len(results), failed_seeds, request.seed_keywords)


@celery_app.task
@with_source_health("tiktok")
def fetch_tiktok_trends() -> dict:
    """Harvest TikTok hashtag/sound micro-trends from the active seed set."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed = TikTokClient(db_client=db_client).harvest_and_store(request)
    return _bulk_status(len(results), failed, request.seed_keywords)


@celery_app.task
@with_source_health("pinterest")
def fetch_pinterest_trends() -> dict:
    """Harvest Pinterest search/board traction from the active seed set."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed = PinterestClient(db_client=db_client).harvest_and_store(request)
    return _bulk_status(len(results), failed, request.seed_keywords)


@celery_app.task
@with_source_health("marketplace")
def fetch_marketplace_suggestions() -> dict:
    """Collect Amazon + Etsy search autocomplete queries for the active seed set."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed_sources = AutocompleteHarvester(db_client=db_client).harvest_and_store(request)
    return _bulk_status(len(results), failed_sources, request.seed_keywords)


@celery_app.task
@with_source_health("x")
def fetch_x_trends() -> dict:
    """Harvest X hashtag/keyword conversation trends from the active seed set."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed = XClient(db_client=db_client).harvest_and_store(request)
    return _bulk_status(len(results), failed, request.seed_keywords)


@celery_app.task
@with_source_health("reddit")
def fetch_reddit_trends() -> dict:
    """Harvest Reddit post titles and subreddit traction from seeds + POD subs."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed = RedditClient(db_client=db_client).harvest_and_store(request)
    return _bulk_status(len(results), failed, request.seed_keywords)


@celery_app.task
@with_source_health("youtube")
def fetch_youtube_trends() -> dict:
    """Harvest YouTube Shorts titles and tags from the active seed set."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed = YouTubeClient(db_client=db_client).harvest_and_store(request)
    return _bulk_status(len(results), failed, request.seed_keywords)


@celery_app.task
@with_source_health("instagram")
def fetch_instagram_trends() -> dict:
    """Harvest Instagram hashtag engagement from the active seed set."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed = InstagramClient(db_client=db_client).harvest_and_store(request)
    return _bulk_status(len(results), failed, request.seed_keywords)


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
