import logging
import os
from collections.abc import Sequence
from typing import Any

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
from store_listing.orchestration.product_scope import enabled_product_types
from store_listing.orchestration.promotion import (
    ORIGIN_DISCOVERY,
    STARTER_SEEDS,
    compute_promotion_score,
)
from store_listing.orchestration.source_health import build_tracker, with_source_health

logger = logging.getLogger(__name__)

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


def _llm_curation_enabled() -> bool:
    return os.environ.get("LLM_SEED_CURATION_ENABLED", "false").lower() in ("1", "true", "yes")


def _llm_design_briefs_enabled() -> bool:
    return os.environ.get("LLM_DESIGN_BRIEFS_ENABLED", "false").lower() in ("1", "true", "yes")


def _require_tagline_text() -> bool:
    """Whether a missing tagline should reject the image.

    Off by default: diffusion drops the secondary line often, and losing the
    whole design over it costs more than shipping it without the tagline.
    """
    return os.environ.get("REQUIRE_TAGLINE_TEXT", "false").lower() in ("1", "true", "yes")


def _artwork_validation_enabled() -> bool:
    """Whether the artwork gate runs.

    On by default: an image that shows the product instead of the design
    cannot be printed (issue #195).
    """
    return os.environ.get("ARTWORK_VALIDATION_ENABLED", "true").lower() in ("1", "true", "yes")


def _build_artwork_validator() -> Any:
    """The validator that judges the generated artwork, or None when it is off."""
    if not _artwork_validation_enabled():
        return None

    from store_listing.orchestration.artwork_validation import ArtworkValidator
    from store_listing.orchestration.llm_client import get_artwork_vision_model, get_vision_client

    return ArtworkValidator(client=get_vision_client(), model=get_artwork_vision_model())


def _comfyui_steps() -> int:
    """The step count the image model runs with.

    Flux Schnell is distilled for 4 steps, but the count is a setting, so a
    run can be tuned without a code change (issue #210).
    """
    from store_listing.orchestration.comfyui_client import DEFAULT_STEPS

    try:
        return int(os.environ.get("COMFYUI_STEPS", str(DEFAULT_STEPS)))
    except ValueError:
        logger.warning("COMFYUI_STEPS is not a number, using %d", DEFAULT_STEPS)
        return DEFAULT_STEPS


def _comfyui_cfg() -> float:
    """The cfg the image model runs with.

    Flux ignores the negative prompt at cfg 1, so the value matters when the
    model is not the distilled one (issue #210).
    """
    from store_listing.orchestration.comfyui_client import DEFAULT_CFG

    try:
        return float(os.environ.get("COMFYUI_CFG", str(DEFAULT_CFG)))
    except ValueError:
        logger.warning("COMFYUI_CFG is not a number, using %s", DEFAULT_CFG)
        return DEFAULT_CFG


def _check_llm_model(client: Any, model: str) -> str:
    """Log the models the endpoint serves when the configured one is missing.

    The cluster pointed at a model the DGX proxy does not serve, so every
    brief and curation call failed and the logs never said which model was
    missing (issue #194).
    """
    from store_listing.orchestration.llm_client import check_model_available

    message = check_model_available(client, model)
    if message:
        logger.error("The configured LLM model is not served: %s", message)
    return message


def _artwork_text_validation_enabled() -> bool:
    """Whether the lettering gate runs.

    On by default: the image model cannot spell, and the text of the design is
    drawn on the artwork afterwards, so lettering in the artwork is lettering
    that is wrong (issue #208).
    """
    return os.environ.get("ARTWORK_TEXT_VALIDATION_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )


def _build_artwork_text_validator() -> Any:
    """The validator that looks for lettering in the artwork, or None when it is off."""
    if not _artwork_text_validation_enabled():
        return None

    from store_listing.orchestration.artwork_validation import ArtworkTextValidator
    from store_listing.orchestration.llm_client import get_artwork_vision_model, get_vision_client

    return ArtworkTextValidator(client=get_vision_client(), model=get_artwork_vision_model())


def _build_hero_validator() -> Any:
    """The strict OCR gate for Qwen-drawn lettering, or None when the path is off.

    The hero path is the only one where the image model draws the words, so it
    needs the stricter gate: one wrong letter is a wrong design, not an OCR
    wobble (issue #254).
    """
    from store_listing.orchestration.typographic_hero import typographic_hero_enabled

    if not typographic_hero_enabled():
        return None

    from store_listing.orchestration.llm_client import get_vision_client, get_vision_model
    from store_listing.orchestration.text_validation import StrictTextValidator

    return StrictTextValidator(client=get_vision_client(), model=get_vision_model())


celery_app.conf.beat_schedule = {
    "seed-context-injection": {
        "task": "store_listing.orchestration.tasks.inject_context_seeds_task",
        "schedule": crontab(hour=5, minute=55),
    },
    "daily-trend-harvest": {
        "task": "store_listing.orchestration.tasks.fetch_daily_trends",
        "schedule": crontab(hour=6, minute=0),
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
    "trend-discovery": {
        "task": "store_listing.orchestration.tasks.fetch_discovery_trends",
        "schedule": crontab(hour="*/6", minute=45),
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


def _parse_cron(env_var: str, default: str) -> crontab:
    """Parse a 5-field cron expression from an env var into a Celery crontab."""
    expr = os.environ.get(env_var, default)
    parts = expr.split()
    if len(parts) != 5:
        parts = default.split()
    return crontab(
        minute=parts[0],
        hour=parts[1],
        day_of_week=parts[4],
        day_of_month=parts[2],
        month_of_year=parts[3],
    )


if _llm_curation_enabled():
    celery_app.conf.beat_schedule["llm-seed-curation"] = {
        "task": "store_listing.orchestration.tasks.curate_seeds_task",
        "schedule": _parse_cron("LLM_CURATION_SCHEDULE", "50 2,6,10,14,18,22 * * *"),
    }

if _llm_design_briefs_enabled():
    celery_app.conf.beat_schedule["llm-design-briefs"] = {
        "task": "store_listing.orchestration.tasks.generate_design_briefs_task",
        "schedule": _parse_cron("LLM_BRIEFS_SCHEDULE", "0 7 * * *"),
    }
    celery_app.conf.beat_schedule["design-image-generation"] = {
        "task": "store_listing.orchestration.tasks.generate_design_images_task",
        "schedule": _parse_cron("LLM_IMAGE_GEN_SCHEDULE", "30 7 * * *"),
    }
    celery_app.conf.beat_schedule["design-brief-reranking"] = {
        "task": "store_listing.orchestration.tasks.rerank_design_briefs_task",
        "schedule": _parse_cron("LLM_RERANK_SCHEDULE", "0 8 * * *"),
    }


def _active_seed_keywords(db_client: DatabaseClient) -> list[str]:
    """The seeds the harvesters expand.

    A discovery is left out: it was found without a seed to start from, so
    expanding it again is the pool talking to itself (issue #201).
    """
    return [seed.query for seed in db_client.list_active_seeds() if seed.origin != ORIGIN_DISCOVERY]


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
def fetch_marketplace_suggestions() -> dict:
    """Collect Amazon + Etsy search autocomplete queries for the active seed set."""
    db_client = DatabaseClient()
    request = TrendHarvestRequest(seed_keywords=_active_seed_keywords(db_client) or STARTER_SEEDS)
    results, failed_sources = AutocompleteHarvester(db_client=db_client).harvest_and_store(request)
    _record_marketplace_health(results, failed_sources, request.seed_keywords)
    return _bulk_status(len(results), failed_sources, request.seed_keywords)


def _record_marketplace_health(
    results: Sequence[object], failed_sources: list[str], seeds: list[str]
) -> None:
    tracker = build_tracker()
    if tracker is None:
        return
    failed_by_source: dict[str, list[str]] = {}
    for entry in failed_sources:
        source, _, seed = entry.partition(":")
        failed_by_source.setdefault(source, []).append(seed)
    for source in ("amazon", "etsy"):
        source_failures = failed_by_source.get(source, [])
        if source_failures and len(source_failures) >= len(seeds):
            tracker.record(source, "failure", error=f"all seeds failed for {source}")
        elif source_failures:
            tracker.record(source, "success")
        else:
            tracker.record(source, "success")


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


@celery_app.task
def inject_context_seeds_task() -> dict:
    """Inject the calendar seeds and the evergreen theme seeds before harvesters run."""
    from store_listing.orchestration.context_seeds import inject_context_seeds

    return inject_context_seeds(DatabaseClient())


@celery_app.task
def fetch_discovery_trends() -> dict:
    """Find trends that do not come from the seed pool (issue #201)."""
    from datetime import UTC, datetime

    from store_listing.ingest.trends.autocomplete import AmazonSuggestionClient
    from store_listing.ingest.trends.discovery import (
        DISCOVERY_CATEGORIES,
        DiscoveryHarvester,
        DiscoveryRequest,
        discovery_categories_per_run,
        rotate_categories,
    )

    today = datetime.now(tz=UTC).date()
    categories = rotate_categories(
        tuple(DISCOVERY_CATEGORIES),
        today.toordinal(),
        discovery_categories_per_run(),
    )
    harvester = DiscoveryHarvester(
        db_client=DatabaseClient(),
        amazon=AmazonSuggestionClient(),
    )
    results, failed = harvester.harvest_and_store(DiscoveryRequest(categories=categories))

    return {
        "status": "failed" if failed and not results else ("partial" if failed else "success"),
        "results_count": len(results),
        "failed_sources": failed,
        "categories": list(categories),
    }


_CURATION_MAX_CHAIN = int(os.environ.get("LLM_CURATION_MAX_CHAIN", "10"))
_CURATION_CHAIN_DELAY = int(os.environ.get("LLM_CURATION_CHAIN_DELAY", "10"))


def _daily_curation_capacity() -> int:
    """Compute how many candidates the normal schedule can process in 24h."""
    from store_listing.orchestration.seed_curation import MAX_CANDIDATES

    cron_expr = os.environ.get("LLM_CURATION_SCHEDULE", "50 2,6,10,14,18,22 * * *")
    hour_field = cron_expr.split()[1] if len(cron_expr.split()) >= 2 else "*"

    if hour_field == "*":
        runs_per_day = 24
    elif "/" in hour_field:
        step = int(hour_field.split("/")[1])
        runs_per_day = 24 // step
    else:
        runs_per_day = len(hour_field.split(","))

    return runs_per_day * MAX_CANDIDATES


@celery_app.task
def curate_seeds_task(chain_depth: int = 0) -> dict:
    """LLM-driven seed curation with automatic catch-up chaining."""
    from store_listing.orchestration.llm_client import (
        configured,
        get_llm_client,
        get_llm_curation_model,
    )
    from store_listing.orchestration.seed_curation import curate_seeds

    if not configured():
        return {"status": "skipped", "reason": "LLM_API_KEY not set"}

    client = get_llm_client()
    model = get_llm_curation_model()
    _check_llm_model(client, model)
    result = curate_seeds(DatabaseClient(), client.chat.completions, model)
    result["llm_model"] = model

    remaining = result.get("remaining", 0)
    daily_capacity = _daily_curation_capacity()
    should_chain = (
        remaining > daily_capacity
        and chain_depth < _CURATION_MAX_CHAIN
        and result.get("status") == "success"
    )
    if should_chain:
        curate_seeds_task.apply_async(args=[chain_depth + 1], countdown=_CURATION_CHAIN_DELAY)  # type: ignore[attr-defined]
        result["chained"] = True
        result["chain_depth"] = chain_depth

    return result


@celery_app.task
def create_custom_brief_task(
    description: str, product_type: str, personal_use: bool = False
) -> dict:
    """Create a design brief from a user's free-text description, then generate its image."""
    from store_listing.orchestration.custom_brief import create_custom_brief
    from store_listing.orchestration.llm_client import configured, get_llm_client, get_llm_model

    if not configured():
        return {"status": "skipped", "reason": "LLM_API_KEY not set"}

    client = get_llm_client()
    model = get_llm_model()
    _check_llm_model(client, model)
    result = create_custom_brief(
        DatabaseClient(),
        client.chat.completions,
        model,
        description,
        product_type,
        personal_use=personal_use,
        enabled=enabled_product_types(DatabaseClient()),
    )
    result["llm_model"] = model

    if result.get("status") == "success" and result.get("brief_id"):
        generate_single_brief_image_task.apply_async(  # type: ignore[attr-defined]
            args=[result["brief_id"]], countdown=5
        )
        result["image_generation"] = "queued"

    return result


@celery_app.task
def generate_design_briefs_task() -> dict:
    """LLM-driven design brief generation from curated trend signals."""
    from store_listing.orchestration.design_briefs import generate_design_briefs
    from store_listing.orchestration.llm_client import configured, get_llm_client, get_llm_model

    if not configured():
        return {"status": "skipped", "reason": "LLM_API_KEY not set"}

    client = get_llm_client()
    model = get_llm_model()
    _check_llm_model(client, model)
    enabled = enabled_product_types(DatabaseClient())
    logger.info("Product types in production: %s", ", ".join(enabled))
    result = generate_design_briefs(
        DatabaseClient(), client.chat.completions, model, enabled=enabled
    )
    result["llm_model"] = model
    result["product_types"] = enabled
    return result


@celery_app.task
def rerank_design_briefs_task() -> dict:
    """Re-evaluate design brief confidence against current trend signals."""
    from store_listing.orchestration.brief_reranking import rerank_briefs
    from store_listing.orchestration.llm_client import (
        configured,
        get_llm_client,
        get_llm_curation_model,
    )

    if not configured():
        return {"status": "skipped", "reason": "LLM_API_KEY not set"}

    client = get_llm_client()
    model = get_llm_curation_model()
    _check_llm_model(client, model)
    result = rerank_briefs(DatabaseClient(), client.chat.completions, model)
    result["llm_model"] = model
    return result


@celery_app.task
def replan_design_briefs_task(batch_id: str, feedback: str | None = None) -> dict:
    """Rewrite the text plans of an existing batch before re-rendering it."""
    from store_listing.orchestration.design_briefs import replan_batch
    from store_listing.orchestration.llm_client import configured, get_llm_client, get_llm_model

    if not configured():
        return {"status": "skipped", "reason": "LLM_API_KEY not set"}

    client = get_llm_client()
    model = get_llm_model()
    _check_llm_model(client, model)
    result = replan_batch(
        client.chat.completions,
        model,
        DatabaseClient(),
        batch_id,
        [feedback] if feedback else None,
    )
    result["llm_model"] = model
    return result


@celery_app.task
def generate_design_images_task() -> dict:
    """Generate images for design briefs via ComfyUI on the DGX, store in S3, update DB."""
    import boto3

    from store_listing.orchestration.comfyui_client import ComfyUIClient
    from store_listing.orchestration.image_pipeline import generate_images_for_briefs
    from store_listing.orchestration.image_storage import S3_BUCKET, S3_ENDPOINT_URL
    from store_listing.orchestration.llm_client import get_vision_client, get_vision_model
    from store_listing.orchestration.text_validation import TextValidator

    s3_kwargs: dict[str, str] = {}
    if S3_ENDPOINT_URL:
        s3_kwargs["endpoint_url"] = S3_ENDPOINT_URL
    s3_client = boto3.client("s3", **s3_kwargs)

    comfyui_client = ComfyUIClient(
        base_url=os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188"),
        timeout_seconds=float(os.environ.get("COMFYUI_TIMEOUT_SECONDS", "300")),
        poll_interval_seconds=float(os.environ.get("COMFYUI_POLL_INTERVAL_SECONDS", "1.0")),
        unet_name=os.environ.get("COMFYUI_UNET_NAME", "flux1-schnell.safetensors"),
        clip_name1=os.environ.get("COMFYUI_CLIP_NAME1", "t5xxl_fp8_e4m3fn.safetensors"),
        clip_name2=os.environ.get("COMFYUI_CLIP_NAME2", "clip_l.safetensors"),
        vae_name=os.environ.get("COMFYUI_VAE_NAME", "ae.safetensors"),
        bg_model_name=os.environ.get("COMFYUI_BG_MODEL_NAME") or None,
        alpha_black_point=int(os.environ.get("COMFYUI_BG_ALPHA_BLACK", "24")),
        alpha_white_point=int(os.environ.get("COMFYUI_BG_ALPHA_WHITE", "240")),
        alpha_erode_pixels=int(os.environ.get("COMFYUI_BG_ERODE_PIXELS", "1")),
        steps=_comfyui_steps(),
        cfg=_comfyui_cfg(),
    )
    text_validator = TextValidator(
        client=get_vision_client(),
        model=get_vision_model(),
        threshold=float(os.environ.get("TEXT_MATCH_THRESHOLD", "0.85")),
        require_tagline=_require_tagline_text(),
    )

    return generate_images_for_briefs(
        db_client=DatabaseClient(),
        comfyui_client=comfyui_client,
        s3_client=s3_client,
        bucket=S3_BUCKET,
        endpoint_url=S3_ENDPOINT_URL,
        text_validator=text_validator,
        artwork_validator=_build_artwork_validator(),
        artwork_text_validator=_build_artwork_text_validator(),
        hero_validator=_build_hero_validator(),
        product_types=enabled_product_types(DatabaseClient()),
    )


@celery_app.task
def generate_single_brief_image_task(brief_id: int) -> dict:
    """Generate image for a single design brief — triggered by requestRegeneration."""
    import boto3

    from store_listing.orchestration.comfyui_client import ComfyUIClient
    from store_listing.orchestration.image_pipeline import generate_image_for_brief
    from store_listing.orchestration.image_storage import S3_BUCKET, S3_ENDPOINT_URL
    from store_listing.orchestration.llm_client import get_vision_client, get_vision_model
    from store_listing.orchestration.text_validation import TextValidator

    s3_kwargs: dict[str, str] = {}
    if S3_ENDPOINT_URL:
        s3_kwargs["endpoint_url"] = S3_ENDPOINT_URL
    s3_client = boto3.client("s3", **s3_kwargs)

    comfyui_client = ComfyUIClient(
        base_url=os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188"),
        timeout_seconds=float(os.environ.get("COMFYUI_TIMEOUT_SECONDS", "300")),
        poll_interval_seconds=float(os.environ.get("COMFYUI_POLL_INTERVAL_SECONDS", "1.0")),
        unet_name=os.environ.get("COMFYUI_UNET_NAME", "flux1-schnell.safetensors"),
        clip_name1=os.environ.get("COMFYUI_CLIP_NAME1", "t5xxl_fp8_e4m3fn.safetensors"),
        clip_name2=os.environ.get("COMFYUI_CLIP_NAME2", "clip_l.safetensors"),
        vae_name=os.environ.get("COMFYUI_VAE_NAME", "ae.safetensors"),
        bg_model_name=os.environ.get("COMFYUI_BG_MODEL_NAME") or None,
        alpha_black_point=int(os.environ.get("COMFYUI_BG_ALPHA_BLACK", "24")),
        alpha_white_point=int(os.environ.get("COMFYUI_BG_ALPHA_WHITE", "240")),
        alpha_erode_pixels=int(os.environ.get("COMFYUI_BG_ERODE_PIXELS", "1")),
        steps=_comfyui_steps(),
        cfg=_comfyui_cfg(),
    )
    text_validator = TextValidator(
        client=get_vision_client(),
        model=get_vision_model(),
        threshold=float(os.environ.get("TEXT_MATCH_THRESHOLD", "0.85")),
        require_tagline=_require_tagline_text(),
    )

    return generate_image_for_brief(
        brief_id=brief_id,
        db_client=DatabaseClient(),
        comfyui_client=comfyui_client,
        s3_client=s3_client,
        bucket=S3_BUCKET,
        endpoint_url=S3_ENDPOINT_URL,
        text_validator=text_validator,
        artwork_validator=_build_artwork_validator(),
        artwork_text_validator=_build_artwork_text_validator(),
        hero_validator=_build_hero_validator(),
    )
