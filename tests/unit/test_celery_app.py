import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

from store_listing.orchestration.promotion import ActiveSeed, SeedCandidate


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


def test_fetch_daily_trends_harvests_active_seeds(redis_url: str) -> None:
    """Daily harvest must feed the unified pool from active_seeds, not hardcoded seeds."""
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="hoodie", promotion_score=0),
        ActiveSeed(id=2, query="gift", promotion_score=0),
    ]
    trends_mock = MagicMock()
    trends_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock) as db_patch,
        patch.object(module, "GoogleTrendsClient", return_value=trends_mock) as trends_patch,
    ):
        result = module.fetch_daily_trends.run()

    request = trends_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == ["hoodie", "gift"]
    assert result["status"] == "success"
    assert db_patch.call_count == 1
    assert trends_patch.call_count == 1


def test_fetch_daily_trends_falls_back_to_starter_seeds(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = []
    trends_mock = MagicMock()
    trends_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "GoogleTrendsClient", return_value=trends_mock),
    ):
        module.fetch_daily_trends.run()

    request = trends_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == module.STARTER_SEEDS
    assert len(request.seed_keywords) > 0


def test_promote_seeds_promotes_only_qualifying_candidates(redis_url: str) -> None:
    """Black-box: only google candidates meeting rules are promoted; others untouched."""
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    pending = [
        SeedCandidate(
            id=11,
            query="mom shirt",
            source="google",
            query_type="rising",
            score=0,
            delta=12000,
            promotion_score=12000,
            status="pending",
        ),
        SeedCandidate(
            id=12,
            query="pickleball",
            source="google",
            query_type="top",
            score=80,
            delta=0,
            promotion_score=80,
            status="pending",
        ),
        SeedCandidate(
            id=13,
            query="tiktok boom",
            source="tiktok",
            query_type="hashtag",
            score=90,
            delta=0,
            promotion_score=0,
            status="pending",
        ),
    ]
    db_mock = MagicMock()
    db_mock.list_pending_candidates.return_value = pending
    db_mock.cross_seed_counts.return_value = {"pickleball": 2}
    db_mock.list_active_seeds.return_value = []

    with patch.object(module, "DatabaseClient", return_value=db_mock):
        result = module.promote_seeds.run()

    assert result["promoted"] == 2
    assert result["archived"] == 0
    promoted_ids = {call.args[0] for call in db_mock.promote_candidate.call_args_list}
    assert promoted_ids == {11, 12}
    assert db_mock.insert_active_seed.call_count == 2


def test_promote_seeds_returns_zero_when_nothing_qualifies(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    db_mock = MagicMock()
    db_mock.list_pending_candidates.return_value = []
    db_mock.cross_seed_counts.return_value = {}

    with patch.object(module, "DatabaseClient", return_value=db_mock):
        result = module.promote_seeds.run()

    assert result == {"status": "success", "promoted": 0, "archived": 0}


def test_beat_schedule_registers_harvest_and_promotion(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    beat = module.celery_app.conf.beat_schedule
    assert (
        beat["daily-trend-harvest"]["task"]
        == "store_listing.orchestration.tasks.fetch_daily_trends"
    )
    assert beat["seed-promotion"]["task"] == "store_listing.orchestration.tasks.promote_seeds"


def _reload(redis_url: str):
    with patch.dict(os.environ, {"REDIS_URL": redis_url}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        return importlib.reload(module)


def test_fetch_tiktok_trends_harvests_active_seeds(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="mom humor", promotion_score=0),
        ActiveSeed(id=2, query="pickleball", promotion_score=0),
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "TikTokClient", return_value=client_mock),
    ):
        result = module.fetch_tiktok_trends.run()

    request = client_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == ["mom humor", "pickleball"]
    assert result["status"] == "success"


def test_fetch_tiktok_trends_partial_when_some_fail(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="pickleball", promotion_score=0)
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], ["#shirttok"])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "TikTokClient", return_value=client_mock),
    ):
        result = module.fetch_tiktok_trends.run()

    assert result["status"] == "partial"


def test_fetch_pinterest_trends_harvests_active_seeds(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="pickleball", promotion_score=0)
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "PinterestClient", return_value=client_mock),
    ):
        result = module.fetch_pinterest_trends.run()

    request = client_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == ["pickleball"]
    assert result["status"] == "success"


def test_fetch_marketplace_suggestions_runs_both_sources(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="mom humor", promotion_score=0),
        ActiveSeed(id=2, query="gift", promotion_score=0),
    ]
    harvester_mock = MagicMock()
    harvester_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "AutocompleteHarvester", return_value=harvester_mock),
    ):
        result = module.fetch_marketplace_suggestions.run()

    request = harvester_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == ["mom humor", "gift"]
    assert result["status"] == "success"


def test_fetch_x_trends_harvests_active_seeds(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="mom humor", promotion_score=0),
        ActiveSeed(id=2, query="pickleball", promotion_score=0),
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "XClient", return_value=client_mock),
    ):
        result = module.fetch_x_trends.run()

    request = client_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == ["mom humor", "pickleball"]
    assert result["status"] == "success"


def test_fetch_x_trends_partial_when_some_fail(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="mom humor", promotion_score=0)
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], ["mom humor"])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "XClient", return_value=client_mock),
    ):
        result = module.fetch_x_trends.run()

    assert result["status"] == "partial"


def test_beat_schedule_registers_micro_trend_tasks_without_tiktok_by_default(
    redis_url: str,
) -> None:
    module = _reload(redis_url)
    beat = module.celery_app.conf.beat_schedule

    assert "tiktok-micro-trends" not in beat
    assert "instagram-micro-trends" not in beat
    assert (
        beat["pinterest-micro-trends"]["task"]
        == "store_listing.orchestration.tasks.fetch_pinterest_trends"
    )
    assert (
        beat["marketplace-suggestions"]["task"]
        == "store_listing.orchestration.tasks.fetch_marketplace_suggestions"
    )
    assert beat["x-micro-trends"]["task"] == "store_listing.orchestration.tasks.fetch_x_trends"
    assert (
        beat["reddit-micro-trends"]["task"]
        == "store_listing.orchestration.tasks.fetch_reddit_trends"
    )
    assert (
        beat["youtube-micro-trends"]["task"]
        == "store_listing.orchestration.tasks.fetch_youtube_trends"
    )


def test_beat_schedule_tiktok_registered_when_enabled(redis_url: str) -> None:
    """TikTok micro-trend harvest is opt-in (TIKTOK_ENABLED=true); when enabled
    it runs weekly (Sunday) to cap Apify spend."""
    with patch.dict(os.environ, {"REDIS_URL": redis_url, "TIKTOK_ENABLED": "true"}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    beat = module.celery_app.conf.beat_schedule
    assert (
        beat["tiktok-micro-trends"]["task"]
        == "store_listing.orchestration.tasks.fetch_tiktok_trends"
    )
    schedule = beat["tiktok-micro-trends"]["schedule"]
    assert schedule.minute == {10}
    assert schedule.hour == {6}
    assert schedule.day_of_week == {0}


def test_beat_schedule_tiktok_omitted_when_flag_false(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url, "TIKTOK_ENABLED": "false"}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    assert "tiktok-micro-trends" not in module.celery_app.conf.beat_schedule


def test_beat_schedule_pinterest_runs_twice_weekly(redis_url: str) -> None:
    """Pinterest reduced to 2x/week (Mon+Thu) to cap Apify spend."""
    module = _reload(redis_url)
    schedule = module.celery_app.conf.beat_schedule["pinterest-micro-trends"]["schedule"]
    assert schedule.day_of_week == {0, 3}


def test_beat_schedule_x_runs_twice_weekly(redis_url: str) -> None:
    """X reduced to 2x/week (Tue+Fri) to cap Apify spend."""
    module = _reload(redis_url)
    schedule = module.celery_app.conf.beat_schedule["x-micro-trends"]["schedule"]
    assert schedule.day_of_week == {1, 4}


def test_beat_schedule_reddit_runs_daily(redis_url: str) -> None:
    module = _reload(redis_url)
    schedule = module.celery_app.conf.beat_schedule["reddit-micro-trends"]["schedule"]
    assert schedule.minute == {30}
    assert schedule.hour == {6}


def test_beat_schedule_youtube_runs_daily(redis_url: str) -> None:
    module = _reload(redis_url)
    schedule = module.celery_app.conf.beat_schedule["youtube-micro-trends"]["schedule"]
    assert schedule.minute == {35}
    assert schedule.hour == {6}


def test_beat_schedule_instagram_registered_when_enabled(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url, "INSTAGRAM_ENABLED": "true"}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    beat = module.celery_app.conf.beat_schedule
    assert (
        beat["instagram-micro-trends"]["task"]
        == "store_listing.orchestration.tasks.fetch_instagram_trends"
    )
    assert beat["instagram-micro-trends"]["schedule"].day_of_week == {5}


def test_beat_schedule_instagram_omitted_when_flag_false(redis_url: str) -> None:
    with patch.dict(os.environ, {"REDIS_URL": redis_url, "INSTAGRAM_ENABLED": "false"}):
        module = importlib.import_module("store_listing.orchestration.tasks")
        importlib.reload(module)

    assert "instagram-micro-trends" not in module.celery_app.conf.beat_schedule


def test_fetch_reddit_trends_harvests_active_seeds(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="mom humor", promotion_score=0),
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "RedditClient", return_value=client_mock),
    ):
        result = module.fetch_reddit_trends.run()

    request = client_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == ["mom humor"]
    assert result["status"] == "success"


def test_fetch_youtube_trends_harvests_active_seeds(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="gym fitness", promotion_score=0),
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "YouTubeClient", return_value=client_mock),
    ):
        result = module.fetch_youtube_trends.run()

    request = client_mock.harvest_and_store.call_args.args[0]
    assert request.seed_keywords == ["gym fitness"]
    assert result["status"] == "success"


class FakeHealthTracker:
    def __init__(self) -> None:
        self.outcomes: list[tuple[str, str, str | None]] = []

    def record(self, source: str, outcome: str, *, error: str | None = None) -> None:
        self.outcomes.append((source, outcome, error))


def _reload_with_health(redis_url: str):
    with patch.dict(os.environ, {"REDIS_URL": redis_url, "SOURCE_HEALTH_ENABLED": "true"}):
        module = importlib.import_module("store_listing.orchestration.tasks")
    return importlib.reload(module)


def test_health_reports_failure_when_harvest_raises(redis_url: str) -> None:
    module = _reload_with_health(redis_url)
    tracker = FakeHealthTracker()
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = []

    with (
        patch("store_listing.orchestration.source_health.build_tracker", return_value=tracker),
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(
            module.GoogleTrendsClient, "harvest_and_store", side_effect=RuntimeError("boom")
        ),
        pytest.raises(RuntimeError),
    ):
        module.fetch_daily_trends.run()

    assert tracker.outcomes[0] == ("google", "failure", "boom")


def test_health_reports_empty_when_no_results(redis_url: str) -> None:
    module = _reload_with_health(redis_url)
    tracker = FakeHealthTracker()
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = []
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([], [])

    with (
        patch("store_listing.orchestration.source_health.build_tracker", return_value=tracker),
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "GoogleTrendsClient", return_value=client_mock),
    ):
        module.fetch_daily_trends.run()

    assert ("google", "empty", None) in tracker.outcomes


def test_health_reports_success_when_results_present(redis_url: str) -> None:
    module = _reload_with_health(redis_url)
    tracker = FakeHealthTracker()
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = []
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch("store_listing.orchestration.source_health.build_tracker", return_value=tracker),
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "GoogleTrendsClient", return_value=client_mock),
    ):
        module.fetch_daily_trends.run()

    assert ("google", "success", None) in tracker.outcomes


def test_health_tracks_tiktok_source_by_tag(redis_url: str) -> None:
    module = _reload_with_health(redis_url)
    tracker = FakeHealthTracker()
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = [
        ActiveSeed(id=1, query="pickleball", promotion_score=0)
    ]
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([], ["#pickleball"])

    with (
        patch("store_listing.orchestration.source_health.build_tracker", return_value=tracker),
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "TikTokClient", return_value=client_mock),
    ):
        result = module.fetch_tiktok_trends.run()

    assert result["status"] == "failed"
    assert tracker.outcomes[0][0] == "tiktok"
    assert tracker.outcomes[0][1] == "failure"


def test_health_disabled_runs_task_without_tracking(redis_url: str) -> None:
    module = _reload(redis_url)
    db_mock = MagicMock()
    db_mock.list_active_seeds.return_value = []
    client_mock = MagicMock()
    client_mock.harvest_and_store.return_value = ([object()], [])

    with (
        patch("store_listing.orchestration.source_health.build_tracker", return_value=None),
        patch.object(module, "DatabaseClient", return_value=db_mock),
        patch.object(module, "GoogleTrendsClient", return_value=client_mock),
    ):
        result = module.fetch_daily_trends.run()

    assert result["status"] == "success"
