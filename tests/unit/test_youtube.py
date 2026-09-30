"""Unit tests for the YouTube Shorts micro-trend harvester."""

from unittest.mock import MagicMock

from store_listing.ingest.trends.schemas import TrendHarvestRequest
from store_listing.ingest.trends.youtube import (
    YouTubeClient,
    YouTubeShort,
)

SHORT_A = YouTubeShort(
    video_id="abc123",
    title="Funniest gym fails compilation",
    tags=("gym", "fitness", "funny"),
    view_count=250000,
    like_count=12000,
    comment_count=890,
    channel_title="FitLaughs",
)

SHORT_B = YouTubeShort(
    video_id="def456",
    title="Mom humor that hits different",
    tags=("mom", "humor", "relatable"),
    view_count=180000,
    like_count=9500,
    comment_count=620,
    channel_title="MomVibes",
)

SHORT_C = YouTubeShort(
    video_id="ghi789",
    title="funny t-shirt",
    tags=(),
    view_count=500,
    like_count=10,
    comment_count=1,
    channel_title="SmallChannel",
)

SHORT_HIGH_VIEWS = YouTubeShort(
    video_id="big001",
    title="Viral trending video",
    tags=("trending", "viral"),
    view_count=5_000_000_000,
    like_count=100000,
    comment_count=50000,
    channel_title="BigChannel",
)

SHORT_HIGH_VIEWS_2 = YouTubeShort(
    video_id="big002",
    title="Another viral video",
    tags=("trending", "funny"),
    view_count=3_000_000_000,
    like_count=80000,
    comment_count=30000,
    channel_title="BigChannel2",
)


class FakeYouTubeGateway:
    def __init__(self, results_by_query=None, failures=()):
        self.results_by_query = dict(results_by_query or {})
        self.failures = set(failures)
        self.calls: list[str] = []

    def search_shorts(self, query: str) -> list[YouTubeShort]:
        self.calls.append(query)
        if query in self.failures:
            raise RuntimeError(f"search failed: {query}")
        return self.results_by_query.get(query, [])


def make_client(gateway: FakeYouTubeGateway) -> YouTubeClient:
    return YouTubeClient(db_client=MagicMock(), gateway=gateway)


def test_harvest_only_returns_tag_results_not_video_titles() -> None:
    gateway = FakeYouTubeGateway(results_by_query={"gym fitness": [SHORT_A, SHORT_B]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["gym fitness"])

    results, failed = client.harvest_and_store(request)

    video_results = [r for r in results if r.query_type == "video"]
    assert len(video_results) == 0
    tag_results = [r for r in results if r.query_type == "hashtag"]
    assert len(tag_results) > 0
    assert all(r.source == "youtube" for r in results)
    assert not failed


def test_harvest_derives_hashtag_results_from_tags() -> None:
    gateway = FakeYouTubeGateway(results_by_query={"gym fitness": [SHORT_A, SHORT_B]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["gym fitness"])

    results, _ = client.harvest_and_store(request)

    tag_results = [r for r in results if r.query_type == "hashtag"]
    tag_queries = [r.query for r in tag_results]
    assert "funny" in tag_queries
    assert "humor" in tag_queries


def test_harvest_skips_seed_as_tag() -> None:
    short_with_seed_tag = YouTubeShort(
        video_id="xyz",
        title="Some video",
        tags=("gym",),
        view_count=1000,
        like_count=10,
        comment_count=1,
        channel_title="Ch",
    )
    gateway = FakeYouTubeGateway(results_by_query={"gym": [short_with_seed_tag]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["gym"])

    results, _ = client.harvest_and_store(request)

    tag_queries = [r.query for r in results if r.query_type == "hashtag"]
    assert "gym" not in tag_queries


def test_no_results_from_video_with_no_tags() -> None:
    gateway = FakeYouTubeGateway(results_by_query={"funny t-shirt": [SHORT_C]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["funny t-shirt"])

    results, _ = client.harvest_and_store(request)

    assert len(results) == 0


def test_tag_score_uses_frequency_weighted_engagement() -> None:
    gateway = FakeYouTubeGateway(
        results_by_query={"trending": [SHORT_HIGH_VIEWS, SHORT_HIGH_VIEWS_2]}
    )
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["trending"])

    results, _ = client.harvest_and_store(request)

    tag_results = {r.query: r for r in results if r.query_type == "hashtag"}
    # "trending" is skipped (matches seed)
    # "viral" appears in 1 video with 5B views: score = max(1, min(10000, 1 * 5000)) = 5000
    assert "viral" in tag_results
    assert tag_results["viral"].score == 5000
    assert tag_results["viral"].delta == 1
    # "funny" appears in 1 video with 3B views: score = max(1, min(10000, 1 * 3000)) = 3000
    assert "funny" in tag_results
    assert tag_results["funny"].score == 3000
    assert tag_results["funny"].delta == 1


def test_tag_score_capped_at_10000() -> None:
    big_short_1 = YouTubeShort(
        video_id="cap1",
        title="V1",
        tags=("mega",),
        view_count=10_000_000_000,
        like_count=0,
        comment_count=0,
        channel_title="C",
    )
    big_short_2 = YouTubeShort(
        video_id="cap2",
        title="V2",
        tags=("mega",),
        view_count=10_000_000_000,
        like_count=0,
        comment_count=0,
        channel_title="C",
    )
    gateway = FakeYouTubeGateway(results_by_query={"test": [big_short_1, big_short_2]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["test"])

    results, _ = client.harvest_and_store(request)

    tag_results = {r.query: r for r in results if r.query_type == "hashtag"}
    # 2 videos * (20B total views / 1M) = 2 * 20000 = 40000, capped at 10000
    assert tag_results["mega"].score == 10000


def test_tag_score_floor_is_tag_count() -> None:
    low_views_short = YouTubeShort(
        video_id="low1",
        title="Low views",
        tags=("niche",),
        view_count=500,
        like_count=1,
        comment_count=0,
        channel_title="C",
    )
    gateway = FakeYouTubeGateway(results_by_query={"test": [low_views_short]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["test"])

    results, _ = client.harvest_and_store(request)

    tag_results = {r.query: r for r in results if r.query_type == "hashtag"}
    # 1 * (500 / 1M) = 0, floored to tag_count = 1
    assert tag_results["niche"].score == 1
    assert tag_results["niche"].delta == 1


def test_per_seed_failure_is_captured_and_harvest_continues() -> None:
    gateway = FakeYouTubeGateway(
        results_by_query={"hoodie": [SHORT_A]},
        failures={"gym fitness"},
    )
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["gym fitness", "hoodie"])

    results, failed = client.harvest_and_store(request)

    assert "gym fitness" in failed
    assert len(results) > 0


def test_store_calls_db_client() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    gateway = FakeYouTubeGateway(results_by_query={"hoodie": [SHORT_A]})
    client = YouTubeClient(db_client=db, gateway=gateway)
    request = TrendHarvestRequest(seed_keywords=["hoodie"])

    client.harvest_and_store(request)

    assert db.insert_trend_query.called
    assert db.insert_trend_score.called
    assert db.upsert_seed_candidate.called


def test_upsert_candidate_uses_correct_source() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    gateway = FakeYouTubeGateway(results_by_query={"hoodie": [SHORT_A]})
    client = YouTubeClient(db_client=db, gateway=gateway)
    request = TrendHarvestRequest(seed_keywords=["hoodie"])

    client.harvest_and_store(request)

    call_kwargs = db.upsert_seed_candidate.call_args
    assert call_kwargs.kwargs["source"] == "youtube"
