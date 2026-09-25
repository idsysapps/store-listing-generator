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


def test_harvest_and_store_writes_video_rows() -> None:
    gateway = FakeYouTubeGateway(results_by_query={"gym fitness": [SHORT_A, SHORT_B]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["gym fitness"])

    results, failed = client.harvest_and_store(request)

    video_results = [r for r in results if r.query_type == "video"]
    assert len(video_results) >= 2
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


def test_harvest_skips_seed_title() -> None:
    gateway = FakeYouTubeGateway(results_by_query={"funny t-shirt": [SHORT_C]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["funny t-shirt"])

    results, _ = client.harvest_and_store(request)

    video_results = [r for r in results if r.query_type == "video"]
    titles = [r.query for r in video_results]
    assert "funny t-shirt" not in titles


def test_harvest_aggregates_tag_views() -> None:
    gateway = FakeYouTubeGateway(results_by_query={"gym": [SHORT_A, SHORT_B]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["gym"])

    results, _ = client.harvest_and_store(request)

    tag_results = {r.query: r for r in results if r.query_type == "hashtag"}
    assert "humor" in tag_results
    assert tag_results["humor"].score == SHORT_B.view_count
    assert tag_results["humor"].delta == 1


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


def test_video_result_score_is_view_count() -> None:
    gateway = FakeYouTubeGateway(results_by_query={"gym": [SHORT_A]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["gym"])

    results, _ = client.harvest_and_store(request)

    video_results = [r for r in results if r.query_type == "video"]
    assert video_results[0].score == 250000
    assert video_results[0].delta == 12000
