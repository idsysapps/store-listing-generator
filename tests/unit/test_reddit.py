"""Unit tests for the Reddit micro-trend harvester."""

from unittest.mock import MagicMock

from store_listing.ingest.trends.reddit import (
    POD_SUBREDDITS,
    RedditClient,
    RedditPost,
)
from store_listing.ingest.trends.schemas import TrendHarvestRequest


POST_A = RedditPost(
    post_id="abc123",
    title="This oddly specific shirt is perfect",
    subreddit="oddlyspecific",
    score=5200,
    num_comments=142,
    upvote_ratio=0.96,
)

POST_B = RedditPost(
    post_id="def456",
    title="Gym bros will understand",
    subreddit="funny",
    score=12300,
    num_comments=430,
    upvote_ratio=0.94,
)

POST_C = RedditPost(
    post_id="ghi789",
    title="funny t-shirt",
    subreddit="TargetedShirts",
    score=800,
    num_comments=22,
    upvote_ratio=0.91,
)


class FakeRedditGateway:
    def __init__(self, search_results=None, sub_results=None, failures=()):
        self.search_results = dict(search_results or {})
        self.sub_results = dict(sub_results or {})
        self.failures = set(failures)
        self.search_calls: list[str] = []
        self.sub_calls: list[str] = []

    def search(self, query: str) -> list[RedditPost]:
        self.search_calls.append(query)
        if query in self.failures:
            raise RuntimeError(f"search failed: {query}")
        return self.search_results.get(query, [])

    def subreddit_hot(self, subreddit: str) -> list[RedditPost]:
        self.sub_calls.append(subreddit)
        if subreddit in self.failures:
            raise RuntimeError(f"sub failed: {subreddit}")
        return self.sub_results.get(subreddit, [])


def make_client(gateway: FakeRedditGateway) -> RedditClient:
    return RedditClient(db_client=MagicMock(), gateway=gateway)


def test_harvest_and_store_writes_search_rows() -> None:
    gateway = FakeRedditGateway(search_results={"funny t-shirt": [POST_A, POST_B]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["funny t-shirt"])

    results, failed = client.harvest_and_store(request)

    search_results = [r for r in results if r.query_type == "search"]
    assert len(search_results) >= 2
    assert all(r.source == "reddit" for r in results)
    assert not failed


def test_harvest_skips_seed_title() -> None:
    gateway = FakeRedditGateway(search_results={"funny t-shirt": [POST_C]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["funny t-shirt"])

    results, _ = client.harvest_and_store(request)

    search_results = [r for r in results if r.query_type == "search"]
    titles = [r.query for r in search_results]
    assert "funny t-shirt" not in titles


def test_harvest_derives_subreddit_results() -> None:
    gateway = FakeRedditGateway(search_results={"funny t-shirt": [POST_A, POST_B]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["funny t-shirt"])

    results, _ = client.harvest_and_store(request)

    sub_results = [r for r in results if r.query_type == "subreddit"]
    sub_queries = [r.query for r in sub_results]
    assert "r/oddlyspecific" in sub_queries
    assert "r/funny" in sub_queries


def test_harvest_scrapes_pod_subreddits() -> None:
    gateway = FakeRedditGateway(sub_results={"funny": [POST_B]})
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=[])

    results, _ = client.harvest_and_store(request)

    assert gateway.sub_calls == list(POD_SUBREDDITS)
    sub_search = [r for r in results if r.query_type == "search"]
    assert any(r.query == "Gym bros will understand" for r in sub_search)


def test_per_seed_failure_is_captured_and_harvest_continues() -> None:
    gateway = FakeRedditGateway(
        search_results={"hoodie": [POST_A]},
        failures={"funny t-shirt"},
    )
    client = make_client(gateway)
    request = TrendHarvestRequest(seed_keywords=["funny t-shirt", "hoodie"])

    results, failed = client.harvest_and_store(request)

    assert "funny t-shirt" in failed
    assert len(results) > 0


def test_store_calls_db_client() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    gateway = FakeRedditGateway(search_results={"hoodie": [POST_A]})
    client = RedditClient(db_client=db, gateway=gateway)
    request = TrendHarvestRequest(seed_keywords=["hoodie"])

    client.harvest_and_store(request)

    assert db.insert_trend_query.called
    assert db.insert_trend_score.called
    assert db.upsert_seed_candidate.called


def test_upsert_candidate_uses_correct_source() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    gateway = FakeRedditGateway(search_results={"hoodie": [POST_A]})
    client = RedditClient(db_client=db, gateway=gateway)
    request = TrendHarvestRequest(seed_keywords=["hoodie"])

    client.harvest_and_store(request)

    call_kwargs = db.upsert_seed_candidate.call_args
    assert call_kwargs.kwargs["source"] == "reddit"
