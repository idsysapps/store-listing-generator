"""Unit tests for the X (Twitter) micro-trend harvester (Apify + free web fallback)."""

import os
from unittest.mock import MagicMock, patch

from store_listing.ingest.trends.schemas import TrendHarvestRequest, TrendResult
from store_listing.ingest.trends.x import (
    DEFAULT_X_ACTOR,
    ApifyXGateway,
    XClient,
    XWebGateway,
)


def test_x_item_mapping_reads_hashtags_and_engagements() -> None:
    raw = {
        "text": "Pickleball gifts for mom #PickleballGift",
        "user": {"username": "pickleballmom"},
        "likeCount": 12000,
        "retweetCount": 3000,
        "replyCount": 500,
        "viewCount": 150000,
        "hashtags": [{"text": "PickleballGift"}],
    }

    from store_listing.ingest.trends.x import _x_tweet

    tweet = _x_tweet(raw)
    assert tweet.hashtags == ("pickleballgift",)
    assert tweet.like_count == 12000
    assert tweet.retweet_count == 3000
    assert tweet.reply_count == 500
    assert tweet.view_count == 150000
    assert tweet.handle == "pickleballmom"


def test_apify_gateway_maps_actor_output() -> None:
    apify_client_cls = MagicMock()
    apify_client_cls.return_value.actor.return_value.call.return_value = {
        "defaultDatasetId": "ds-x"
    }
    apify_client_cls.return_value.dataset.return_value.list_items.return_value.items = [
        {
            "fullText": "funny pickleball mom shirts #momhumor #pickleballgift",
            "user": {"username": "seed"},
            "likeCount": 100000,
            "retweetCount": 20000,
            "replyCount": 1000,
            "viewCount": 900000,
            "hashtags": [{"text": "momhumor"}, {"text": "pickleballgift"}],
        }
    ]
    gateway = ApifyXGateway(token="tok123", apify_client_cls=apify_client_cls)
    tweets = gateway.scrape_query("mom humor")

    assert len(tweets) == 1
    assert tweets[0].hashtags == ("momhumor", "pickleballgift")
    assert tweets[0].like_count == 100000
    apify_client_cls.assert_called_once_with("tok123")
    call_kwargs = apify_client_cls.return_value.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["query"] == "mom humor"


SAMPLE_HTML = """<!doctype html><html><head><script type="application/json">{
  "data": [
    {"text": "resort pickleball #pickleballgift", "likeCount": 100000, "retweetCount": 20000, "replyCount": 1000, "viewCount": 900000, "hashtags": ["pickleballgift"]},
    {"text": "gear for moms #momsvg #pickleballgift", "likeCount": 1500, "retweetCount": 200, "replyCount": 0, "viewCount": 4000, "hashtags": ["momsvg", "pickleballgift"]}
  ]
}</script></head><body></body></html>"""


def test_web_gateway_parses_embedded_state() -> None:
    tweets = XWebGateway()._parse_html(SAMPLE_HTML)

    assert len(tweets) == 2
    assert tweets[0].hashtags == ("pickleballgift",)
    assert tweets[0].like_count == 100000
    assert tweets[1].hashtags == ("momsvg", "pickleballgift")


class StubGateway:
    def __init__(self, tweets) -> None:
        self.tweets = tweets

    def scrape_query(self, query: str):
        return self.tweets


def _stub_with_sample() -> StubGateway:
    return StubGateway(XWebGateway._parse_html(SAMPLE_HTML))


def test_web_gateway_parses_empty_html() -> None:
    assert XWebGateway()._parse_html("<html><body></body></html>") == []


def test_client_builds_apify_gateway_when_token_set() -> None:
    with patch.dict(os.environ, {"APIFY_API_TOKEN": "tok123"}):
        client = XClient(db_client=MagicMock())

    assert isinstance(client.gateway, ApifyXGateway)
    assert client.gateway.token == "tok123"


def test_client_falls_back_to_web_gateway_without_token() -> None:
    with patch.dict(os.environ, {"APIFY_API_TOKEN": ""}):
        client = XClient(db_client=MagicMock())

    assert isinstance(client.gateway, XWebGateway)


def test_default_x_actor_is_configured() -> None:
    assert DEFAULT_X_ACTOR


def test_harvest_aggregates_hashtags_and_keywords() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    client = XClient(db_client=db, gateway=_stub_with_sample())

    results, failed = client.harvest_and_store(TrendHarvestRequest(seed_keywords=["mom humor"]))

    assert failed == []
    assert results
    by_query = {r.query: r for r in results}

    pickle = by_query["#pickleballgift"]
    assert pickle.score == 1026700
    assert pickle.delta == 2
    assert pickle.source == "x"
    assert pickle.query_type == "hashtag"

    moms = by_query["#momsvg"]
    assert moms.score == 5700
    assert moms.delta == 1

    keyword = by_query["pickleball"]
    assert keyword.query_type == "search"
    assert keyword.score == 1021000
    assert keyword.delta == 1

    assert "resort" in by_query
    assert "gear" in by_query

    records = [call.kwargs["source"] for call in db.insert_trend_query.call_args_list]
    assert records and set(records) == {"x"}
    db.upsert_seed_candidate.assert_called()


def test_harvest_records_failed_queries() -> None:
    class Boom:
        def scrape_query(self, query: str):
            raise RuntimeError("boom")

    db = MagicMock()
    client = XClient(db_client=db, gateway=Boom())

    results, failed = client.harvest_and_store(
        TrendHarvestRequest(seed_keywords=["mom humor", "dad jokes"])
    )

    assert results == []
    assert failed == ["mom humor", "dad jokes"]
    assert isinstance(results, list)


def test_harvest_skips_seed_words_and_own_hashtag() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    client = XClient(db_client=db, gateway=_stub_with_sample())

    results, _ = client.harvest_and_store(TrendHarvestRequest(seed_keywords=["pickleball"]))

    queries = [r.query for r in results]
    assert "pickleball" not in queries
    assert "gear" in queries
    assert all(isinstance(r, TrendResult) for r in results)
