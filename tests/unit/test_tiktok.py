"""Unit tests for the TikTok micro-trend harvester (Apify + free web fallback)."""

import os
from unittest.mock import MagicMock, patch

from store_listing.ingest.trends.schemas import TrendHarvestRequest, TrendResult
from store_listing.ingest.trends.tiktok import (
    DEFAULT_TIKTOK_ACTOR,
    TARGET_HASHTAGS,
    ApifyTikTokGateway,
    TikTokClient,
    TikTokItem,
    TikTokWebGateway,
)

SAMPLE_HTML = """<!doctype html><html><head><script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">{
  "state": {
    "LoaderInitialData": {
      "videos": [
        {"id": "7123", "desc": "pickleball gifts for mom #pickleballgift #momhumor",
         "stats": {"playCount": 100000, "diggCount": 500, "shareCount": 10, "commentCount": 20},
         "musicMeta": {"musicName": "sunset tones"},
         "hashtags": [{"name": "pickleballgift"}, {"name": "momhumor"}]},
        {"id": "7124", "desc": "rush pickleball gear #pickleballgear",
         "stats": {"playCount": 80000, "diggCount": 300, "shareCount": 5, "commentCount": 8},
         "musicMeta": {"musicName": "sunset tones"},
         "hashtags": [{"name": "pickleballgear"}]}
      ]
    }
  }
}</script></head><body></body></html>
"""

ITEM_A = TikTokItem(
    video_id="7123",
    text="pickleball gifts for mom",
    music="sunset tones",
    music_author="composer",
    play_count=100000,
    digg_count=500,
    share_count=10,
    comment_count=20,
    hashtags=("pickleballgift", "momhumor"),
)

ITEM_B = TikTokItem(
    video_id="7124",
    text="rush pickleball gear",
    music="sunset tones",
    music_author="composer",
    play_count=80000,
    digg_count=300,
    share_count=5,
    comment_count=8,
    hashtags=("pickleballgear",),
)


class FakeTikTokGateway:
    def __init__(self, items_by_hashtag=None, failures=()) -> None:
        self.items_by_hashtag = dict(items_by_hashtag or {})
        self.failures = set(failures)
        self.calls: list[str] = []

    def scrape_hashtag(self, hashtag: str) -> list[TikTokItem]:
        self.calls.append(hashtag)
        if hashtag in self.failures:
            raise RuntimeError(f"scrape failed: {hashtag}")
        return self.items_by_hashtag.get(hashtag, [])


def make_client(gateway: FakeTikTokGateway) -> TikTokClient:
    return TikTokClient(db_client=MagicMock(), gateway=gateway)


def test_target_hashtags_normalize_seeds_and_include_issue_targets() -> None:
    client = make_client(FakeTikTokGateway())
    tags = client.target_hashtags(["Mom Humor", "gym fitness", "pickleball"])

    assert tags == ["#momhumor", "#gymfitness", "#pickleball", "#shirttok", "#gymhumor"]
    assert all(tag in tags for tag in TARGET_HASHTAGS)
    assert len(tags) == len(set(tags))


def test_harvest_and_store_writes_hashtag_sound_rows() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    client = TikTokClient(
        db_client=db,
        gateway=FakeTikTokGateway(items_by_hashtag={"#pickleball": [ITEM_A, ITEM_B]}),
    )

    results, failed = client.harvest_and_store(TrendHarvestRequest(seed_keywords=["pickleball"]))

    assert failed == []
    hashtags = [r.query for r in results if r.query_type == "hashtag"]
    sounds = [r for r in results if r.query_type == "sound"]

    assert "#pickleballgift" in hashtags
    assert "#pickleballgear" in hashtags
    assert len(sounds) == 1
    assert sounds[0].query == "sunset tones"
    assert sounds[0].score == 180000
    assert sounds[0].delta == 2

    for result in results:
        assert result.source == "tiktok"
        assert result.region == "US"
        assert isinstance(result, TrendResult)

    db.insert_trend_query.assert_called()
    assert db.insert_trend_query.call_args.kwargs["source"] == "tiktok"
    db.upsert_seed_candidate.assert_called()


def test_harvest_skips_seed_hashtag_and_stores_candidates() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    item_self_tag = TikTokItem(
        video_id="999",
        text="pickleball edit",
        music="track",
        music_author="artist",
        play_count=5000,
        digg_count=10,
        share_count=1,
        comment_count=2,
        hashtags=("pickleball", "newpickleballthing"),
    )
    client = TikTokClient(
        db_client=db,
        gateway=FakeTikTokGateway(items_by_hashtag={"#pickleball": [item_self_tag]}),
    )

    results, failed = client.harvest_and_store(TrendHarvestRequest(seed_keywords=["pickleball"]))

    assert failed == []
    queries = [r.query for r in results]
    assert "#pickleball" not in queries
    assert "#newpickleballthing" in queries

    upsert_calls = db.upsert_seed_candidate.call_args_list
    for call in upsert_calls:
        assert call.kwargs["source"] == "tiktok"
        assert call.kwargs["query"] != "#pickleball"


def test_per_seed_failure_is_captured_and_harvest_continues() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    gateway = FakeTikTokGateway(
        items_by_hashtag={"#pickleball": [ITEM_A]},
        failures={"#shirttok"},
    )
    client = TikTokClient(db_client=db, gateway=gateway)

    results, failed = client.harvest_and_store(
        TrendHarvestRequest(seed_keywords=["pickleball", "shirt"])
    )

    assert failed == ["#shirttok"]
    assert len(results) > 0


def test_apify_gateway_selected_when_token_present() -> None:
    with patch.dict(os.environ, {"APIFY_API_TOKEN": "tok123"}):
        client = TikTokClient(db_client=MagicMock())

    assert isinstance(client.gateway, ApifyTikTokGateway)
    assert client.gateway.token == "tok123"
    assert client.gateway.actor_id == DEFAULT_TIKTOK_ACTOR


def test_web_gateway_selected_when_no_token() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("APIFY_API_TOKEN", None)
        client = TikTokClient(db_client=MagicMock())

    assert isinstance(client.gateway, TikTokWebGateway)


def test_apify_gateway_maps_actor_output() -> None:
    apify_client_cls = MagicMock()
    apify_client_cls.return_value.actor.return_value.call.return_value = {
        "defaultDatasetId": "ds-1"
    }
    apify_client_cls.return_value.dataset.return_value.list_items.return_value.items = [
        {
            "text": "pickleball gifts for mom",
            "stats": {"playCount": 100000, "diggCount": 500, "shareCount": 10, "commentCount": 20},
            "musicMeta": {"musicName": "sunset tones", "musicAuthor": "composer"},
            "hashtags": [{"name": "pickleballgift"}],
        }
    ]
    gateway = ApifyTikTokGateway(token="tok123", apify_client_cls=apify_client_cls)
    items = gateway.scrape_hashtag("#pickleball")

    assert len(items) == 1
    assert items[0].hashtags == ("pickleballgift",)
    assert items[0].play_count == 100000
    apify_client_cls.assert_called_once_with("tok123")
    call_kwargs = apify_client_cls.return_value.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["hashtags"] == ["pickleball"]


def test_web_gateway_parses_rehydrated_json() -> None:
    items = TikTokWebGateway()._parse_html(SAMPLE_HTML)

    assert len(items) == 2
    assert items[0].music == "sunset tones"
    assert items[0].play_count == 100000
    assert items[0].hashtags == ("pickleballgift", "momhumor")


def test_web_gateway_parses_empty_html() -> None:
    assert TikTokWebGateway()._parse_html("<html><body></body></html>") == []
