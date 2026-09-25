"""Unit tests for the Pinterest micro-trend harvester (Apify + free web fallback)."""

import os
from unittest.mock import MagicMock, patch

from store_listing.ingest.trends.pinterest import (
    DEFAULT_PINTEREST_ACTOR,
    ApifyPinterestGateway,
    Pin,
    PinterestClient,
    PinterestWebGateway,
)
from store_listing.ingest.trends.schemas import TrendHarvestRequest, TrendResult

SAMPLE_HTML = """<!doctype html><html><head><script id="__PWS_DATA__" type="application/json">{
  "props": {
    "pageProps": {
      "pins": [
        {"id": "p1", "title": "Pickleball Gifts for Mom", "description": "funny pickleball svg",
         "board": {"name": "Pickleball Gifts"}, "repin_count": 600, "favorite_count": 100, "comment_count": 20},
        {"id": "p2", "title": "Pickleball Shirt", "description": "sporty tee",
         "board": {"name": "Pickleball Gifts"}, "repin_count": 500, "favorite_count": 4, "comment_count": 1}
      ]
    }
  }
}</script></head><body></body></html>
"""

PIN_A = Pin(
    id="p1",
    title="Pickleball Gifts for Mom",
    description="funny pickleball svg",
    board="Pickleball Gifts",
    repins=600,
    favorites=100,
    comments=20,
)

PIN_B = Pin(
    id="p2",
    title="Pickleball Shirt",
    description="sporty tee",
    board="Pickleball Gifts",
    repins=500,
    favorites=4,
    comments=1,
)


class FakePinterestGateway:
    def __init__(self, pins_by_query=None, failures=()) -> None:
        self.pins_by_query = dict(pins_by_query or {})
        self.failures = set(failures)
        self.calls: list[str] = []

    def scrape_search(self, query: str) -> list[Pin]:
        self.calls.append(query)
        if query in self.failures:
            raise RuntimeError(f"scrape failed: {query}")
        return self.pins_by_query.get(query, [])


def make_client(gateway: FakePinterestGateway) -> PinterestClient:
    return PinterestClient(db_client=MagicMock(), gateway=gateway)


def test_harvest_and_store_writes_search_and_board_results() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    client = PinterestClient(
        db_client=db,
        gateway=FakePinterestGateway(pins_by_query={"pickleball": [PIN_A, PIN_B]}),
    )

    results, failed = client.harvest_and_store(TrendHarvestRequest(seed_keywords=["pickleball"]))

    assert failed == []
    search = [r for r in results if r.query_type == "search"]
    boards = [r for r in results if r.query_type == "board"]

    assert {r.query for r in search} == {"Pickleball Gifts for Mom", "Pickleball Shirt"}
    by_title = {r.query: r for r in search}
    assert by_title["Pickleball Gifts for Mom"].score == 600
    assert by_title["Pickleball Gifts for Mom"].delta == 120
    assert by_title["Pickleball Shirt"].score == 500
    assert by_title["Pickleball Shirt"].delta == 5

    assert len(boards) == 1
    assert boards[0].query == "Pickleball Gifts"
    assert boards[0].score == 1100
    assert boards[0].delta == 2

    for result in results:
        assert result.source == "pinterest"
        assert result.region == "US"
        assert isinstance(result, TrendResult)

    db.insert_trend_query.assert_called()
    assert db.insert_trend_query.call_args.kwargs["source"] == "pinterest"
    db.upsert_seed_candidate.assert_called()


def test_harvest_skips_query_equal_to_seed() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    self_title_pin = Pin(
        id="p3",
        title="pickleball",
        description="",
        board="Pickleball Fanatics",
        repins=800,
        favorites=50,
        comments=5,
    )
    client = PinterestClient(
        db_client=db,
        gateway=FakePinterestGateway(pins_by_query={"pickleball": [self_title_pin]}),
    )

    results, failed = client.harvest_and_store(TrendHarvestRequest(seed_keywords=["pickleball"]))

    assert failed == []
    assert all(r.query != "pickleball" for r in results if r.query_type == "search")
    upsert_calls = db.upsert_seed_candidate.call_args_list
    for call in upsert_calls:
        assert call.kwargs["query"].lower() != "pickleball"
        assert call.kwargs["source"] == "pinterest"


def test_per_seed_failure_is_captured_and_harvest_continues() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1
    gateway = FakePinterestGateway(
        pins_by_query={"pickleball": [PIN_A]},
        failures={"shirt"},
    )
    client = PinterestClient(db_client=db, gateway=gateway)

    results, failed = client.harvest_and_store(
        TrendHarvestRequest(seed_keywords=["pickleball", "shirt"])
    )

    assert failed == ["shirt"]
    assert len(results) > 0


def test_apify_gateway_selected_when_token_present() -> None:
    with patch.dict(os.environ, {"APIFY_API_TOKEN": "tok123"}):
        client = PinterestClient(db_client=MagicMock())

    assert isinstance(client.gateway, ApifyPinterestGateway)
    assert client.gateway.token == "tok123"
    assert client.gateway.actor_id == DEFAULT_PINTEREST_ACTOR


def test_web_gateway_selected_when_no_token() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("APIFY_API_TOKEN", None)
        client = PinterestClient(db_client=MagicMock())

    assert isinstance(client.gateway, PinterestWebGateway)


def test_apify_gateway_maps_actor_output() -> None:
    apify_client_cls = MagicMock()
    apify_client_cls.return_value.actor.return_value.call.return_value = {
        "defaultDatasetId": "ds-2"
    }
    apify_client_cls.return_value.dataset.return_value.list_items.return_value.items = [
        {
            "title": "Pickleball Gifts for Mom",
            "description": "funny pickleball svg",
            "board": {"name": "Pickleball Gifts"},
            "repins_count": 600,
            "favorite_count": 100,
            "comment_count": 20,
        }
    ]
    gateway = ApifyPinterestGateway(token="tok123", apify_client_cls=apify_client_cls)
    pins = gateway.scrape_search("pickleball")

    assert len(pins) == 1
    assert pins[0].board == "Pickleball Gifts"
    assert pins[0].repins == 600
    apify_client_cls.assert_called_once_with("tok123")
    call_kwargs = apify_client_cls.return_value.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["query"] == "pickleball"
    assert call_kwargs["run_input"]["maxPins"] == 25
    actor_id = apify_client_cls.return_value.actor.call_args.args[0]
    assert actor_id == "automation-lab~pinterest-scraper"


def test_web_gateway_parses_embedded_state() -> None:
    pins = PinterestWebGateway()._parse_html(SAMPLE_HTML)

    assert len(pins) == 2
    assert pins[0].title == "Pickleball Gifts for Mom"
    assert pins[0].repins == 600
    assert pins[1].board == "Pickleball Gifts"


def test_web_gateway_parses_empty_html() -> None:
    assert PinterestWebGateway()._parse_html("<html><body></body></html>") == []
