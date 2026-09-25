"""Unit tests for the Amazon + Etsy search autocomplete harvesters."""

from unittest.mock import MagicMock

import httpx

from store_listing.ingest.trends.autocomplete import (
    AmazonSuggestionClient,
    AutocompleteHarvester,
    EtsySuggestionClient,
)
from store_listing.ingest.trends.schemas import TrendHarvestRequest, TrendResult


def _transport(payload) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=payload, request=request)

    return httpx.MockTransport(handler), captured


def _amazon_client(payload) -> tuple[AmazonSuggestionClient, list[httpx.Request]]:
    transport, captured = _transport(payload)
    return AmazonSuggestionClient(client=httpx.Client(transport=transport)), captured


def _etsy_client(payload) -> tuple[EtsySuggestionClient, list[httpx.Request]]:
    transport, captured = _transport(payload)
    return EtsySuggestionClient(
        client=httpx.Client(transport=transport), api_key="test-key"
    ), captured


def test_amazon_suggest_parses_suggestions() -> None:
    client, captured = _amazon_client(
        {"suggestions": [{"value": "mom shirt"}, {"value": "mom shirt svg"}]}
    )

    assert client.suggest("mom sho") == ["mom shirt", "mom shirt svg"]

    request = captured[0]
    assert "completion.amazon.com" in str(request.url)
    assert request.url.params["prefix"] == "mom sho"
    assert request.url.params["alias"] == "aps"
    assert request.url.params["mid"] == "ATVPDKIKX0DER"


def test_amazon_suggest_ignores_empty_values() -> None:
    client, _ = _amazon_client({"suggestions": [{"value": ""}, {"value": "hoodie"}]})
    assert client.suggest("hoodie") == ["hoodie"]


def test_etsy_suggest_parses_values() -> None:
    client, captured = _etsy_client(
        {
            "count": 2,
            "results": [
                {"title": "Pickleball Mug", "listing_id": 1},
                {"title": "Pickleball Mom Shirt", "listing_id": 2},
            ],
        }
    )

    assert client.suggest("pickleball") == ["pickleball mug", "pickleball mom shirt"]

    request = captured[0]
    assert "openapi.etsy.com" in str(request.url)
    assert request.url.params["keywords"] == "pickleball"
    assert request.headers["x-api-key"] == "test-key"


def test_harvester_writes_amazon_and_etsy_discoveries() -> None:
    class AmazonFake:
        def suggest(self, prefix: str) -> list[str]:
            return ["mom shirt", "mom shirt svg"]

    class EtsyFake:
        def suggest(self, prefix: str) -> list[str]:
            return ["pickleball mom shirt"]  # Etsy API returns lowercased titles

    db = MagicMock()
    db.insert_trend_query.return_value = 1
    harvester = AutocompleteHarvester(
        db_client=db,
        amazon=AmazonFake(),
        etsy=EtsyFake(),
    )

    results, failed = harvester.harvest_and_store(
        TrendHarvestRequest(seed_keywords=["mom humor", "pickleball"])
    )

    assert failed == []
    sources = {r.source for r in results}
    assert sources == {"amazon", "etsy"}
    amazon_results = [r for r in results if r.source == "amazon"]
    assert amazon_results[0].query_type == "search"
    assert amazon_results[0].score == 1

    assert len(db.insert_trend_query.call_args_list) == len(results)
    kwargs_sources = {call.kwargs["source"] for call in db.insert_trend_query.call_args_list}
    assert kwargs_sources == {"amazon", "etsy"}
    db.upsert_seed_candidate.assert_called()


def test_harvester_skips_suggestion_equal_to_seed() -> None:
    db = MagicMock()
    db.insert_trend_query.return_value = 1

    class Echo:
        def suggest(self, prefix: str) -> list[str]:
            return [prefix, f"{prefix} svg"]

    harvester = AutocompleteHarvester(
        db_client=db,
        amazon=Echo(),
        etsy=Echo(),
    )

    results, failed = harvester.harvest_and_store(TrendHarvestRequest(seed_keywords=["mom humor"]))

    assert failed == []
    assert "mom humor" not in [r.query for r in results]
    assert "mom humor svg" in [r.query for r in results]


def test_harvester_captures_per_source_failures() -> None:
    class Boom:
        def suggest(self, prefix: str) -> list[str]:
            raise RuntimeError("boom")

    class Fine:
        def suggest(self, prefix: str) -> list[str]:
            return ["mom shirt svg"]

    db = MagicMock()
    db.insert_trend_query.return_value = 1
    harvester = AutocompleteHarvester(
        db_client=db,
        amazon=Boom(),
        etsy=Fine(),
    )

    results, failed = harvester.harvest_and_store(TrendHarvestRequest(seed_keywords=["mom humor"]))

    assert failed == ["amazon:mom humor"]
    assert all(r.source == "etsy" for r in results)
    assert isinstance(results[0], TrendResult)
