"""Amazon + Etsy search autocomplete harvesters (free, no Apify involved).

Parse each marketplace's public search-autocomplete endpoint for real-time buyer
queries. Suggestions are written to the unified trend store (trend_queries +
trend_scores) tagged ``source='amazon'`` / ``source='etsy'`` and fed to the seed
pool via ``seed_candidates``.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import httpx

from store_listing.orchestration.promotion import compute_promotion_score

from .google_trends import DatabaseClient
from .schemas import Source, TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

AMAZON_SUGGEST_URL: Final[str] = "https://completion.amazon.com/api/2017/suggestions"

DEFAULT_ETSY_SUGGEST_URL: Final[str] = "https://www.etsy.com/autosuggest"

ETSY_SUGGEST_URL: Final[str] = os.environ.get("ETSY_SUGGEST_URL") or DEFAULT_ETSY_SUGGEST_URL


class SuggestionSource(Protocol):
    def suggest(self, prefix: str) -> list[str]: ...


class AmazonSuggestionClient:
    """Amazon completion API: returns real-time buyer queries for a prefix."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._http = client or httpx.Client(headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=10)

    def suggest(self, prefix: str, alias: str = "aps") -> list[str]:
        response = self._http.get(
            AMAZON_SUGGEST_URL,
            params={
                "limit": 11,
                "prefix": prefix,
                "suggestion-type": "WIDGET",
                "alias": alias,
            },
        )
        response.raise_for_status()
        payload = response.json()
        suggestions = payload.get("suggestions") if isinstance(payload, dict) else None
        if not isinstance(suggestions, list):
            return []
        return [
            str(entry["value"]).strip()
            for entry in suggestions
            if isinstance(entry, dict) and entry.get("value")
        ]


class EtsySuggestionClient:
    """Etsy site search-autocomplete endpoint (URL overridable via ETSY_SUGGEST_URL)."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        endpoint: str | None = None,
    ) -> None:
        self._http = client or httpx.Client(headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=10)
        self._endpoint = endpoint or ETSY_SUGGEST_URL or DEFAULT_ETSY_SUGGEST_URL

    def suggest(self, prefix: str) -> list[str]:
        response = self._http.get(self._endpoint, params={"search_query": prefix})
        response.raise_for_status()
        return _extract_etsy_suggestions(response.json())


def _extract_etsy_suggestions(payload: Any) -> list[str]:
    """Defensive extraction across the Etsy autosuggest response shapes."""
    if isinstance(payload, dict):
        body = payload.get("result") or payload.get("results") or payload
        entries = body if isinstance(body, list) else []
    elif isinstance(payload, list):
        entries = []
        for entry in payload:
            if isinstance(entry, list):
                entries.extend(entry)
            elif isinstance(entry, dict):
                entries.append(entry)
    else:
        entries = []

    suggestions: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value") or entry.get("display") or entry.get("text") or entry.get("name")
        if value:
            suggestions.append(str(value).strip())
    return suggestions


class AutocompleteHarvester:
    """Two-step contract for Amazon + Etsy: raw rows to trend store, discoveries to seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        amazon: SuggestionSource | None = None,
        etsy: SuggestionSource | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.amazon = amazon or AmazonSuggestionClient()
        self.etsy = etsy or EtsySuggestionClient()

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        all_results: list[TrendResult] = []
        failed: list[str] = []

        for seed in request.seed_keywords:
            sources: tuple[tuple[Source, SuggestionSource], ...] = (
                ("amazon", self.amazon),
                ("etsy", self.etsy),
            )
            for source_name, source in sources:
                try:
                    suggestions = source.suggest(seed)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to fetch %s suggestions for %s: %s", source_name, seed, e
                    )
                    failed.append(f"{source_name}:{seed}")
                    continue

                for suggestion in suggestions:
                    query = suggestion.strip().lower()
                    if not query or query == seed.lower():
                        continue
                    result = self._store(seed, source_name, query)
                    all_results.append(result)

        return all_results, failed

    def _store(self, seed: str, source_name: Source, query: str) -> TrendResult:
        result = TrendResult(
            query=query,
            score=1,
            delta=0,
            region="US",
            query_type="search",
            source=source_name,
            fetched_at=datetime.now(UTC),
        )
        query_id = self.db_client.insert_trend_query(seed, result.query, source=source_name)
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source=source_name,
            trend_direction=result.trend_direction,
        )
        promotion_score = compute_promotion_score(source_name, "search", 1, 0)
        self.db_client.upsert_seed_candidate(
            source_seed=seed,
            query=result.query,
            source=source_name,
            query_type="search",
            score=1,
            delta=0,
            promotion_score=promotion_score,
        )
        return result
