"""Pinterest micro-trend harvester.

Scrapes pin saves/traction and boards under target seed searches. Uses an Apify
actor when APIFY_API_TOKEN is set; otherwise falls back to a free web scrape of
the search-pins page (best-effort). Discoveries are written to the unified
trend store (trend_queries + trend_scores with ``source='pinterest'``) and fed
to the seed pool via ``seed_candidates``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

import httpx

from store_listing.orchestration.promotion import compute_promotion_score

from .google_trends import DatabaseClient
from .schemas import TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

DEFAULT_PINTEREST_ACTOR: Final[str] = "epctex/pinterest-scraper"

_SCRIPT_RE = re.compile(r'<script[^>]*id="__PWS_DATA__"[^>]*>(.*?)</script>', re.DOTALL)

_DEFAULT_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


@dataclass(frozen=True)
class Pin:
    id: str
    title: str
    description: str
    board: str
    repins: int
    favorites: int
    comments: int


@runtime_checkable
class PinterestScraperGateway(Protocol):
    def scrape_search(self, query: str) -> list[Pin]: ...


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _pin(item: dict[str, Any]) -> Pin:
    board = item.get("board")
    if isinstance(board, dict):
        board_name = str(board.get("name") or "")
    else:
        board_name = str(board or item.get("board_name") or "")
    return Pin(
        id=str(item.get("id") or ""),
        title=str(item.get("title") or ""),
        description=str(item.get("description") or ""),
        board=board_name,
        repins=int(item.get("repin_count") or item.get("repins_count") or 0),
        favorites=int(item.get("favorite_count") or 0),
        comments=int(item.get("comment_count") or 0),
    )


class ApifyPinterestGateway:
    """Apify actor wrapper (epctex/pinterest-scraper)."""

    def __init__(
        self,
        token: str,
        actor_id: str = DEFAULT_PINTEREST_ACTOR,
        apify_client_cls: Callable[[str], Any] | None = None,
    ) -> None:
        self.token = token
        self.actor_id = actor_id
        self._apify_client_cls = apify_client_cls
        self._client: Any | None = None

    def _apify(self) -> Any:
        if self._client is None:
            self._client = self._resolve_client_class()(self.token)
        return self._client

    def _resolve_client_class(self) -> Any:
        if self._apify_client_cls is not None:
            return self._apify_client_cls
        from apify_client import ApifyClient  # type: ignore[import-not-found]

        return ApifyClient

    def scrape_search(self, query: str) -> list[Pin]:
        run = self._apify().actor(self.actor_id).call(run_input={"query": query})
        raw_items = self._apify().dataset(run["defaultDatasetId"]).list_items().items
        return [_pin(item) for item in raw_items]


class PinterestWebGateway:
    """Free fallback: parse the search-pins page's embedded state (best-effort)."""

    BASE_URL = "https://www.pinterest.com/search/pins/?q={query}"

    def scrape_search(self, query: str) -> list[Pin]:
        url = self.BASE_URL.format(query=query.replace(" ", "%20"))
        response = httpx.get(
            url,
            headers=_DEFAULT_HEADERS,
            timeout=20,
            follow_redirects=True,
        )
        response.raise_for_status()
        return self._parse_html(response.text)

    @staticmethod
    def _parse_html(html: str) -> list[Pin]:
        match = _SCRIPT_RE.search(html)
        if match is None:
            return []
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        entries = [
            item
            for item in _iter_dicts(data)
            if isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and item.get("title")
            and (
                isinstance(item.get("repin_count"), (int, float))
                or isinstance(item.get("repins_count"), (int, float))
            )
        ]
        return [_pin(item) for item in entries]


class PinterestClient:
    """Two-step contract: raw rows to trend store, discoveries to seed pool."""

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        gateway: PinterestScraperGateway | None = None,
    ) -> None:
        self.db_client = db_client or DatabaseClient()
        self.gateway = gateway or self._build_gateway()

    @staticmethod
    def _build_gateway() -> PinterestScraperGateway:
        token = os.environ.get("APIFY_API_TOKEN")
        if token:
            actor_id = os.environ.get("APIFY_ACTOR_PINTEREST") or DEFAULT_PINTEREST_ACTOR
            return ApifyPinterestGateway(token=token, actor_id=actor_id)
        return PinterestWebGateway()

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        all_results: list[TrendResult] = []
        failed: list[str] = []

        for seed in request.seed_keywords:
            try:
                pins = self.gateway.scrape_search(seed)
                results = self._derive_results(seed, pins)
                for result in results:
                    self._store(seed, result)
                    all_results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to scrape Pinterest search %s: %s", seed, e)
                failed.append(seed)

        return all_results, failed

    def _store(self, seed: str, result: TrendResult) -> None:
        query_id = self.db_client.insert_trend_query(seed, result.query, source="pinterest")
        self.db_client.insert_trend_score(
            query_id=query_id,
            score=result.score,
            delta=result.delta,
            region=result.region,
            query_type=result.query_type,
            source="pinterest",
            trend_direction=result.trend_direction,
        )
        self._upsert_candidate(seed, result)

    @staticmethod
    def _derive_results(seed: str, pins: list[Pin]) -> list[TrendResult]:
        timestamp = datetime.now(UTC)
        results: list[TrendResult] = []
        seen_titles: set[str] = set()
        board_repins: dict[str, int] = {}
        board_count: dict[str, int] = {}

        for pin in pins:
            title = (pin.title or pin.description).strip()
            if title and title.lower() != seed.lower() and title not in seen_titles:
                seen_titles.add(title)
                results.append(
                    TrendResult(
                        query=title,
                        score=pin.repins,
                        delta=pin.favorites + pin.comments,
                        region="US",
                        query_type="search",
                        source="pinterest",
                        fetched_at=timestamp,
                    )
                )
            if pin.board:
                board_repins[pin.board] = board_repins.get(pin.board, 0) + pin.repins
                board_count[pin.board] = board_count.get(pin.board, 0) + 1

        for board in sorted(board_repins):
            results.append(
                TrendResult(
                    query=board,
                    score=board_repins[board],
                    delta=board_count[board],
                    region="US",
                    query_type="board",
                    source="pinterest",
                    fetched_at=timestamp,
                )
            )
        return results

    def _upsert_candidate(self, seed: str, result: TrendResult) -> None:
        if result.query_type not in ("search", "board"):
            return
        if result.query.lower() == seed.lower():
            return
        promotion_score = compute_promotion_score(
            result.source, result.query_type, result.score, result.delta
        )
        self.db_client.upsert_seed_candidate(
            source_seed=seed,
            query=result.query,
            source=result.source,
            query_type=result.query_type,
            score=result.score,
            delta=result.delta,
            promotion_score=promotion_score,
        )
