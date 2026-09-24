import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol, TypeVar, cast, runtime_checkable

import pandas as pd
import psycopg2
import requests
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq

from .schemas import QueryType, TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class TrendRequestGateway(Protocol):
    cookies: dict[str, str]

    def build_payload(self, keywords: list[str], timeframe: str, geo: str) -> None: ...

    def interest_by_region(self) -> pd.DataFrame: ...

    def related_queries(self) -> dict[str, Any]: ...

    def interest_over_time(self) -> pd.DataFrame: ...

    def GetGoogleCookie(self) -> dict[str, str]: ...


class DatabaseClient:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.connection_params = {
            "host": host or os.environ.get("DATABASE_HOST") or "localhost",
            "port": port if port is not None else int(os.environ.get("DATABASE_PORT") or 5432),
            "database": database or os.environ.get("DATABASE_NAME") or "store_listing",
            "user": user if user is not None else os.environ.get("DATABASE_USER"),
            "password": password if password is not None else os.environ.get("DATABASE_PASSWORD"),
        }

    def connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(**self.connection_params)

    def insert_trend_query(self, seed_keyword: str, query: str) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trend_queries (seed_keyword, query)
                VALUES (%s, %s)
                ON CONFLICT (seed_keyword, query) DO UPDATE SET created_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (seed_keyword, query),
            )
            result = cur.fetchone()
            if result is None:
                msg = "Failed to insert trend query"
                raise RuntimeError(msg)
            return result[0]

    def insert_trend_score(
        self,
        query_id: int,
        score: int,
        delta: int,
        region: str,
        query_type: QueryType = "interest",
        trend_direction: int | None = None,
    ) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trend_scores (query_id, score, delta, region, query_type, trend_direction)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (query_id, score, delta, region, query_type, trend_direction),
            )
            result = cur.fetchone()
            if result is None:
                msg = "Failed to insert trend score"
                raise RuntimeError(msg)
            return result[0]


class GoogleTrendsClient:
    DEFAULT_SEEDS: ClassVar[list[str]] = [
        "funny t-shirt",
        "hoodie",
        "gift",
        "mom humor",
        "gym fitness",
    ]

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        trend_req: TrendRequestGateway | None = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        request_timeout: tuple[int, int] = (5, 30),
        consent_max_retries: int = 2,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.request_timeout = request_timeout
        self.consent_max_retries = consent_max_retries
        self.pytrends: TrendRequestGateway = trend_req or cast(
            TrendRequestGateway,
            TrendReq(
                hl="en-US",
                tz=360,
                timeout=request_timeout,
                retries=max_retries,
                backoff_factor=int(backoff_factor),
            ),
        )
        self.db_client = db_client or DatabaseClient()

    def refresh_cookies(self) -> None:
        """Google's consent/NID cookie expires; fetch a fresh one and retry."""
        self.pytrends.cookies.update(self.pytrends.GetGoogleCookie())

    def _call_with_retry(self, func: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(self.consent_max_retries + 1):
            try:
                return func()
            except ResponseError as e:
                status = getattr(e.response, "status_code", None)
                if status == 404 and attempt < self.consent_max_retries:
                    self.refresh_cookies()
                    continue
                raise
            except requests.RequestException as e:
                if attempt >= self.consent_max_retries:
                    raise
                self.refresh_cookies()
                last_error = e
        assert last_error is not None
        raise last_error

    def _build_payload(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> dict[str, Any]:
        self._call_with_retry(
            lambda: self.pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
        )
        return {
            "interest_by_region": self._call_with_retry(self.pytrends.interest_by_region),
            "related_queries": self._fetch_related_queries(),
            "interest_over_time": self._fetch_interest_over_time(),
        }

    def _fetch_related_queries(self) -> dict[str, Any]:
        """Best-effort: never abort the seed if Google fails a related-queries call."""
        try:
            related = self._call_with_retry(self.pytrends.related_queries)
            return related if isinstance(related, dict) else {}
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch related queries (ignored): %s", e)
            return {}

    def _fetch_interest_over_time(self) -> pd.DataFrame:
        """Best-effort: never abort the seed if Google fails an interest-over-time call."""
        try:
            data = self._call_with_retry(self.pytrends.interest_over_time)
            return data if isinstance(data, pd.DataFrame) else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch interest over time (ignored): %s", e)
            return pd.DataFrame()

    def _calculate_delta(self, current_data: pd.DataFrame, keyword: str) -> int:
        if current_data.empty or keyword not in current_data.columns:
            return 0
        values = current_data[keyword].dropna()
        if len(values) < 2:
            return 0
        latest = values.iloc[-1]
        previous = values.iloc[-2]
        if previous == 0:
            return 100 if latest > 0 else 0
        return int(((latest - previous) / previous) * 100)

    def fetch_trending(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> list[TrendResult]:
        return self._regional_results(self._build_payload(keyword, timeframe, geo), keyword)

    def _regional_results(self, payload: dict[str, Any], keyword: str) -> list[TrendResult]:
        region_data = payload.get("interest_by_region")
        if not isinstance(region_data, pd.DataFrame):
            region_data = pd.DataFrame()

        results = []
        for region, row in region_data.iterrows():
            value = row.get(keyword)
            if value is not None and pd.notna(value) and value > 0:
                score = int(value)
                delta = self._calculate_delta(region_data, keyword)
                results.append(
                    TrendResult(
                        query=keyword,
                        score=score,
                        delta=delta,
                        region=region if isinstance(region, str) else str(region),
                        query_type="interest",
                        fetched_at=datetime.now(UTC),
                    )
                )
        return results

    @staticmethod
    def _parse_rise_pct(value: object) -> int:
        """Parse pytrends rising value strings like "+250000%"/"Breakout" into an int delta."""
        if value is None:
            return 0
        text = str(value).strip()
        if text.lower() == "breakout":
            return 5000
        digits = re.sub(r"[^0-9.]", "", text)
        try:
            return int(float(digits or 0))
        except ValueError:
            return 0

    def fetch_related_queries(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> list[TrendResult]:
        """Rising + top related queries for a seed, tagged by type/region."""
        return self._related_results(self._build_payload(keyword, timeframe, geo), keyword, geo)

    def _related_results(
        self, payload: dict[str, Any], keyword: str, geo: str
    ) -> list[TrendResult]:
        related = payload.get("related_queries") or {}
        entry = related.get(keyword)
        if not isinstance(entry, dict):
            return []

        rising = entry.get("rising")
        top = entry.get("top")

        results: list[TrendResult] = []
        if isinstance(rising, pd.DataFrame) and not rising.empty:
            for _, row in rising.iterrows():
                results.append(
                    TrendResult(
                        query=str(row.get("query", keyword)),
                        score=0,
                        delta=self._parse_rise_pct(row.get("value")),
                        region=geo,
                        query_type="rising",
                        fetched_at=datetime.now(UTC),
                    )
                )
        if isinstance(top, pd.DataFrame) and not top.empty:
            for _, row in top.iterrows():
                results.append(
                    TrendResult(
                        query=str(row.get("query", keyword)),
                        score=int(self._parse_rise_pct(row.get("value"))),
                        delta=0,
                        region="GLOBAL",
                        query_type="top",
                        fetched_at=datetime.now(UTC),
                    )
                )
        return results

    def fetch_global_trend(
        self, keyword: str, timeframe: str = "today 3-m", geo: str = "US"
    ) -> list[TrendResult]:
        """Interest-over-time trajectory for a seed; one GLOBAL result with direction."""
        return self._global_results(self._build_payload(keyword, timeframe, geo), keyword)

    def _global_results(self, payload: dict[str, Any], keyword: str) -> list[TrendResult]:
        over_time = payload.get("interest_over_time")
        if not isinstance(over_time, pd.DataFrame):
            over_time = pd.DataFrame()
        if over_time.empty or keyword not in over_time.columns:
            return []

        values = over_time[keyword].dropna()
        if values.empty:
            return []

        latest = int(values.iloc[-1])
        delta = self._calculate_delta(over_time, keyword)
        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
        return [
            TrendResult(
                query=keyword,
                score=latest,
                delta=delta,
                region="GLOBAL",
                query_type="top",
                trend_direction=direction,
                fetched_at=datetime.now(UTC),
            )
        ]

    def _harvest_seed(self, keyword: str, timeframe: str, geo: str) -> list[TrendResult]:
        """Regional interest + related rising/top + global trend for one seed."""
        payload = self._build_payload(keyword, timeframe, geo)
        return (
            self._regional_results(payload, keyword)
            + self._related_results(payload, keyword, geo)
            + self._global_results(payload, keyword)
        )

    def harvest_and_store(
        self, request: TrendHarvestRequest
    ) -> tuple[list[TrendResult], list[str]]:
        """Harvest requested seeds; return (stored results, failed seed keywords)."""
        all_results: list[TrendResult] = []
        failed_seeds: list[str] = []

        for seed in request.seed_keywords:
            try:
                results = self._harvest_seed(seed, request.timeframe, request.region)

                for result in results:
                    query_id = self.db_client.insert_trend_query(seed, result.query)
                    self.db_client.insert_trend_score(
                        query_id=query_id,
                        score=result.score,
                        delta=result.delta,
                        region=result.region,
                        query_type=result.query_type,
                        trend_direction=result.trend_direction,
                    )
                    all_results.append(result)

            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to fetch trends for seed %s: %s", seed, e)
                failed_seeds.append(seed)

        return all_results, failed_seeds
