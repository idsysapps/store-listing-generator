import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol, TypeVar, cast, runtime_checkable

import pandas as pd
import psycopg2
import requests
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq

from .schemas import TrendHarvestRequest, TrendResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class TrendRequestGateway(Protocol):
    cookies: dict[str, str]

    def build_payload(self, keywords: list[str], timeframe: str, geo: str) -> None: ...

    def interest_by_region(self) -> pd.DataFrame: ...

    def trending_searches(self) -> pd.DataFrame: ...

    def GetGoogleCookie(self) -> dict[str, str]: ...


class DatabaseClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "store_listing",
        user: str | None = None,
        password: str | None = None,
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
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

    def insert_trend_score(self, query_id: int, score: int, delta: int, region: str) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trend_scores (query_id, score, delta, region)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (query_id, score, delta, region),
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
            "trending_searches": self._call_with_retry(self.pytrends.trending_searches),
        }

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
        payload = self._build_payload(keyword, timeframe, geo)
        region_data = payload["interest_by_region"]

        results = []
        for region, row in region_data.iterrows():
            if keyword in row and pd.notna(row[keyword]) and row[keyword] > 0:
                score = int(row[keyword])
                delta = self._calculate_delta(region_data, keyword)
                results.append(
                    TrendResult(
                        query=keyword,
                        score=score,
                        delta=delta,
                        region=region if isinstance(region, str) else str(region),
                        fetched_at=datetime.now(UTC),
                    )
                )
        return results

    def harvest_and_store(self, request: TrendHarvestRequest) -> list[TrendResult]:
        all_results: list[TrendResult] = []

        for seed in request.seed_keywords:
            try:
                results = self.fetch_trending(
                    keyword=seed,
                    timeframe=request.timeframe,
                    geo=request.region,
                )

                for result in results:
                    query_id = self.db_client.insert_trend_query(
                        seed_keyword=seed,
                        query=result.query,
                    )
                    self.db_client.insert_trend_score(
                        query_id=query_id,
                        score=result.score,
                        delta=result.delta,
                        region=result.region,
                    )
                    all_results.append(result)

            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to fetch trends for seed %s: %s", seed, e)
                continue

        return all_results
