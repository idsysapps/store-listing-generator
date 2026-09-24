import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from pytrends.exceptions import ResponseError

from store_listing.ingest.trends.google_trends import DatabaseClient, GoogleTrendsClient
from store_listing.ingest.trends.schemas import TrendHarvestRequest, TrendResult


class FakeTrendReq:
    def __init__(self, errors=None) -> None:
        self.errors = list(errors or [])
        self.build_payload_calls = 0
        self.interest_by_region_calls = 0
        self.cookie_fetches = 0
        self.cookies = {}

    def build_payload(self, *args, **kwargs) -> None:
        self.build_payload_calls += 1
        if self.errors:
            raise self.errors.pop(0)

    def interest_by_region(self) -> pd.DataFrame:
        self.interest_by_region_calls += 1
        return pd.DataFrame({"funny t-shirt": [45]}, index=["US"])

    def trending_searches(self) -> pd.DataFrame:
        return pd.DataFrame()

    def GetGoogleCookie(self) -> dict[str, str]:
        self.cookie_fetches += 1
        return self.cookies


def response_404() -> ResponseError:
    class _Resp:
        status_code = 404

    return ResponseError.from_response(_Resp())


class TestGoogleTrendsClient:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:
        return MagicMock(spec=DatabaseClient)

    @pytest.fixture
    def client(self, mock_db_client: MagicMock) -> GoogleTrendsClient:
        return GoogleTrendsClient(
            db_client=mock_db_client,
            trend_req=FakeTrendReq(),
            consent_max_retries=2,
        )

    def test_fetch_trending_returns_trend_results(self, client: GoogleTrendsClient) -> None:
        mock_region_data = pd.DataFrame(
            {
                "funny t-shirt": [45, 50, 55],
            },
            index=["US", "CA", "UK"],
        )

        with patch.object(client, "_build_payload") as mock_payload:
            mock_payload.return_value = {
                "interest_by_region": mock_region_data,
                "trending_searches": pd.DataFrame(),
            }

            results = client.fetch_trending("funny t-shirt")

            assert len(results) > 0
            assert all(isinstance(r, TrendResult) for r in results)

    def test_fetch_trending_handles_empty_data(self, client: GoogleTrendsClient) -> None:
        empty_df = pd.DataFrame()

        with patch.object(client, "_build_payload") as mock_payload:
            mock_payload.return_value = {
                "interest_by_region": empty_df,
                "trending_searches": pd.DataFrame(),
            }

            results = client.fetch_trending("nonexistent keyword")
            assert results == []

    def test_calculate_delta_returns_percentage(self, client: GoogleTrendsClient) -> None:
        data = pd.DataFrame(
            {
                "test": [100, 150, 200],
            }
        )

        delta = client._calculate_delta(data, "test")
        assert delta == 33

    def test_calculate_delta_handles_zero_previous(self, client: GoogleTrendsClient) -> None:
        data = pd.DataFrame(
            {
                "test": [0, 50, 100],
            }
        )

        delta = client._calculate_delta(data, "test")
        assert delta == 100

    def test_harvest_and_store_inserts_data(
        self, client: GoogleTrendsClient, mock_db_client: MagicMock
    ) -> None:
        mock_db_client.insert_trend_query.return_value = 1
        mock_db_client.insert_trend_score.return_value = 1

        mock_region_data = pd.DataFrame(
            {
                "hoodie": [30],
            },
            index=["US"],
        )

        with patch.object(client, "_build_payload") as mock_payload:
            mock_payload.return_value = {
                "interest_by_region": mock_region_data,
                "trending_searches": pd.DataFrame(),
            }

            request = TrendHarvestRequest(seed_keywords=["hoodie"])
            results, failed_seeds = client.harvest_and_store(request)

            assert len(results) > 0
            assert failed_seeds == []
            mock_db_client.insert_trend_query.assert_called()
            mock_db_client.insert_trend_score.assert_called()


class TestDatabaseClient:
    def test_connect_creates_connection(self) -> None:
        with patch("store_listing.ingest.trends.google_trends.psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            db_client = DatabaseClient(host="localhost", database="test")
            db_client.connect()

            mock_connect.assert_called_once_with(
                host="localhost",
                port=5432,
                database="test",
                user=None,
                password=None,
            )

    def test_connect_uses_env_when_no_args_passed(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "DATABASE_HOST": "store-listing-store-listing-postgres",
                    "DATABASE_PORT": "5432",
                    "DATABASE_NAME": "store_listing",
                    "DATABASE_USER": "store_listing",
                    "DATABASE_PASSWORD": "store_listing",
                },
            ),
            patch("store_listing.ingest.trends.google_trends.psycopg2.connect") as mock_connect,
        ):
            mock_connect.return_value = MagicMock()

            DatabaseClient().connect()

            mock_connect.assert_called_once_with(
                host="store-listing-store-listing-postgres",
                port=5432,
                database="store_listing",
                user="store_listing",
                password="store_listing",
            )

    def test_explicit_args_override_env(self) -> None:
        with (
            patch.dict(os.environ, {"DATABASE_HOST": "env-host"}),
            patch("store_listing.ingest.trends.google_trends.psycopg2.connect") as mock_connect,
        ):
            mock_connect.return_value = MagicMock()

            DatabaseClient(host="explicit-host").connect()

            mock_connect.assert_called_once_with(
                host="explicit-host",
                port=5432,
                database="store_listing",
                user=None,
                password=None,
            )

    def test_insert_trend_query_returns_id(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(DatabaseClient, "connect", return_value=mock_conn):
            db_client = DatabaseClient()
            result = db_client.insert_trend_query("hoodie", "funny hoodie")

            assert result == 42

    def test_insert_trend_score_returns_id(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (99,)

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(DatabaseClient, "connect", return_value=mock_conn):
            db_client = DatabaseClient()
            result = db_client.insert_trend_score(query_id=1, score=75, delta=10, region="US")

            assert result == 99


class TestPytrendsHardening:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:
        return MagicMock(spec=DatabaseClient)

    def test_constructor_passes_timeout_and_retry_to_pytrends(
        self, mock_db_client: MagicMock
    ) -> None:
        with patch("store_listing.ingest.trends.google_trends.TrendReq") as mock_trend_req:
            GoogleTrendsClient(
                db_client=mock_db_client,
                max_retries=4,
                backoff_factor=3.0,
                request_timeout=(5, 30),
            )

            mock_trend_req.assert_called_once_with(
                hl="en-US",
                tz=360,
                timeout=(5, 30),
                retries=4,
                backoff_factor=3.0,
            )

    def test_404_refreshes_cookies_then_succeeds(self, mock_db_client: MagicMock) -> None:
        fake = FakeTrendReq(errors=[response_404()])
        client = GoogleTrendsClient(
            db_client=mock_db_client,
            trend_req=fake,
            consent_max_retries=2,
        )

        results = client.fetch_trending("funny t-shirt")

        assert len(results) == 1
        assert fake.build_payload_calls == 2
        assert fake.cookie_fetches == 1

    def test_transient_network_error_refreshes_and_retries(self, mock_db_client: MagicMock) -> None:
        fake = FakeTrendReq(errors=[requests.exceptions.ConnectionError("boom")])
        client = GoogleTrendsClient(
            db_client=mock_db_client,
            trend_req=fake,
            consent_max_retries=2,
        )

        results = client.fetch_trending("funny t-shirt")

        assert len(results) == 1
        assert fake.build_payload_calls == 2
        assert fake.cookie_fetches == 1

    def test_repeated_404_exhausts_and_harvest_continues(self, mock_db_client: MagicMock) -> None:
        fake = FakeTrendReq(errors=[response_404(), response_404(), response_404()])
        client = GoogleTrendsClient(
            db_client=mock_db_client,
            trend_req=fake,
            consent_max_retries=2,
        )

        with patch("store_listing.ingest.trends.google_trends.logger.warning") as mock_warning:
            results, failed_seeds = client.harvest_and_store(
                TrendHarvestRequest(seed_keywords=["funny t-shirt"])
            )

        assert results == []
        assert failed_seeds == ["funny t-shirt"]
        assert fake.build_payload_calls == 3
        assert fake.cookie_fetches == 2
        mock_warning.assert_called_once()

    def test_trending_searches_failure_does_not_abort_harvest(
        self, mock_db_client: MagicMock
    ) -> None:
        """Trending-searches 404 (Google removed hottrends endpoint) must not kill the seed.

        Only interest_by_region feeds trends; a trending_searches failure must
        degrade to an empty frame and still store the region data.
        """

        class Trending404(FakeTrendReq):
            def trending_searches(self) -> pd.DataFrame:
                raise response_404()

        fake = Trending404()
        client = GoogleTrendsClient(
            db_client=mock_db_client,
            trend_req=fake,
            consent_max_retries=0,
        )
        mock_db_client.insert_trend_query.return_value = 1
        mock_db_client.insert_trend_score.return_value = 1

        with patch("store_listing.ingest.trends.google_trends.logger.warning") as mock_warning:
            results, failed_seeds = client.harvest_and_store(
                TrendHarvestRequest(seed_keywords=["funny t-shirt"])
            )

        assert len(results) == 1
        assert failed_seeds == []
        mock_db_client.insert_trend_query.assert_called_once()
        mock_db_client.insert_trend_score.assert_called_once()
        mock_warning.assert_called_once()
