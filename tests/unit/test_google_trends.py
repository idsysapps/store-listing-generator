from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from store_listing.ingest.trends.google_trends import DatabaseClient, GoogleTrendsClient
from store_listing.ingest.trends.schemas import TrendHarvestRequest, TrendResult


class TestGoogleTrendsClient:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:
        return MagicMock(spec=DatabaseClient)

    @pytest.fixture
    def client(self, mock_db_client: MagicMock) -> GoogleTrendsClient:
        return GoogleTrendsClient(db_client=mock_db_client)

    def test_fetch_trending_returns_trend_results(self, client: GoogleTrendsClient) -> None:
        mock_region_data = pd.DataFrame({
            "funny t-shirt": [45, 50, 55],
        }, index=["US", "CA", "UK"])

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
        data = pd.DataFrame({
            "test": [100, 150, 200],
        })

        delta = client._calculate_delta(data, "test")
        assert delta == 33

    def test_calculate_delta_handles_zero_previous(self, client: GoogleTrendsClient) -> None:
        data = pd.DataFrame({
            "test": [0, 50, 100],
        })

        delta = client._calculate_delta(data, "test")
        assert delta == 100

    def test_harvest_and_store_inserts_data(
        self, client: GoogleTrendsClient, mock_db_client: MagicMock
    ) -> None:
        mock_db_client.insert_trend_query.return_value = 1
        mock_db_client.insert_trend_score.return_value = 1

        mock_region_data = pd.DataFrame({
            "hoodie": [30],
        }, index=["US"])

        with patch.object(client, "_build_payload") as mock_payload:
            mock_payload.return_value = {
                "interest_by_region": mock_region_data,
                "trending_searches": pd.DataFrame(),
            }

            request = TrendHarvestRequest(seed_keywords=["hoodie"])
            results = client.harvest_and_store(request)

            assert len(results) > 0
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

    def test_insert_trend_query_returns_id(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(DatabaseClient, "connect", return_value=mock_conn):
            db_client = DatabaseClient()
            result = db_client.insert_trend_query("hoodie", "funny hoodie")

            assert result == 42

    def test_insert_trend_score_returns_id(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (99,)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(DatabaseClient, "connect", return_value=mock_conn):
            db_client = DatabaseClient()
            result = db_client.insert_trend_score(query_id=1, score=75, delta=10, region="US")

            assert result == 99
