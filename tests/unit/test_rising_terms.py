"""Unit: harvest rising queries, top related queries, and global interest-over-time signal.

The scraper must store three signals per seed beyond plain regional interest:
- related rising queries (query_type="rising"), tagged with the harvest region
- related top queries (query_type="top"), tagged GLOBAL
- the seed's own global interest-over-time trajectory (query_type="top",
  region="GLOBAL", trend_direction set)

None of these may abort the seed when Google misbehaves, mirroring the
best-effort contract for auxiliary signals.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pytrends.exceptions import ResponseError

from store_listing.ingest.trends.google_trends import DatabaseClient, GoogleTrendsClient
from store_listing.ingest.trends.schemas import TrendHarvestRequest


def response_404() -> ResponseError:
    class _Resp:
        status_code = 404

    return ResponseError.from_response(_Resp())


class FakeTrendReq:
    def __init__(self) -> None:
        self.cookies: dict[str, str] = {}

    def build_payload(self, *args, **kwargs) -> None: ...

    def interest_by_region(self) -> pd.DataFrame:
        return pd.DataFrame({"funny t-shirt": [45]}, index=["US"])

    def related_queries(self) -> dict[str, dict[str, pd.DataFrame]]:
        return {}

    def interest_over_time(self) -> pd.DataFrame:
        return pd.DataFrame()

    def GetGoogleCookie(self) -> dict[str, str]:
        return self.cookies


class RisingFake(FakeTrendReq):
    def related_queries(self) -> dict[str, dict[str, pd.DataFrame]]:
        return {
            "hoodie": {
                "top": pd.DataFrame(
                    {"query": ["custom hoodie", "hoodie amazon"], "value": ["100", "88"]}
                ),
                "rising": pd.DataFrame({"query": ["hoodie sweatshirt"], "value": ["+250000%"]}),
            }
        }

    def interest_over_time(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"hoodie": [30, 45, 60]},
            index=pd.date_range("2026-09-01", periods=3, freq="D"),
        )


class Rising404(FakeTrendReq):
    def related_queries(self) -> dict[str, dict[str, pd.DataFrame]]:
        raise response_404()


class TestRisingTerms:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:
        return MagicMock(spec=DatabaseClient)

    def test_parse_rise_percentage(self) -> None:
        client = GoogleTrendsClient(trend_req=FakeTrendReq())
        assert client._parse_rise_pct("+250000%") == 250000
        assert client._parse_rise_pct("+1,000%") == 1000
        assert client._parse_rise_pct("2500") == 2500
        assert client._parse_rise_pct("Breakout") == 5000
        assert client._parse_rise_pct(None) == 0

    def test_harvest_stores_rising_top_and_global_signals(self, mock_db_client: MagicMock) -> None:
        client = GoogleTrendsClient(db_client=mock_db_client, trend_req=RisingFake())
        mock_db_client.insert_trend_query.return_value = 1
        mock_db_client.insert_trend_score.return_value = 1

        results, failed_seeds = client.harvest_and_store(
            TrendHarvestRequest(seed_keywords=["hoodie"], region="US")
        )

        assert failed_seeds == []

        rising = [r for r in results if r.query_type == "rising"]
        assert len(rising) == 1
        assert rising[0].query == "hoodie sweatshirt"
        assert rising[0].delta == 250000
        assert rising[0].region == "US"

        top = [r for r in results if r.query_type == "top"]
        assert len(top) == 3
        assert all(r.region == "GLOBAL" for r in top)

        global_trend = [r for r in results if r.query == "hoodie"]
        assert len(global_trend) == 1
        assert global_trend[0].trend_direction == 1

        query_calls = [c.args for c in mock_db_client.insert_trend_query.call_args_list]
        assert ("hoodie", "hoodie sweatshirt") in query_calls
        assert ("hoodie", "custom hoodie") in query_calls

        score_kwargs = [c.kwargs for c in mock_db_client.insert_trend_score.call_args_list]
        assert any(k.get("query_type") == "rising" for k in score_kwargs)
        assert any(k.get("query_type") == "top" for k in score_kwargs)

    def test_related_queries_failure_does_not_abort_harvest(
        self, mock_db_client: MagicMock
    ) -> None:
        mock_db_client.insert_trend_query.return_value = 1
        mock_db_client.insert_trend_score.return_value = 1
        client = GoogleTrendsClient(db_client=mock_db_client, trend_req=Rising404())

        with patch("store_listing.ingest.trends.google_trends.logger.warning") as mock_warning:
            results, failed_seeds = client.harvest_and_store(
                TrendHarvestRequest(seed_keywords=["funny t-shirt"])
            )

        assert len(results) == 1
        assert failed_seeds == []
        mock_db_client.insert_trend_query.assert_called_once()
        mock_db_client.insert_trend_score.assert_called_once()
        mock_warning.assert_called_once()

    def test_global_result_has_fetched_at(self, mock_db_client: MagicMock) -> None:
        client = GoogleTrendsClient(db_client=mock_db_client, trend_req=RisingFake())

        results = client.fetch_global_trend("hoodie")

        assert len(results) == 1
        assert isinstance(results[0].fetched_at, datetime)
        assert results[0].fetched_at.tzinfo == UTC
