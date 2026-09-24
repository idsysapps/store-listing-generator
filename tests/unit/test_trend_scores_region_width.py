"""Unit: trend_scores.region must accommodate US metro/state names.

Google's interest_by_region() returns metro/state labels (e.g. "District of
Columbia", "New York") as the region index. The schema column and the
TrendScore model previously capped region at 10 chars, aborting the harvest
seed on any longer name.

The DB column and the pydantic bound must stay wide enough to hold an actual
region name.
"""

from pathlib import Path

from store_listing.ingest.trends.schemas import TrendScore

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_SQL = REPO_ROOT / "src" / "store_listing" / "db" / "schema.sql"
SCHEMAS_PY = REPO_ROOT / "src" / "store_listing" / "ingest" / "trends" / "schemas.py"


def test_trend_scores_region_column_accepts_long_names() -> None:
    """GREEN: a 21-char region ("District of Columbia") fits in the column."""
    schema = SCHEMA_SQL.read_text()
    assert "region VARCHAR(255)" in schema, "trend_scores.region must be VARCHAR(255)"


def test_trend_score_model_accepts_long_region() -> None:
    """GREEN: TrendScore region validation is not narrower than the column."""
    TrendScore(
        query_id=1,
        score=45,
        delta=0,
        region="District of Columbia",
    )
