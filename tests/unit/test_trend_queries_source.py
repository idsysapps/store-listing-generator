"""Contract: trend_queries must carry the ingester `source` like trend_scores.

Issue #2 writes raw TikTok/Pinterest/Etsy discoveries into the unified ranked
trend store (trend_queries + trend_scores). The prerequisite (#72) only added
`source` to trend_scores; this contract pins the matching column on trend_queries
in every SQL artifact (schema.sql + helm configmap + migrations).
"""

import re
from pathlib import Path

from store_listing.ingest.trends.schemas import Source

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_SQL = REPO_ROOT / "src" / "store_listing" / "db" / "schema.sql"
CONFIGMAP_YAML = REPO_ROOT / "helm" / "store-listing" / "templates" / "configmap.yaml"
SCHEMAS_PY = REPO_ROOT / "src" / "store_listing" / "ingest" / "trends" / "schemas.py"

TREND_QUERIES_BLOCK = r"CREATE TABLE (?:IF NOT EXISTS )?trend_queries \((.*?)\);"


def trend_queries_block(sql: str) -> str:
    match = re.search(TREND_QUERIES_BLOCK, sql, re.DOTALL)
    assert match is not None, "trend_queries CREATE TABLE block not found"
    return match.group(1)


def test_schema_sql_gives_trend_queries_a_source_column() -> None:
    block = trend_queries_block(SCHEMA_SQL.read_text())
    assert "source VARCHAR(20)" in block, "trend_queries must carry source VARCHAR(20)"
    assert "NOT NULL DEFAULT 'google'" in block


def test_configmap_schema_matches_source_column() -> None:
    block = trend_queries_block(CONFIGMAP_YAML.read_text())
    assert "source VARCHAR(20)" in block


def test_migrations_add_source_to_trend_queries_idempotently() -> None:
    migrations = CONFIGMAP_YAML.read_text()
    assert (
        "ALTER TABLE trend_queries ADD COLUMN IF NOT EXISTS source VARCHAR(20) NOT NULL DEFAULT 'google';"
        in migrations
    )


def test_etsy_is_a_valid_source() -> None:
    """RED: Source literal must admit the Etsy suggestion source."""
    assert "etsy" in Source.__args__  # type: ignore[attr-defined]


def test_schemas_py_declares_all_five_sources() -> None:
    text = SCHEMAS_PY.read_text()
    for source in ["google", "tiktok", "pinterest", "amazon", "etsy"]:
        assert source in text
