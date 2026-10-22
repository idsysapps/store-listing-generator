"""Unit: the material product types are stored as an array, in the chart too.

Cost-book publishes the product types a material can serve as a JSON array
(`["substrate","dtf_apparel"]`). The column was a `VARCHAR(50)` that held the
raw JSON text, so a list of three or more types did not fit and the equality
filter never matched (issue #186).

The chart is the schema of the deployed database, so the fresh schema and the
migration have to agree with the column the code reads.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGMAP = REPO_ROOT / "helm" / "store-listing" / "templates" / "configmap.yaml"
TEST_SCHEMA = (
    REPO_ROOT / "trend-service" / "src" / "test" / "resources" / "db" / "fabrication_materials.sql"
)


def _configmap_block(key: str) -> str:
    """The body of a block of the configmap, e.g. 'migrations.sql'."""
    lines = CONFIGMAP.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"{key}: |")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith("    "):
            break
        body.append(line.removeprefix("    "))
    return "\n".join(body)


def _table_block(sql: str, table: str) -> str:
    """The CREATE TABLE block of one table."""
    start = sql.index(f"CREATE TABLE IF NOT EXISTS {table} (")
    end = sql.index(");", start)
    return sql[start:end]


class TestProductTypeColumn:
    def test_the_fresh_schema_stores_the_product_types_as_an_array(self) -> None:
        sql = _table_block(_configmap_block("schema.sql"), "fabrication_materials")

        assert "product_type TEXT[]" in sql
        assert "product_type VARCHAR(50)" not in sql

    def test_the_migration_converts_the_column_to_an_array(self) -> None:
        sql = _configmap_block("migrations.sql")

        assert "ALTER COLUMN product_type TYPE TEXT[]" in sql

    def test_the_migration_only_converts_a_column_that_is_not_an_array_yet(self) -> None:
        # The migration job runs on every deploy, so a conversion that empties
        # the column again would throw the cache away every time.
        sql = _configmap_block("migrations.sql")

        assert "data_type <> 'ARRAY'" in sql

    def test_the_test_schema_stores_the_product_types_as_an_array(self) -> None:
        sql = TEST_SCHEMA.read_text()

        assert "product_type TEXT[]" in sql
