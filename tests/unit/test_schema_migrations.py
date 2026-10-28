"""The schema has to be in place before the code reads it.

The migration that adds the seed columns ran the insert that names them
before the alter that adds them, and the migration job runs with
ON_ERROR_STOP=1, so the whole migration failed (issue #206). These tests
read the SQL and keep the order right.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGMAP = REPO_ROOT / "helm" / "store-listing" / "templates" / "configmap.yaml"
FRESH_SCHEMA = REPO_ROOT / "src" / "store_listing" / "db" / "schema.sql"
APP_DEPLOYMENT = REPO_ROOT / "helm" / "store-listing" / "templates" / "app" / "deployment.yaml"

SEED_COLUMNS = ("origin", "theme", "seasonal_window_end")


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


def _migration_sql() -> str:
    return _configmap_block("migrations.sql")


def _schema_sql() -> str:
    return _configmap_block("schema.sql")


class TestMigrationOrder:
    def test_the_migration_adds_the_seed_columns_before_it_writes_them(self) -> None:
        sql = _migration_sql()
        insert_at = sql.index("INSERT INTO active_seeds")

        for column in SEED_COLUMNS:
            alter_at = sql.index(f"ADD COLUMN IF NOT EXISTS {column}")
            assert alter_at < insert_at, f"{column} is added after the insert that uses it"

    def test_the_fresh_schema_adds_the_seed_columns_before_it_writes_them(self) -> None:
        sql = _schema_sql()
        insert_at = sql.index("INSERT INTO active_seeds")

        for column in SEED_COLUMNS:
            alter_at = sql.index(f"ADD COLUMN IF NOT EXISTS {column}")
            assert alter_at < insert_at, f"{column} is added after the insert that uses it"

    def test_the_standalone_schema_adds_the_seed_columns_before_it_writes_them(self) -> None:
        sql = FRESH_SCHEMA.read_text()
        insert_at = sql.index("INSERT INTO active_seeds")

        for column in SEED_COLUMNS:
            alter_at = sql.index(f"ADD COLUMN IF NOT EXISTS {column}")
            assert alter_at < insert_at, f"{column} is added after the insert that uses it"

    def test_the_starter_seeds_are_inserted_as_evergreen(self) -> None:
        sql = _migration_sql()
        insert = sql[sql.index("INSERT INTO active_seeds") : sql.index("ON CONFLICT (query)")]

        assert "'funny t-shirt', 0, 'evergreen'" in insert
        assert "'dad jokes', 0, 'evergreen'" in insert

    def test_the_migration_backfills_the_starter_seeds(self) -> None:
        sql = _migration_sql()

        assert "SET origin = 'evergreen'" in sql

    def test_the_migration_is_safe_to_run_twice(self) -> None:
        sql = _migration_sql()

        for column in SEED_COLUMNS:
            assert f"ADD COLUMN IF NOT EXISTS {column}" in sql
        assert "ON CONFLICT (query) DO NOTHING" in sql


class TestCopyModeMigration:
    def test_the_mode_is_not_backfilled_on_existing_rows(self) -> None:
        sql = _migration_sql()
        add_column = next(
            line for line in sql.splitlines() if "ADD COLUMN IF NOT EXISTS copy_mode" in line
        )

        # A default on ADD COLUMN would write 'caption' into every legacy row
        # and lose the no-text/hero meaning those rows already had.
        assert "DEFAULT" not in add_column
        assert "ALTER COLUMN copy_mode SET DEFAULT 'caption'" in sql


class TestAppWaitsForTheSchema:
    def test_the_app_waits_for_the_schema_before_it_starts(self) -> None:
        deployment = APP_DEPLOYMENT.read_text()

        assert "wait-for-schema" in deployment
        assert "store_listing.db.wait_for_schema" in deployment
