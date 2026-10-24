"""Waits until the database has the columns the code needs.

The schema migration runs as a post-upgrade hook, so a new pod can start
before the migration has finished. The pod waits here instead of failing its
first query, and it never waits forever: after the timeout the pod starts
anyway and the failing query says what is wrong.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final

import psycopg2

logger = logging.getLogger(__name__)

# The columns the seed pool needs. A pod that starts before the migration has
# run waits for them.
REQUIRED_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("active_seeds", "origin"),
    ("active_seeds", "theme"),
    ("active_seeds", "seasonal_window_end"),
    # The text-plan pass writes this, so a pod must not serve a brief batch
    # before the migration has added it (issue #246).
    ("design_briefs", "text_plan"),
)

DEFAULT_TIMEOUT_SECONDS: Final[int] = 120
DEFAULT_INTERVAL_SECONDS: Final[int] = 3


def connect() -> Any:
    """A connection to the database the app uses."""
    return psycopg2.connect(
        host=os.environ.get("DATABASE_HOST") or "localhost",
        port=int(os.environ.get("DATABASE_PORT") or 5432),
        dbname=os.environ.get("DATABASE_NAME") or "store_listing",
        user=os.environ.get("DATABASE_USER"),
        password=os.environ.get("DATABASE_PASSWORD"),
    )


def missing_columns(
    connect_fn: Callable[[], Any],
    required: Iterable[tuple[str, str]] = REQUIRED_COLUMNS,
) -> list[str]:
    """The required columns the database does not have yet."""
    wanted = list(required)
    tables = {table for table, _ in wanted}

    with connect_fn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_name = ANY(%s)",
            (list(tables),),
        )
        present = {(row[0], row[1]) for row in cur.fetchall()}

    return [f"{table}.{column}" for table, column in wanted if (table, column) not in present]


def wait_for_schema(
    connect_fn: Callable[[], Any] = connect,
    required: Sequence[tuple[str, str]] = REQUIRED_COLUMNS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait until the database has the required columns.

    True when the schema is ready, False when the wait ran out.
    """
    deadline = time.monotonic() + timeout_seconds

    while True:
        try:
            missing = missing_columns(connect_fn, required)
            if not missing:
                logger.info("The database schema is ready")
                return True
            logger.info("Waiting for the database schema: %s", ", ".join(missing))
        except Exception as e:  # noqa: BLE001
            logger.warning("The database is not ready yet: %s", e)

        if time.monotonic() >= deadline:
            logger.error("The database schema is still not ready after %s seconds", timeout_seconds)
            return False

        sleep(interval_seconds)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    wait_for_schema()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
