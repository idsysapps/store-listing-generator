"""Unit tests for the schema wait of the app pods."""

from typing import Any, Self

from store_listing.db.wait_for_schema import (
    REQUIRED_COLUMNS,
    missing_columns,
    wait_for_schema,
)


class _FakeCursor:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.sql = sql

    def fetchall(self) -> list[tuple[str, str]]:
        return self._rows

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return None


class _FakeConn:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._cursor = _FakeCursor(rows)

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return None


def _connect_with(rows: list[tuple[str, str]]):
    return lambda: _FakeConn(rows)


class TestMissingColumns:
    def test_a_migrated_database_is_missing_nothing(self) -> None:
        rows = [(table, column) for table, column in REQUIRED_COLUMNS]

        assert missing_columns(_connect_with(rows)) == []

    def test_an_unmigrated_database_is_missing_the_seed_columns(self) -> None:
        missing = missing_columns(_connect_with([]))

        assert missing == [
            "active_seeds.origin",
            "active_seeds.theme",
            "active_seeds.seasonal_window_end",
        ]

    def test_a_half_migrated_database_is_missing_the_rest(self) -> None:
        missing = missing_columns(_connect_with([("active_seeds", "origin")]))

        assert missing == ["active_seeds.theme", "active_seeds.seasonal_window_end"]


class TestWaitForSchema:
    def test_it_returns_when_the_schema_is_ready(self) -> None:
        rows = [(table, column) for table, column in REQUIRED_COLUMNS]
        slept: list[float] = []

        ready = wait_for_schema(
            connect_fn=_connect_with(rows),
            sleep=slept.append,
        )

        assert ready is True
        assert slept == []

    def test_it_keeps_waiting_until_the_schema_arrives(self) -> None:
        calls = {"count": 0}

        def connect() -> Any:
            calls["count"] += 1
            if calls["count"] < 3:
                return _FakeConn([])
            return _FakeConn([(table, column) for table, column in REQUIRED_COLUMNS])

        ready = wait_for_schema(connect_fn=connect, sleep=lambda _seconds: None)

        assert ready is True
        assert calls["count"] == 3

    def test_it_gives_up_after_the_timeout(self) -> None:
        ready = wait_for_schema(
            connect_fn=_connect_with([]),
            timeout_seconds=0,
            sleep=lambda _seconds: None,
        )

        assert ready is False
