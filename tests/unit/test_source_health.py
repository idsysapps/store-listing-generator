"""Unit tests for source-health tracking and auto-created GitHub issues."""

import json
from datetime import UTC, datetime

import pytest
from httpx import Client, MockTransport, Response

from store_listing.orchestration.source_health import (
    GithubReporter,
    HealthState,
    HealthStore,
    SourceHealthTracker,
    build_issue_body,
    build_tracker,
    issue_title,
    record,
    should_open_issue,
)


def _state(**overrides) -> HealthState:
    defaults = {
        "consecutive_failures": 0,
        "consecutive_empty": 0,
        "last_error": None,
        "last_error_at": None,
        "last_success_at": None,
        "issue_open": False,
        "last_issue_url": None,
        "last_issue_number": None,
    }
    defaults.update(overrides)
    return HealthState(**defaults)


NOW = datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC)


# --- pure record transitions -------------------------------------------------


def test_record_success_resets_failure_and_empty_counters() -> None:
    before = _state(
        consecutive_failures=5,
        consecutive_empty=2,
        issue_open=True,
        last_issue_url="https://github.com/x/issues/9",
        last_issue_number=9,
    )
    after = record(before, "success", now=NOW)
    assert after.consecutive_failures == 0
    assert after.consecutive_empty == 0
    assert after.last_error is None
    assert after.last_success_at == NOW
    assert after.issue_open is True
    assert after.last_issue_url == "https://github.com/x/issues/9"


def test_record_failure_increments_and_captures_error() -> None:
    before = _state(consecutive_failures=1)
    after = record(before, "failure", now=NOW, error="boom")
    assert after.consecutive_failures == 2
    assert after.last_error == "boom"
    assert after.last_error_at == NOW
    assert after.consecutive_empty == before.consecutive_empty


def test_record_empty_counts_separately_without_clearing_failures() -> None:
    before = _state(consecutive_failures=1, consecutive_empty=1)
    after = record(before, "empty", now=NOW)
    assert after.consecutive_empty == 2
    assert after.consecutive_failures == 1
    assert after.last_success_at is None


def test_record_first_outcome_from_empty_state() -> None:
    after = record(None, "failure", now=NOW, error="boom")
    assert after.consecutive_failures == 1
    assert after.consecutive_empty == 0


# --- threshold logic ----------------------------------------------------------


def test_should_open_issue_only_after_failure_threshold() -> None:
    assert (
        should_open_issue(_state(consecutive_failures=1), failure_threshold=2, empty_threshold=3)
        is False
    )
    assert (
        should_open_issue(_state(consecutive_failures=2), failure_threshold=2, empty_threshold=3)
        is True
    )


def test_should_open_issue_uses_separate_empty_threshold() -> None:
    assert (
        should_open_issue(_state(consecutive_empty=2), failure_threshold=2, empty_threshold=3)
        is False
    )
    assert (
        should_open_issue(_state(consecutive_empty=3), failure_threshold=2, empty_threshold=3)
        is True
    )


def test_should_open_issue_respects_open_issue_flag() -> None:
    state = _state(consecutive_failures=2, issue_open=True)
    assert should_open_issue(state, failure_threshold=2, empty_threshold=3) is False


# --- issue title + body --------------------------------------------------------


def test_issue_title_is_namespaced_with_source() -> None:
    assert issue_title("tiktok") == "[auto-health] tiktok harvester is failing consistently"


def test_build_issue_body_includes_runbook_and_metrics() -> None:
    state = _state(
        consecutive_failures=2,
        last_error="TimeoutError: handshake operation timed out",
        last_error_at=NOW,
    )
    body = build_issue_body("tiktok", state)
    assert "tiktok" in body
    assert "failures: 2" in body
    assert "TimeoutError" in body
    assert "ingest/trends/tiktok.py" in body
    assert "APIFY_API_TOKEN" in body


# --- GitHubReporter -------------------------------------------------------------


def test_github_reporter_creates_issue_with_label() -> None:
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return Response(201, json={"html_url": "https://github.com/o/r/issues/9", "number": 9})

    reporter = GithubReporter(
        token="tok",
        repo="o/r",
        client=Client(transport=MockTransport(handler), base_url="https://api.github.com"),
    )
    url, number = reporter.create("tiktok", issue_title("tiktok"), "body")
    assert url == "https://github.com/o/r/issues/9"
    assert number == 9
    assert captured["method"] == "POST"
    assert captured["path"] == "/repos/o/r/issues"
    assert captured["payload"]["title"] == issue_title("tiktok")
    assert captured["payload"]["labels"] == ["auto-health"]


def test_github_reporter_has_open_ignores_closed_and_prs() -> None:
    open_pr = {"title": issue_title("tiktok"), "pull_request": {}}
    closed = {"title": issue_title("tiktok"), "state": "closed"}
    other = {"title": "[auto-health] pinterest harvester is failing consistently"}

    def handler(request):
        return Response(200, json=[open_pr, closed, other])

    reporter = GithubReporter(
        token="tok",
        repo="o/r",
        client=Client(transport=MockTransport(handler), base_url="https://api.github.com"),
    )
    assert reporter.has_open("tiktok", issue_title("tiktok")) is False
    assert reporter.has_open("pinterest", issue_title("pinterest")) is True


# --- HealthStore -----------------------------------------------------------------


class FakeCursor:
    def __init__(self) -> None:
        self.fetchone_value: object = None
        self.executed_params: list = []

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed_params.append((sql, params))

    def fetchone(self):
        return self.fetchone_value

    def fetchall(self):
        return []

    def __enter__(self) -> "FakeCursor":  # noqa: PYI034
        return self

    def __exit__(self, *args) -> None:
        return None


class FakeConn:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> FakeCursor:
        return self._cursor

    def __enter__(self) -> "FakeConn":  # noqa: PYI034
        return self

    def __exit__(self, *args) -> None:
        return None


class StubDb:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def connect(self) -> FakeConn:
        return FakeConn(self._cursor)


def test_health_store_get_none_when_missing() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = None
    store = HealthStore(db_client=StubDb(cursor))
    assert store.get("tiktok") is None


def test_health_store_get_maps_row() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = (3, 1, "err", NOW, NOW, True, "https://x/1", 1)
    store = HealthStore(db_client=StubDb(cursor))
    state = store.get("tiktok")
    assert state is not None
    assert state.consecutive_failures == 3
    assert state.consecutive_empty == 1
    assert state.last_error == "err"
    assert state.issue_open is True
    sql, params = cursor.executed_params[0]
    assert "source_health" in sql
    assert params == ("tiktok",)


def test_health_store_save_upserts() -> None:
    cursor = FakeCursor()
    store = HealthStore(db_client=StubDb(cursor))
    store.save("tiktok", _state(consecutive_failures=2))
    sql, params = cursor.executed_params[0]
    assert "ON CONFLICT (source) DO UPDATE" in sql
    assert params[0] == "tiktok"
    assert params[1] == 2


# --- SourceHealthTracker -----------------------------------------------------------


class FakeStore:
    def __init__(self) -> None:
        self.data: dict[str, HealthState] = {}

    def get(self, source: str) -> HealthState | None:
        return self.data.get(source)

    def save(self, source: str, state: HealthState) -> None:
        self.data[source] = state


class FakeReporter:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.remote_open: bool = False

    def has_open(self, source: str, title: str) -> bool:
        return self.remote_open

    def create(self, source: str, title: str, body: str) -> tuple[str, int]:
        self.created.append(title)
        return ("https://github.com/o/r/issues/88", 88)


def _tracker(
    store: FakeStore | None = None, reporter: FakeReporter | None = None
) -> tuple[SourceHealthTracker, FakeReporter]:
    store = store or FakeStore()
    reporter = reporter or FakeReporter()
    return SourceHealthTracker(store=store, reporter=reporter), reporter


def test_tracker_opens_issue_after_two_failures_once() -> None:
    tracker, reporter = _tracker()

    tracker.record("tiktok", "failure", error="oops one")
    state1 = tracker.record("tiktok", "failure", error="oops two")

    assert reporter.created == [issue_title("tiktok")]
    assert state1.issue_open is True
    assert state1.last_issue_url == "https://github.com/o/r/issues/88"

    # third failure while issue still open -> no duplicate
    tracker.record("tiktok", "failure", error="oops three")
    assert reporter.created == [issue_title("tiktok")]


def test_tracker_opens_issue_after_empty_threshold() -> None:
    tracker, reporter = _tracker()
    for _ in range(3):
        tracker.record("pinterest", "empty")
    assert reporter.created == [issue_title("pinterest")]


def test_tracker_success_resets_and_reopens_after_stale_issue_closed() -> None:
    tracker, reporter = _tracker()
    reporter.remote_open = True
    tracker.record("tiktok", "failure", error="a")
    tracker.record("tiktok", "failure", error="b")
    assert reporter.created == []

    # issue now closed upstream; next success clears the stale open flag
    reporter.remote_open = False
    state = tracker.record("tiktok", "success")
    assert state.issue_open is False
    assert state.last_issue_url is None
    assert state.consecutive_failures == 0

    # recovery then another regression -> re-files the issue
    tracker.record("tiktok", "failure", error="x")
    state = tracker.record("tiktok", "failure", error="y")
    assert reporter.created == [issue_title("tiktok")]
    assert state.consecutive_failures == 2
    assert state.issue_open is True


def test_tracker_without_reporter_records_but_never_issues() -> None:
    tracker = SourceHealthTracker(store=FakeStore(), reporter=None)
    state = tracker.record("tiktok", "failure", error="boom")
    tracker.record("tiktok", "failure", error="boom2")
    assert state.consecutive_failures >= 1
    assert state.issue_open is False
    assert state.last_issue_url is None


# --- build_tracker (env driven) -------------------------------------------------------


def test_build_tracker_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_HEALTH_ENABLED", "false")
    assert build_tracker() is None


def test_build_tracker_none_without_github_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_HEALTH_ENABLED", "true")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    tracker = build_tracker()
    assert tracker is not None
    assert tracker.reporter is None


def test_build_tracker_with_github_token_and_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_HEALTH_ENABLED", "true")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPO", "o/r")
    monkeypatch.setenv("HEALTH_FAILURE_THRESHOLD", "3")
    monkeypatch.setenv("HEALTH_EMPTY_THRESHOLD", "4")
    tracker = build_tracker()
    assert tracker is not None
    assert isinstance(tracker.reporter, GithubReporter)
    assert tracker.failure_threshold == 3
    assert tracker.empty_threshold == 4
