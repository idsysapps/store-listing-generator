"""Source-health tracking with automatic GitHub issue filing.

Free-path-first strategy: when a harvester fails or returns nothing repeatedly,
the tracker opens (and de-duplicates) a GitHub issue so the fix loop can repair
the blocked source. Recording is always on when the tracker is active; GitHub
issue creation is best-effort and disabled without a token.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Literal, Protocol, TypeVar, cast

import httpx

from store_listing.ingest.trends.google_trends import DatabaseClient

logger = logging.getLogger(__name__)

Outcome = Literal["success", "failure", "empty"]

DEFAULT_FAILURE_THRESHOLD = 2
DEFAULT_EMPTY_THRESHOLD = 3
ISSUE_LABEL = "auto-health"


@dataclass(frozen=True)
class HealthState:
    consecutive_failures: int = 0
    consecutive_empty: int = 0
    last_error: str | None = None
    last_error_at: datetime | None = None
    last_success_at: datetime | None = None
    issue_open: bool = False
    last_issue_url: str | None = None
    last_issue_number: int | None = None


def record(
    state: HealthState | None,
    outcome: Outcome,
    *,
    now: datetime,
    error: str | None = None,
) -> HealthState:
    """Transition pure-health model to the next state for one harvest outcome."""
    current = state or HealthState()
    if outcome == "success":
        return replace(
            current,
            consecutive_failures=0,
            consecutive_empty=0,
            last_error=None,
            last_error_at=None,
            last_success_at=now,
        )
    if outcome == "failure":
        return replace(
            current,
            consecutive_failures=current.consecutive_failures + 1,
            last_error=error,
            last_error_at=now,
        )
    return replace(current, consecutive_empty=current.consecutive_empty + 1)


def should_open_issue(
    state: HealthState,
    *,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    empty_threshold: int = DEFAULT_EMPTY_THRESHOLD,
) -> bool:
    if state.issue_open:
        return False
    if state.consecutive_failures >= failure_threshold:
        return True
    return state.consecutive_empty >= empty_threshold


def issue_title(source: str) -> str:
    return f"[auto-health] {source} harvester is failing consistently"


def build_issue_body(source: str, state: HealthState) -> str:
    lines = [
        "### What",
        (
            f"The **{source}** trend harvester has been failing consistently "
            f"(failures: {state.consecutive_failures}, empties: {state.consecutive_empty})."
        ),
        "",
        "### Last error",
        state.last_error or "(no error captured)",
        "",
        "### Where to look",
        f"- `src/store_listing/ingest/trends/{source}.py` — gateway/actor wiring",
        f"- `src/store_listing/orchestration/tasks.py` — `fetch_{source}_trends` task",
        (
            "- Free-path-first: verify the no-token web gateway; the Apify actor is only used "
            "when `APIFY_API_TOKEN` is set."
        ),
        "",
        "### How to fix",
        "1. Reproduce: `uv pytest tests/unit/` for the affected source.",
        "2. Fix the gateway (or its env wiring) so the free path succeeds.",
        (
            "3. Close this issue once the source recovers; closing resets tracking so a fresh "
            "regression re-opens it after the next 2 failed runs."
        ),
    ]
    return "\n".join(lines)


class DBConnector(Protocol):
    """Minimum surface of DatabaseClient used by the health tracker."""

    def connect(self) -> Any: ...


class IssueReporter(Protocol):
    def create(self, source: str, title: str, body: str) -> tuple[str, int]: ...

    def has_open(self, source: str, title: str) -> bool: ...


class HealthGateway(Protocol):
    def get(self, source: str) -> HealthState | None: ...

    def save(self, source: str, state: HealthState) -> None: ...


class GithubReporter:
    """Thin GitHub REST client: create issues and detect an existing open one."""

    def __init__(
        self,
        token: str,
        repo: str,
        client: httpx.Client | None = None,
        label: str = ISSUE_LABEL,
    ) -> None:
        self.token = token
        self.repo = repo
        self.label = label
        self._client = client or httpx.Client(
            base_url="https://api.github.com",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def create(self, source: str, title: str, body: str) -> tuple[str, int]:
        response = self._client.post(
            f"/repos/{self.repo}/issues",
            json={"title": title, "body": body, "labels": [self.label]},
        )
        response.raise_for_status()
        payload = response.json()
        return payload["html_url"], int(payload["number"])

    def has_open(self, source: str, title: str) -> bool:
        response = self._client.get(
            f"/repos/{self.repo}/issues", params={"state": "open", "per_page": 100}
        )
        response.raise_for_status()
        return any(
            item.get("title") == title
            and "pull_request" not in item
            and item.get("state", "open") == "open"
            for item in response.json()
        )


class HealthStore:
    """Persist/load HealthState per source in the `source_health` table."""

    def __init__(self, db_client: DBConnector | None = None) -> None:
        self.db_client = db_client or DatabaseClient()

    def get(self, source: str) -> HealthState | None:
        with self.db_client.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT consecutive_failures, consecutive_empty, last_error,
                       last_error_at, last_success_at,
                       issue_open, last_issue_url, last_issue_number
                FROM source_health
                WHERE source = %s
                """,
                (source,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        (
            failures,
            empty,
            last_error,
            last_error_at,
            last_success_at,
            issue_open,
            last_issue_url,
            last_issue_number,
        ) = row
        return HealthState(
            consecutive_failures=failures,
            consecutive_empty=empty,
            last_error=last_error,
            last_error_at=last_error_at,
            last_success_at=last_success_at,
            issue_open=bool(issue_open),
            last_issue_url=last_issue_url,
            last_issue_number=last_issue_number,
        )

    def save(self, source: str, state: HealthState) -> None:
        with self.db_client.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_health (
                    source, consecutive_failures, consecutive_empty,
                    last_error, last_error_at, last_success_at,
                    issue_open, last_issue_url, last_issue_number
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source) DO UPDATE SET
                    consecutive_failures = EXCLUDED.consecutive_failures,
                    consecutive_empty = EXCLUDED.consecutive_empty,
                    last_error = EXCLUDED.last_error,
                    last_error_at = EXCLUDED.last_error_at,
                    last_success_at = EXCLUDED.last_success_at,
                    issue_open = EXCLUDED.issue_open,
                    last_issue_url = EXCLUDED.last_issue_url,
                    last_issue_number = EXCLUDED.last_issue_number,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    source,
                    state.consecutive_failures,
                    state.consecutive_empty,
                    state.last_error,
                    state.last_error_at,
                    state.last_success_at,
                    state.issue_open,
                    state.last_issue_url,
                    state.last_issue_number,
                ),
            )


class SourceHealthTracker:
    """Compose the store and reporter: record outcomes and manage issue state."""

    def __init__(
        self,
        store: HealthGateway,
        reporter: IssueReporter | None,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        empty_threshold: int = DEFAULT_EMPTY_THRESHOLD,
    ) -> None:
        self.store = store
        self.reporter = reporter
        self.failure_threshold = failure_threshold
        self.empty_threshold = empty_threshold

    def record(
        self,
        source: str,
        outcome: Outcome,
        *,
        error: str | None = None,
    ) -> HealthState:
        previous = self.store.get(source)
        state = record(previous, outcome, now=datetime.now(UTC), error=error)

        if outcome == "success" and previous is not None and previous.issue_open:
            state = self._reconcile_open_issue(source, state)

        if should_open_issue(
            state,
            failure_threshold=self.failure_threshold,
            empty_threshold=self.empty_threshold,
        ):
            state = self._attach_issue(source, state)

        self.store.save(source, state)
        return state

    def _reconcile_open_issue(self, source: str, state: HealthState) -> HealthState:
        """If the tracked issue was closed upstream, clear the stale open flag."""
        if self.reporter is None:
            return state
        try:
            if not self.reporter.has_open(source, issue_title(source)):
                return replace(state, issue_open=False, last_issue_url=None, last_issue_number=None)
        except Exception:
            logger.warning("could not reconcile open issue for %s", source, exc_info=True)
        return state

    def _attach_issue(self, source: str, state: HealthState) -> HealthState:
        if state.issue_open or state.last_issue_url is not None:
            return replace(state, issue_open=True)
        if self.reporter is None:
            return state
        try:
            if self.reporter.has_open(source, issue_title(source)):
                # Someone already filed it (operator or previous tracker): remember it.
                return state
            url, number = self.reporter.create(
                source, issue_title(source), build_issue_body(source, state)
            )
            return replace(state, issue_open=True, last_issue_url=url, last_issue_number=number)
        except Exception:
            logger.exception("failed to open source-health issue for %s", source)
            return state


def build_tracker() -> SourceHealthTracker | None:
    """Build a tracker from the environment, or None when health is disabled."""
    if os.environ.get("SOURCE_HEALTH_ENABLED", "true").lower() not in ("1", "true", "yes"):
        return None
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPO")
    reporter: IssueReporter | None = None
    if token and repo:
        reporter = GithubReporter(token=token, repo=repo)
    elif token or repo:
        logger.warning("source-health enabled but GitHub token/repo missing -> recording only")
    return SourceHealthTracker(
        store=HealthStore(),
        reporter=reporter,
        failure_threshold=int(
            os.environ.get("HEALTH_FAILURE_THRESHOLD", DEFAULT_FAILURE_THRESHOLD)
        ),
        empty_threshold=int(os.environ.get("HEALTH_EMPTY_THRESHOLD", DEFAULT_EMPTY_THRESHOLD)),
    )


def _classify(result: Any, error: BaseException | None) -> tuple[Outcome, str | None]:
    """Map a task result (or raised error) to health outcome.

    Task functions return status dicts with ``status`` and ``results_count``,
    but the decorator is generic over any callable return.
    """
    if error is not None:
        return "failure", str(error)
    if result is None:
        return "empty", None
    if isinstance(result, dict) and result.get("status") == "failed":
        return "failure", None
    if isinstance(result, dict) and int(result.get("results_count", 0) or 0) > 0:
        return "success", None
    return "empty", None


F = TypeVar("F", bound=Callable[..., object])


def with_source_health(source: str) -> Callable[[F], F]:
    """Decorate a task function so its outcome is recorded on the health tracker."""

    def decorator(task_fn: F) -> F:
        @wraps(task_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracker = build_tracker()
            if tracker is None:
                return task_fn(*args, **kwargs)
            try:
                result = task_fn(*args, **kwargs)
            except Exception as exc:
                outcome, error = _classify(None, exc)
                tracker.record(source, outcome, error=error)
                raise
            outcome, error = _classify(result, None)
            tracker.record(source, outcome, error=error)
            return result

        return cast(F, wrapper)

    return decorator
