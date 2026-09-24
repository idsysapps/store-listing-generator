from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from store_listing.ingest.trends.google_trends import DatabaseClient
from store_listing.orchestration.promotion import ActiveSeed, SeedCandidate


class FakeCursor:
    def __init__(self) -> None:
        self.fetchone_value: object = None
        self.fetchall_value: list = []
        self.executed_params: list = []
        self.rowcount: int = 1

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed_params.append((sql, params))

    def fetchone(self):
        return self.fetchone_value

    def fetchall(self):
        return self.fetchall_value

    def __enter__(self) -> "FakeCursor":  # noqa: PYI034
        return self

    def __exit__(self, *args) -> None:
        return None


class FakeMetaConn:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self) -> "FakeMetaConn":  # noqa: PYI034
        return self

    def __exit__(self, *args) -> None:
        return None


@contextmanager
def connected_db(cursor: FakeCursor) -> Iterator[tuple[DatabaseClient, FakeMetaConn]]:
    """Yield a DatabaseClient whose connect() is stubbed for the duration of the call."""
    conn = FakeMetaConn(cursor)
    db = DatabaseClient()
    with patch.object(DatabaseClient, "connect", return_value=conn):
        yield db, conn


def test_insert_trend_score_defaults_source_to_google() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = (1,)
    with connected_db(cursor) as (db, _conn):
        db.insert_trend_score(query_id=1, score=75, delta=10, region="US", query_type="rising")

    sql, params = cursor.executed_params[0]
    assert "source" in sql
    assert params[5] == "google"


def test_insert_trend_score_accepts_explicit_source() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = (1,)
    with connected_db(cursor) as (db, _conn):
        db.insert_trend_score(query_id=1, score=75, delta=10, region="US", source="tiktok")

    _sql, params = cursor.executed_params[0]
    assert params[5] == "tiktok"


def test_upsert_seed_candidate_returns_id() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = (7,)
    with connected_db(cursor) as (db, _conn):
        result = db.upsert_seed_candidate(
            source_seed="hoodie",
            query="matching hoodie",
            source="google",
            query_type="rising",
            score=0,
            delta=12000,
            promotion_score=12000,
        )

    assert result == 7
    assert "ON CONFLICT (query)" in cursor.executed_params[0][0]


def test_upsert_seed_candidate_returns_existing_when_conflict_locked() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = None
    with connected_db(cursor) as (db, _conn):
        result = db.upsert_seed_candidate(
            source_seed="hoodie",
            query="matching hoodie",
            source="google",
            query_type="top",
            score=50,
            delta=0,
            promotion_score=50,
        )

    assert result is None
    assert len(cursor.executed_params) == 2  # insert then fallback select


def test_list_pending_candidates_returns_seed_candidates() -> None:
    cursor = FakeCursor()
    cursor.fetchall_value = [
        (1, "mom shirt", "google", "rising", 0, 12000, 12000, "pending"),
        (2, "pickleball", "google", "top", 80, 0, 80, "pending"),
    ]
    with connected_db(cursor) as (db, _conn):
        result = db.list_pending_candidates()

    assert result == [
        SeedCandidate(
            id=1,
            query="mom shirt",
            source="google",
            query_type="rising",
            score=0,
            delta=12000,
            promotion_score=12000,
            status="pending",
        ),
        SeedCandidate(
            id=2,
            query="pickleball",
            source="google",
            query_type="top",
            score=80,
            delta=0,
            promotion_score=80,
            status="pending",
        ),
    ]


def test_cross_seed_counts_groups_by_query() -> None:
    cursor = FakeCursor()
    cursor.fetchall_value = [("pickleball", 2), ("mom shirt", 1)]
    with connected_db(cursor) as (db, _conn):
        assert db.cross_seed_counts() == {"pickleball": 2, "mom shirt": 1}


def test_list_active_seeds_returns_only_active() -> None:
    cursor = FakeCursor()
    cursor.fetchall_value = [(1, "hoodie", 0), (2, "gift", 0)]
    with connected_db(cursor) as (db, _conn):
        result = db.list_active_seeds()

    assert result == [
        ActiveSeed(id=1, query="hoodie", promotion_score=0),
        ActiveSeed(id=2, query="gift", promotion_score=0),
    ]


def test_count_active_seeds_counts_unarchived() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = (42,)
    with connected_db(cursor) as (db, _conn):
        assert db.count_active_seeds() == 42


def test_insert_active_seed_returns_id() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = (3,)
    with connected_db(cursor) as (db, _conn):
        result = db.insert_active_seed(query="running", promotion_score=100, candidate_id=9)

    assert result == 3
    sql, params = cursor.executed_params[0]
    assert "ON CONFLICT (query) DO NOTHING" in sql
    assert params == ("running", 100, 9)


def test_insert_active_seed_allows_starter_without_candidate() -> None:
    cursor = FakeCursor()
    cursor.fetchone_value = (4,)
    with connected_db(cursor) as (db, _conn):
        result = db.insert_active_seed(query="socks", promotion_score=0, candidate_id=None)

    assert result == 4


def test_promote_candidate_marks_promoted() -> None:
    cursor = FakeCursor()
    with connected_db(cursor) as (db, _conn):
        db.promote_candidate(5)

    sql, params = cursor.executed_params[0]
    assert "status" in sql and "promoted" in sql
    assert params == (5,)


def test_archive_active_seed_marks_archived() -> None:
    cursor = FakeCursor()
    with connected_db(cursor) as (db, _conn):
        db.archive_active_seed("hoodie")

    sql, params = cursor.executed_params[0]
    assert "archived_at" in sql
    assert params == ("hoodie",)
