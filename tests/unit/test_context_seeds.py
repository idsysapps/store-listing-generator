"""Unit tests for the rule-based calendar seed injector."""

from datetime import date
from unittest.mock import MagicMock

from store_listing.orchestration.context_seeds import (
    SEASONAL_SEED_RULES,
    get_seeds_to_inject,
    get_upcoming_events,
    inject_seasonal_seeds,
)


class TestGetUpcomingEvents:
    def test_halloween_within_window(self) -> None:
        today = date(2026, 9, 25)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "halloween" in names

    def test_halloween_outside_window(self) -> None:
        today = date(2026, 8, 1)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "halloween" not in names

    def test_christmas_60_day_window(self) -> None:
        today = date(2026, 10, 26)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "christmas" in names

    def test_christmas_too_early(self) -> None:
        today = date(2026, 10, 1)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "christmas" not in names

    def test_valentines_day_wraps_year(self) -> None:
        today = date(2026, 12, 30)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "valentines_day" not in names

    def test_valentines_day_within_window(self) -> None:
        today = date(2026, 1, 20)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "valentines_day" in names

    def test_multiple_overlapping_events(self) -> None:
        today = date(2026, 11, 10)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "christmas" in names

    def test_on_event_day(self) -> None:
        today = date(2026, 10, 31)
        events = get_upcoming_events(today)
        names = [e["name"] for e in events]
        assert "halloween" in names

    def test_days_until_correct(self) -> None:
        today = date(2026, 10, 21)
        events = get_upcoming_events(today)
        halloween = next(e for e in events if e["name"] == "halloween")
        assert halloween["days_until"] == 10

    def test_no_events_in_summer(self) -> None:
        today = date(2026, 6, 20)
        events = get_upcoming_events(today)
        assert len(events) == 0


class TestGetSeedsToInject:
    def test_returns_seed_event_pairs(self) -> None:
        today = date(2026, 9, 25)
        pairs = get_seeds_to_inject(today)
        seeds = [p[0] for p in pairs]
        events = [p[1] for p in pairs]
        assert seeds
        assert set(seeds) <= set(SEASONAL_SEED_RULES["halloween"].seeds)
        assert all(e == "halloween" for e in events)

    def test_only_a_few_seeds_per_event_are_injected(self) -> None:
        # A holiday used to inject its whole seed list every day, which is
        # what made the pool all-Halloween (issue #198).
        today = date(2026, 9, 25)
        pairs = get_seeds_to_inject(today, per_event=2)
        assert len(pairs) == 2

    def test_empty_when_no_events(self) -> None:
        today = date(2026, 6, 20)
        pairs = get_seeds_to_inject(today)
        assert pairs == []


class TestInjectSeasonalSeeds:
    def test_inserts_seeds_and_logs(self) -> None:
        db = MagicMock()
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1

        result = inject_seasonal_seeds(db, today=date(2026, 9, 25), per_event=3)

        assert result["status"] == "success"
        assert result["injected"] == 3
        assert "halloween" in result["events"]
        assert db.insert_active_seed.call_count == 3
        assert db.insert_curation_log.call_count == result["injected"]

    def test_the_seeds_of_an_event_are_injected_over_the_season(self) -> None:
        # Every seed of the event gets its turn, not only the first three.
        db = MagicMock()
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1

        injected: set[str] = set()
        for day in range(16, 31):
            inject_seasonal_seeds(db, today=date(2026, 9, day), per_event=3)
            injected.update(call.kwargs["query"] for call in db.insert_active_seed.call_args_list)
            db.insert_active_seed.reset_mock()

        assert injected == set(SEASONAL_SEED_RULES["halloween"].seeds)

    def test_skips_duplicate_seeds(self) -> None:
        db = MagicMock()
        db.insert_active_seed.return_value = None
        db.insert_curation_log.return_value = 1

        result = inject_seasonal_seeds(db, today=date(2026, 9, 25))

        assert result["injected"] == 0
        assert db.insert_curation_log.call_count == 0

    def test_no_events_returns_zero(self) -> None:
        db = MagicMock()

        result = inject_seasonal_seeds(db, today=date(2026, 6, 20))

        assert result["injected"] == 0
        assert result["events"] == []
        assert not db.insert_active_seed.called

    def test_logs_with_calendar_inject_action(self) -> None:
        db = MagicMock()
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1

        inject_seasonal_seeds(db, today=date(2026, 9, 25))

        log_call = db.insert_curation_log.call_args
        assert log_call.kwargs["action"] == "calendar_inject"
        assert log_call.kwargs["event_context"] == "halloween"
        assert log_call.kwargs["llm_model"] is None

    def test_the_seed_knows_the_event_it_belongs_to(self) -> None:
        db = MagicMock()
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1

        inject_seasonal_seeds(db, today=date(2026, 9, 25))

        kwargs = db.insert_active_seed.call_args.kwargs
        assert kwargs["origin"] == "calendar"
        assert kwargs["theme"] == "halloween"
        assert kwargs["seasonal_window_end"] == date(2026, 10, 31)
