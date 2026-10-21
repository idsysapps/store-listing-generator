"""Unit tests for the seed pool balance (issue #198).

The pool is fed by a calendar of holidays and by standing themes. Without a
rule the season takes every slot, so the pool has to keep both.
"""

from datetime import date

from store_listing.orchestration.context_seeds import (
    EVERGREEN_SEED_RULES,
    get_evergreen_seeds_to_inject,
    get_seeds_to_inject,
    inject_evergreen_seeds,
    inject_seasonal_seeds,
    rotate_seeds,
)
from store_listing.orchestration.promotion import (
    ActiveSeed,
    enforce_cap,
    select_briefable_seeds,
)


def _seed(
    seed_id: int,
    query: str,
    score: int = 100,
    origin: str = "trend",
    theme: str | None = None,
    window_end: date | None = None,
) -> ActiveSeed:
    return ActiveSeed(
        id=seed_id,
        query=query,
        promotion_score=score,
        origin=origin,
        theme=theme,
        seasonal_window_end=window_end,
    )


class TestCalendarInjection:
    def test_only_a_few_seeds_per_event_are_injected(self) -> None:
        pairs = get_seeds_to_inject(date(2026, 10, 20), per_event=3)

        assert len(pairs) == 3
        assert all(event == "halloween" for _, event in pairs)

    def test_the_injected_seeds_rotate_between_days(self) -> None:
        first = [seed for seed, _ in get_seeds_to_inject(date(2026, 10, 20), per_event=3)]
        second = [seed for seed, _ in get_seeds_to_inject(date(2026, 10, 21), per_event=3)]

        assert first != second

    def test_every_seed_of_the_event_is_used_over_the_season(self) -> None:
        used: set[str] = set()
        for day in range(20, 31):
            used.update(
                seed
                for seed, event in get_seeds_to_inject(date(2026, 10, day), per_event=3)
                if event == "halloween"
            )

        assert len(used) == 8

    def test_the_injected_seed_says_where_it_came_from(self) -> None:
        from unittest.mock import MagicMock

        db = MagicMock()
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1

        inject_seasonal_seeds(db, today=date(2026, 10, 20), per_event=2)

        kwargs = db.insert_active_seed.call_args_list[0].kwargs
        assert kwargs["origin"] == "calendar"
        assert kwargs["theme"] == "halloween"
        assert kwargs["seasonal_window_end"] == date(2026, 10, 31)

    def test_rotate_seeds_keeps_the_order_of_the_list(self) -> None:
        seeds = ["a", "b", "c", "d"]

        assert rotate_seeds(seeds, 1, 2) == ["b", "c"]

    def test_rotate_seeds_wraps_around(self) -> None:
        seeds = ["a", "b", "c"]

        assert rotate_seeds(seeds, 2, 2) == ["c", "a"]


class TestEvergreenInjection:
    def test_the_evergreen_seeds_are_not_tied_to_a_date(self) -> None:
        # The same themes are available in July as in October.
        summer = get_evergreen_seeds_to_inject(date(2026, 7, 4), per_day=4)
        autumn = get_evergreen_seeds_to_inject(date(2026, 10, 20), per_day=4)

        assert summer and autumn
        assert all(theme for _, theme, _ in summer)

    def test_only_a_few_evergreen_seeds_are_injected_per_day(self) -> None:
        pairs = get_evergreen_seeds_to_inject(date(2026, 10, 20), per_day=4)

        assert len(pairs) == 4

    def test_the_evergreen_seeds_rotate_between_days(self) -> None:
        first = [seed for seed, _, _ in get_evergreen_seeds_to_inject(date(2026, 10, 20), 4)]
        second = [seed for seed, _, _ in get_evergreen_seeds_to_inject(date(2026, 10, 21), 4)]

        assert first != second

    def test_every_theme_is_used_over_the_season(self) -> None:
        used: set[str] = set()
        day = date(2026, 10, 1)
        for _ in range(59):
            used.update(theme for _, theme, _ in get_evergreen_seeds_to_inject(day, 4))
            day = date.fromordinal(day.toordinal() + 1)

        assert used == set(EVERGREEN_SEED_RULES)

    def test_the_injected_seed_says_it_is_evergreen(self) -> None:
        from unittest.mock import MagicMock

        db = MagicMock()
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1

        result = inject_evergreen_seeds(db, today=date(2026, 10, 20), per_day=3)

        assert result["injected"] == 3
        kwargs = db.insert_active_seed.call_args_list[0].kwargs
        assert kwargs["origin"] == "evergreen"
        assert kwargs["theme"] in EVERGREEN_SEED_RULES


class TestEnforceCap:
    def test_a_seasonal_seed_goes_before_an_evergreen_one(self) -> None:
        seeds = [
            _seed(1, "halloween mug", score=10, origin="calendar", theme="halloween"),
            _seed(2, "cat lover mug", score=10, origin="evergreen", theme="pets"),
        ]

        archives = enforce_cap(seeds, incoming=0, max_active=1, briefable_ids={1, 2}, reserve=0)

        assert archives == ["halloween mug"]

    def test_an_expired_seasonal_seed_goes_first(self) -> None:
        seeds = [
            _seed(
                1,
                "halloween mug",
                score=10,
                origin="calendar",
                theme="halloween",
                window_end=date(2026, 10, 31),
            ),
            _seed(
                2,
                "christmas mug",
                score=10,
                origin="calendar",
                theme="christmas",
                window_end=date(2026, 12, 25),
            ),
            _seed(3, "cat lover mug", score=10, origin="evergreen", theme="pets"),
        ]

        archives = enforce_cap(
            seeds,
            incoming=0,
            max_active=2,
            briefable_ids={1, 2, 3},
            reserve=0,
            today=date(2026, 11, 2),
        )

        assert "halloween mug" in archives
        assert "cat lover mug" not in archives

    def test_the_evergreen_reserve_is_never_broken(self) -> None:
        seeds = [
            _seed(1, "cat lover mug", score=10, origin="evergreen", theme="pets"),
            _seed(2, "dog mom shirt", score=10, origin="evergreen", theme="pets"),
            _seed(3, "halloween mug", score=10, origin="calendar", theme="halloween"),
        ]

        archives = enforce_cap(
            seeds,
            incoming=0,
            max_active=0,
            briefable_ids={1, 2, 3},
            reserve=0,
            evergreen_reserve=2,
            today=date(2026, 10, 20),
        )

        assert archives == ["halloween mug"]

    def test_a_retired_seed_still_goes_first(self) -> None:
        seeds = [
            _seed(1, "used up seed", score=10, origin="trend"),
            _seed(2, "halloween mug", score=10, origin="calendar", theme="halloween"),
        ]

        archives = enforce_cap(seeds, incoming=0, max_active=1, briefable_ids={2}, reserve=0)

        assert archives == ["used up seed"]


class TestSelectBriefableSeeds:
    def test_evergreen_seeds_survive_a_dominant_season(self) -> None:
        seeds = [
            _seed(index, f"halloween {index}", score=1000, origin="calendar", theme="halloween")
            for index in range(1, 21)
        ] + [
            _seed(21, "cat lover mug", score=1, origin="evergreen", theme="pets"),
        ]

        chosen = select_briefable_seeds(seeds, limit=20)

        assert "cat lover mug" in [seed.query for seed in chosen]

    def test_the_quotas_hold(self) -> None:
        seeds = (
            [
                _seed(index, f"halloween {index}", score=1000, origin="calendar", theme="halloween")
                for index in range(1, 11)
            ]
            + [
                _seed(100 + index, f"pet {index}", score=500, origin="evergreen", theme="pets")
                for index in range(1, 11)
            ]
            + [
                _seed(200 + index, f"trend {index}", score=100, origin="trend")
                for index in range(1, 11)
            ]
        )

        chosen = select_briefable_seeds(
            seeds,
            limit=20,
            seasonal_quota=6,
            evergreen_quota=8,
            trend_quota=6,
        )

        seasonal = [s for s in chosen if s.is_seasonal]
        evergreen = [s for s in chosen if s.is_evergreen]
        trend = [s for s in chosen if not s.is_seasonal and not s.is_evergreen]
        assert len(seasonal) == 6
        assert len(evergreen) == 8
        assert len(trend) == 6

    def test_an_expired_seasonal_seed_is_dropped(self) -> None:
        seeds = [
            _seed(
                1,
                "halloween mug",
                score=1000,
                origin="calendar",
                theme="halloween",
                window_end=date(2026, 10, 31),
            ),
            _seed(2, "cat lover mug", score=10, origin="evergreen", theme="pets"),
        ]

        chosen = select_briefable_seeds(seeds, limit=20, today=date(2026, 11, 2))

        assert [seed.query for seed in chosen] == ["cat lover mug"]

    def test_the_shortfall_is_filled_from_the_rest(self) -> None:
        seeds = [
            _seed(1, "halloween mug", score=1000, origin="calendar", theme="halloween"),
            _seed(2, "cat lover mug", score=10, origin="evergreen", theme="pets"),
        ]

        chosen = select_briefable_seeds(seeds, limit=20)

        assert len(chosen) == 2

    def test_the_limit_is_respected(self) -> None:
        seeds = [
            _seed(index, f"seed {index}", score=index, origin="trend") for index in range(1, 51)
        ]

        chosen = select_briefable_seeds(seeds, limit=20)

        assert len(chosen) == 20
