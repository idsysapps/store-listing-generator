"""Unit tests for the seasonal/evergreen mix of a batch of design briefs."""

from dataclasses import dataclass, field

from store_listing.orchestration.brief_mix import (
    BriefMix,
    classify_brief_mix,
    event_theme_for_query,
    seed_themes_for_queries,
)
from store_listing.orchestration.brief_mix_report import group_brief_mix, render


@dataclass
class _Brief:
    concept: str
    confidence: int = 80
    source_seed_ids: list[int] = field(default_factory=list)


class TestEventThemeForQuery:
    def test_a_calendar_seed_belongs_to_its_event(self) -> None:
        assert event_theme_for_query("halloween costume shirt") == "halloween"

    def test_a_seasonal_word_in_a_query_marks_the_event(self) -> None:
        assert event_theme_for_query("spooky halloween mug") == "halloween"

    def test_an_evergreen_seed_has_no_event(self) -> None:
        assert event_theme_for_query("dad jokes") is None

    def test_a_starter_seed_is_never_seasonal(self) -> None:
        # "mom humor" and "dad jokes" are evergreen starters, but the mother's
        # day and father's day rules contain seeds that start the same way.
        assert event_theme_for_query("mom humor") is None
        assert event_theme_for_query("dad jokes") is None

    def test_the_query_is_matched_case_insensitively(self) -> None:
        assert event_theme_for_query("Halloween Mug") == "halloween"


class TestClassifyBriefMix:
    def test_counts_seasonal_and_evergreen_briefs(self) -> None:
        briefs = [
            _Brief("Pumpkin Party", source_seed_ids=[1]),
            _Brief("Coffee Cat", source_seed_ids=[2]),
        ]
        seed_themes = {1: "halloween", 2: None}

        mix = classify_brief_mix(briefs, seed_themes)

        assert mix.total == 2
        assert mix.seasonal == 1
        assert mix.evergreen == 1

    def test_groups_by_theme(self) -> None:
        briefs = [
            _Brief("Pumpkin Party", source_seed_ids=[1]),
            _Brief("Witch Brew", source_seed_ids=[2]),
            _Brief("Coffee Cat", source_seed_ids=[3]),
        ]
        seed_themes = {1: "halloween", 2: "halloween", 3: None}

        mix = classify_brief_mix(briefs, seed_themes)

        assert mix.by_theme == {"halloween": 2}

    def test_a_brief_without_source_seeds_is_evergreen(self) -> None:
        briefs = [_Brief("Custom Idea")]

        mix = classify_brief_mix(briefs, {})

        assert mix.seasonal == 0
        assert mix.evergreen == 1

    def test_a_brief_with_several_seeds_counts_its_event_once(self) -> None:
        briefs = [_Brief("Pumpkin Party", source_seed_ids=[1, 2])]
        seed_themes = {1: "halloween", 2: None}

        mix = classify_brief_mix(briefs, seed_themes)

        assert mix.seasonal == 1
        assert mix.by_theme == {"halloween": 1}

    def test_only_the_listed_themes_count_as_seasonal(self) -> None:
        # A theme that is not a calendar event (an evergreen category) is
        # evergreen even though the seed has a theme.
        briefs = [_Brief("Golden Retriever", source_seed_ids=[1])]
        seed_themes = {1: "pets"}

        mix = classify_brief_mix(briefs, seed_themes, seasonal_themes=["halloween"])

        assert mix.seasonal == 0
        assert mix.evergreen == 1

    def test_the_summary_names_the_counts(self) -> None:
        mix = BriefMix(total=3, seasonal=1, evergreen=2, by_theme={"halloween": 1})

        summary = mix.summary()

        assert "seasonal=1" in summary
        assert "evergreen=2" in summary
        assert "halloween" in summary


class TestSeedThemesForQueries:
    def test_maps_a_seed_id_to_its_theme(self) -> None:
        themes = seed_themes_for_queries([(1, "halloween mug"), (2, "dad jokes")])

        assert themes == {1: "halloween", 2: None}


class TestGroupBriefMix:
    def test_groups_the_links_by_batch(self) -> None:
        links = [
            ("2026-10-20", "Pumpkin Party", "halloween costume shirt"),
            ("2026-10-20", "Coffee Cat", "dad jokes"),
            ("2026-10-21", "Witch Brew", "spooky mug"),
        ]

        batches = group_brief_mix(links)

        by_id = {batch.batch_id: batch for batch in batches}
        assert by_id["2026-10-20"].total == 2
        assert by_id["2026-10-20"].seasonal == 1
        assert by_id["2026-10-20"].evergreen == 1
        assert by_id["2026-10-21"].by_theme == {"halloween": 1}

    def test_a_brief_with_several_seeds_is_counted_once(self) -> None:
        links = [
            ("2026-10-20", "Pumpkin Party", "halloween costume shirt"),
            ("2026-10-20", "Pumpkin Party", "dad jokes"),
        ]

        batches = group_brief_mix(links)

        assert batches[0].total == 1
        assert batches[0].seasonal == 1

    def test_the_report_totals_every_batch(self) -> None:
        links = [
            ("2026-10-20", "Pumpkin Party", "halloween costume shirt"),
            ("2026-10-21", "Coffee Cat", "dad jokes"),
        ]

        report = render(group_brief_mix(links))

        assert "ALL: total=2 seasonal=1 evergreen=1" in report
