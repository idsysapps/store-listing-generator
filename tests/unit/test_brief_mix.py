"""Unit tests for the seasonal/evergreen mix of a batch of design briefs."""

from dataclasses import dataclass, field

from store_listing.orchestration.brief_mix import (
    BriefMix,
    classify_brief_mix,
    enforce_style_mix,
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
            ("2026-10-20", "Pumpkin Party", "halloween costume shirt", "halloween"),
            ("2026-10-20", "Coffee Cat", "dad jokes", None),
            ("2026-10-21", "Witch Brew", "spooky mug", "halloween"),
        ]

        batches = group_brief_mix(links)

        by_id = {batch.batch_id: batch for batch in batches}
        assert by_id["2026-10-20"].total == 2
        assert by_id["2026-10-20"].seasonal == 1
        assert by_id["2026-10-20"].evergreen == 1
        assert by_id["2026-10-21"].by_theme == {"halloween": 1}

    def test_a_brief_with_several_seeds_is_counted_once(self) -> None:
        links = [
            ("2026-10-20", "Pumpkin Party", "halloween costume shirt", "halloween"),
            ("2026-10-20", "Pumpkin Party", "dad jokes", None),
        ]

        batches = group_brief_mix(links)

        assert batches[0].total == 1
        assert batches[0].seasonal == 1

    def test_a_seed_without_a_theme_is_guessed_from_its_query(self) -> None:
        links = [("2026-10-20", "Pumpkin Party", "halloween costume shirt", None)]

        batches = group_brief_mix(links)

        assert batches[0].seasonal == 1

    def test_the_report_totals_every_batch(self) -> None:
        links = [
            ("2026-10-20", "Pumpkin Party", "halloween costume shirt", "halloween"),
            ("2026-10-21", "Coffee Cat", "dad jokes", None),
        ]

        report = render(group_brief_mix(links))

        assert "ALL: total=2 seasonal=1 evergreen=1" in report


class _StyleBrief:
    """A brief with a look and a voice."""

    def __init__(
        self,
        concept: str,
        voice: str = "deadpan",
        style_preset: str | None = "halftone_comic",
        confidence: int = 80,
    ) -> None:
        self.concept = concept
        self.voice = voice
        self.style_preset = style_preset
        self.confidence = confidence


class TestEnforceStyleMix:
    def test_a_batch_of_one_look_is_thinned_out(self) -> None:
        # A batch that came back with the same look three times over looks
        # like one design repeated (issue #227).
        briefs = [_StyleBrief(f"Design {index}") for index in range(5)]

        kept, dropped = enforce_style_mix(briefs, max_per_preset=3)

        assert len(kept) == 3
        assert len(dropped) == 2

    def test_the_looks_of_a_batch_are_kept(self) -> None:
        briefs = [
            _StyleBrief("A", style_preset="halftone_comic"),
            _StyleBrief("B", style_preset="kawaii_flat"),
            _StyleBrief("C", style_preset="art_nouveau"),
        ]

        kept, dropped = enforce_style_mix(briefs, max_per_preset=1)

        assert len(kept) == 3
        assert dropped == []

    def test_the_voices_of_a_batch_are_kept(self) -> None:
        briefs = [
            _StyleBrief("A", voice="deadpan", style_preset="halftone_comic"),
            _StyleBrief("B", voice="loud", style_preset="kawaii_flat"),
            _StyleBrief("C", voice="cozy", style_preset="art_nouveau"),
        ]

        kept, dropped = enforce_style_mix(briefs, max_per_voice=1)

        assert len(kept) == 3
        assert dropped == []

    def test_one_voice_does_not_fill_the_batch(self) -> None:
        briefs = [_StyleBrief(f"Design {index}", voice="deadpan") for index in range(4)]

        kept, dropped = enforce_style_mix(briefs, max_per_voice=2)

        assert len(kept) == 2
        assert len(dropped) == 2

    def test_a_brief_without_a_look_is_never_dropped_for_it(self) -> None:
        briefs = [_StyleBrief(f"Design {index}", style_preset=None) for index in range(6)]

        kept, dropped = enforce_style_mix(briefs, max_per_preset=1)

        assert len(kept) == 6
        assert dropped == []

    def test_the_best_briefs_are_the_ones_kept(self) -> None:
        briefs = [
            _StyleBrief("Low", confidence=10),
            _StyleBrief("High", confidence=99),
        ]

        kept, _ = enforce_style_mix(briefs, max_per_preset=1)

        assert kept[0].concept == "High"
