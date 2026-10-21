"""Unit tests for the diversity of a batch of design briefs (issue #200).

A batch of briefs used to be whatever the model felt like on the day, and
the rules the prompts shared used a holiday as their example, so a season
took the whole batch.
"""

import json
import re
from dataclasses import dataclass, field

from store_listing.orchestration.brief_mix import (
    enforce_brief_mix,
    evergreen_count,
)
from store_listing.orchestration.brief_prompts import (
    DESIGN_ONLY_RULES,
    DIVERSITY_RULES,
    LAYOUT_RULES,
    PRODUCT_CATALOG,
    SEASONAL_WORDS,
)
from store_listing.orchestration.design_briefs import build_brief_prompt, diversity_rules


@dataclass
class _Brief:
    concept: str
    confidence: int = 80
    source_seed_ids: list[int] = field(default_factory=list)


class TestSharedRulesHaveNoSeasonalExample:
    def test_the_shared_rules_mention_no_holiday(self) -> None:
        rules = (f"{PRODUCT_CATALOG} {LAYOUT_RULES} {DESIGN_ONLY_RULES} {DIVERSITY_RULES}").lower()

        for word in SEASONAL_WORDS:
            assert not re.search(rf"\b{re.escape(word)}", rules), f"the rules mention {word}"


class TestDiversityRules:
    def test_the_rules_name_the_quota_per_event(self) -> None:
        rules = diversity_rules(max_per_theme=2, min_evergreen=4)

        assert "At most 2 briefs" in rules

    def test_the_rules_name_the_evergreen_minimum(self) -> None:
        rules = diversity_rules(max_per_theme=2, min_evergreen=4)

        assert "At least 4 briefs" in rules

    def test_the_rules_say_the_calendar_is_not_the_default(self) -> None:
        rules = diversity_rules().lower()

        assert "not the default" in rules


class TestBuildBriefPrompt:
    def _messages(self, **kwargs):
        from store_listing.orchestration.design_briefs import BriefableCandidate

        candidate = BriefableCandidate(
            active_seed_id=1,
            query="dad jokes shirt",
            source="google",
            query_type="rising",
            score=100,
            delta=5000,
            promotion_score=5000,
        )
        return build_brief_prompt(
            candidates=[candidate],
            context=kwargs.pop("context", []),
            today=kwargs.pop("today", __import__("datetime").date(2026, 10, 21)),
            **kwargs,
        )

    def test_the_prompt_caps_the_briefs_per_event(self) -> None:
        messages = self._messages()

        assert "At most" in messages[0]["content"]

    def test_the_prompt_requires_evergreen_briefs(self) -> None:
        messages = self._messages()

        assert "evergreen" in messages[0]["content"]

    def test_the_prompt_does_not_pass_the_event_seed_lists(self) -> None:
        messages = self._messages(
            context=[{"name": "halloween", "days_until": 10, "seeds": ["spooky mug"]}]
        )

        payload = json.loads(messages[1]["content"])
        assert payload["upcoming_events"] == [
            {"name": "halloween", "days_until": 10, "theme": "halloween"}
        ]
        assert "spooky mug" not in messages[1]["content"]

    def test_the_prompt_asks_for_evergreen_reasoning(self) -> None:
        messages = self._messages()

        assert "timely or evergreen" in messages[0]["content"]

    def test_extra_instructions_reach_the_prompt(self) -> None:
        messages = self._messages(extra_instructions=["Write more evergreen briefs."])

        assert "Write more evergreen briefs." in messages[0]["content"]


class TestEnforceBriefMix:
    def test_the_briefs_past_the_event_quota_are_dropped(self) -> None:
        briefs = [
            _Brief("Pumpkin Party", source_seed_ids=[1]),
            _Brief("Witch Brew", source_seed_ids=[1]),
            _Brief("Ghost Rider", source_seed_ids=[1]),
        ]
        seed_themes = {1: "halloween"}

        kept, dropped = enforce_brief_mix(briefs, seed_themes, max_per_theme=2)

        assert len(kept) == 2
        assert len(dropped) == 1

    def test_the_kept_briefs_are_the_most_confident_ones(self) -> None:
        briefs = [
            _Brief("Low", confidence=40, source_seed_ids=[1]),
            _Brief("High", confidence=90, source_seed_ids=[1]),
            _Brief("Middle", confidence=60, source_seed_ids=[1]),
        ]
        seed_themes = {1: "halloween"}

        kept, _dropped = enforce_brief_mix(briefs, seed_themes, max_per_theme=2)

        assert [brief.concept for brief in kept] == ["High", "Middle"]

    def test_two_events_keep_their_own_quota(self) -> None:
        briefs = [
            _Brief("Pumpkin Party", source_seed_ids=[1]),
            _Brief("Witch Brew", source_seed_ids=[1]),
            _Brief("Christmas Sweater", source_seed_ids=[2]),
            _Brief("Santa Mug", source_seed_ids=[2]),
        ]
        seed_themes = {1: "halloween", 2: "christmas"}

        kept, dropped = enforce_brief_mix(briefs, seed_themes, max_per_theme=2)

        assert len(kept) == 4
        assert dropped == []

    def test_evergreen_briefs_are_never_dropped(self) -> None:
        briefs = [
            _Brief("Pumpkin Party", source_seed_ids=[1]),
            _Brief("Witch Brew", source_seed_ids=[1]),
            _Brief("Coffee Cat", source_seed_ids=[2]),
            _Brief("Garden Gnome", source_seed_ids=[3]),
        ]
        seed_themes = {1: "halloween", 2: None, 3: "pets"}

        kept, _dropped = enforce_brief_mix(briefs, seed_themes, max_per_theme=2)

        assert "Coffee Cat" in [brief.concept for brief in kept]
        assert "Garden Gnome" in [brief.concept for brief in kept]

    def test_the_evergreen_count_ignores_the_event_briefs(self) -> None:
        briefs = [
            _Brief("Pumpkin Party", source_seed_ids=[1]),
            _Brief("Coffee Cat", source_seed_ids=[2]),
        ]
        seed_themes = {1: "halloween", 2: None}

        assert evergreen_count(briefs, seed_themes) == 1
