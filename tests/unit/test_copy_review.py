"""Tests for the LLM copy editor.

The editor is the taste-maker for the text of a design. The code only builds
the context, reads the answer and applies it; it must not decide that a line
is bad because of a word list.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.copy_review import (
    CopyBlockReview,
    CopyReview,
    apply_copy_reviews,
    build_copy_review_prompt,
    copy_review_enabled,
    parse_copy_review_response,
    review_batch_copy,
)
from store_listing.orchestration.design_briefs import DesignBrief


def _brief(
    *,
    concept: str = "A garden gnome driving a night train through a kitchen",
    headline: str | None = "NIGHT TRAIN",
    kicker: str | None = "GNOME ON THE MOVE",
    tagline: str | None = "THROUGH A KITCHEN",
    scene: str | None = "A garden gnome driving a night train through a kitchen",
    copy_mode: str = "type_hero",
) -> DesignBrief:
    return DesignBrief(
        concept=concept,
        product_type="dtf_apparel",
        specific_products=[],
        audience="halloween shoppers",
        visual_style="retro",
        confidence=80,
        reasoning="test",
        source_seed_ids=[],
        layout_type="type_hero",
        headline_text=headline,
        tagline_text=tagline,
        copy_mode=copy_mode,
        kicker_text=kicker,
        scene_description=scene,
    )


def _raw(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    parsed = MagicMock(choices=[choice])
    raw = MagicMock()
    raw.headers = {}
    raw.parse.return_value = parsed
    return raw


def _response(
    *,
    index: int = 1,
    design_stands_alone: bool = True,
    headline: dict[str, Any] | None = None,
    kicker: dict[str, Any] | None = None,
    tagline: dict[str, Any] | None = None,
    overall_action: str = "KEEP",
    overall_reason: str = "The design is wearable.",
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "reviews": [
                {
                    "index": index,
                    "design_stands_alone": design_stands_alone,
                    "headline": headline
                    or {"action": "KEEP", "rewrite": None, "reason": "Strong hero."},
                    "kicker": kicker
                    or {"action": "KEEP", "rewrite": None, "reason": "Sets up the hero."},
                    "tagline": tagline
                    or {"action": "DROP", "rewrite": None, "reason": "The design stands alone."},
                    "overall_action": overall_action,
                    "overall_reason": overall_reason,
                    "confidence": confidence,
                }
            ]
        }
    )


class TestBuildCopyReviewPrompt:
    def test_includes_the_design_and_the_sellability_rules(self) -> None:
        messages = build_copy_review_prompt([_brief()])

        system = messages[0]["content"].lower()
        user = messages[1]["content"]

        assert "design_stands_alone" in system
        assert "drop" in system
        assert "rewrite" in system
        assert "would a stranger wear" in system
        assert "do not explain" in system
        assert "not a list of words" in system
        assert "never names a brand" in system
        assert "night train" in user.lower()
        assert "through a kitchen" in user.lower()

    def test_does_not_include_a_no_text_brief(self) -> None:
        messages = build_copy_review_prompt(
            [_brief(headline=None, kicker=None, tagline=None, copy_mode="no_text")]
        )

        user = json.loads(messages[1]["content"])
        assert user["briefs"] == []

    def test_says_the_headline_can_be_dropped_when_the_picture_explains_it(self) -> None:
        system = build_copy_review_prompt([_brief()])[0]["content"].lower()

        assert "drop the headline" in system
        assert "art-only" in system
        assert "describes the picture" in system

    def test_judges_the_idea_before_the_copy(self) -> None:
        system = build_copy_review_prompt([_brief()])[0]["content"].lower()

        assert "the concept is the idea" in system
        assert "only names a scene" in system
        assert "judge the idea before the copy" in system


class TestParseCopyReviewResponse:
    def test_reads_decisions(self) -> None:
        reviews, error = parse_copy_review_response(
            _response(
                tagline={
                    "action": "REWRITE",
                    "rewrite": "KITCHEN COMMUTE",
                    "reason": "Adds a second joke.",
                }
            ),
            expected_indexes={1},
        )

        assert error is None
        assert len(reviews) == 1
        assert reviews[0].tagline.action == "rewrite"
        assert reviews[0].tagline.rewrite == "KITCHEN COMMUTE"

    def test_an_unknown_action_is_ignored(self) -> None:
        reviews, error = parse_copy_review_response(
            _response(tagline={"action": "MAYBE", "rewrite": None, "reason": "?"}),
            expected_indexes={1},
        )

        assert reviews == []
        assert error is not None

    def test_a_missing_review_is_reported(self) -> None:
        reviews, error = parse_copy_review_response('{"reviews": []}', expected_indexes={1})

        assert reviews == []
        assert error is not None

    def test_a_dropped_headline_is_read(self) -> None:
        reviews, error = parse_copy_review_response(
            _response(
                headline={
                    "action": "DROP",
                    "rewrite": None,
                    "reason": "The picture explains it.",
                }
            ),
            expected_indexes={1},
        )

        assert error is None
        assert reviews[0].headline.action == "drop"


class TestApplyCopyReviews:
    def test_a_drop_removes_the_tagline(self) -> None:
        brief = _brief()
        review = CopyReview(
            index=1,
            design_stands_alone=True,
            headline=CopyBlockReview("keep"),
            kicker=CopyBlockReview("keep"),
            tagline=CopyBlockReview("drop", reason="The design stands alone."),
        )

        result = apply_copy_reviews([brief], [review])

        assert result[0].tagline_text is None
        assert result[0].headline_text == "NIGHT TRAIN"

    def test_a_rewrite_replaces_the_tagline(self) -> None:
        brief = _brief()
        review = CopyReview(
            index=1,
            design_stands_alone=False,
            headline=CopyBlockReview("keep"),
            kicker=CopyBlockReview("keep"),
            tagline=CopyBlockReview("rewrite", rewrite="KITCHEN COMMUTE"),
        )

        result = apply_copy_reviews([brief], [review])

        assert result[0].tagline_text == "KITCHEN COMMUTE"

    def test_a_reject_drops_the_brief(self) -> None:
        brief = _brief()
        review = CopyReview(
            index=1,
            design_stands_alone=False,
            headline=CopyBlockReview("keep"),
            kicker=CopyBlockReview("keep"),
            tagline=CopyBlockReview("drop"),
            overall_action="reject",
            overall_reason="The headline is not wearable.",
        )

        result = apply_copy_reviews([brief], [review])

        assert result == []

    def test_a_headline_rewrite_is_applied(self) -> None:
        brief = _brief(headline="Paper to Flight")
        review = CopyReview(
            index=1,
            design_stands_alone=False,
            headline=CopyBlockReview("rewrite", rewrite="PAPER CRANE"),
            kicker=CopyBlockReview("keep"),
            tagline=CopyBlockReview("drop"),
        )

        result = apply_copy_reviews([brief], [review])

        assert result[0].headline_text == "PAPER CRANE"

    def test_a_dropped_headline_makes_the_design_art_only(self) -> None:
        brief = _brief()
        review = CopyReview(
            index=1,
            design_stands_alone=True,
            headline=CopyBlockReview("drop", reason="The picture explains it."),
            kicker=CopyBlockReview("drop"),
            tagline=CopyBlockReview("drop"),
        )

        result = apply_copy_reviews([brief], [review])

        assert result[0].copy_mode == "no_text"
        assert result[0].layout_type == "full_bleed"
        assert result[0].headline_text is None
        assert result[0].kicker_text is None
        assert result[0].tagline_text is None

    def test_a_dropped_headline_clears_a_surviving_kicker_and_tagline(self) -> None:
        # An art-only design carries no words at all: a kicker or tagline that
        # survived the drop would be text with no headline to anchor it.
        brief = _brief()
        review = CopyReview(
            index=1,
            design_stands_alone=True,
            headline=CopyBlockReview("drop"),
            kicker=CopyBlockReview("keep"),
            tagline=CopyBlockReview("keep"),
        )

        result = apply_copy_reviews([brief], [review])

        assert result[0].copy_mode == "no_text"
        assert result[0].kicker_text is None
        assert result[0].tagline_text is None


class TestReviewBatchCopy:
    def _llm(self, *contents: str) -> MagicMock:
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [_raw(content) for content in contents]
        return llm

    def test_a_drop_uses_only_one_call(self) -> None:
        llm = self._llm(_response())

        result, issues = review_batch_copy(llm, "model", [_brief()])

        assert result[0].tagline_text is None
        assert issues == []
        assert llm.with_raw_response.create.call_count == 1

    def test_a_rewrite_is_re_reviewed(self) -> None:
        llm = self._llm(
            _response(
                design_stands_alone=False,
                tagline={
                    "action": "REWRITE",
                    "rewrite": "KITCHEN COMMUTE",
                    "reason": "Adds a second joke.",
                },
            ),
            _response(
                design_stands_alone=False,
                tagline={
                    "action": "KEEP",
                    "rewrite": None,
                    "reason": "The rewrite adds a second joke.",
                },
            ),
        )

        result, issues = review_batch_copy(llm, "model", [_brief()])

        assert result[0].tagline_text == "KITCHEN COMMUTE"
        assert issues == []
        assert llm.with_raw_response.create.call_count == 2

    def test_a_failed_rewrite_is_dropped(self) -> None:
        llm = self._llm(
            _response(
                design_stands_alone=False,
                tagline={
                    "action": "REWRITE",
                    "rewrite": "A GNOME ON A TRAIN",
                    "reason": "Try again.",
                },
            ),
            _response(
                tagline={
                    "action": "DROP",
                    "rewrite": None,
                    "reason": "Still a scene description.",
                }
            ),
        )

        result, _ = review_batch_copy(llm, "model", [_brief()])

        assert result[0].tagline_text is None

    def test_a_no_text_brief_does_not_call_the_model(self) -> None:
        llm = MagicMock()
        brief = _brief(headline=None, kicker=None, tagline=None, copy_mode="no_text")

        result, issues = review_batch_copy(llm, "model", [brief])

        assert result == [brief]
        assert issues == []
        llm.with_raw_response.create.assert_not_called()

    def test_every_brief_is_reviewed_on_its_own(self) -> None:
        # Batching the editor changes its mind: the same design is judged
        # differently depending on the briefs next to it. One design per call
        # makes the decision a property of that design.
        llm = self._llm(_response(index=1), _response(index=1))
        briefs = [
            _brief(headline="NIGHT TRAIN", tagline="THROUGH A KITCHEN"),
            _brief(headline="CURSES AND COFFEE", tagline="WEEKEND WITCH"),
        ]

        review_batch_copy(llm, "model", briefs)

        assert llm.with_raw_response.create.call_count == 2
        for call in llm.with_raw_response.create.call_args_list:
            user = json.loads(call.kwargs["messages"][1]["content"])
            assert len(user["briefs"]) == 1

    def test_a_dropped_connection_is_asked_again(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            RuntimeError("boom"),
            _raw(_response()),
        ]

        result, issues = review_batch_copy(llm, "model", [_brief()])

        assert result[0].tagline_text is None
        assert issues == []
        assert llm.with_raw_response.create.call_count == 2

    def test_an_unreadable_answer_keeps_the_original_copy(self) -> None:
        llm = self._llm("not json", "not json", "not json")
        brief = _brief()

        result, issues = review_batch_copy(llm, "model", [brief])

        assert result[0].tagline_text == "THROUGH A KITCHEN"
        assert issues

    def test_a_low_confidence_review_keeps_the_original_copy(self) -> None:
        llm = self._llm(_response(confidence=0.2))
        brief = _brief()

        result, issues = review_batch_copy(llm, "model", [brief])

        assert result[0].tagline_text == "THROUGH A KITCHEN"
        assert issues
        assert "confidence" in issues[0]


class TestFeatureFlag:
    def test_is_on_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_COPY_REVIEW", raising=False)
        assert copy_review_enabled() is True

    def test_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_COPY_REVIEW", "false")
        assert copy_review_enabled() is False
