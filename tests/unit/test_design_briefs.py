"""Unit tests for LLM-driven design brief generation."""

import json
import logging
import re
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.brief_prompts import SEASONAL_WORDS
from store_listing.orchestration.design_briefs import (
    BriefableCandidate,
    DesignBrief,
    assign_batch_devices,
    assign_batch_templates,
    build_brief_prompt,
    generate_design_briefs,
    parse_brief_response,
    read_brief_response,
)
from store_listing.orchestration.idea_devices import DEVICE_IDS
from store_listing.orchestration.typography import family_for_voice


def _briefable(
    active_seed_id: int,
    query: str,
    source: str = "google",
    query_type: str = "rising",
    score: int = 100,
    delta: int = 5000,
    product_tags: list[str] | None = None,
) -> BriefableCandidate:
    return BriefableCandidate(
        active_seed_id=active_seed_id,
        query=query,
        source=source,
        query_type=query_type,
        score=score,
        delta=delta,
        promotion_score=5000,
        product_tags=product_tags or [],
    )


class TestBuildBriefPrompt:
    def test_the_prompt_does_not_let_the_model_pick_the_geometry(self) -> None:
        # The model used to answer text_top every time, and the batch came
        # back as one silhouette (issue #238).
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
        )

        system = messages[0]["content"]
        assert "'auto'" in system
        assert "text_top" not in system
        assert "badge" not in system
        assert "stack_side" not in system

    def test_the_prompt_asks_for_print_artwork_not_a_product_photo(self) -> None:
        # A scene that describes a garment or a mug is rendered as a mockup, and
        # a mockup is not a printable file (issue #188).
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
        )

        system = messages[0]["content"].lower()
        assert "print artwork" in system
        assert "not a photo of a product" in system

    def test_the_prompt_asks_for_a_headline_in_words(self) -> None:
        # The headline is drawn with a font, and the font has no emoji.
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
        )

        assert "no emoji" in messages[0]["content"].lower()

    def test_the_prompt_says_the_concept_is_not_the_product(self) -> None:
        # A concept that names the product ends up in the image prompt and the
        # model draws the product instead of the design (issue #195).
        messages = build_brief_prompt(
            candidates=[_briefable(1, "halloween tote")],
            context=[],
            today=date(2026, 10, 21),
        )

        system = messages[0]["content"].lower()
        assert "not the product" in system
        assert "weekend tote with coffee beans" in system

    def test_the_prompt_says_the_scene_never_names_the_product(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "halloween tote")],
            context=[],
            today=date(2026, 10, 21),
        )

        system = messages[0]["content"].lower()
        assert "never mention a bag" in system

    def test_the_prompt_offers_a_typography_only_layout(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
        )

        assert "text_only" in messages[0]["content"]

    def test_returns_system_and_user_messages(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[{"name": "halloween", "days_until": 10}],
            today=date(2026, 10, 21),
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_seed_data(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(42, "dad jokes shirt", source="amazon")],
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert "amazon" in user_data["signals_by_source"]
        seeds = user_data["signals_by_source"]["amazon"]
        assert len(seeds) == 1
        assert seeds[0]["id"] == 42
        assert seeds[0]["query"] == "dad jokes shirt"

    def test_groups_seeds_by_source(self) -> None:
        messages = build_brief_prompt(
            candidates=[
                _briefable(1, "dad jokes shirt", source="amazon"),
                _briefable(2, "dad jokes 2026", source="google"),
                _briefable(3, "funny mugs", source="amazon"),
            ],
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert len(user_data["signals_by_source"]["amazon"]) == 2
        assert len(user_data["signals_by_source"]["google"]) == 1

    def test_includes_date_and_context(self) -> None:
        messages = build_brief_prompt(
            candidates=[],
            context=[{"name": "halloween", "days_until": 5}],
            today=date(2026, 10, 26),
        )

        user_data = json.loads(messages[1]["content"])
        assert user_data["date"] == "2026-10-26"
        assert user_data["upcoming_events"][0]["name"] == "halloween"

    def test_includes_product_tags_per_seed(self) -> None:
        messages = build_brief_prompt(
            candidates=[
                _briefable(1, "dad jokes shirt", product_tags=["dtf_apparel"]),
            ],
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        source_seeds = user_data["signals_by_source"]["google"]
        assert source_seeds[0]["product_tags"] == ["dtf_apparel"]

    def test_respects_max_seeds_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import store_listing.orchestration.design_briefs as mod

        monkeypatch.setattr(mod, "MAX_SEEDS_PER_BRIEF", 2)

        candidates = [
            _briefable(1, "a"),
            _briefable(2, "b"),
            _briefable(3, "c"),
        ]
        messages = build_brief_prompt(candidates, context=[], today=date(2026, 10, 21))

        user_data = json.loads(messages[1]["content"])
        total = sum(len(seeds) for seeds in user_data["signals_by_source"].values())
        assert total == 2


class TestParseBriefResponse:
    def test_parses_well_formed_json(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Dad Jokes That'll Get You Divorced",
                        "product_type": "dtf_apparel",
                        "specific_products": ["t-shirt", "hoodie"],
                        "audience": "men 30-50, gift buyers",
                        "visual_style": "bold serif text",
                        "confidence": 85,
                        "reasoning": "Amazon buyer intent + Google rising",
                        "source_seed_ids": [1, 2],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Dad Jokes That'll Get You Divorced"
        assert result[0].product_type == "dtf_apparel"
        assert result[0].specific_products == ["t-shirt", "hoodie"]
        assert result[0].confidence == 85
        assert result[0].source_seed_ids == [1, 2]

    def test_handles_json_with_surrounding_text(self) -> None:
        response = 'Here is my analysis:\n{"briefs": [{"concept": "Test", "product_type": "dtf_apparel", "confidence": 50, "source_seed_ids": [1]}]}\nDone.'

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Test"

    def test_handles_malformed_json(self) -> None:
        result = parse_brief_response("this is not json at all")
        assert result == []

    def test_reads_a_batch_the_model_appended_something_to(self) -> None:
        # The model answered with the batch and then another object. The
        # parser sliced from the first "{" to the last "}", read two values as
        # one and threw the whole batch away (issue #209).
        response = (
            json.dumps(
                {
                    "briefs": [
                        {
                            "concept": "Coffee Beans",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [1],
                        }
                    ]
                }
            )
            + "\n"
            + json.dumps({"notes": {"summary": "a second object"}})
        )

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Coffee Beans"

    def test_reads_a_batch_the_model_appended_prose_to(self) -> None:
        response = (
            json.dumps(
                {
                    "briefs": [
                        {
                            "concept": "Coffee Beans",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [1],
                        }
                    ]
                }
            )
            + "\nI also wrote a note {like this}."
        )

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Coffee Beans"

    def test_merges_a_batch_the_model_split_in_two(self) -> None:
        first = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "One",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )
        second = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Two",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = parse_brief_response(first + "\n" + second)

        assert {brief.concept for brief in result} == {"One", "Two"}

    def test_does_not_duplicate_briefs_when_the_batch_was_repeated(self) -> None:
        batch = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "One",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = parse_brief_response(batch + "\n" + batch)

        assert len(result) == 1
        assert result[0].concept == "One"

    def test_handles_empty_response(self) -> None:
        result = parse_brief_response("")
        assert result == []

    def test_filters_invalid_product_types(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Good brief",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    },
                    {
                        "concept": "Bad brief",
                        "product_type": "invalid_type",
                        "confidence": 50,
                        "source_seed_ids": [2],
                    },
                ]
            }
        )

        result = parse_brief_response(response)

        assert len(result) == 1
        assert result[0].concept == "Good brief"

    def test_filters_briefs_without_concept(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    },
                    {
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [2],
                    },
                ]
            }
        )

        result = parse_brief_response(response)
        assert result == []

    def test_clamps_confidence_to_valid_range(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Over",
                        "product_type": "dtf_apparel",
                        "confidence": 150,
                        "source_seed_ids": [1],
                    },
                    {
                        "concept": "Under",
                        "product_type": "sublimation",
                        "confidence": -10,
                        "source_seed_ids": [2],
                    },
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].confidence == 100
        assert result[1].confidence == 0

    def test_filters_non_integer_source_ids(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1, "bad", None, 2],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].source_seed_ids == [1, 2]

    def test_handles_missing_optional_fields(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Minimal",
                        "product_type": "sticker_vinyl",
                        "confidence": 70,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].specific_products == []
        assert result[0].audience == ""
        assert result[0].visual_style == ""
        assert result[0].reasoning == ""

    def test_non_list_briefs_returns_empty(self) -> None:
        response = json.dumps({"briefs": "not a list"})
        assert parse_brief_response(response) == []

    def test_non_dict_brief_entries_skipped(self) -> None:
        response = json.dumps({"briefs": ["not a dict", 42]})
        assert parse_brief_response(response) == []

    def test_parses_layout_fields(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Born To Be Spooky",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_top",
                        "headline_text": "Born To Be Spooky",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "text_top"
        assert result[0].headline_text == "Born To Be Spooky"
        assert result[0].tagline_text is None

    def test_parses_text_top_bottom_layout(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Rescue Dog Halloween",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                        "layout_type": "text_top_bottom",
                        "headline_text": "Adopted My",
                        "tagline_text": "Costume Too",
                        "font_color": "#FF6600",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "text_top_bottom"
        assert result[0].headline_text == "Adopted My"
        assert result[0].tagline_text == "Costume Too"
        assert result[0].font_color == "#FF6600"

    def test_defaults_layout_to_full_bleed(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Abstract Pattern",
                        "product_type": "sublimation",
                        "confidence": 70,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "full_bleed"
        assert result[0].headline_text is None
        assert result[0].tagline_text is None

    def test_invalid_layout_type_defaults_to_full_bleed(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                        "layout_type": "invalid_layout",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "full_bleed"

    def test_parses_a_typography_only_brief(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Candy Scatter",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_only",
                        "headline_text": "Candy Scatter",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].layout_type == "text_only"
        assert result[0].headline_text == "Candy Scatter"

    def test_a_typography_only_brief_needs_no_scene(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Candy Scatter",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_only",
                        "headline_text": "Candy Scatter",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].scene_description is None

    def test_a_scene_that_asks_for_words_loses_them(self) -> None:
        # The image model cannot spell: a scene that quotes the punchline makes
        # it draw the words wrong (issue #208).
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Pumpkin joke",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_top_bottom",
                        "headline_text": "Why Did the Pumpkin",
                        "tagline_text": "Cross the Road?",
                        "scene_description": (
                            "A grinning pumpkin with a speech bubble saying "
                            "'To get to the other side!' with a small ghost peeking out"
                        ),
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        scene = result[0].scene_description
        assert scene is not None
        assert "other side" not in scene
        assert "speech bubble" not in scene
        assert "pumpkin" in scene

    def test_a_scene_that_is_only_words_is_dropped(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Pumpkin joke",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_top_bottom",
                        "headline_text": "Why Did the Pumpkin",
                        "scene_description": "the words 'TO GET TO THE OTHER SIDE'",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result == []

    def test_a_typography_only_brief_falls_back_to_a_short_concept(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Candy Scatter",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_only",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].headline_text == "Candy Scatter"

    def test_a_typography_only_brief_without_a_usable_headline_is_dropped(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "A concept that is far too long to be a headline for a shirt",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "layout_type": "text_only",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result == []


class _ApiError(Exception):
    """An error that carries an HTTP status, like the OpenAI SDK raises."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class TestGenerateDesignBriefs:
    def _mock_db(self, candidates: list[BriefableCandidate] | None = None) -> MagicMock:
        db = MagicMock()
        db.list_briefable_seeds.return_value = candidates or []
        db.list_active_materials.return_value = []
        db.insert_design_brief.return_value = 1
        db.insert_brief_source.return_value = 1
        return db

    def _mock_llm(self, response_json: dict) -> MagicMock:
        llm = MagicMock()
        llm.with_raw_response.create.return_value = self._raw(response_json)
        return llm

    def _raw(self, response_json: dict) -> MagicMock:
        choice = MagicMock()
        choice.message.content = json.dumps(response_json)
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        return raw

    def _raw_text(self, content: str) -> MagicMock:
        choice = MagicMock()
        choice.message.content = content
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        return raw

    def test_no_candidates_returns_early(self) -> None:
        db = self._mock_db()
        llm = MagicMock()

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "success"
        assert result["briefs_created"] == 0
        assert not llm.with_raw_response.create.called

    def test_no_candidates_says_why_it_skipped(self) -> None:
        # A bare zero used to look like a silent failure (issue #192).
        db = self._mock_db()
        llm = MagicMock()

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["reason"] == "no briefable seeds"
        assert result["briefable"] == 0

    def test_a_model_the_endpoint_does_not_serve_is_reported_clearly(self) -> None:
        # The cluster pointed at a model the DGX proxy does not serve, so
        # every brief call failed with a generic LLM error that did not say
        # which model was missing (issue #194).
        db = self._mock_db([_briefable(1, "funny t-shirt")])
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = _ApiError(
            400,
            "Unknown model: qwen2.5. Available: ['qwen38_briefs', 'qwen_vl_ocr']",
        )

        result = generate_design_briefs(
            db, llm, "qwen2.5:32b-instruct-q4_K_M", today=date(2026, 10, 1)
        )

        assert result["status"] == "error"
        assert "qwen38_briefs" in result["error"]

    def test_the_result_says_which_model_was_asked(self) -> None:
        db = self._mock_db([_briefable(1, "funny t-shirt")])
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = _ApiError(500, "boom")

        result = generate_design_briefs(db, llm, "qwen38_briefs", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "boom" in result["error"]

    def test_an_unreadable_answer_is_a_failure_not_an_empty_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A parse failure used to look like a successful batch of zero
        # briefs, so the whole run was silently lost (issue #209).
        import store_listing.orchestration.design_briefs as mod

        monkeypatch.setattr(mod, "MIN_EVERGREEN_BRIEFS", 0)
        db = self._mock_db([_briefable(1, "funny t-shirt")])
        llm = MagicMock()
        llm.with_raw_response.create.return_value = self._raw_text("I could not do it.")

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"

    def test_an_unreadable_answer_is_asked_again_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import store_listing.orchestration.design_briefs as mod

        monkeypatch.setattr(mod, "MIN_EVERGREEN_BRIEFS", 0)
        db = self._mock_db([_briefable(1, "funny t-shirt")])
        llm = MagicMock()
        llm.with_raw_response.create.return_value = self._raw_text("I could not do it.")

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert llm.with_raw_response.create.call_count == 2

    def test_the_second_attempt_can_recover_the_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import store_listing.orchestration.design_briefs as mod

        monkeypatch.setattr(mod, "MIN_EVERGREEN_BRIEFS", 0)
        db = self._mock_db([_briefable(1, "funny t-shirt")])
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            self._raw_text("I could not do it."),
            self._raw(
                {
                    "briefs": [
                        {
                            "concept": "Coffee Beans",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [1],
                        }
                    ]
                }
            ),
        ]

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "success"
        assert result["briefs_created"] == 1

    def test_the_raw_answer_is_logged_when_it_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The raw answer is the only way to see what the model actually sent
        # (issue #209).
        import store_listing.orchestration.design_briefs as mod

        monkeypatch.setattr(mod, "MIN_EVERGREEN_BRIEFS", 0)
        db = self._mock_db([_briefable(1, "funny t-shirt")])
        llm = MagicMock()
        llm.with_raw_response.create.return_value = self._raw_text("I could not do it.")

        with caplog.at_level(logging.ERROR):
            generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert "I could not do it." in caplog.text

    def test_reports_how_many_seeds_were_available(self) -> None:
        candidates = [_briefable(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [42],
                    }
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["briefable"] == 1

    def test_the_briefs_past_the_event_quota_are_not_created(self) -> None:
        # Three briefs for the same event is a season taking the batch
        # (issue #200): only the quota is kept.
        candidates = [_briefable(1, "halloween costume shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": f"Pumpkin {index}",
                        "product_type": "dtf_apparel",
                        "confidence": 80 - index,
                        "source_seed_ids": [1],
                    }
                    for index in range(3)
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["briefs_created"] == 2
        assert result["dropped"] == 1
        assert db.insert_design_brief.call_count == 2

    def test_a_seasonal_batch_is_asked_for_once_more(self) -> None:
        candidates = [_briefable(1, "halloween costume shirt"), _briefable(2, "cat lover mug")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Pumpkin Party",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert llm.with_raw_response.create.call_count == 2

    def test_the_second_answer_is_used_when_it_has_more_evergreen_briefs(self) -> None:
        candidates = [_briefable(1, "halloween costume shirt"), _briefable(2, "cat lover mug")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            self._raw(
                {
                    "briefs": [
                        {
                            "concept": "Pumpkin Party",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [1],
                        }
                    ]
                }
            ),
            self._raw(
                {
                    "briefs": [
                        {
                            "concept": "Coffee Cat",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [2],
                        },
                        {
                            "concept": "Garden Gnome",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [2],
                        },
                    ]
                }
            ),
        ]

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["briefs_created"] == 2
        assert result["evergreen"] == 2
        concepts = [call.kwargs["concept"] for call in db.insert_design_brief.call_args_list]
        assert concepts == ["Coffee Cat", "Garden Gnome"]

    def test_a_batch_that_is_asked_for_again_still_gets_templates(self) -> None:
        candidates = [_briefable(1, "halloween costume shirt"), _briefable(2, "cat lover mug")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            self._raw(
                {
                    "briefs": [
                        {
                            "concept": "Pumpkin Party",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [1],
                        }
                    ]
                }
            ),
            self._raw(
                {
                    "briefs": [
                        {
                            "concept": "Coffee Cat",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [2],
                            "headline_text": "COFFEE CAT",
                        },
                        {
                            "concept": "Garden Gnome",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [2],
                            "headline_text": "GARDEN GNOME",
                        },
                    ]
                }
            ),
        ]

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        layouts = [call.kwargs["layout_type"] for call in db.insert_design_brief.call_args_list]
        assert len(set(layouts)) == len(layouts)
        assert not {"auto", "text_top", "text_top_bottom", "full_bleed"} & set(layouts)

    def test_warns_when_a_brief_has_no_usable_source_seed(self, caplog) -> None:
        candidates = [_briefable(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [999],
                    }
                ]
            }
        )

        with caplog.at_level("WARNING"):
            generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert "999" in caplog.text
        assert not db.insert_brief_source.called

    def test_generates_briefs_from_llm(self) -> None:
        candidates = [_briefable(42, "dad jokes shirt", source="amazon")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Dad Jokes That'll Get You Divorced",
                        "product_type": "dtf_apparel",
                        "specific_products": ["t-shirt"],
                        "audience": "men 30-50",
                        "visual_style": "bold text",
                        "confidence": 85,
                        "reasoning": "buyer intent",
                        "source_seed_ids": [42],
                    }
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "success"
        assert result["briefs_created"] == 1
        db.insert_design_brief.assert_called_once()
        db.insert_brief_source.assert_called_once_with(brief_id=1, active_seed_id=42)

    def test_logs_the_seasonal_evergreen_mix(self, caplog) -> None:
        # A batch that is all seasonal is the symptom this reports (issue #199).
        candidates = [_briefable(1, "halloween costume shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Pumpkin Party",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        with caplog.at_level("INFO"):
            result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert "seasonal=1" in caplog.text
        assert result["seasonal"] == 1
        assert result["evergreen"] == 0

    def test_reports_an_evergreen_batch_as_evergreen(self) -> None:
        candidates = [_briefable(1, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Dad Jokes",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["seasonal"] == 0
        assert result["evergreen"] == 1

    def test_validates_source_ids_against_actual_candidates(self) -> None:
        candidates = [_briefable(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [42, 999],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        source_calls = db.insert_brief_source.call_args_list
        assert len(source_calls) == 1
        assert source_calls[0].kwargs["active_seed_id"] == 42

    def test_handles_llm_error_gracefully(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "API down" in result["error"]

    def test_skips_briefs_with_invalid_product_type(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Bad",
                        "product_type": "invalid",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["briefs_created"] == 0
        assert not db.insert_design_brief.called

    def test_inserts_brief_source_for_each_valid_seed_id(self) -> None:
        candidates = [
            _briefable(10, "dad jokes"),
            _briefable(20, "funny mugs"),
        ]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Multi-source brief",
                        "product_type": "dtf_apparel",
                        "confidence": 75,
                        "source_seed_ids": [10, 20],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        source_calls = db.insert_brief_source.call_args_list
        assert len(source_calls) == 2
        seed_ids = {c.kwargs["active_seed_id"] for c in source_calls}
        assert seed_ids == {10, 20}

    def test_batch_id_is_todays_date(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["batch_id"] == "2026-10-01"

    def test_passes_model_to_insert(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Test",
                        "product_type": "dtf_apparel",
                        "confidence": 50,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["llm_model"] == "test-model"

    def test_queries_materials_and_passes_to_prompt(self) -> None:
        candidates = [_briefable(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        db.list_active_materials.return_value = [
            {
                "external_id": "bc3001-white",
                "name": "Bella Canvas 3001",
                "brand": "Bella Canvas",
                "substrate_type": "t-shirt",
                "color": "White",
                "product_type": "dtf_apparel",
                "sku": "BC3001-WHT",
                "unit_cost": 4.50,
            }
        ]
        llm = self._mock_llm(
            {
                "briefs": [
                    {
                        "concept": "Hoodie Season",
                        "product_type": "dtf_apparel",
                        "confidence": 75,
                        "source_seed_ids": [1],
                        "recommended_materials": [
                            {
                                "external_id": "bc3001-white",
                                "name": "Bella Canvas 3001",
                                "color": "White",
                                "substrate_type": "t-shirt",
                                "unit_cost": 4.50,
                            }
                        ],
                    }
                ]
            }
        )

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["briefs_created"] == 1
        db.list_active_materials.assert_called_once()
        call_kwargs = db.insert_design_brief.call_args.kwargs
        assert call_kwargs["recommended_materials"] is not None


class TestBuildBriefPromptWithMaterials:
    def _sample_materials(self) -> list[dict[str, Any]]:
        return [
            {
                "external_id": "bc3001-white",
                "name": "Bella Canvas 3001",
                "brand": "Bella Canvas",
                "substrate_type": "t-shirt",
                "color": "White",
                "product_type": "dtf_apparel",
                "sku": "BC3001-WHT",
                "unit_cost": 4.50,
            },
            {
                "external_id": "mug-11oz-white",
                "name": "11oz Ceramic Mug",
                "brand": "Generic",
                "substrate_type": "mug",
                "color": "White",
                "product_type": "sublimation",
                "sku": "MUG-11OZ-WHT",
                "unit_cost": 2.75,
            },
        ]

    def test_includes_available_materials_in_user_message(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
            materials=self._sample_materials(),
        )

        user_data = json.loads(messages[1]["content"])
        assert "available_materials" in user_data
        assert len(user_data["available_materials"]) == 2
        assert user_data["available_materials"][0]["external_id"] == "bc3001-white"

    def test_omits_available_materials_when_none(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
        )

        user_data = json.loads(messages[1]["content"])
        assert "available_materials" not in user_data

    def test_omits_available_materials_when_empty(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
            materials=[],
        )

        user_data = json.loads(messages[1]["content"])
        assert "available_materials" not in user_data

    def test_system_prompt_mentions_materials(self) -> None:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 21),
            materials=self._sample_materials(),
        )

        system_content = messages[0]["content"]
        assert "recommended_materials" in system_content


class TestParseBriefResponseWithMaterials:
    def test_parses_recommended_materials(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Dad Jokes Shirt",
                        "product_type": "dtf_apparel",
                        "confidence": 85,
                        "source_seed_ids": [1],
                        "recommended_materials": [
                            {
                                "external_id": "bc3001-white",
                                "name": "Bella Canvas 3001",
                                "color": "White",
                                "substrate_type": "t-shirt",
                                "unit_cost": 4.50,
                            }
                        ],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].recommended_materials is not None
        assert len(result[0].recommended_materials) == 1
        assert result[0].recommended_materials[0]["external_id"] == "bc3001-white"

    def test_handles_missing_recommended_materials(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "No Materials Brief",
                        "product_type": "dtf_apparel",
                        "confidence": 70,
                        "source_seed_ids": [1],
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].recommended_materials is None

    def test_handles_non_list_recommended_materials(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "Bad Materials",
                        "product_type": "dtf_apparel",
                        "confidence": 70,
                        "source_seed_ids": [1],
                        "recommended_materials": "not a list",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].recommended_materials is None


class TestTheVoiceOfThePrompt:
    def _prompt(self) -> str:
        messages = build_brief_prompt(
            candidates=[_briefable(1, "funny t-shirt")],
            context=[],
            today=date(2026, 10, 21),
        )
        return messages[0]["content"]

    def test_the_prompt_asks_for_a_voice(self) -> None:
        # The store sells three voices and the operator curates afterwards,
        # so every brief has to name the one it is written in (issue #225).
        assert "voice" in self._prompt().lower()

    def test_the_prompt_teaches_the_three_voices(self) -> None:
        prompt = self._prompt().lower()
        for voice in ("deadpan", "loud", "cozy"):
            assert voice in prompt

    def test_the_prompt_shows_what_a_line_looks_like(self) -> None:
        prompt = self._prompt()
        assert "GOOD:" in prompt
        assert "BAD:" in prompt

    def test_the_examples_stay_evergreen(self) -> None:
        # A seasonal example anchors every brief in the batch, which is what
        # happened when the rules taught the design-vs-product rule with a
        # holiday example (issue #200).
        prompt = self._prompt().lower()
        for word in SEASONAL_WORDS:
            assert not re.search(rf"\b{re.escape(word)}\b", prompt), f"{word} anchors the prompt"

    def test_the_prompt_asks_the_scene_for_one_idea(self) -> None:
        prompt = self._prompt().lower()
        assert "one idea" in prompt

    def test_the_scene_has_room_for_the_idea(self) -> None:
        # The scene used to be capped at twenty words, which is enough to
        # name objects and not enough to invent anything.
        assert "under 20 words" not in self._prompt().lower()
        assert "under 35 words" in self._prompt().lower()


class TestTheTextGate:
    def _brief(self, headline: str, **extra: Any) -> str:
        brief = {
            "concept": "A Design",
            "product_type": "dtf_apparel",
            "confidence": 80,
            "source_seed_ids": [1],
            "layout_type": "text_top",
            "headline_text": headline,
            "scene_description": "a stone castle with a wide gate",
        }
        brief.update(extra)
        return json.dumps({"briefs": [brief]})

    def test_a_line_is_kept(self) -> None:
        result = parse_brief_response(self._brief("Gobble Til The Drawbridge Drops"))
        assert len(result) == 1

    def test_a_headline_that_names_a_brand_is_dropped(self) -> None:
        # The design that started this (issue #225).
        result = parse_brief_response(self._brief("Thanksgiving in the Magic Kingdom"))
        assert result == []

    def test_a_noun_list_is_dropped(self) -> None:
        result = parse_brief_response(self._brief("Feast, fun, and fireworks"))
        assert result == []

    def test_a_headline_that_only_restates_the_picture_is_dropped(self) -> None:
        result = parse_brief_response(
            self._brief(
                "A Stone Castle With A Wide Gate",
                scene_description="a stone castle with a wide gate",
            )
        )
        assert result == []

    def test_the_reason_is_reported(self) -> None:
        rejections: list[str] = []

        read_brief_response(self._brief("Feast, fun, and fireworks"), rejections=rejections)

        assert rejections
        assert "list" in rejections[0].lower()

    def test_the_voice_is_parsed(self) -> None:
        result = parse_brief_response(self._brief("Gobble Til The Drawbridge Drops", voice="loud"))
        assert result[0].voice == "loud"

    def test_an_unknown_voice_falls_back(self) -> None:
        result = parse_brief_response(
            self._brief("Gobble Til The Drawbridge Drops", voice="shouty")
        )
        assert result[0].voice == "deadpan"


class TestTheTextIsAskedForAgain:
    def _mock_db(self) -> MagicMock:
        db = MagicMock()
        db.list_briefable_seeds.return_value = [_briefable(1, "funny t-shirt")]
        db.list_active_materials.return_value = []
        db.insert_design_brief.return_value = 1
        db.insert_brief_source.return_value = 1
        return db

    def _raw(self, response_json: dict) -> MagicMock:
        choice = MagicMock()
        choice.message.content = json.dumps(response_json)
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        return raw

    def _bad_brief(self) -> dict:
        return {
            "briefs": [
                {
                    "concept": "A Castle Design",
                    "product_type": "dtf_apparel",
                    "confidence": 80,
                    "source_seed_ids": [1],
                    "layout_type": "text_top",
                    "headline_text": "Thanksgiving in the Magic Kingdom",
                    "scene_description": "a stone castle with a wide gate",
                }
            ]
        }

    def _good_brief(self) -> dict:
        return {
            "briefs": [
                {
                    "concept": "A Castle Design",
                    "product_type": "dtf_apparel",
                    "confidence": 80,
                    "source_seed_ids": [1],
                    "layout_type": "text_top",
                    "headline_text": "Gobble Til The Drawbridge Drops",
                    "scene_description": "a stone castle with a wide gate",
                }
            ]
        }

    def test_a_dropped_brief_is_asked_for_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The batch is worth more with the brief in it than without it, so the
        # model gets one more chance with the reason (issue #225).
        monkeypatch.setattr(
            "store_listing.orchestration.design_briefs.BRIEF_MIX_REGENERATE_ONCE", False
        )
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            self._raw(self._bad_brief()),
            self._raw(self._good_brief()),
        ]
        db = self._mock_db()

        result = generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert llm.with_raw_response.create.call_count == 2
        assert result["briefs_created"] == 1

    def test_the_second_attempt_is_told_what_was_wrong(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "store_listing.orchestration.design_briefs.BRIEF_MIX_REGENERATE_ONCE", False
        )
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            self._raw(self._bad_brief()),
            self._raw(self._good_brief()),
        ]
        db = self._mock_db()

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        second_prompt = llm.with_raw_response.create.call_args_list[1].kwargs["messages"][0][
            "content"
        ]
        assert "brand" in second_prompt.lower()

    def test_the_voice_is_stored_with_the_brief(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            self._raw(
                {
                    "briefs": [
                        {
                            "concept": "A Castle Design",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [1],
                            "layout_type": "text_top",
                            "headline_text": "Gobble Til The Drawbridge Drops",
                            "scene_description": "a stone castle with a wide gate",
                            "voice": "loud",
                        }
                    ]
                }
            )
        ]
        db = self._mock_db()

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        assert db.insert_design_brief.call_args.kwargs["voice"] == "loud"


class TestTheTypeOfTheBrief:
    def _raw(self, response_json: dict) -> MagicMock:
        choice = MagicMock()
        choice.message.content = json.dumps(response_json)
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        return raw

    def test_the_font_family_follows_the_voice(self) -> None:
        # The model picks a tone, the type follows the art direction. The
        # model never picks a font file (issue #226).
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "A Design",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                        "layout_type": "text_top",
                        "headline_text": "Gobble Til The Drawbridge Drops",
                        "scene_description": "a stone castle with a wide gate",
                        "voice": "loud",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].font_family == family_for_voice("loud")

    def test_the_font_family_is_stored_with_the_brief(self) -> None:
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = [
            self._raw(
                {
                    "briefs": [
                        {
                            "concept": "A Castle Design",
                            "product_type": "dtf_apparel",
                            "confidence": 80,
                            "source_seed_ids": [1],
                            "layout_type": "text_top",
                            "headline_text": "Gobble Til The Drawbridge Drops",
                            "scene_description": "a stone castle with a wide gate",
                            "voice": "cozy",
                        }
                    ]
                }
            )
        ]
        db = MagicMock()
        db.list_briefable_seeds.return_value = [_briefable(1, "funny t-shirt")]
        db.list_active_materials.return_value = []
        db.insert_design_brief.return_value = 1
        db.insert_brief_source.return_value = 1

        generate_design_briefs(db, llm, "test-model", today=date(2026, 10, 1))

        stored = db.insert_design_brief.call_args.kwargs
        assert stored["voice"] == "cozy"
        assert stored["font_family"] == family_for_voice("cozy")


class TestTheBatchTemplates:
    def _brief(self, concept: str, **overrides: Any) -> DesignBrief:
        defaults: dict[str, Any] = {
            "concept": concept,
            "product_type": "dtf_apparel",
            "specific_products": ["t-shirt"],
            "audience": "everyone",
            "visual_style": "bold",
            "confidence": 50,
            "reasoning": "trending",
            "source_seed_ids": [1],
            "layout_type": "auto",
            "headline_text": "GOOD LINE",
        }
        defaults.update(overrides)
        return DesignBrief(**defaults)

    def test_the_batch_covers_the_templates(self) -> None:
        briefs = [self._brief(f"concept {index}") for index in range(8)]

        assigned = assign_batch_templates(briefs, "2026-10-22")

        assert len({brief.layout_type for brief in assigned}) == 8

    def test_a_legacy_layout_the_model_named_is_overridden(self) -> None:
        # The model answered text_top eight times; the batch still carries
        # eight different silhouettes (issue #238).
        briefs = [self._brief(f"concept {index}", layout_type="text_top") for index in range(8)]

        assigned = assign_batch_templates(briefs, "2026-10-22")

        assert "text_top" not in {brief.layout_type for brief in assigned}
        assert len({brief.layout_type for brief in assigned}) == 8

    def test_a_brief_without_copy_is_left_art_only(self) -> None:
        briefs = [self._brief("no copy", headline_text=None, tagline_text=None)]

        assigned = assign_batch_templates(briefs, "2026-10-22")

        assert assigned[0].layout_type == "full_bleed"

    def test_a_text_only_brief_keeps_its_layout(self) -> None:
        briefs = [self._brief("only type", layout_type="text_only")]

        assigned = assign_batch_templates(briefs, "2026-10-22")

        assert assigned[0].layout_type == "text_only"


class TestTheIdeaDevices:
    def _messages(self):
        return build_brief_prompt(
            candidates=[_briefable(1, "dad jokes shirt")],
            context=[],
            today=date(2026, 10, 22),
        )

    def _brief(self, concept: str, **overrides: Any) -> DesignBrief:
        defaults: dict[str, Any] = {
            "concept": concept,
            "product_type": "dtf_apparel",
            "specific_products": ["t-shirt"],
            "audience": "everyone",
            "visual_style": "bold",
            "confidence": 50,
            "reasoning": "trending",
            "source_seed_ids": [1],
            "layout_type": "auto",
            "headline_text": "GOOD LINE",
        }
        defaults.update(overrides)
        return DesignBrief(**defaults)

    def test_the_prompt_hands_each_brief_slot_a_device(self) -> None:
        payload = json.loads(self._messages()[1]["content"])

        devices = [slot["idea_device"] for slot in payload["brief_slots"]]

        assert len(devices) == payload["max_briefs"]
        assert len(set(devices)) >= 6

    def test_the_system_prompt_names_every_device(self) -> None:
        system = self._messages()[0]["content"]

        for device in DEVICE_IDS:
            assert device in system

    def test_a_brief_keeps_the_device_it_names(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "A raccoon heist",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                        "layout_type": "auto",
                        "headline_text": "SNACK RUN",
                        "idea_device": "narrative",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].idea_device == "narrative"

    def test_an_unknown_device_is_dropped(self) -> None:
        response = json.dumps(
            {
                "briefs": [
                    {
                        "concept": "A raccoon heist",
                        "product_type": "dtf_apparel",
                        "confidence": 80,
                        "source_seed_ids": [1],
                        "layout_type": "auto",
                        "headline_text": "SNACK RUN",
                        "idea_device": "something_else",
                    }
                ]
            }
        )

        result = parse_brief_response(response)

        assert result[0].idea_device is None

    def test_the_batch_rotation_covers_the_vocabulary(self) -> None:
        briefs = [self._brief(f"concept {index}") for index in range(8)]

        assigned = assign_batch_devices(briefs, "2026-10-22")

        assert len({brief.idea_device for brief in assigned}) >= 6

    def test_the_batch_rotation_is_deterministic(self) -> None:
        briefs = [self._brief(f"concept {index}") for index in range(8)]

        assert assign_batch_devices(briefs, "2026-10-22") == assign_batch_devices(
            briefs, "2026-10-22"
        )
