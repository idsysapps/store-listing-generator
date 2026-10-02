"""Unit tests for LLM-driven seed curation (per-candidate evaluation)."""

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from store_listing.orchestration.promotion import ActiveSeed, SeedCandidate
from store_listing.orchestration.seed_curation import (
    PRODUCT_TAGS,
    CurationResult,
    build_curation_prompt,
    curate_seeds,
    parse_curation_response,
)


def _candidate(
    cid: int,
    query: str,
    source: str = "google",
    query_type: str = "rising",
    score: int = 100,
    delta: int = 5000,
) -> SeedCandidate:
    return SeedCandidate(
        id=cid,
        query=query,
        source=source,
        query_type=query_type,
        score=score,
        delta=delta,
        promotion_score=5000,
        status="pending",
    )


def _active(query: str, score: int = 0) -> ActiveSeed:
    return ActiveSeed(id=1, query=query, promotion_score=score)


def _eval_response(evaluations: list[dict], event_seeds: list[dict] | None = None) -> str:
    data: dict = {"evaluations": evaluations}
    if event_seeds is not None:
        data["event_seeds"] = event_seeds
    return json.dumps(data)


class TestBuildCurationPrompt:
    def test_returns_system_and_user_messages(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "dad jokes shirt")],
            active_seeds=[_active("hoodie")],
            cross_seed_counts={"dad jokes shirt": 2},
            context=[{"name": "halloween", "days_until": 10}],
            today=date(2026, 10, 21),
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_trend_data(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(42, "dad jokes shirt")],
            active_seeds=[],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        user_content = messages[1]["content"]
        assert "dad jokes shirt" in user_content
        assert "42" in user_content

    def test_includes_date_and_context(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "test")],
            active_seeds=[],
            cross_seed_counts={},
            context=[{"name": "halloween", "days_until": 5}],
            today=date(2026, 10, 26),
        )

        user_content = messages[1]["content"]
        assert "2026-10-26" in user_content
        assert "halloween" in user_content

    def test_includes_current_seeds(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "test")],
            active_seeds=[_active("hoodie"), _active("gift")],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        user_content = messages[1]["content"]
        assert "hoodie" in user_content
        assert "gift" in user_content

    def test_system_prompt_contains_action_types(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "test")],
            active_seeds=[],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        system = messages[0]["content"]
        assert "PROMOTE" in system
        assert "PIVOT" in system
        assert "REJECT" in system

    def test_user_message_includes_output_format(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "test")],
            active_seeds=[],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        user_content = messages[1]["content"]
        assert "evaluations" in user_content
        assert "event_seeds" in user_content


class TestParseCurationResponse:
    def test_parses_well_formed_evaluations(self) -> None:
        response = _eval_response(
            [
                {
                    "id": 42,
                    "reasoning": "strong visual hook",
                    "action": "PROMOTE",
                    "product_tags": ["dtf_apparel"],
                    "design_style": "retro typography",
                    "pivot_to": None,
                },
                {
                    "id": 99,
                    "reasoning": "generic category",
                    "action": "REJECT",
                    "product_tags": [],
                    "design_style": None,
                    "pivot_to": None,
                },
            ]
        )

        result = parse_curation_response(response)

        assert len(result.evaluations) == 2
        assert result.evaluations[0].id == 42
        assert result.evaluations[0].action == "PROMOTE"
        assert result.evaluations[0].reasoning == "strong visual hook"
        assert result.evaluations[0].product_tags == ["dtf_apparel"]
        assert result.evaluations[0].design_style == "retro typography"
        assert result.evaluations[1].id == 99
        assert result.evaluations[1].action == "REJECT"

    def test_parses_pivot_evaluation(self) -> None:
        response = _eval_response(
            [
                {
                    "id": 10,
                    "reasoning": "trademark risk",
                    "action": "PIVOT",
                    "product_tags": ["dtf_apparel"],
                    "design_style": None,
                    "pivot_to": "angel rider",
                },
            ]
        )

        result = parse_curation_response(response)

        assert result.evaluations[0].action == "PIVOT"
        assert result.evaluations[0].pivot_to == "angel rider"

    def test_parses_event_seeds(self) -> None:
        response = _eval_response(
            [],
            event_seeds=[
                {"seed": "halloween candy shirt", "product_tags": ["dtf_apparel"]},
            ],
        )

        result = parse_curation_response(response)

        assert len(result.event_seeds) == 1
        assert result.event_seeds[0].seed == "halloween candy shirt"

    def test_handles_json_with_surrounding_text(self) -> None:
        response = 'Here is my analysis:\n{"evaluations": [{"id": 1, "action": "REJECT", "reasoning": "noise"}]}\nDone.'

        result = parse_curation_response(response)

        assert len(result.evaluations) == 1

    def test_handles_malformed_json(self) -> None:
        result = parse_curation_response("this is not json at all")

        assert result.evaluations == []
        assert result.event_seeds == []

    def test_handles_empty_response(self) -> None:
        result = parse_curation_response("")

        assert result == CurationResult()

    def test_filters_invalid_actions(self) -> None:
        response = _eval_response(
            [
                {"id": 1, "action": "PROMOTE", "reasoning": "ok"},
                {"id": 2, "action": "INVALID", "reasoning": "bad"},
                {"id": 3, "action": "REJECT", "reasoning": "ok"},
            ]
        )

        result = parse_curation_response(response)

        assert len(result.evaluations) == 2
        assert result.evaluations[0].id == 1
        assert result.evaluations[1].id == 3

    def test_filters_non_integer_ids(self) -> None:
        response = _eval_response(
            [
                {"id": 1, "action": "PROMOTE", "reasoning": "ok"},
                {"id": "bad", "action": "REJECT", "reasoning": "bad"},
            ]
        )

        result = parse_curation_response(response)

        assert len(result.evaluations) == 1

    def test_filters_invalid_product_tags(self) -> None:
        response = _eval_response(
            [
                {
                    "id": 1,
                    "action": "PROMOTE",
                    "reasoning": "ok",
                    "product_tags": ["dtf_apparel", "furniture", "sublimation"],
                },
            ]
        )

        result = parse_curation_response(response)

        assert result.evaluations[0].product_tags == ["dtf_apparel", "sublimation"]

    def test_action_case_insensitive(self) -> None:
        response = _eval_response([{"id": 1, "action": "promote", "reasoning": "ok"}])

        result = parse_curation_response(response)

        assert result.evaluations[0].action == "PROMOTE"


class TestCurateSeeds:
    def _mock_db(self, candidates=None, active_seeds=None) -> MagicMock:
        db = MagicMock()
        db.list_pending_candidates.return_value = candidates or []
        db.list_active_seeds.return_value = active_seeds or []
        db.cross_seed_counts.return_value = {}
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1
        return db

    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        llm.with_raw_response.create.return_value = raw
        return llm

    def test_no_candidates_returns_early(self) -> None:
        db = self._mock_db()
        llm = MagicMock()

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "success"
        assert result["promoted"] == 0
        assert result["remaining"] == 0
        assert not llm.with_raw_response.create.called

    def test_promotes_candidates(self) -> None:
        candidates = [_candidate(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {
                        "id": 42,
                        "action": "PROMOTE",
                        "reasoning": "great hook",
                        "product_tags": ["dtf_apparel"],
                        "design_style": "bold typography",
                    },
                ]
            )
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["promoted"] == 1
        db.insert_active_seed.assert_any_call(
            query="dad jokes shirt", promotion_score=5000, candidate_id=42
        )
        db.promote_candidate.assert_called_with(42)

    def test_rejects_candidates(self) -> None:
        candidates = [_candidate(99, "compression socks")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {"id": 99, "action": "REJECT", "reasoning": "generic product"},
                ]
            )
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["rejected"] == 1
        db.reject_candidate.assert_called_with(99)

    def test_pivots_reject_original_and_insert_new(self) -> None:
        candidates = [_candidate(10, "ghost rider hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {
                        "id": 10,
                        "action": "PIVOT",
                        "reasoning": "trademark",
                        "product_tags": ["dtf_apparel"],
                        "pivot_to": "angel rider hoodie",
                    },
                ]
            )
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["pivots"] == 1
        db.reject_candidate.assert_called_with(10)
        db.insert_active_seed.assert_any_call(
            query="angel rider hoodie", promotion_score=0, candidate_id=None
        )

    def test_pivot_uses_original_query_when_no_pivot_to(self) -> None:
        candidates = [_candidate(10, "broad hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {
                        "id": 10,
                        "action": "PIVOT",
                        "reasoning": "too broad",
                        "product_tags": ["dtf_apparel"],
                        "pivot_to": None,
                    },
                ]
            )
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_active_seed.assert_any_call(
            query="broad hoodie", promotion_score=0, candidate_id=None
        )

    def test_inserts_event_seeds(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [{"id": 1, "action": "REJECT", "reasoning": "noise"}],
                event_seeds=[{"seed": "halloween candy shirt", "product_tags": ["dtf_apparel"]}],
            )
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["event_seeds"] == 1
        db.insert_active_seed.assert_any_call(
            query="halloween candy shirt", promotion_score=0, candidate_id=None
        )

    def test_logs_all_actions_with_reasoning(self) -> None:
        candidates = [_candidate(42, "dad jokes"), _candidate(99, "socks")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {"id": 42, "action": "PROMOTE", "reasoning": "good hook"},
                    {"id": 99, "action": "REJECT", "reasoning": "generic"},
                ]
            )
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        log_calls = db.insert_curation_log.call_args_list
        actions = [c.kwargs["action"] for c in log_calls]
        assert "promote" in actions
        assert "reject" in actions
        assert all(c.kwargs["llm_model"] == "test-model" for c in log_calls)
        promote_call = next(c for c in log_calls if c.kwargs["action"] == "promote")
        assert promote_call.kwargs["reasoning"] == "good hook"

    def test_handles_llm_error_gracefully(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError("API down")

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "API down" in result["error"]

    def test_ignores_unknown_candidate_ids(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {"id": 999, "action": "PROMOTE", "reasoning": "phantom"},
                ]
            )
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["promoted"] == 0
        assert not db.promote_candidate.called

    def test_empty_choices_returns_error(self) -> None:
        candidates = [_candidate(1, "test seed")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = MagicMock(choices=[])
        llm.with_raw_response.create.return_value = raw

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "error"
        assert "empty choices" in result["error"]

    def test_returns_remaining_backlog_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import store_listing.orchestration.seed_curation as mod

        monkeypatch.setattr(mod, "MAX_CANDIDATES", 50)
        candidates = [_candidate(i, f"seed-{i}") for i in range(75)]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response([{"id": i, "action": "REJECT", "reasoning": "noise"} for i in range(50)])
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["remaining"] == 25

    def test_uses_temperature_zero(self) -> None:
        candidates = [_candidate(1, "test")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(_eval_response([{"id": 1, "action": "REJECT", "reasoning": "x"}]))

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        call_kwargs = llm.with_raw_response.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    def test_429_returns_rate_limited_status(self) -> None:
        candidates = [_candidate(1, "test")]
        db = self._mock_db(candidates=candidates)
        llm = MagicMock()
        llm.with_raw_response.create.side_effect = RuntimeError(
            "Error code: 429 - Rate limit exceeded"
        )

        result = curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        assert result["status"] == "rate_limited"
        assert "429" in result["error"]


class TestProductTags:
    def test_product_tags_contains_three_categories(self) -> None:
        assert "dtf_apparel" in PRODUCT_TAGS
        assert "sublimation" in PRODUCT_TAGS
        assert "sticker_vinyl" in PRODUCT_TAGS
        assert len(PRODUCT_TAGS) == 3


class TestCurateSeedsProductTags:
    def _mock_db(self, candidates=None, active_seeds=None) -> MagicMock:
        db = MagicMock()
        db.list_pending_candidates.return_value = candidates or []
        db.list_active_seeds.return_value = active_seeds or []
        db.cross_seed_counts.return_value = {}
        db.insert_active_seed.return_value = 1
        db.insert_curation_log.return_value = 1
        return db

    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        parsed = MagicMock(choices=[choice])
        raw = MagicMock()
        raw.headers = {}
        raw.parse.return_value = parsed
        llm.with_raw_response.create.return_value = raw
        return llm

    def test_promoted_seed_gets_product_tags(self) -> None:
        candidates = [_candidate(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {
                        "id": 42,
                        "action": "PROMOTE",
                        "reasoning": "great",
                        "product_tags": ["dtf_apparel"],
                    },
                ]
            )
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_called_once()
        call_args = db.insert_product_tags.call_args
        assert call_args.kwargs["active_seed_id"] == 1
        assert call_args.kwargs["tags"] == ["dtf_apparel"]

    def test_pivot_seed_gets_product_tags(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {
                        "id": 1,
                        "action": "PIVOT",
                        "reasoning": "too broad",
                        "product_tags": ["dtf_apparel"],
                        "pivot_to": "vintage hoodie",
                    },
                ]
            )
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_called_once()
        call_args = db.insert_product_tags.call_args
        assert call_args.kwargs["active_seed_id"] == 1
        assert call_args.kwargs["tags"] == ["dtf_apparel"]

    def test_event_seed_gets_product_tags(self) -> None:
        candidates = [_candidate(1, "hoodie")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [{"id": 1, "action": "REJECT", "reasoning": "x"}],
                event_seeds=[{"seed": "halloween mug", "product_tags": ["sublimation"]}],
            )
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_called_once()
        call_args = db.insert_product_tags.call_args
        assert call_args.kwargs["active_seed_id"] == 1
        assert call_args.kwargs["tags"] == ["sublimation"]

    def test_no_tags_means_no_insert(self) -> None:
        candidates = [_candidate(42, "dad jokes shirt")]
        db = self._mock_db(candidates=candidates)
        llm = self._mock_llm(
            _eval_response(
                [
                    {"id": 42, "action": "PROMOTE", "reasoning": "ok", "product_tags": []},
                ]
            )
        )

        curate_seeds(db, llm, "test-model", today=date(2026, 10, 1))

        db.insert_product_tags.assert_not_called()

    def test_prompt_includes_product_tags_instruction(self) -> None:
        messages = build_curation_prompt(
            candidates=[_candidate(1, "dad jokes shirt")],
            active_seeds=[],
            cross_seed_counts={},
            context=[],
            today=date(2026, 10, 21),
        )

        assert "product_tags" in messages[0]["content"]
        assert "dtf_apparel" in messages[0]["content"]
