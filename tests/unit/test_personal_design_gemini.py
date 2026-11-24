"""Tests for the Gemini full-sentence personal design research path (issue #356)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from store_listing.orchestration.personal_design import PersonalDesign, SearchNote
from store_listing.orchestration.personal_design_gemini import (
    gemini_research_sentence,
    research_personal_design_gemini,
)

# ---------------------------------------------------------------------------
# Fake Gemini client helpers
# ---------------------------------------------------------------------------


def _fake_genai_client(response_text: str) -> Any:
    """A minimal google-genai client stub that returns a fixed response."""

    class _FakePart:
        text = response_text

    class _FakeContent:
        parts: ClassVar = [_FakePart()]

    class _FakeGrounding:
        grounding_chunks: ClassVar = []

    class _FakeCandidate:
        content = _FakeContent()
        grounding_metadata = _FakeGrounding()

    class _FakeResponse:
        text = response_text
        candidates: ClassVar = [_FakeCandidate()]

    class _FakeModels:
        def generate_content(self, model: str, contents: str, config: Any) -> _FakeResponse:
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    return _FakeClient()


def _fake_apify_client(raw_items: list) -> type:
    """A minimal Apify client stub whose dataset returns raw_items."""

    def _list_items(raw: list) -> Any:
        return type("Result", (), {"items": raw})()

    class _FakeDataset:
        def list_items(self) -> Any:
            return _list_items(raw_items)

    class _FakeRun:
        default_dataset_id = "ds-test"

    class _FakeActor:
        def call(self, **kwargs: Any) -> _FakeRun:
            return _FakeRun()

    class _FakeClient:
        def __init__(self, token: str) -> None:
            pass

        def actor(self, actor_id: str) -> _FakeActor:
            return _FakeActor()

        def dataset(self, dataset_id: str) -> _FakeDataset:
            return _FakeDataset()

    return _FakeClient


# ---------------------------------------------------------------------------
# gemini_research_sentence
# ---------------------------------------------------------------------------


def test_gemini_research_sentence_returns_subjects_and_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    payload = {
        "subjects": [
            {
                "name": "Mickey Mouse",
                "description": "Anthropomorphic black mouse with round ears and red shorts.",
                "image_query": "Mickey Mouse Disney character art",
            },
            {
                "name": "Tron",
                "description": "Heroic program in a black bodysuit with glowing blue neon lines.",
                "image_query": "Tron Disney character neon blue suit light cycle",
            },
        ],
        "scene": "Mickey Mouse waves a Pilgrim hat while riding Tron's glowing light cycle, autumn leaves swirling around them.",
        "customer_line": None,
        "voice_found": False,
        "lettering": None,
    }
    client = _fake_genai_client(json.dumps(payload))
    result = gemini_research_sentence(
        "Mickey Mouse, Tron -- Thanksgiving themed",
        genai_client=client,
    )

    assert result is not None
    assert len(result.subjects) == 2
    assert result.subjects[0].name == "Mickey Mouse"
    assert result.subjects[1].name == "Tron"
    assert "light cycle" in result.scene
    assert result.customer_line is None


def test_gemini_research_sentence_returns_none_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = gemini_research_sentence("Mickey Mouse Thanksgiving")
    assert result is None


def test_gemini_research_sentence_returns_none_on_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = _fake_genai_client("   ")
    result = gemini_research_sentence("Mickey Mouse", genai_client=client)
    assert result is None


def test_gemini_research_sentence_returns_none_on_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = _fake_genai_client("not json at all")
    result = gemini_research_sentence("Mickey Mouse", genai_client=client)
    assert result is None


# ---------------------------------------------------------------------------
# research_personal_design_gemini
# ---------------------------------------------------------------------------


def test_research_uses_gemini_scene_as_subjects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("APIFY_API_TOKEN", "test-apify")

    payload = {
        "subjects": [
            {
                "name": "Mickey Mouse",
                "description": "Black mouse with red shorts and round ears.",
                "image_query": "Mickey Mouse Disney character art",
            },
        ],
        "scene": "Mickey Mouse in a Pilgrim hat carries a cornucopia past Cinderella Castle.",
        "customer_line": None,
        "voice_found": False,
        "lettering": None,
    }
    client = _fake_genai_client(json.dumps(payload))
    image_items = [
        {"imageUrl": "https://example.com/mickey.jpg", "title": "Mickey", "source": "disney.com"}
    ]

    design = research_personal_design_gemini(
        "Mickey Mouse -- Thanksgiving themed",
        seed=0,
        genai_client=client,
        apify_client_cls=_fake_apify_client(image_items),
    )

    assert design is not None
    assert isinstance(design, PersonalDesign)
    assert "Pilgrim hat" in design.subjects or "cornucopia" in design.subjects
    assert len(design.reference_urls) >= 1
    assert design.reference_urls[0] == "https://example.com/mickey.jpg"


def test_research_uses_context_aware_image_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apify is called with Gemini's specific image_query, not just the subject name."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("APIFY_API_TOKEN", "test-apify")

    queries_sent: list[str] = []

    payload = {
        "subjects": [
            {
                "name": "Tron",
                "description": "Heroic program in neon blue suit.",
                "image_query": "Tron Disney character neon blue suit light cycle",
            },
        ],
        "scene": "Tron speeds through a neon grid on a light cycle.",
        "customer_line": None,
        "voice_found": False,
        "lettering": None,
    }

    def _fake_search(query: str, apify_client_cls: Any = None) -> list[SearchNote]:
        queries_sent.append(query)
        return [SearchNote(query, "Tron art", "Tron character.", "https://example.com/tron.jpg")]

    client = _fake_genai_client(json.dumps(payload))

    research_personal_design_gemini(
        "Tron -- Magic Kingdom",
        seed=0,
        genai_client=client,
        image_search=_fake_search,
    )

    assert any("neon blue suit" in q for q in queries_sent), (
        f"Expected context-aware query, got: {queries_sent}"
    )


def test_research_returns_none_without_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = research_personal_design_gemini("Mickey Mouse Thanksgiving", seed=0)
    assert result is None


def test_research_holiday_theme_in_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("APIFY_API_TOKEN", "test-apify")

    payload = {
        "subjects": [
            {
                "name": "Mickey Mouse",
                "description": "Iconic Disney mouse.",
                "image_query": "Mickey Mouse Disney",
            }
        ],
        "scene": "Mickey Mouse in a Pilgrim hat surrounded by autumn harvest cornucopias and golden leaves.",
        "customer_line": None,
        "voice_found": False,
        "lettering": None,
    }
    client = _fake_genai_client(json.dumps(payload))

    design = research_personal_design_gemini(
        "Mickey Mouse -- Thanksgiving themed",
        seed=0,
        genai_client=client,
        apify_client_cls=_fake_apify_client([]),
    )

    assert design is not None
    assert any(
        word in design.subjects.lower()
        for word in ["thanksgiving", "pilgrim", "harvest", "cornucopia", "autumn"]
    )


def test_gemini_surfaces_theme_song_as_lettering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Gemini finds the subject's theme song it becomes the design lettering."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("APIFY_API_TOKEN", "test-apify")

    payload = {
        "subjects": [
            {
                "name": "Mickey's Runaway Railway",
                "description": "A cartoon-styled dark ride in the Paul Rudish aesthetic.",
                "image_query": "Mickey's Runaway Railway Disney World ride vehicle cartoon",
            }
        ],
        "scene": "Mickey Mouse at the wheel of his bright red runaway train car, zipping through a cartoon landscape.",
        "customer_line": None,
        "voice_found": True,
        "lettering": "Nothing Can Stop Us Now",
    }
    client = _fake_genai_client(json.dumps(payload))

    design = research_personal_design_gemini(
        "Mickey's Runaway Railway",
        seed=0,
        genai_client=client,
        apify_client_cls=_fake_apify_client([]),
    )

    assert design is not None
    assert design.lettering == "Nothing Can Stop Us Now"
    assert not design.customer_lettering


def test_gemini_no_lettering_when_voice_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("APIFY_API_TOKEN", "test-apify")

    payload = {
        "subjects": [
            {
                "name": "Space Mountain",
                "description": "A futuristic roller coaster.",
                "image_query": "Space Mountain Disney roller coaster",
            }
        ],
        "scene": "Space Mountain rocket car launches into a neon starfield.",
        "customer_line": None,
        "voice_found": False,
        "lettering": None,
    }
    client = _fake_genai_client(json.dumps(payload))

    design = research_personal_design_gemini(
        "Space Mountain",
        seed=0,
        genai_client=client,
        apify_client_cls=_fake_apify_client([]),
    )

    assert design is not None
    assert design.lettering is None
