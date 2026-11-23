"""The personal custom design: search, one line, one print-ready graphic."""

from __future__ import annotations

import json

import httpx
import pytest

from store_listing.orchestration.personal_design import (
    PersonalDesign,
    PhotoLook,
    SearchNote,
    arrangement_line,
    fetch_reference,
    graphic_prompt,
    look_at_reference,
    photo_look_from_answer,
    rearrange,
    research_personal_design,
    resolve_lettering,
    search_wikipedia,
)


def test_customer_written_words_are_the_only_line() -> None:
    line = resolve_lettering(
        customer_line="Get there eventually",
        proposed_line="Mesa so very, very thankful!",
        voice_found=True,
    )

    assert line == "Get there eventually"


def test_a_voice_in_the_notes_allows_one_new_line() -> None:
    line = resolve_lettering(
        customer_line=None,
        proposed_line="Mesa so very, very thankful!",
        voice_found=True,
    )

    assert line == "Mesa so very, very thankful!"


def test_no_voice_leaves_the_shirt_without_a_line() -> None:
    line = resolve_lettering(
        customer_line="  ",
        proposed_line="Mesa so very, very thankful!",
        voice_found=False,
    )

    assert line is None


def test_the_graphic_is_a_magenta_key_and_names_its_line_once() -> None:
    prompt = graphic_prompt(
        subjects="a tall Gungan with long ears beside a round freighter",
        lettering="Mesa so very, very thankful!",
        arrangement=arrangement_line(1),
        facts=("Jar Jar speaks in a Gungan voice.",),
    )

    assert "solid pure magenta" in prompt
    assert "no backdrop" in prompt
    assert prompt.count("Mesa so very, very thankful!") == 1
    assert "photograph of the place" not in prompt
    assert "Jar Jar speaks in a Gungan voice." in prompt


def test_a_second_pass_keeps_the_facts_and_changes_the_arrangement() -> None:
    design = PersonalDesign(
        prompt=graphic_prompt(
            subjects="a tall Gungan beside a round freighter",
            lettering="Mesa so very, very thankful!",
            arrangement=arrangement_line(1),
            facts=("Jar Jar speaks in a Gungan voice.",),
        ),
        lettering="Mesa so very, very thankful!",
        reference_urls=("https://example.com/jar.jpg",),
        facts=("Jar Jar speaks in a Gungan voice.",),
        subjects="a tall Gungan beside a round freighter",
    )

    again = rearrange(design, seed=2)

    assert again.prompt != design.prompt
    assert "Jar Jar speaks in a Gungan voice." in again.prompt
    assert again.lettering == design.lettering
    assert again.reference_urls == design.reference_urls


def test_research_writes_one_line_when_the_notes_have_a_voice() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["jar jar speech"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": True,
                    "lettering": "Mesa so very, very thankful!",
                    "subjects": "a tall Gungan with long ears beside a round freighter",
                    "references": [{"url": "https://example.com/jar.jpg", "keep": True}],
                }
            ),
        ]
    )
    note = SearchNote(
        query="jar jar speech",
        title="Jar Jar",
        text="Jar Jar speaks in a Gungan voice.",
        image_url="https://example.com/jar.jpg",
    )

    design = research_personal_design(
        "hollywood studios, jar jar, thanksgiving",
        complete=lambda _messages: next(answers),
        search=lambda _query: [note],
        seed=1,
    )

    assert design.lettering == "Mesa so very, very thankful!"
    assert design.prompt.count("Mesa so very, very thankful!") == 1
    assert "solid pure magenta" in design.prompt
    assert design.reference_urls == ("https://example.com/jar.jpg",)
    assert any("Jar Jar speaks in a Gungan voice." in fact for fact in design.facts)


def test_research_drops_a_line_the_notes_cannot_support() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["thanksgiving table"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": "Mesa so very, very thankful!",
                    "subjects": "a harvest table",
                    "references": [],
                }
            ),
        ]
    )

    design = research_personal_design(
        "thanksgiving table",
        complete=lambda _messages: next(answers),
        search=lambda _query: [],
        seed=1,
    )

    assert design.lettering is None
    assert "thankful" not in design.prompt


def test_research_keeps_at_most_six_references() -> None:
    refs = [{"url": f"https://example.com/{index}.jpg", "keep": True} for index in range(8)]
    answers = iter(
        [
            json.dumps({"queries": ["ship"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a round freighter",
                    "references": refs,
                }
            ),
        ]
    )

    design = research_personal_design(
        "a round freighter",
        complete=lambda _messages: next(answers),
        search=lambda _query: [],
        seed=4,
    )

    assert len(design.reference_urls) == 6


def test_a_rejected_reference_is_not_kept() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["ship"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a round freighter",
                    "references": [
                        {"url": "https://example.com/crowd.jpg", "keep": False},
                        {"url": "https://example.com/ship.jpg", "keep": True},
                    ],
                }
            ),
        ]
    )

    design = research_personal_design(
        "a round freighter",
        complete=lambda _messages: next(answers),
        search=lambda _query: [
            SearchNote("ship", "Ship", "A round freighter.", "https://example.com/crowd.jpg")
        ],
        seed=1,
    )

    assert design.reference_urls == ("https://example.com/ship.jpg",)


def test_a_search_keeps_a_photo_and_drops_a_logo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["piprop"] == "thumbnail"
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "title": "Park",
                            "extract": "A theme park.",
                            "thumbnail": {"source": "https://upload.wikimedia.org/park.jpg"},
                        },
                        "2": {
                            "title": "Logo",
                            "extract": "A studio logo.",
                            "thumbnail": {"source": "https://upload.wikimedia.org/logo.svg"},
                        },
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_wikipedia("park", client=client)
    urls = [note.image_url for note in notes]

    assert "https://upload.wikimedia.org/park.jpg" in urls
    assert "https://upload.wikimedia.org/logo.svg" not in urls


def test_a_reference_download_retries_a_rate_limit_and_rejects_a_logo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.time.sleep", lambda _seconds: None
    )
    jpeg = b"\xff\xd8\xff" + b"photo"
    photo_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        photo_calls["n"] += 1
        if photo_calls["n"] == 1:
            return httpx.Response(429, content=b"slow", headers={"content-type": "text/html"})
        return httpx.Response(200, content=jpeg, headers={"content-type": "image/jpeg"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert fetch_reference("https://example.com/logo.svg", client=client) is None
        assert fetch_reference("https://example.com/park.jpg", client=client) == jpeg


def test_a_logo_answer_is_not_a_subject() -> None:
    logo = photo_look_from_answer('{"look": "a studio logo", "kind": "logo"}')
    mountain = photo_look_from_answer('{"look": "a snowy mountain", "kind": "subject"}')
    unread = photo_look_from_answer("not json")

    assert logo == PhotoLook("a studio logo", keep=False)
    assert mountain == PhotoLook("a snowy mountain", keep=True)
    assert unread == PhotoLook("", keep=True)


def test_a_vision_failure_is_an_empty_look() -> None:
    class _Completions:
        def create(self, **_kwargs: object) -> object:
            raise RuntimeError("vision is down")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    assert look_at_reference(b"img", client=_Client(), model="qwen_vl_ocr") == PhotoLook(
        "", keep=True
    )


def test_a_photo_look_fills_the_gaps_and_does_not_rename_the_mountain() -> None:
    sentence = "Animal Kingdom, Expedition Everest, Zootopia zoogether sleuth"
    answers = iter(
        [
            json.dumps({"queries": ["expedition everest"]}),
            json.dumps(
                {
                    "customer_line": sentence,
                    "voice_found": True,
                    "lettering": "Get there eventually",
                    "subjects": "a roller coaster train",
                    "additions": "a rabbit detective",
                    "references": [
                        {"url": "https://example.com/mountain.jpg", "keep": True},
                        {"url": "https://example.com/logo.jpg", "keep": True},
                    ],
                }
            ),
        ]
    )
    seen: list[list[dict[str, str]]] = []

    def complete(messages: list[dict[str, str]]) -> str:
        seen.append(messages)
        return next(answers)

    def look(image: bytes) -> PhotoLook:
        if image == b"logo":
            return PhotoLook("a studio logo", keep=False)
        return PhotoLook("a snowy mountain", keep=True)

    design = research_personal_design(
        sentence,
        complete=complete,
        search=lambda _query: [
            SearchNote(
                "expedition everest",
                "Expedition Everest",
                "Expedition Everest is a steel roller coaster.",
                "https://example.com/mountain.jpg",
            ),
            SearchNote("logo", "Logo", "A studio logo.", "https://example.com/logo.jpg"),
        ],
        seed=1,
        look=look,
        fetch=lambda url: b"logo" if url.endswith("logo.jpg") else b"mountain",
    )

    instructions = " ".join(message["content"] for messages in seen for message in messages)
    assert "Fill the gaps" in instructions
    assert "Photo shows: a snowy mountain" in instructions
    assert "a snowy mountain" in design.prompt
    assert "a rabbit detective" in design.prompt
    assert "roller" not in design.prompt
    assert "train" not in design.prompt
    assert design.lettering == "Get there eventually"
    assert design.prompt.count("Get there eventually") == 1
    assert "zoogether sleuth" not in design.prompt
    assert design.reference_urls == ("https://example.com/mountain.jpg",)


def test_a_failed_look_stays_on_the_article() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["expedition everest"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a roller coaster train",
                    "references": [{"url": "https://example.com/mountain.jpg", "keep": True}],
                }
            ),
        ]
    )

    design = research_personal_design(
        "Expedition Everest",
        complete=lambda _messages: next(answers),
        search=lambda _query: [
            SearchNote(
                "expedition everest",
                "Expedition Everest",
                "Expedition Everest is a steel roller coaster.",
                "https://example.com/mountain.jpg",
            )
        ],
        seed=1,
        look=lambda _image: PhotoLook("", keep=True),
        fetch=lambda _url: b"mountain",
    )

    assert "roller coaster train" in design.prompt
    assert "steel roller coaster" in design.prompt
