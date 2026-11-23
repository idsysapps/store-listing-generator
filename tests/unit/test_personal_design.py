"""The personal custom design: search, one line, one print-ready graphic."""

from __future__ import annotations

import json

import httpx
import pytest

from store_listing.orchestration.personal_design import (
    PersonalDesign,
    SearchNote,
    arrangement_line,
    character_from_notes,
    character_role,
    fetch_reference,
    graphic_prompt,
    quoted_lettering,
    rearrange,
    research_personal_design,
    resolve_lettering,
    search_commons,
    search_duckduckgo,
    search_personal,
    search_wikipedia,
    specified_characters,
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


def test_double_quotes_are_the_shirt_line() -> None:
    assert quoted_lettering('slow moving with text "Get There Eventually"') == (
        "Get There Eventually"
    )
    assert quoted_lettering('slow moving with text ""Get There Eventually""') == (
        "Get There Eventually"
    )


def test_curly_quotes_are_the_shirt_line() -> None:
    assert quoted_lettering("slow moving with text “Get There Eventually”") == (
        "Get There Eventually"
    )


def test_a_name_before_character_is_that_character() -> None:
    names = specified_characters(
        'Judy Hopps character showing slow moving with text "Get There Eventually"'
    )

    assert names == ("Judy Hopps",)


def test_a_name_after_character_is_that_character() -> None:
    assert specified_characters("the character Judy Hopps, slow moving") == ("Judy Hopps",)
    assert specified_characters("characters Judy Hopps and Nick Wilde") == (
        "Judy Hopps",
        "Nick Wilde",
    )


def test_plural_characters_are_not_one_specific_character() -> None:
    assert specified_characters("lifelike Disney Zootopia characters") == ()
    assert specified_characters("Zootopia sleuth character") == ()
    assert character_role("lifelike Disney Zootopia characters") is None
    assert character_role("Judy Hopps character") is None


def test_a_role_is_the_search_for_an_unnamed_character() -> None:
    assert (
        character_role(
            'Zootopia sleuth character showing slow moving with text "Get There Eventually"'
        )
        == "Zootopia sleuth character"
    )


def test_one_named_result_is_the_character() -> None:
    notes = [
        SearchNote(
            "Zootopia sleuth character",
            "Flash Slothmore",
            "Flash Slothmore is a sloth.",
            "https://example.com/flash.jpg",
        )
    ]

    assert character_from_notes("Zootopia sleuth character", notes) == "Flash Slothmore"


def test_a_crowd_of_names_is_not_resolved() -> None:
    notes = [
        SearchNote(
            "Zootopia sleuth character",
            "Zootopia",
            "Judy Hopps, Nick Wilde and Flash Slothmore appear.",
            None,
        )
    ]

    assert character_from_notes("Zootopia sleuth character", notes) is None


def test_research_replaces_a_role_with_the_character_the_search_names() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["zootopia movie"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": None,
                    "subjects": "a grey rabbit beside an orange tiger",
                    "references": [],
                }
            ),
        ]
    )
    queries: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries.append(query)
        if query == "Zootopia sleuth character":
            return [
                SearchNote(
                    query,
                    "Flash Slothmore",
                    "Flash Slothmore is a sloth.",
                    "https://example.com/flash.jpg",
                )
            ]
        if query == "Flash Slothmore":
            return [
                SearchNote(
                    query,
                    "Flash Slothmore",
                    "Flash Slothmore is a sloth in Zootopia.",
                    "https://example.com/flash-ref.jpg",
                )
            ]
        return [
            SearchNote(
                query,
                "Zootopia",
                "Judy Hopps and Nick Wilde are police officers.",
                None,
            )
        ]

    design = research_personal_design(
        'Zootopia sleuth character showing slow moving with text "Get There Eventually"',
        complete=lambda _messages: next(answers),
        search=search,
        seed=1,
    )

    assert queries[0] == "Zootopia sleuth character"
    assert "Flash Slothmore" in queries
    assert "Flash Slothmore" in design.subjects
    assert "not a different character" in design.subjects
    assert "Judy Hopps" not in design.subjects
    assert design.reference_urls[0] == "https://example.com/flash-ref.jpg"
    assert design.lettering == "Get There Eventually"


def test_research_prints_the_quoted_line_and_locks_the_named_character() -> None:
    answers = iter(
        [
            json.dumps({"queries": ["zootopia movie"]}),
            json.dumps(
                {
                    "customer_line": None,
                    "voice_found": False,
                    "lettering": "zoogether sleuth",
                    "subjects": "a grey rabbit beside an orange tiger detective",
                    "references": [{"url": "https://example.com/poster.jpg", "keep": False}],
                }
            ),
        ]
    )
    queries: list[str] = []

    def search(query: str) -> list[SearchNote]:
        queries.append(query)
        if query == "Judy Hopps":
            return [
                SearchNote(
                    query,
                    "Judy Hopps",
                    "Judy Hopps is a rabbit police officer.",
                    "https://example.com/judy.jpg",
                ),
                SearchNote(
                    query,
                    "1953 British Mount Everest expedition",
                    "A mountaineering expedition.",
                    "https://example.com/mountain.jpg",
                ),
            ]
        return [
            SearchNote(query, "Zootopia", "A film.", "https://example.com/film.jpg"),
        ]

    design = research_personal_design(
        'Judy Hopps character showing slow moving with text "Get There Eventually"',
        complete=lambda _messages: next(answers),
        search=search,
        seed=1,
    )

    assert design.customer_lettering is True
    assert design.lettering == "Get There Eventually"
    assert "Judy Hopps" in design.subjects
    assert "not a different character" in design.subjects
    assert "Get There Eventually" not in design.prompt
    assert "No lettering" in design.prompt
    assert queries[0] == "Judy Hopps"
    assert design.reference_urls[0] == "https://example.com/judy.jpg"
    assert "https://example.com/mountain.jpg" not in design.reference_urls
    assert "mountaineering" not in " ".join(design.facts)


def test_duckduckgo_adds_a_short_note_and_a_picture() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "Judy Hopps"
        return httpx.Response(
            200,
            json={
                "Heading": "Judy Hopps",
                "AbstractText": "Judy Hopps is a rabbit police officer.",
                "Image": "https://example.com/judy.png",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_duckduckgo("Judy Hopps", client=client)

    assert notes[0].title == "Judy Hopps"
    assert notes[0].image_url == "https://example.com/judy.png"


def test_commons_keeps_a_character_picture_and_drops_a_logo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["gsrnamespace"] == "6"
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "title": "File:Judy Hopps.png",
                            "imageinfo": [
                                {
                                    "thumburl": "https://upload.wikimedia.org/judy.png",
                                    "mime": "image/png",
                                }
                            ],
                        },
                        "2": {
                            "title": "File:Studio logo.svg",
                            "imageinfo": [
                                {
                                    "thumburl": "https://upload.wikimedia.org/logo.svg",
                                    "mime": "image/svg+xml",
                                }
                            ],
                        },
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        notes = search_commons("Judy Hopps", client=client)

    assert [note.image_url for note in notes] == ["https://upload.wikimedia.org/judy.png"]


def test_personal_search_asks_every_source(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def note(source: str) -> list[SearchNote]:
        seen.append(source)
        return [SearchNote("Judy Hopps", source, f"From {source}.", None)]

    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.search_wikipedia",
        lambda query, client=None: note("wikipedia"),
    )
    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.search_duckduckgo",
        lambda query, client=None: note("duckduckgo"),
    )
    monkeypatch.setattr(
        "store_listing.orchestration.personal_design.search_commons",
        lambda query, client=None: note("commons"),
    )

    notes = search_personal("Judy Hopps")

    assert seen == ["wikipedia", "duckduckgo", "commons"]
    assert [note.title for note in notes] == ["wikipedia", "duckduckgo", "commons"]


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
