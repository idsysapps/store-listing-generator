"""Research a personal custom sentence into one DTF shirt graphic.

The customer's sentence stays the request. A search supplies what the named
things look like and how a subject speaks. The lettering rules below decide
whether one line is printed. FLUX.2 dev draws the graphic; this module does
not call it.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx

from store_listing.orchestration.json_response import extract_json_objects

logger = logging.getLogger(__name__)

MAX_REFERENCES: int = 6
MAX_QUERIES: int = 4
_USER_AGENT = "store-listing-generator/1.0 (personal design research)"
_QUOTED = re.compile(r"“([^”\n]{1,120})”|\"([^\"\n]{1,120})\"")
_NAME = r"[A-Z][\w'.-]*"
_NAME_SEQ = rf"{_NAME}(?:\s+{_NAME}){{0,3}}"

_ARRANGEMENTS: tuple[str, ...] = (
    "The subject is large on the left and the other objects are smaller beside it.",
    "The subject is centered and the other objects sit above and to one side.",
    "The subject is on the right and the other objects are clustered to the left.",
)

_QUERY_SYSTEM = (
    "You choose web searches for one shirt design. The customer's sentence is the "
    'request. Return JSON only: {"queries": ["..."]}. Search for the places, '
    "rides, shows, and objects, and for how a named speaker talks. Give each "
    "place, ride, show, and character its own query, and search a character by "
    "that character's name alone. A place or a ride is not "
    "a movie. Do not merge two names into one search. At most four queries."
)

_DESIGN_SYSTEM = (
    "You turn search notes into one DTF shirt graphic. Return JSON only with "
    "customer_line (the phrase inside double quotes, copied exactly, or null; "
    "the picture description is not a line), "
    "voice_found (true only when the notes show how the subject speaks), "
    "lettering (one new line in that voice for this sentence, or null), "
    "subjects (how the named places, rides, characters, and objects look; "
    "keep a character the customer named, in those words), and "
    'references (up to six objects {"url", "keep"} for a clear photo of the '
    "subject or the place). A logo, a poster, a map, a costume crowd, a tight "
    "crop, or a photo full of unrelated people is keep false. Do not invent a "
    "line when voice_found is false. Do not replace words the customer wrote."
)


@dataclass(frozen=True)
class SearchNote:
    """One visual note from a search."""

    query: str
    title: str
    text: str
    image_url: str | None = None


@dataclass(frozen=True)
class PersonalDesign:
    """The graphic a personal sentence became."""

    prompt: str
    lettering: str | None
    reference_urls: tuple[str, ...]
    facts: tuple[str, ...]
    subjects: str
    customer_lettering: bool = False


def resolve_lettering(
    *,
    customer_line: str | None,
    proposed_line: str | None,
    voice_found: bool,
) -> str | None:
    """The one line for the shirt, or none.

    Words the customer already wrote stay as written. A new line is allowed
    only when the search showed how the subject speaks.
    """
    written = (customer_line or "").strip()
    if written:
        return written
    if not voice_found:
        return None
    proposed = (proposed_line or "").strip()
    return proposed or None


def quoted_lettering(sentence: str) -> str | None:
    """The shirt line the customer put in double quotes, copied exactly."""
    lines: list[str] = []
    for match in _QUOTED.finditer(sentence or ""):
        text = next(group for group in match.groups() if group)
        cleaned = " ".join(text.split())
        if cleaned and cleaned not in lines:
            lines.append(cleaned)
    if not lines:
        return None
    return " ".join(lines)


def _without_quotes(sentence: str) -> str:
    """The sentence with the shirt line taken out of the picture description."""
    cleaned = _QUOTED.sub(" ", sentence or "")
    cleaned = re.sub(r"\bwith text\b", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _names_in(fragment: str) -> list[str]:
    """Title-case names at the start of a fragment, split on and/or."""
    names: list[str] = []
    for part in re.split(r"\s+(?:and|or)\s+|,\s*|&\s*", fragment):
        match = re.match(rf"\s*({_NAME_SEQ})\b", part)
        if match is None:
            break
        name = " ".join(match.group(1).split())
        if name:
            names.append(name)
    return names


def specified_characters(sentence: str) -> tuple[str, ...]:
    """Characters the phrase names, not a generic 'characters' mention.

    ``Judy Hopps character`` and ``the character Judy Hopps`` name one
    character. ``Zootopia characters`` does not.
    """
    bare = _without_quotes(sentence)
    found: list[str] = []
    for match in re.finditer(
        r"\b(characters?)\b\s+(?:named\s+|called\s+)?",
        bare,
        flags=re.IGNORECASE,
    ):
        names = _names_in(bare[match.end() :])
        if match.group(1).lower() == "character":
            names = names[:1]
        found.extend(names)
    for match in re.finditer(r"\bcharacter\b", bare, flags=re.IGNORECASE):
        words = re.findall(r"[A-Za-z][\w'.-]*", bare[: match.start()])
        picked: list[str] = []
        for word in reversed(words):
            if word[:1].isupper():
                picked.append(word)
            else:
                break
        if picked:
            found.append(" ".join(reversed(picked)))
    unique: list[str] = []
    seen = {name.lower() for name in unique}
    for name in found:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return tuple(unique)


_ROLE_STOP = frozenset(
    {"a", "an", "the", "showing", "with", "text", "and", "or", "on", "in", "of", "for"}
)
_TITLE_FILLER = _ROLE_STOP | frozenset(
    {"list", "film", "movie", "series", "character", "characters"}
)


def character_role(sentence: str) -> str | None:
    """A singular character described by a role, ready to be searched.

    ``Zootopia sleuth character`` is a search. ``Judy Hopps character`` is
    already a name, and ``Zootopia characters`` is not one character.
    """
    if specified_characters(sentence):
        return None
    bare = _without_quotes(sentence)
    match = re.search(r"\bcharacter\b", bare, flags=re.IGNORECASE)
    if match is None or re.match(r"characters\b", bare[match.start() :], flags=re.IGNORECASE):
        return None
    words = re.findall(r"[A-Za-z][\w'.-]*", bare[: match.start()])
    picked: list[str] = []
    for word in reversed(words):
        if word.lower() in _ROLE_STOP:
            break
        picked.append(word)
        if len(picked) == 5:
            break
    if not picked or all(word[:1].isupper() for word in picked):
        return None
    return f"{' '.join(reversed(picked))} character"


def _proper_names(text: str, query_words: set[str]) -> list[str]:
    names: list[str] = []
    for match in re.finditer(rf"\b({_NAME_SEQ})\b", text or ""):
        kept = [
            word
            for word in match.group(1).split()
            if word.lower() not in query_words and word.lower() not in _TITLE_FILLER
        ]
        name = " ".join(kept)
        if name and name.lower() not in {item.lower() for item in names}:
            names.append(name)
    return names


def character_from_notes(query: str, notes: Sequence[SearchNote]) -> str | None:
    """The one character a role search names, or none when the notes disagree.

    A page titled Flash Slothmore is that character. A film page that lists
    Judy Hopps, Nick Wilde and Flash Slothmore is not a decision.
    """
    query_words = set(_title_text(query).split())
    titled: list[str] = []
    for note in notes:
        for name in _proper_names(note.title, query_words):
            if name.lower() not in {item.lower() for item in titled}:
                titled.append(name)
    if len(titled) == 1:
        return titled[0]
    if titled:
        return None
    mentioned: list[str] = []
    for note in notes:
        for name in _proper_names(note.text, query_words):
            if name.lower() not in {item.lower() for item in mentioned}:
                mentioned.append(name)
    if len(mentioned) == 1:
        return mentioned[0]
    return None


def lock_characters(subjects: str, characters: Sequence[str]) -> str:
    """The picture description, with every named character still in it."""
    text = " ".join((subjects or "").split())
    missing = [name for name in characters if name.lower() not in text.lower()]
    if missing:
        text = f"{text} The character is {', '.join(missing)}.".strip()
    if characters:
        names = ", ".join(characters)
        text = (
            f"{text} Draw {names} as that character, "
            "not a different character and not a generic substitute."
        )
    return " ".join(text.split())


def _title_text(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def matching_notes(query: str, notes: Sequence[SearchNote]) -> list[SearchNote]:
    """Notes whose title is the thing that was searched.

    A query for one ride also returns the mountain. When any title contains
    the query, the others are a different subject and are dropped.
    """
    phrase = _title_text(query)
    exact = [note for note in notes if phrase and phrase in _title_text(note.title)]
    return exact or list(notes)


def _add_query(queries: list[str], query: str) -> None:
    cleaned = " ".join(query.split())
    if not cleaned:
        return
    if any(cleaned.lower() == existing.lower() for existing in queries):
        return
    queries.append(cleaned)


def arrangement_line(seed: int) -> str:
    """One arrangement of the same facts. A new seed is a new arrangement."""
    return _ARRANGEMENTS[seed % len(_ARRANGEMENTS)]


def graphic_prompt(
    *,
    subjects: str,
    lettering: str | None,
    arrangement: str,
    facts: Sequence[str] = (),
    leave_text_blank: bool = False,
) -> str:
    """The ComfyUI prompt: a flat magenta key, the subjects, and one line."""
    fact_text = " ".join(fact.strip() for fact in facts if fact and fact.strip())
    known = f"Known facts: {fact_text} " if fact_text else ""
    if lettering:
        line = f' One line of lettering, spelled exactly: "{lettering}". No other text.'
    elif leave_text_blank:
        line = " No lettering and no words in the picture."
    else:
        line = ""
    placed = arrangement.strip()
    if placed and not placed.endswith("."):
        placed = f"{placed}."
    return (
        "Shirt print artwork. A solid pure magenta field fills every pixel that is "
        "not the subject, one flat colour edge to edge. No street, no sky, no ground, "
        "no buildings, no crowd, no place photograph, no frame, no backdrop. "
        f"{known}{subjects.strip()} {placed}{line}"
    )


def rearrange(design: PersonalDesign, seed: int) -> PersonalDesign:
    """The same facts and line in a different arrangement."""
    prompt = graphic_prompt(
        subjects=design.subjects,
        lettering=None if design.customer_lettering else design.lettering,
        arrangement=arrangement_line(seed),
        facts=design.facts,
        leave_text_blank=design.customer_lettering,
    )
    return replace(design, prompt=prompt)


def research_personal_design(
    sentence: str,
    *,
    complete: Callable[[list[dict[str, str]]], str],
    search: Callable[[str], list[SearchNote]],
    seed: int,
) -> PersonalDesign:
    """Search the sentence and turn the notes into one graphic.

    ``complete`` is one Qwen call. ``search`` is one query. Both are injected
    so a test never touches the network.
    """
    named = list(specified_characters(sentence))
    role = character_role(sentence)
    quotes = quoted_lettering(sentence)
    notes: list[SearchNote] = []
    character_notes: list[SearchNote] = []

    def _found(query: str) -> list[SearchNote]:
        try:
            found = search(query)
        except Exception:  # noqa: BLE001
            logger.warning("Personal design search failed for %r", query)
            found = []
        return matching_notes(query, found)

    role_notes = _found(role) if role else []
    notes.extend(role_notes)
    resolved = character_from_notes(role, role_notes) if role else None
    characters = list(named)
    if resolved is not None and resolved.lower() not in {name.lower() for name in characters}:
        characters.insert(0, resolved)
    queries: list[str] = []
    for name in characters:
        _add_query(queries, name)
    for query in _queries(complete, sentence):
        if len(queries) >= MAX_QUERIES:
            break
        _add_query(queries, query)

    character_keys = {name.lower() for name in characters}
    for query in queries:
        found = _found(query)
        if query.lower() in character_keys:
            character_notes.extend(found)
        notes.extend(found)
    if resolved is not None:
        character_notes.extend(
            note
            for note in role_notes
            if resolved.lower() in _title_text(note.title) or resolved.lower() in note.text.lower()
        )

    facts = tuple(_fact(note) for note in notes if _fact(note))
    design = _read_design(complete, sentence, notes)
    picture = _without_quotes(sentence).strip() or sentence.strip()
    if design is None:
        subjects = lock_characters(picture, characters)
        return PersonalDesign(
            prompt=graphic_prompt(
                subjects=subjects,
                lettering=None,
                arrangement=arrangement_line(seed),
                facts=facts,
                leave_text_blank=quotes is not None,
            ),
            lettering=quotes,
            reference_urls=_merged_urls(character_notes, ()),
            facts=facts,
            subjects=subjects,
            customer_lettering=quotes is not None,
        )

    lettering = (
        quotes
        if quotes
        else resolve_lettering(
            customer_line=design["customer_line"],
            proposed_line=design["lettering"],
            voice_found=bool(design["voice_found"]),
        )
    )
    subjects = lock_characters(
        _without_quotes(design["subjects"]).strip() or picture,
        characters,
    )
    urls = _merged_urls(character_notes, _reference_urls(design["references"], notes))
    return PersonalDesign(
        prompt=graphic_prompt(
            subjects=subjects,
            lettering=None if quotes else lettering,
            arrangement=arrangement_line(seed),
            facts=facts,
            leave_text_blank=quotes is not None,
        ),
        lettering=lettering,
        reference_urls=urls,
        facts=facts,
        subjects=subjects,
        customer_lettering=quotes is not None,
    )


def search_wikipedia(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """Short visual notes from Wikipedia. A failed call is an empty list."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": "3",
                "prop": "extracts|pageimages",
                "exintro": "1",
                "explaintext": "1",
                "exchars": "400",
                "piprop": "thumbnail",
                "pithumbsize": "1024",
                "format": "json",
            },
            headers={"User-Agent": "store-listing-generator/1.0 (personal design research)"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        logger.warning("Wikipedia search failed for %r", query)
        return []
    finally:
        if owns_client:
            http.close()

    pages = (payload.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return []
    notes: list[SearchNote] = []
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "").strip()
        text = str(page.get("extract") or "").strip()
        original = page.get("thumbnail") or page.get("original")
        source = original.get("source") if isinstance(original, dict) else None
        image_url = source if isinstance(source, str) and _is_raster_url(source) else None
        if title and text:
            notes.append(SearchNote(query=query, title=title, text=text, image_url=image_url))
    return notes


def search_duckduckgo(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """One short note from DuckDuckGo. A failed call is an empty list."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = http.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        logger.warning("DuckDuckGo search failed for %r", query)
        return []
    finally:
        if owns_client:
            http.close()

    if not isinstance(payload, dict):
        return []
    title = str(payload.get("Heading") or "").strip()
    text = str(payload.get("AbstractText") or "").strip()
    image = payload.get("Image")
    image_url = image if isinstance(image, str) and _is_raster_url(image) else None
    if not title or not text:
        return []
    return [SearchNote(query=query, title=title, text=text, image_url=image_url)]


def search_commons(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """Character and place pictures from Wikimedia Commons."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        response = http.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": "4",
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "iiurlwidth": "1024",
                "format": "json",
            },
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        logger.warning("Commons search failed for %r", query)
        return []
    finally:
        if owns_client:
            http.close()

    pages = (payload.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return []
    notes: list[SearchNote] = []
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "").removeprefix("File:").strip()
        info = page.get("imageinfo")
        thumb = info[0] if isinstance(info, list) and info else {}
        source = thumb.get("thumburl") if isinstance(thumb, dict) else None
        image_url = source if isinstance(source, str) and _is_raster_url(source) else None
        if title and image_url:
            notes.append(SearchNote(query=query, title=title, text=title, image_url=image_url))
    return notes


def search_personal(query: str, *, client: httpx.Client | None = None) -> list[SearchNote]:
    """Wikipedia, DuckDuckGo, and Commons notes for one query."""
    owns_client = client is None
    http = client or httpx.Client(timeout=10.0)
    try:
        notes: list[SearchNote] = []
        for search in (search_wikipedia, search_duckduckgo, search_commons):
            notes.extend(search(query, client=http))
        return notes
    finally:
        if owns_client:
            http.close()


_RASTER_TYPES = ("image/jpeg", "image/jpg", "image/png", "image/webp")
_FETCH_ATTEMPTS = 3


def _is_raster_url(url: str) -> bool:
    path = url.split("?", 1)[0].lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".webp"))


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """How long to wait after a rate limit. The server's number wins."""
    header = response.headers.get("retry-after", "").strip()
    try:
        requested = float(header)
    except ValueError:
        requested = float(attempt * 4)
    return min(max(requested, 1.0), 15.0)


def fetch_reference(url: str, *, client: httpx.Client | None = None) -> bytes | None:
    """The bytes of one reference photo, or None when it cannot be fetched.

    A logo is not a photo. A rate limit is waited out and tried again.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=20.0, follow_redirects=True)
    try:
        if not _is_raster_url(url):
            return None
        for attempt in range(1, _FETCH_ATTEMPTS + 1):
            try:
                response = http.get(
                    url,
                    headers={
                        "User-Agent": "store-listing-generator/1.0 (personal design research)"
                    },
                )
            except httpx.HTTPError:
                logger.warning("Reference image could not be fetched")
                return None
            if response.status_code in (429, 503) and attempt < _FETCH_ATTEMPTS:
                time.sleep(_retry_delay(response, attempt))
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPError:
                logger.warning("Reference image could not be fetched")
                return None
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type not in _RASTER_TYPES or not response.content:
                return None
            return response.content
        return None
    finally:
        if owns_client:
            http.close()


def _queries(complete: Callable[[list[dict[str, str]]], str], sentence: str) -> list[str]:
    raw = complete(
        [
            {"role": "system", "content": _QUERY_SYSTEM},
            {"role": "user", "content": sentence},
        ]
    )
    parsed = _first_object(raw)
    queries = parsed.get("queries") if isinstance(parsed, dict) else None
    if not isinstance(queries, list):
        return [sentence.strip()]
    cleaned = [str(query).strip() for query in queries if str(query).strip()]
    return cleaned[:MAX_QUERIES] or [sentence.strip()]


def _read_design(
    complete: Callable[[list[dict[str, str]]], str],
    sentence: str,
    notes: Sequence[SearchNote],
) -> dict[str, Any] | None:
    note_text = "\n".join(
        f"- {note.title}: {note.text}" + (f" image: {note.image_url}" if note.image_url else "")
        for note in notes
    )
    raw = complete(
        [
            {"role": "system", "content": _DESIGN_SYSTEM},
            {"role": "user", "content": f"Sentence: {sentence}\n\nNotes:\n{note_text}"},
        ]
    )
    parsed = _first_object(raw)
    if not isinstance(parsed, dict):
        return None
    subjects = parsed.get("subjects")
    if not isinstance(subjects, str):
        return None
    customer_line = parsed.get("customer_line")
    proposed = parsed.get("lettering")
    references = parsed.get("references")
    return {
        "customer_line": customer_line if isinstance(customer_line, str) else None,
        "voice_found": bool(parsed.get("voice_found")),
        "lettering": proposed if isinstance(proposed, str) else None,
        "subjects": subjects,
        "references": references if isinstance(references, list) else [],
    }


def _reference_urls(references: list[Any], notes: Sequence[SearchNote]) -> tuple[str, ...]:
    chosen: list[Any] = references
    if not chosen:
        chosen = [{"url": note.image_url, "keep": True} for note in notes if note.image_url]
    urls: list[str] = []
    for entry in chosen:
        if not isinstance(entry, dict) or entry.get("keep") is False:
            continue
        url = entry.get("url")
        if isinstance(url, str) and url.strip() and url.strip() not in urls:
            urls.append(url.strip())
    return tuple(urls[:MAX_REFERENCES])


def _merged_urls(character_notes: Sequence[SearchNote], chosen: Sequence[str]) -> tuple[str, ...]:
    """Character pictures first, then the other references, at most six."""
    urls: list[str] = []
    for note in character_notes:
        url = note.image_url
        if isinstance(url, str) and url.strip() and url.strip() not in urls:
            urls.append(url.strip())
    for url in chosen:
        if url not in urls:
            urls.append(url)
    return tuple(urls[:MAX_REFERENCES])


def _fact(note: SearchNote) -> str:
    text = note.text.strip()
    if not text or text.lower() == note.title.strip().lower():
        return ""
    sentence = text.split(". ", 1)[0].strip()
    if sentence and not sentence.endswith("."):
        sentence = f"{sentence}."
    return f"{note.title}: {sentence}" if sentence else note.title


def _first_object(text: str) -> Any:
    for value in extract_json_objects(text or ""):
        if isinstance(value, dict):
            return value
    return None
