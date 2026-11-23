"""Research a personal custom sentence into one DTF shirt graphic.

The customer's sentence stays the request. A search supplies what the named
things look like and how a subject speaks. The lettering rules below decide
whether one line is printed. FLUX.2 dev draws the graphic; this module does
not call it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx

from store_listing.orchestration.json_response import extract_json_objects

logger = logging.getLogger(__name__)

MAX_REFERENCES: int = 6
MAX_QUERIES: int = 4

_ARRANGEMENTS: tuple[str, ...] = (
    "The subject is large on the left and the other objects are smaller beside it.",
    "The subject is centered and the other objects sit above and to one side.",
    "The subject is on the right and the other objects are clustered to the left.",
)

_QUERY_SYSTEM = (
    "You choose web searches for one shirt design. The customer's sentence is the "
    'request. Return JSON only: {"queries": ["..."]}. Search for the places, '
    "rides, shows, and objects, and for how a named speaker talks. Give each "
    "place, ride, show, and character its own query. A place or a ride is not "
    "a movie. Do not merge two names into one search. At most four queries."
)

_DESIGN_SYSTEM = (
    "You turn search notes into one DTF shirt graphic. Return JSON only with "
    "customer_line (a phrase the customer asked to print, or null; the picture "
    "description is not a line), "
    "voice_found (true only when the notes show how the subject speaks), "
    "lettering (one new line in that voice for this sentence, or null), "
    "subjects (how the named places, rides, characters, and objects look), and "
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


def arrangement_line(seed: int) -> str:
    """One arrangement of the same facts. A new seed is a new arrangement."""
    return _ARRANGEMENTS[seed % len(_ARRANGEMENTS)]


def graphic_prompt(
    *,
    subjects: str,
    lettering: str | None,
    arrangement: str,
    facts: Sequence[str] = (),
) -> str:
    """The ComfyUI prompt: a flat magenta key, the subjects, and one line."""
    fact_text = " ".join(fact.strip() for fact in facts if fact and fact.strip())
    known = f"Known facts: {fact_text} " if fact_text else ""
    line = ""
    if lettering:
        line = f' One line of lettering, spelled exactly: "{lettering}". No other text.'
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
        lettering=design.lettering,
        arrangement=arrangement_line(seed),
        facts=design.facts,
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
    queries = _queries(complete, sentence)
    notes: list[SearchNote] = []
    for query in queries:
        try:
            found = search(query)
        except Exception:  # noqa: BLE001
            logger.warning("Personal design search failed for %r", query)
            found = []
        notes.extend(found)

    facts = tuple(_fact(note) for note in notes if _fact(note))
    design = _read_design(complete, sentence, notes)
    if design is None:
        return PersonalDesign(
            prompt=graphic_prompt(
                subjects=sentence.strip(),
                lettering=None,
                arrangement=arrangement_line(seed),
                facts=facts,
            ),
            lettering=None,
            reference_urls=(),
            facts=facts,
            subjects=sentence.strip(),
        )

    lettering = resolve_lettering(
        customer_line=design["customer_line"],
        proposed_line=design["lettering"],
        voice_found=bool(design["voice_found"]),
    )
    subjects = design["subjects"].strip() or sentence.strip()
    urls = _reference_urls(design["references"], notes)
    return PersonalDesign(
        prompt=graphic_prompt(
            subjects=subjects,
            lettering=lettering,
            arrangement=arrangement_line(seed),
            facts=facts,
        ),
        lettering=lettering,
        reference_urls=urls,
        facts=facts,
        subjects=subjects,
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


def _fact(note: SearchNote) -> str:
    text = note.text.strip()
    if not text:
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
