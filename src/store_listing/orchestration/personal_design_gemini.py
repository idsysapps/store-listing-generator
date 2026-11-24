"""Full-sentence personal design research using Gemini with Google Search grounding.

One Gemini call with the customer's sentence figures out every named subject,
produces a visual description and a context-aware image search query for each,
then fetches reference images via Apify Google Images using those targeted
queries. This replaces the per-query Wikipedia/Commons/Wikidata loop for
personal briefs when GEMINI_API_KEY is set.

Key improvement over the per-query path: Gemini knows that "Tron" in the
context of "Magic Kingdom" is the Disney character, not a Swedish historical
painting. Its image_query values carry that context into Apify searches.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from store_listing.orchestration.json_response import extract_json_objects
from store_listing.orchestration.personal_design import (
    _HOLIDAY_TERMS,
    MAX_REFERENCES,
    PersonalDesign,
    SearchNote,
    arrangement_line,
    graphic_prompt,
    lock_characters,
    quoted_lettering,
    search_google_images,
)

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-3.8-flash"
_GEMINI_RESEARCH_SYSTEM = (
    "You research a custom shirt design request using Google Search. "
    "The customer's sentence is what they want on a shirt. "
    "Return JSON only with: "
    "subjects (array of objects for NAMED CHARACTERS AND CREATURES ONLY — do NOT include "
    "parks, theme parks, places, or location names as subjects; parks and locations set the "
    "scene and theme but are not drawn as characters; e.g. for 'Disney Animal Kingdom - "
    "Mickey, Simba, Zazu' the subjects are Mickey, Simba, and Zazu — NOT Animal Kingdom; "
    "each subject has name (the character as the customer wrote it, verbatim), "
    "description (2-3 sentences of visual appearance — colors, distinctive features, "
    "costume, style — grounded in what you found), and image_query (a specific Google "
    "Images search string to find high-quality character art — be specific, add "
    "'flat cartoon illustration' or 'official Disney art' to avoid photos and Etsy fan art; "
    "include the franchise name, e.g. "
    "'Simba Lion King Disney cub official cartoon art flat illustration', "
    "'Mickey Mouse Disney safari outfit flat cartoon illustration')), "
    "scene (1-2 sentences as a creative illustrator brief in flat 2D cartoon vector "
    "style — bold outlines, clean color fills, no photorealism, no gradients; one "
    "primary character or element anchors the center foreground as the focal point, "
    "supporting characters and elements radiate outward and upward around it in a "
    "pyramid or cluster composition; active verbs, all elements interacting with the "
    "theme; no sky, no ground, no crowd, no backdrop — isolated on a flat field), "
    "customer_line (phrase in double quotes from the sentence, copied exactly, or null), "
    "voice_found (true when results reveal the subject's catchphrase, motto, theme song, "
    "or signature line — search for it; for a ride or show look for its theme song or "
    "opening lyric; for a character look for their catchphrase; only true when you find "
    "actual words, not a general description), "
    "lettering (the exact catchphrase, theme song line, or motto you found — copy it "
    "verbatim, do not paraphrase; null when voice_found is false)."
)
_GEMINI_ATTEMPTS = 3


@dataclass(frozen=True)
class GeminiSubject:
    """One named subject from Gemini's full-sentence research."""

    name: str
    description: str
    image_query: str


@dataclass(frozen=True)
class GeminiResearch:
    """Structured output from one Gemini full-sentence research call."""

    subjects: tuple[GeminiSubject, ...]
    scene: str
    customer_line: str | None
    voice_found: bool
    lettering: str | None


def gemini_research_sentence(
    sentence: str,
    *,
    genai_client: Any | None = None,
) -> GeminiResearch | None:
    """Research the full customer sentence in one Gemini call with Search grounding.

    Returns None when GEMINI_API_KEY is not set, the call fails, or the
    response cannot be parsed. The caller falls back to the per-query path.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        from google.genai import types  # type: ignore[import-not-found]

        if genai_client is None:
            from google import genai  # type: ignore[import-not-found]

            genai_client = genai.Client(
                api_key=api_key,
                http_options={"api_version": "v1beta"},
            )

        last_exc: Exception | None = None
        for attempt in range(1, _GEMINI_ATTEMPTS + 1):
            try:
                response = genai_client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=f"Customer sentence: {sentence}",
                    config=types.GenerateContentConfig(
                        system_instruction=_GEMINI_RESEARCH_SYSTEM,
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        max_output_tokens=1500,
                    ),
                )
                break
            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) and attempt < _GEMINI_ATTEMPTS:
                    time.sleep(min(4.0 * attempt, 15.0))
                    continue
                raise
        else:
            raise last_exc  # type: ignore[misc]

        text = (response.text or "").strip()
        if not text:
            return None
        return _parse_research(text)
    except Exception:  # noqa: BLE001
        logger.warning("Gemini full-sentence research failed for %r", sentence)
        return None


def _parse_research(text: str) -> GeminiResearch | None:
    for obj in extract_json_objects(text):
        if not isinstance(obj, dict):
            continue
        raw_subjects = obj.get("subjects")
        if not isinstance(raw_subjects, list) or not raw_subjects:
            continue
        subjects = []
        for s in raw_subjects:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "").strip()
            description = str(s.get("description") or "").strip()
            image_query = str(s.get("image_query") or name).strip()
            if name and description:
                subjects.append(
                    GeminiSubject(name=name, description=description, image_query=image_query)
                )
        if not subjects:
            continue
        scene = str(obj.get("scene") or "").strip()
        if not scene:
            continue
        return GeminiResearch(
            subjects=tuple(subjects),
            scene=scene,
            customer_line=str(obj.get("customer_line") or "").strip() or None,
            voice_found=bool(obj.get("voice_found")),
            lettering=str(obj.get("lettering") or "").strip() or None,
        )
    return None


def research_personal_design_gemini(
    sentence: str,
    *,
    seed: int,
    genai_client: Any | None = None,
    apify_client_cls: Any | None = None,
    image_search: Callable[[str], list[SearchNote]] | None = None,
) -> PersonalDesign | None:
    """Full-sentence research path: one Gemini call + targeted Apify image searches.

    Returns None when GEMINI_API_KEY is not set or Gemini fails, so the
    caller can fall back to the per-query research_personal_design path.

    ``image_search`` is injectable for testing. When not provided, it uses
    ``search_google_images`` with the Apify actor.
    """
    research = gemini_research_sentence(sentence, genai_client=genai_client)
    if research is None:
        return None

    quotes = quoted_lettering(sentence)

    # Collect context-aware reference images for each subject
    reference_urls: list[str] = []
    _apify = apify_client_cls

    def _fetch_images(q: str) -> list[SearchNote]:
        if image_search is not None:
            return image_search(q)
        return search_google_images(q, apify_client_cls=_apify)

    for subject in research.subjects:
        if len(reference_urls) >= MAX_REFERENCES:
            break
        notes = _fetch_images(subject.image_query)
        for note in notes:
            if note.image_url and note.image_url not in reference_urls:
                reference_urls.append(note.image_url)
                if len(reference_urls) >= MAX_REFERENCES:
                    break

    # Build facts from subject descriptions
    facts = tuple(
        f"{s.name}: {s.description.split('.')[0].strip()}."
        for s in research.subjects
        if s.description
    )

    # Subject names for the lock instruction
    character_names = [s.name for s in research.subjects]

    # Holiday fallback: if Gemini's scene dropped a holiday the customer named, inject it
    scene = research.scene
    sentence_lower = sentence.lower()
    subjects_lower = scene.lower()
    missed_holidays = [t for t in _HOLIDAY_TERMS if t in sentence_lower and t not in subjects_lower]
    if missed_holidays:
        holiday_text = " and ".join(t.title() for t in missed_holidays)
        scene = f"{scene} The scene is {holiday_text}-themed."

    subjects = lock_characters(scene, character_names)

    lettering = quotes if quotes else (research.lettering if research.voice_found else None)

    prompt = graphic_prompt(
        subjects=subjects,
        lettering=None if quotes else lettering,
        arrangement=arrangement_line(seed),
        facts=facts,
        leave_text_blank=quotes is not None,
    )

    return PersonalDesign(
        prompt=prompt,
        lettering=lettering,
        reference_urls=tuple(reference_urls[:MAX_REFERENCES]),
        facts=facts,
        subjects=subjects,
        customer_lettering=quotes is not None,
        search_query_count=len(research.subjects),
        note_count=len(research.subjects),
    )
