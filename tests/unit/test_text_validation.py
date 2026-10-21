"""Tests for the OCR text-validation gate."""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import MagicMock

from PIL import Image

from store_listing.orchestration.text_validation import (
    TextValidator,
    encode_image_for_vision,
    missing_words,
    normalize_text,
    text_matches,
)


def _png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    image = Image.new("RGB", size, (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _vision_client(observation: str) -> Any:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=observation))]
    )
    return client


class TestNormalizeText:
    def test_folds_case(self) -> None:
        assert normalize_text("SPOOKY SEASON") == "spooky season"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("spooky   season\n") == "spooky season"

    def test_strips_punctuation(self) -> None:
        assert normalize_text("Spooky, Season!") == "spooky season"

    def test_empty_string_stays_empty(self) -> None:
        assert normalize_text("") == ""


class TestTextMatches:
    def test_identical_text_matches_perfectly(self) -> None:
        matched, score = text_matches("SPOOKY SEASON", "spooky season")
        assert matched is True
        assert score == 1.0

    def test_a_single_wrong_character_still_matches(self) -> None:
        matched, score = text_matches("SPOOKY SEASON", "SPOOKY SEAS0N")
        assert matched is True
        assert score >= 0.85

    def test_different_text_does_not_match(self) -> None:
        matched, score = text_matches("SPOOKY SEASON", "HAPPY HALLOWEEN")
        assert matched is False
        assert score < 0.85

    def test_a_stray_extra_word_does_not_break_the_match(self) -> None:
        matched, _ = text_matches("SPOOKY SEASON", "the spooky season is here")
        assert matched is True

    def test_empty_observation_does_not_match(self) -> None:
        matched, score = text_matches("SPOOKY SEASON", "")
        assert matched is False
        assert score == 0.0

    def test_empty_expectation_matches(self) -> None:
        matched, _ = text_matches("", "anything")
        assert matched is True


class TestWordLevelMatching:
    """The characters of a line are not enough: a word has to be the word.

    A character score of 0.85 let a misspelled word through - the bubble on
    the image of brief #100 read "To get to the onthe sid" and the gate passed
    it (issue #208).
    """

    def test_a_misspelled_word_fails(self) -> None:
        matched, _ = text_matches("To get to the other side", "To get to the onthe sid")
        assert matched is False

    def test_a_wrong_word_fails(self) -> None:
        matched, _ = text_matches("Cross the Road?", "Cross the toad?")
        assert matched is False

    def test_a_dropped_letter_in_a_short_word_fails(self) -> None:
        matched, _ = text_matches("SPOOKY SEASON", "SPOOKY SEAS")
        assert matched is False

    def test_a_dropped_letter_in_a_long_word_is_tolerated(self) -> None:
        # The text is drawn with a font, so a long word that is one character
        # off is the vision model misreading it, not a typo in the design.
        matched, _ = text_matches("SPOOKY SEASON", "SPOOKY SEAS0N")
        assert matched is True

    def test_the_whole_text_must_be_there(self) -> None:
        matched, _ = text_matches("Why Did the Pumpkin", "Why Did the")
        assert matched is False

    def test_a_word_split_over_a_line_break_is_still_found(self) -> None:
        matched, _ = text_matches("Pumpkin Patch Crew", "Pumpkin\nPatch Crew")
        assert matched is True

    def test_extra_words_around_the_text_do_not_break_the_match(self) -> None:
        matched, _ = text_matches("SPOOKY SEASON", "the spooky season is here")
        assert matched is True

    def test_the_missing_words_are_reported(self) -> None:
        assert missing_words("Cross the Road?", "Cross the toad?") == ["road"]

    def test_nothing_is_missing_when_the_text_matches(self) -> None:
        assert missing_words("Cross the Road?", "Cross the Road?") == []


class TestEncodeImageForVision:
    def test_returns_a_jpeg_data_url(self) -> None:
        url = encode_image_for_vision(_png_bytes())
        assert url.startswith("data:image/jpeg;base64,")

    def test_the_decoded_payload_is_a_valid_jpeg(self) -> None:
        url = encode_image_for_vision(_png_bytes())
        payload = base64.b64decode(url.split(",", 1)[1])
        assert Image.open(io.BytesIO(payload)).format == "JPEG"

    def test_downscales_to_the_max_side(self) -> None:
        url = encode_image_for_vision(_png_bytes((2048, 1024)), max_side=1024)
        payload = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as image:
            assert max(image.size) <= 1024

    def test_leaves_small_images_alone(self) -> None:
        url = encode_image_for_vision(_png_bytes((64, 64)), max_side=1024)
        payload = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as image:
            assert image.size == (64, 64)


class TestTextValidator:
    def test_matches_when_the_image_carries_the_expected_text(self) -> None:
        client = _vision_client("SPOOKY SEASON")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="SPOOKY SEASON", tagline=None)

        assert result.matched is True
        assert result.error is None
        assert result.expected == ["SPOOKY SEASON"]

    def test_does_not_match_when_the_text_is_wrong(self) -> None:
        client = _vision_client("HAPPY HALLOWEEN")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="SPOOKY SEASON", tagline=None)

        assert result.matched is False

    def test_no_text_in_the_image_does_not_match(self) -> None:
        client = _vision_client("NO_TEXT")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="SPOOKY SEASON", tagline=None)

        assert result.matched is False
        assert result.error is not None

    def test_a_model_error_fails_closed(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("vision model is down")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="SPOOKY SEASON", tagline=None)

        assert result.matched is False
        assert result.error is not None
        assert "vision model is down" in result.error

    def test_briefs_without_text_are_not_sent_to_the_model(self) -> None:
        client = _vision_client("NO_TEXT")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline=None, tagline=None)

        assert result.matched is True
        client.chat.completions.create.assert_not_called()

    def test_a_missing_tagline_does_not_fail_the_gate(self) -> None:
        # Diffusion drops the secondary line far more often than the headline. The
        # headline carries the message, so only it gates the image.
        client = _vision_client("SPOOKY SEASON")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(
            _png_bytes(), headline="SPOOKY SEASON", tagline="TRICK OR TREAT"
        )

        assert result.matched is True
        assert result.expected == ["SPOOKY SEASON", "TRICK OR TREAT"]
        assert result.missing == ["TRICK OR TREAT"]

    def test_a_missing_headline_fails_even_when_the_tagline_matches(self) -> None:
        client = _vision_client("TRICK OR TREAT")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(
            _png_bytes(), headline="SPOOKY SEASON", tagline="TRICK OR TREAT"
        )

        assert result.matched is False
        assert result.missing == ["SPOOKY SEASON"]

    def test_a_headline_that_is_not_text_is_not_gated(self) -> None:
        # The OCR model never transcribes a symbol verbatim, so gating on it would
        # reject every image that carries one.
        client = _vision_client("Spooky Vibes Only")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="🎃", tagline="Spooky Vibes Only")

        assert result.matched is True
        assert result.unverifiable == ["🎃"]

    def test_strict_mode_requires_the_tagline(self) -> None:
        client = _vision_client("SPOOKY SEASON")
        validator = TextValidator(client=client, model="qwen_vl_ocr", require_tagline=True)

        result = validator.validate(
            _png_bytes(), headline="SPOOKY SEASON", tagline="TRICK OR TREAT"
        )

        assert result.matched is False
        assert result.missing == ["TRICK OR TREAT"]

    def test_a_headline_and_a_tagline_on_their_own_lines_both_match(self) -> None:
        client = _vision_client("SPOOKY SEASON\nTRICK OR TREAT")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(
            _png_bytes(), headline="SPOOKY SEASON", tagline="TRICK OR TREAT"
        )

        assert result.matched is True
        assert result.missing == []

    def test_text_the_model_drew_twice_is_reported(self) -> None:
        client = _vision_client("SPOOKY SEASON\nSPOOKY SEASON")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="SPOOKY SEASON", tagline=None)

        assert result.matched is True
        assert result.duplicated == ["SPOOKY SEASON"]

    def test_a_headline_wrapped_over_two_lines_is_still_found(self) -> None:
        client = _vision_client("SPOOKY\nSEASON")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="SPOOKY SEASON", tagline=None)

        assert result.matched is True
        assert result.missing == []

    def test_the_transcription_is_kept_verbatim(self) -> None:
        client = _vision_client("Bag of... Spooky Fun.")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        result = validator.validate(_png_bytes(), headline="Bag of Spooky Fun", tagline=None)

        assert result.observed == "Bag of... Spooky Fun."

    def test_sends_the_image_to_the_vision_model(self) -> None:
        client = _vision_client("SPOOKY SEASON")
        validator = TextValidator(client=client, model="qwen_vl_ocr")

        validator.validate(_png_bytes(), headline="SPOOKY SEASON", tagline=None)

        call = client.chat.completions.create.call_args
        assert call.kwargs["model"] == "qwen_vl_ocr"
        content = call.kwargs["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
