"""Tests for the design image generation pipeline (ComfyUI + OCR gate)."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, call

import pytest
from PIL import Image

from store_listing.orchestration.artwork_validation import (
    ArtworkTextResult,
    ArtworkValidationResult,
    NoTextResult,
)
from store_listing.orchestration.comfyui_client import ComfyUIError, ComfyUIResult
from store_listing.orchestration.image_pipeline import (
    generate_image_for_brief,
    generate_images_for_briefs,
)
from store_listing.orchestration.text_validation import StrictTextResult, TextValidationResult


def _real_png(width: int = 64, height: int = 64, color: str = "white") -> bytes:
    img = Image.new("RGBA", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_db_client(briefs: list[dict[str, Any]] | None = None) -> MagicMock:
    client = MagicMock()
    client.list_briefs_without_images.return_value = briefs or []
    return client


def _comfyui_client(raw: bytes = b"raw-bytes", transparent: bytes = b"transparent") -> MagicMock:
    client = MagicMock()
    client.generate_image.return_value = ComfyUIResult(image_bytes=raw, prompt_id="p1")
    client.remove_background.return_value = ComfyUIResult(image_bytes=transparent, prompt_id="p2")
    return client


def _validator(*verdicts: bool) -> MagicMock:
    validator = MagicMock()
    validator.validate.side_effect = [
        TextValidationResult(
            matched=verdict,
            expected=["SPOOKY SEASON"],
            observed="SPOOKY SEASON" if verdict else "SOMETHING ELSE",
            score=1.0 if verdict else 0.2,
        )
        for verdict in verdicts
    ]
    return validator


def _validator_with_missing(*missing: list[str]) -> MagicMock:
    """A validator that accepts the image but reports text it could not read back."""
    validator = MagicMock()
    validator.validate.side_effect = [
        TextValidationResult(
            matched=True,
            expected=["SPOOKY SEASON", "TRICK OR TREAT"],
            observed="SPOOKY SEASON",
            score=1.0,
            missing=dropped,
        )
        for dropped in missing
    ]
    return validator


def _distinct_colours(image: Image.Image) -> set[tuple[int, int, int]]:
    """The colours that occur in the image."""
    colours = image.getcolors(maxcolors=1_000_000) or []
    return {
        (int(colour[0]), int(colour[1]), int(colour[2]))
        for _count, colour in colours
        if isinstance(colour, tuple) and len(colour) == 3
    }


def _brief(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": 42,
        "concept": "Skeleton yoga",
        "product_type": "sublimation",
        "audience": "yoga fans",
        "visual_style": "retro vintage",
        "layout_type": "full_bleed",
        "headline_text": None,
        "tagline_text": None,
        "font_color": None,
        "scene_description": "a skeleton doing yoga on a beach towel",
    }
    row.update(overrides)
    return row


class TestGenerateImagesForBriefs:
    def test_skips_when_no_briefs_need_images(self) -> None:
        db = _mock_db_client(briefs=[])
        result = generate_images_for_briefs(
            db_client=db, comfyui_client=_comfyui_client(), s3_client=MagicMock()
        )
        assert result["images_generated"] == 0

    def test_the_sweep_only_takes_the_briefs_of_the_types_in_production(self) -> None:
        # A store that focuses on DTF apparel does not spend GPU time on the
        # old sublimation briefs (issue #215).
        db = _mock_db_client(briefs=[])

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            product_types=["dtf_apparel"],
        )

        db.list_briefs_without_images.assert_called_once_with(20, ["dtf_apparel"])

    def test_generates_and_stores_the_image(self) -> None:
        db = _mock_db_client(briefs=[_brief()])
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, comfyui_client=_comfyui_client(), s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        assert result["errors"] == 0
        db.update_brief_image_keys.assert_called_once()
        call_kwargs = db.update_brief_image_keys.call_args.kwargs
        assert call_kwargs["brief_id"] == 42
        assert "raw" in call_kwargs["image_key_raw"]

    def test_dtf_brief_uploads_the_transparent_png(self) -> None:
        db = _mock_db_client(briefs=[_brief(product_type="dtf_apparel")])
        comfyui = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, comfyui_client=comfyui, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        comfyui.remove_background.assert_called_once()
        assert s3.put_object.call_count == 2
        call_kwargs = db.update_brief_image_keys.call_args.kwargs
        assert "transparent" in call_kwargs["image_key_transparent"]

    def test_a_background_removal_failure_still_uploads_the_raw_image(self) -> None:
        db = _mock_db_client(briefs=[_brief(product_type="dtf_apparel")])
        comfyui = _comfyui_client()
        comfyui.remove_background.side_effect = ComfyUIError("BiRefNet exploded")
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, comfyui_client=comfyui, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        assert s3.put_object.call_count == 1
        call_kwargs = db.update_brief_image_keys.call_args.kwargs
        assert call_kwargs["image_key_transparent"] is None

    def test_continues_on_generation_failure(self) -> None:
        db = _mock_db_client(briefs=[_brief(id=1), _brief(id=2)])
        comfyui = _comfyui_client()
        comfyui.generate_image.side_effect = [
            ComfyUIError("ComfyUI is down"),
            ComfyUIResult(image_bytes=b"raw-bytes", prompt_id="p1"),
        ]
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, comfyui_client=comfyui, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        assert result["errors"] == 1


class TestTextValidationGate:
    def test_a_matching_verdict_uploads_normally(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        validator = _validator(True)
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            text_validator=validator,
        )

        assert result["images_generated"] == 1
        validator.validate.assert_called_once()

    def test_a_persistent_mismatch_uploads_nothing(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        validator = _validator(False, False, False)
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            text_validator=validator,
        )

        assert result["images_generated"] == 0
        assert result["errors"] == 1
        s3.put_object.assert_not_called()
        db.update_brief_image_keys.assert_not_called()

    def test_a_mismatch_then_a_match_uploads_the_second_image(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        validator = _validator(False, True)
        comfyui = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=s3,
            bucket="test-bucket",
            text_validator=validator,
        )

        assert result["images_generated"] == 1
        assert comfyui.generate_image.call_count == 2
        db.update_brief_image_keys.assert_called_once()

    def test_each_retry_uses_a_different_seed(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        validator = _validator(False, True)
        comfyui = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            text_validator=validator,
        )

        seeds = [call.kwargs["seed"] for call in comfyui.generate_image.call_args_list]
        assert len(seeds) == 2
        assert seeds[0] != seeds[1]

    def test_a_retry_asks_for_clear_space_where_the_text_goes(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        validator = _validator(False, True)
        comfyui = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            text_validator=validator,
        )

        second_prompt = comfyui.generate_image.call_args_list[1].kwargs["prompt"]
        # The text is composited onto the artwork, so a retry asks for a
        # clearer area for it instead of asking the model to spell it.
        assert "clear" in second_prompt.lower()
        assert "must read exactly" not in second_prompt

    def test_a_retry_does_not_ask_the_model_to_spell(self) -> None:
        db = _mock_db_client(
            briefs=[
                _brief(
                    layout_type="text_top_bottom",
                    headline_text="SPOOKY SEASON",
                    tagline_text="TRICK OR TREAT",
                )
            ]
        )
        validator = _validator(False, True)
        comfyui = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            text_validator=validator,
        )

        second_prompt = comfyui.generate_image.call_args_list[1].kwargs["prompt"]
        assert "SPOOKY SEASON" not in second_prompt
        assert "TRICK OR TREAT" not in second_prompt

    def test_a_dropped_tagline_is_uploaded_and_reported(self, caplog) -> None:
        db = _mock_db_client(
            briefs=[
                _brief(
                    layout_type="text_top_bottom",
                    headline_text="SPOOKY SEASON",
                    tagline_text="TRICK OR TREAT",
                )
            ]
        )
        validator = _validator_with_missing(["TRICK OR TREAT"])
        s3 = MagicMock()

        with caplog.at_level(logging.WARNING):
            result = generate_images_for_briefs(
                db_client=db,
                comfyui_client=_comfyui_client(),
                s3_client=s3,
                bucket="test-bucket",
                text_validator=validator,
            )

        assert result["images_generated"] == 1
        s3.put_object.assert_called_once()
        assert "TRICK OR TREAT" in caplog.text

    def test_briefs_without_text_never_reach_the_validator(self) -> None:
        db = _mock_db_client(briefs=[_brief()])
        validator = _validator()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            bucket="test-bucket",
            text_validator=validator,
        )

        validator.validate.assert_not_called()

    def test_the_composited_text_is_uploaded(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(raw=_real_png()),
            s3_client=s3,
            bucket="test-bucket",
        )

        body = s3.put_object.call_args.kwargs["Body"]
        image = Image.open(BytesIO(body)).convert("RGB")
        width, height = image.size
        top_zone = image.crop((0, 0, width, int(height * 0.15)))
        colours = _distinct_colours(top_zone)
        assert colours != {(255, 255, 255)}, "the headline should have been drawn on the artwork"

    def test_the_validator_sees_the_composited_image(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        validator = _validator(True)

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(raw=_real_png()),
            s3_client=MagicMock(),
            bucket="test-bucket",
            text_validator=validator,
        )

        validated = validator.validate.call_args.args[0]
        image = Image.open(BytesIO(validated)).convert("RGB")
        width, height = image.size
        top_zone = image.crop((0, 0, width, int(height * 0.15)))
        colours = _distinct_colours(top_zone)
        assert colours != {(255, 255, 255)}, "the validated image should carry the headline"

    def test_the_transparent_image_carries_the_text_too(self) -> None:
        db = _mock_db_client(
            briefs=[
                _brief(
                    product_type="dtf_apparel",
                    layout_type="text_top",
                    headline_text="SPOOKY SEASON",
                )
            ]
        )
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(raw=_real_png(), transparent=_real_png()),
            s3_client=s3,
            bucket="test-bucket",
        )

        transparent_body = s3.put_object.call_args_list[1].kwargs["Body"]
        image = Image.open(BytesIO(transparent_body)).convert("RGB")
        width, height = image.size
        top_zone = image.crop((0, 0, width, int(height * 0.15)))
        colours = _distinct_colours(top_zone)
        assert colours != {(255, 255, 255)}, "the print file should carry the headline"

    def test_no_validator_skips_the_gate(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            bucket="test-bucket",
        )

        assert result["images_generated"] == 1


class TestGenerateImageForBrief:
    def test_returns_error_when_brief_not_found(self) -> None:
        db = MagicMock()
        db.get_brief_by_id.return_value = None

        result = generate_image_for_brief(
            brief_id=999,
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
        )

        assert result["status"] == "error"
        assert "999" in result["error"]

    def test_generates_image_for_specific_brief(self) -> None:
        db = MagicMock()
        db.get_brief_by_id.return_value = _brief(id=7)

        result = generate_image_for_brief(
            brief_id=7,
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            bucket="test-bucket",
        )

        assert result["status"] == "success"
        assert result["brief_id"] == 7
        db.update_brief_image_keys.assert_called_once()

    def test_a_failed_text_validation_reports_an_error(self) -> None:
        db = MagicMock()
        db.get_brief_by_id.return_value = _brief(
            id=8, layout_type="text_top", headline_text="SPOOKY SEASON"
        )
        s3 = MagicMock()

        result = generate_image_for_brief(
            brief_id=8,
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            text_validator=_validator(False, False, False),
        )

        assert result["status"] == "error"
        s3.put_object.assert_not_called()
        db.update_brief_image_keys.assert_not_called()

    def test_the_prompt_carries_the_regeneration_feedback(self) -> None:
        db = MagicMock()
        db.get_brief_by_id.return_value = _brief(id=7, regeneration_feedback="more vibrant colors")
        comfyui = _comfyui_client()

        generate_image_for_brief(
            brief_id=7,
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
        )

        prompt = comfyui.generate_image.call_args.kwargs["prompt"]
        assert "more vibrant colors" in prompt


class TestTypographyOnlyDesigns:
    def test_a_typography_only_brief_never_reaches_the_image_model(self) -> None:
        db = _mock_db_client(
            briefs=[_brief(layout_type="text_only", headline_text="SPOOKY SEASON")]
        )
        comfyui = _comfyui_client()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
        )

        assert result["images_generated"] == 1
        comfyui.generate_image.assert_not_called()

    def test_a_typography_only_dtf_design_uploads_a_raw_and_a_transparent_file(self) -> None:
        db = _mock_db_client(
            briefs=[
                _brief(
                    product_type="dtf_apparel",
                    layout_type="text_only",
                    headline_text="SPOOKY SEASON",
                )
            ]
        )
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
        )

        assert result["images_generated"] == 1
        assert s3.put_object.call_count == 2
        call_kwargs = db.update_brief_image_keys.call_args.kwargs
        assert "transparent" in call_kwargs["image_key_transparent"]

    def test_the_typography_only_dtf_raw_is_the_key_colour(self) -> None:
        db = _mock_db_client(
            briefs=[
                _brief(
                    product_type="dtf_apparel",
                    layout_type="text_only",
                    headline_text="SPOOKY SEASON",
                )
            ]
        )
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
        )

        body = s3.put_object.call_args_list[0].kwargs["Body"]
        image = Image.open(BytesIO(body)).convert("RGB")
        assert image.getpixel((0, 0)) == (255, 0, 255)

    def test_the_typography_only_print_file_carries_the_headline(self) -> None:
        db = _mock_db_client(
            briefs=[_brief(layout_type="text_only", headline_text="SPOOKY SEASON")]
        )
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
        )

        body = s3.put_object.call_args.kwargs["Body"]
        image = Image.open(BytesIO(body)).convert("RGB")
        width, height = image.size
        drawn = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if image.getpixel((x, y)) != (255, 255, 255)
        ]
        assert drawn, "the headline should have been drawn on the canvas"

    def test_a_typography_only_design_still_goes_through_the_text_gate(self) -> None:
        db = _mock_db_client(
            briefs=[_brief(layout_type="text_only", headline_text="SPOOKY SEASON")]
        )
        validator = _validator(True)

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            bucket="test-bucket",
            text_validator=validator,
        )

        validator.validate.assert_called_once()

    def test_a_typography_only_design_that_fails_the_text_gate_is_not_uploaded(self) -> None:
        db = _mock_db_client(
            briefs=[_brief(layout_type="text_only", headline_text="SPOOKY SEASON")]
        )
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            text_validator=_validator(False),
        )

        assert result["images_generated"] == 0
        s3.put_object.assert_not_called()

    def test_a_brief_without_a_scene_is_rendered_as_typography_only(self) -> None:
        # The concept may name the product, so it is never used as the visual
        # prompt. Without a scene the brief has no artwork - the typography is
        # still worth printing (issue #195).
        db = _mock_db_client(
            briefs=[
                _brief(
                    concept="Trick or Treat Bag with Candy Scatter",
                    scene_description=None,
                    layout_type="text_top",
                    headline_text="Trick or Treat",
                )
            ]
        )
        comfyui = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=s3,
            bucket="test-bucket",
        )

        assert result["images_generated"] == 1
        comfyui.generate_image.assert_not_called()
        s3.put_object.assert_called()

    def test_a_brief_whose_scene_is_only_words_is_rendered_as_typography_only(self) -> None:
        # A scene that only asks the image model to spell something leaves no
        # picture to draw (issue #208).
        db = _mock_db_client(
            briefs=[
                _brief(
                    scene_description="a speech bubble saying 'To get to the other side!'",
                    layout_type="text_top",
                    headline_text="Why Did the Pumpkin",
                )
            ]
        )
        comfyui = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=s3,
            bucket="test-bucket",
        )

        assert result["images_generated"] == 1
        comfyui.generate_image.assert_not_called()
        s3.put_object.assert_called()

    def test_a_full_bleed_brief_without_a_scene_is_left_pending(self) -> None:
        db = _mock_db_client(briefs=[_brief(scene_description=None)])
        comfyui = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=s3,
            bucket="test-bucket",
        )

        assert result["images_generated"] == 0
        assert result["errors"] == 1
        comfyui.generate_image.assert_not_called()
        s3.put_object.assert_not_called()


def _artwork_validator(*verdicts: str) -> MagicMock:
    """A validator that reports the given artwork verdicts in order."""
    validator = MagicMock()
    validator.validate.side_effect = [
        ArtworkValidationResult(is_product=verdict == "product", verdict=verdict)
        for verdict in verdicts
    ]
    return validator


def _artwork_text_validator(*verdicts: bool) -> MagicMock:
    """A validator that reports lettering in the artwork, in order."""
    validator = MagicMock()
    validator.validate.side_effect = [
        ArtworkTextResult(has_text=verdict, verdict="text" if verdict else "nolettering")
        for verdict in verdicts
    ]
    return validator


def _no_text_validator(*results: NoTextResult) -> MagicMock:
    """A validator that reports the inverted no-text verdicts, in order."""
    validator = MagicMock()
    validator.validate_no_text.side_effect = list(results)
    return validator


class TestArtworkLetteringGate:
    """The artwork must not carry words: the image model cannot spell.

    Brief #100 drew a speech bubble that read "To get to the onthe sid"
    (issue #208).
    """

    def _caption_brief(self, **overrides: Any) -> dict[str, Any]:
        return _brief(
            layout_type="text_top",
            headline_text="SKELETON YOGA",
            **overrides,
        )

    def test_artwork_with_lettering_is_retried(self) -> None:
        db = _mock_db_client(briefs=[self._caption_brief()])
        comfyui = _comfyui_client()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_text_validator=_artwork_text_validator(True, False),
        )

        assert result["images_generated"] == 1
        assert comfyui.generate_image.call_count == 2

    def test_the_retry_asks_for_no_lettering(self) -> None:
        db = _mock_db_client(briefs=[self._caption_brief()])
        comfyui = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_text_validator=_artwork_text_validator(True, False),
        )

        second_prompt = comfyui.generate_image.call_args_list[1].kwargs["prompt"]
        assert "no lettering" in second_prompt.lower()

    def test_artwork_with_lettering_falls_back_to_the_typography_only_design(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            artwork_text_validator=_artwork_text_validator(True, True, True),
        )

        assert result["images_generated"] == 1
        s3.put_object.assert_called()

    def test_the_gate_is_not_run_for_a_typography_only_design(self) -> None:
        db = _mock_db_client(
            briefs=[_brief(layout_type="text_only", headline_text="SPOOKY SEASON")]
        )
        validator = _artwork_text_validator()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_text_validator=validator,
        )

        validator.validate.assert_not_called()

    def test_the_gate_reads_the_artwork_before_the_text_is_drawn_on_it(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        validator = _artwork_text_validator(False)

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_text_validator=validator,
        )

        checked = validator.validate.call_args.args[0]
        assert checked == b"raw-bytes"


class TestNoTextDesigns:
    """A no-text design is the illustration and nothing else.

    The text compositor never runs, the text OCR gate is never asked, and the
    inverted lettering gate is the only thing standing between the render and
    the shop. It fails closed: an unreadable answer is not proof that the
    design is free of words.
    """

    def _no_text_brief(self, **overrides: Any) -> dict[str, Any]:
        row = _brief(
            copy_mode="no_text",
            layout_type="full_bleed",
            headline_text=None,
            tagline_text=None,
            scene_description="a ghost hiding inside a pumpkin",
        )
        row.update(overrides)
        return row

    def _passed(self) -> NoTextResult:
        return NoTextResult(matched=True, verdict="nolettering")

    def _lettered(self) -> NoTextResult:
        return NoTextResult(matched=False, verdict="text")

    def _unreadable(self) -> NoTextResult:
        return NoTextResult(matched=False, verdict="unknown", error="vision is down")

    def test_a_no_text_design_never_reaches_the_text_compositor_or_ocr(self) -> None:
        db = _mock_db_client(briefs=[self._no_text_brief()])
        comfy = _comfyui_client()
        text_validator = _validator()
        lettering = _no_text_validator(self._passed())
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            bucket="test-bucket",
            text_validator=text_validator,
            artwork_text_validator=lettering,
        )

        assert result["images_generated"] == 1
        text_validator.validate.assert_not_called()
        lettering.validate.assert_not_called()
        lettering.validate_no_text.assert_called_once()
        assert s3.put_object.call_args.kwargs["Body"] == b"raw-bytes"

    def test_the_prompt_asks_for_an_illustration_with_no_words(self) -> None:
        db = _mock_db_client(briefs=[self._no_text_brief()])
        comfy = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_text_validator=_no_text_validator(self._passed()),
        )

        prompt = comfy.generate_image.call_args.kwargs["prompt"].lower()
        assert "no text" in prompt
        assert "no lettering" in prompt
        assert "illustration alone carries the idea" in prompt

    def test_lettering_is_retried_and_then_the_design_is_rejected(self) -> None:
        db = _mock_db_client(briefs=[self._no_text_brief()])
        comfy = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            bucket="test-bucket",
            artwork_text_validator=_no_text_validator(self._lettered(), self._lettered()),
        )

        assert result["images_generated"] == 0
        assert result["errors"] == 1
        assert comfy.generate_image.call_count == 2
        s3.put_object.assert_not_called()
        db.update_brief_image_keys.assert_not_called()

    def test_the_retry_asks_for_no_words_at_all(self) -> None:
        db = _mock_db_client(briefs=[self._no_text_brief()])
        comfy = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_text_validator=_no_text_validator(self._lettered(), self._lettered()),
        )

        second_prompt = comfy.generate_image.call_args_list[1].kwargs["prompt"].lower()
        assert "no words" in second_prompt

    def test_an_unreadable_no_text_verdict_fails_closed(self) -> None:
        db = _mock_db_client(briefs=[self._no_text_brief()])
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            artwork_text_validator=_no_text_validator(self._unreadable(), self._unreadable()),
        )

        assert result["images_generated"] == 0
        s3.put_object.assert_not_called()

    def test_a_no_text_design_never_falls_back_to_typography(self) -> None:
        # There is no copy to fall back to, and a lettered illustration is not
        # the design that was asked for.
        db = _mock_db_client(briefs=[self._no_text_brief()])
        comfy = _comfyui_client()
        text_validator = _validator()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            bucket="test-bucket",
            text_validator=text_validator,
            artwork_text_validator=_no_text_validator(self._lettered(), self._lettered()),
        )

        assert result["errors"] == 1
        comfy.generate_image.assert_called()
        text_validator.validate.assert_not_called()
        s3.put_object.assert_not_called()

    def test_a_dtf_no_text_design_uploads_the_keyed_artwork(self) -> None:
        db = _mock_db_client(briefs=[self._no_text_brief(product_type="dtf_apparel")])
        comfy = _comfyui_client(raw=b"raw", transparent=b"transparent")
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            bucket="test-bucket",
            artwork_text_validator=_no_text_validator(self._passed()),
        )

        assert result["images_generated"] == 1
        bodies = [upload.kwargs["Body"] for upload in s3.put_object.call_args_list]
        assert bodies == [b"raw", b"transparent"]

    def test_a_no_text_design_without_a_validator_still_renders(self) -> None:
        db = _mock_db_client(briefs=[self._no_text_brief()])
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
        )

        assert result["images_generated"] == 1
        s3.put_object.assert_called_once()

    def test_a_stale_headline_on_a_no_text_row_is_ignored(self) -> None:
        db = _mock_db_client(
            briefs=[
                self._no_text_brief(
                    layout_type="text_top",
                    headline_text="BOO JEE",
                    tagline_text="A GHOST",
                )
            ]
        )
        text_validator = _validator()
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            text_validator=text_validator,
            artwork_text_validator=_no_text_validator(self._passed()),
        )

        text_validator.validate.assert_not_called()
        assert s3.put_object.call_args.kwargs["Body"] == b"raw-bytes"


class TestBrief100Regression:
    """Brief #100 is the image with the bubble that read "To get to the onthe sid"."""

    SCENE = (
        "A grinning pumpkin with a speech bubble saying 'To get to the other side!' "
        "with a small ghost peeking out"
    )

    def _brief_100(self) -> dict[str, Any]:
        return _brief(
            id=100,
            concept="Why Did the Pumpkin Cross the Road",
            layout_type="text_top_bottom",
            headline_text="Why Did the Pumpkin",
            tagline_text="Cross the Road?",
            scene_description=self.SCENE,
        )

    def test_the_model_is_never_asked_to_spell_the_punchline(self) -> None:
        db = _mock_db_client(briefs=[self._brief_100()])
        comfyui = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_text_validator=_artwork_text_validator(False),
        )

        prompt = comfyui.generate_image.call_args.kwargs["prompt"]
        assert "other side" not in prompt
        assert "speech bubble" not in prompt
        assert "pumpkin" in prompt

    def test_the_artwork_with_the_bubble_is_not_published(self) -> None:
        db = _mock_db_client(briefs=[self._brief_100()])
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            artwork_text_validator=_artwork_text_validator(True, True, True),
        )

        # The artwork with the misspelled bubble is never uploaded. The design
        # falls back to the typography-only version, whose text is exact.
        assert result["images_generated"] == 1
        s3.put_object.assert_called()


class TestArtworkGate:
    def test_a_product_artwork_is_retried(self) -> None:
        db = _mock_db_client(briefs=[_brief()])
        comfyui = _comfyui_client()
        validator = _artwork_validator("product", "artwork")

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_validator=validator,
        )

        assert result["images_generated"] == 1
        assert comfyui.generate_image.call_count == 2

    def test_the_retry_asks_for_the_flat_graphic_only(self) -> None:
        db = _mock_db_client(briefs=[_brief()])
        comfyui = _comfyui_client()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_validator=_artwork_validator("product", "artwork"),
        )

        second_prompt = comfyui.generate_image.call_args_list[1].kwargs["prompt"]
        assert "no bag" in second_prompt.lower()

    def test_a_product_artwork_is_never_uploaded(self) -> None:
        db = _mock_db_client(briefs=[_brief()])
        comfyui = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=s3,
            bucket="test-bucket",
            artwork_validator=_artwork_validator("product", "product", "product"),
        )

        assert result["images_generated"] == 0
        assert result["errors"] == 1
        s3.put_object.assert_not_called()

    def test_a_product_artwork_falls_back_to_the_typography_only_design(self) -> None:
        db = _mock_db_client(briefs=[_brief(layout_type="text_top", headline_text="SPOOKY SEASON")])
        comfyui = _comfyui_client()
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfyui,
            s3_client=s3,
            bucket="test-bucket",
            artwork_validator=_artwork_validator("product", "product", "product"),
        )

        assert result["images_generated"] == 1
        s3.put_object.assert_called()

    def test_an_unknown_verdict_keeps_the_image(self) -> None:
        db = _mock_db_client(briefs=[_brief()])
        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=s3,
            bucket="test-bucket",
            artwork_validator=_artwork_validator("unknown"),
        )

        assert result["images_generated"] == 1
        s3.put_object.assert_called()

    def test_the_artwork_gate_is_not_run_for_a_typography_only_design(self) -> None:
        db = _mock_db_client(
            briefs=[_brief(layout_type="text_only", headline_text="SPOOKY SEASON")]
        )
        validator = _artwork_validator()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(),
            s3_client=MagicMock(),
            bucket="test-bucket",
            artwork_validator=validator,
        )

        validator.validate.assert_not_called()


class TestTheTypeOfTheDesign:
    def _rendered(self, **overrides: Any) -> bytes:
        db = _mock_db_client(briefs=[_brief(**overrides)])
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(raw=_real_png()),
            s3_client=s3,
            bucket="test-bucket",
        )

        return s3.put_object.call_args.kwargs["Body"]

    def test_the_font_family_of_the_brief_is_used(self) -> None:
        # The family of the brief decides the type, so the whole shop does not
        # look like one shop (issue #226).
        one = self._rendered(
            layout_type="text_top",
            headline_text="GOBBLE TIL THE BELL",
            font_family="rye",
        )
        other = self._rendered(
            layout_type="text_top",
            headline_text="GOBBLE TIL THE BELL",
            font_family="bebas",
        )

        assert one != other

    def test_a_brief_without_a_family_still_gets_a_headline(self) -> None:
        # The briefs that were written before the column existed have no
        # family, and they still have to be rendered.
        body = self._rendered(
            layout_type="text_top",
            headline_text="GOBBLE TIL THE BELL",
            font_family=None,
        )

        image = Image.open(BytesIO(body)).convert("RGB")
        width, height = image.size
        top_zone = image.crop((0, 0, width, int(height * 0.15)))
        assert _distinct_colours(top_zone) != {(255, 255, 255)}


class TestTheTextureOfTheDesign:
    def _render(self, style_preset: str) -> Image.Image:
        db = _mock_db_client(
            briefs=[
                _brief(
                    layout_type="text_top",
                    headline_text="GOBBLE TIL THE BELL",
                    style_preset=style_preset,
                )
            ]
        )
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(raw=_real_png()),
            s3_client=s3,
            bucket="test-bucket",
        )

        body = s3.put_object.call_args.kwargs["Body"]
        return Image.open(BytesIO(body)).convert("RGB")

    def test_a_look_with_a_texture_gets_one(self) -> None:
        # The grain, the halftone and the cracked ink are what make a flat
        # vector read as a print, and they cannot be asked of the model: a
        # texture it paints lands on the background too (issue #227).
        textured = self._render("vintage_screenprint")
        plain = self._render("kawaii_flat")

        textured_colours = textured.getcolors(maxcolors=1_000_000) or []
        plain_colours = plain.getcolors(maxcolors=1_000_000) or []
        assert len(textured_colours) > len(plain_colours) * 2

    def test_a_look_without_a_texture_is_left_alone(self) -> None:
        # The look without a texture keeps the flat colour of the render.
        plain = self._render("kawaii_flat")

        colours = plain.getcolors(maxcolors=1_000_000) or []
        white = sum(count for count, colour in colours if colour == (255, 255, 255))
        assert white > 0


class TestTheLookOfARender:
    def test_the_type_follows_the_look(self) -> None:
        # The operator can render the same design in another look, and the
        # type follows the look (issue #228).
        db = _mock_db_client(
            briefs=[
                _brief(
                    layout_type="text_top",
                    headline_text="GOBBLE TIL THE BELL",
                    style_preset="halftone_comic",
                )
            ]
        )
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(raw=_real_png()),
            s3_client=s3,
            bucket="test-bucket",
        )

        body = s3.put_object.call_args.kwargs["Body"]
        comic = Image.open(BytesIO(body)).convert("RGB")

        db = _mock_db_client(
            briefs=[
                _brief(
                    layout_type="text_top",
                    headline_text="GOBBLE TIL THE BELL",
                    style_preset="art_nouveau",
                )
            ]
        )
        s3 = MagicMock()
        generate_images_for_briefs(
            db_client=db,
            comfyui_client=_comfyui_client(raw=_real_png()),
            s3_client=s3,
            bucket="test-bucket",
        )
        body = s3.put_object.call_args.kwargs["Body"]
        nouveau = Image.open(BytesIO(body)).convert("RGB")

        assert comic.tobytes() != nouveau.tobytes()


class TestTypographicHeroPath:
    """The Qwen wordmark path is opt-in and only draws the copy it can spell."""

    def _hero_brief(self, **overrides: Any) -> dict[str, Any]:
        row = _brief(
            id=254,
            product_type="sublimation",
            layout_type="type_hero",
            headline_text="GROOVY",
            tagline_text="GROOVY GHOST VIBES",
            scene_description="a cute groovy ghost",
        )
        row.update(overrides)
        return row

    def _hero_validator(self, *, matched: bool = True) -> MagicMock:
        validator = MagicMock()
        validator.validate.return_value = StrictTextResult(
            matched=matched,
            expected=["GROOVY"],
            observed="GROOVY" if matched else "CROOVY",
            score=1.0 if matched else 0.0,
        )
        return validator

    def test_a_hero_brief_is_drawn_by_qwen_and_uploaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        db = _mock_db_client(briefs=[self._hero_brief()])
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero", seed=1)

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            hero_validator=self._hero_validator(),
        )

        assert result == {"status": "success", "images_generated": 1, "errors": 0}
        comfy.generate_qwen_image.assert_called_once()
        comfy.generate_image.assert_not_called()
        prompt = comfy.generate_qwen_image.call_args.kwargs["prompt"]
        assert "GROOVY" in prompt
        assert "GROOVY GHOST VIBES" not in prompt
        db.update_brief_image_keys.assert_called_once()
        keys = db.update_brief_image_keys.call_args.kwargs
        assert keys["image_key_raw"].endswith("_raw.png")
        assert keys["image_key_transparent"] is None

    def test_the_dtf_hero_is_keyed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        brief = self._hero_brief(product_type="dtf_apparel")
        db = _mock_db_client(briefs=[brief])
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero")
        comfy.remove_background.return_value = ComfyUIResult(image_bytes=b"keyed")

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            hero_validator=self._hero_validator(),
        )

        comfy.remove_background.assert_called_once_with(b"hero")
        keys = db.update_brief_image_keys.call_args.kwargs
        assert keys["image_key_transparent"].endswith("_transparent.png")

    def test_the_path_is_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPOGRAPHIC_HERO_ENABLED", raising=False)
        db = _mock_db_client(briefs=[self._hero_brief()])
        comfy = _comfyui_client()
        comfy.generate_image.return_value = ComfyUIResult(image_bytes=_real_png())

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            text_validator=_validator(True),
            hero_validator=self._hero_validator(),
        )

        assert result["images_generated"] == 1
        comfy.generate_image.assert_called_once()
        comfy.generate_qwen_image.assert_not_called()

    def test_a_copy_mode_hero_uses_qwen_even_with_a_normal_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        db = _mock_db_client(
            briefs=[
                self._hero_brief(
                    layout_type="text_top",
                    copy_mode="type_hero",
                    headline_text="BOO JEE",
                    kicker_text="IN MY SPOOKY ERA",
                )
            ]
        )
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero")

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            hero_validator=self._hero_validator(),
        )

        assert result["images_generated"] == 1
        comfy.generate_qwen_image.assert_called_once()
        comfy.generate_image.assert_not_called()
        prompt = comfy.generate_qwen_image.call_args.kwargs["prompt"]
        assert "BOO JEE" in prompt
        assert "IN MY SPOOKY ERA" in prompt

    def test_a_copy_mode_hero_falls_back_to_a_caption_when_the_path_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TYPOGRAPHIC_HERO_ENABLED", raising=False)
        db = _mock_db_client(
            briefs=[
                self._hero_brief(
                    layout_type="text_top",
                    copy_mode="type_hero",
                    headline_text="BOO JEE",
                )
            ]
        )
        comfy = _comfyui_client()
        comfy.generate_image.return_value = ComfyUIResult(image_bytes=_real_png())

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            text_validator=_validator(True),
            hero_validator=self._hero_validator(),
        )

        assert result["images_generated"] == 1
        comfy.generate_image.assert_called_once()
        comfy.generate_qwen_image.assert_not_called()

    def test_a_hero_that_fails_the_gate_is_not_uploaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        monkeypatch.setenv("TYPOGRAPHIC_HERO_CANDIDATES", "2")
        db = _mock_db_client(briefs=[self._hero_brief()])
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero")

        result = generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=MagicMock(),
            hero_validator=self._hero_validator(matched=False),
        )

        assert result["errors"] == 1
        assert comfy.generate_qwen_image.call_count == 2
        comfy.generate_image.assert_not_called()
        db.update_brief_image_keys.assert_not_called()

    def test_the_descriptor_is_composited_into_the_raw_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        monkeypatch.delenv("TYPOGRAPHIC_HERO_DESCRIPTOR", raising=False)
        composed = MagicMock(return_value=b"composed")
        monkeypatch.setattr(
            "store_listing.orchestration.image_pipeline.compose_hero_descriptor", composed
        )
        db = _mock_db_client(briefs=[self._hero_brief()])
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero")
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            hero_validator=self._hero_validator(),
        )

        composed.assert_called_once_with(b"hero", "GROOVY GHOST VIBES", transparent=False)
        bodies = [upload.kwargs["Body"] for upload in s3.put_object.call_args_list]
        assert b"composed" in bodies

    def test_the_descriptor_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        monkeypatch.setenv("TYPOGRAPHIC_HERO_DESCRIPTOR", "0")
        composed = MagicMock(return_value=b"composed")
        monkeypatch.setattr(
            "store_listing.orchestration.image_pipeline.compose_hero_descriptor", composed
        )
        db = _mock_db_client(briefs=[self._hero_brief()])
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero")
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            hero_validator=self._hero_validator(),
        )

        composed.assert_not_called()
        bodies = [upload.kwargs["Body"] for upload in s3.put_object.call_args_list]
        assert bodies == [b"hero"]

    def test_the_dtf_descriptor_is_composited_on_the_keyed_artwork(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        monkeypatch.delenv("TYPOGRAPHIC_HERO_DESCRIPTOR", raising=False)
        composed = MagicMock(side_effect=[b"composed-raw", b"composed-keyed"])
        monkeypatch.setattr(
            "store_listing.orchestration.image_pipeline.compose_hero_descriptor", composed
        )
        db = _mock_db_client(briefs=[self._hero_brief(product_type="dtf_apparel")])
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero")
        comfy.remove_background.return_value = ComfyUIResult(image_bytes=b"keyed")
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            hero_validator=self._hero_validator(),
        )

        assert composed.call_args_list == [
            call(b"hero", "GROOVY GHOST VIBES", transparent=False),
            call(b"keyed", "GROOVY GHOST VIBES", transparent=True),
        ]
        bodies = [upload.kwargs["Body"] for upload in s3.put_object.call_args_list]
        assert b"composed-raw" in bodies
        assert b"composed-keyed" in bodies

    def test_a_hero_without_a_descriptor_skips_the_compositor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TYPOGRAPHIC_HERO_ENABLED", "1")
        composed = MagicMock(return_value=b"composed")
        monkeypatch.setattr(
            "store_listing.orchestration.image_pipeline.compose_hero_descriptor", composed
        )
        db = _mock_db_client(briefs=[self._hero_brief(tagline_text=None)])
        comfy = _comfyui_client()
        comfy.generate_qwen_image.return_value = ComfyUIResult(image_bytes=b"hero")
        s3 = MagicMock()

        generate_images_for_briefs(
            db_client=db,
            comfyui_client=comfy,
            s3_client=s3,
            hero_validator=self._hero_validator(),
        )

        composed.assert_not_called()
        bodies = [upload.kwargs["Body"] for upload in s3.put_object.call_args_list]
        assert bodies == [b"hero"]
