"""Tests for the design image generation pipeline (ComfyUI + OCR gate)."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

from PIL import Image

from store_listing.orchestration.artwork_validation import ArtworkValidationResult
from store_listing.orchestration.comfyui_client import ComfyUIError, ComfyUIResult
from store_listing.orchestration.image_pipeline import (
    generate_image_for_brief,
    generate_images_for_briefs,
)
from store_listing.orchestration.text_validation import TextValidationResult


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
