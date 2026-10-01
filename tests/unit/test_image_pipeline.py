"""Tests for the design image generation pipeline."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

from PIL import Image

from store_listing.orchestration.image_pipeline import (
    generate_image_for_brief,
    generate_images_for_briefs,
)


def _real_png(width: int = 64, height: int = 64, color: str = "white") -> bytes:
    img = Image.new("RGBA", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_bedrock_response(image_bytes: bytes) -> dict[str, Any]:
    return {
        "body": MagicMock(
            read=lambda: json.dumps({"images": [base64.b64encode(image_bytes).decode()]}).encode()
        ),
    }


def _mock_db_client(
    briefs: list[dict[str, Any]] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.list_briefs_without_images.return_value = briefs or []
    return client


class TestGenerateImagesForBriefs:
    def test_skips_when_no_briefs_need_images(self) -> None:
        db = _mock_db_client(briefs=[])
        result = generate_images_for_briefs(
            db_client=db, bedrock_client=MagicMock(), s3_client=MagicMock()
        )
        assert result["images_generated"] == 0

    def test_generates_and_stores_dtf_image(self) -> None:
        db = _mock_db_client(
            briefs=[
                {
                    "id": 42,
                    "concept": "Skeleton yoga",
                    "product_type": "dtf_apparel",
                    "audience": "yoga fans",
                    "visual_style": "retro vintage",
                }
            ]
        )

        fake_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        fake_transparent = b"\x89PNG\r\n\x1a\n" + b"\x01" * 50
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = [
            _mock_bedrock_response(fake_raw),
            _mock_bedrock_response(fake_transparent),
        ]

        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, bedrock_client=bedrock, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        assert s3.put_object.call_count == 2
        db.update_brief_image_keys.assert_called_once()
        call_kwargs = db.update_brief_image_keys.call_args.kwargs
        assert call_kwargs["brief_id"] == 42
        assert "raw" in call_kwargs["image_key_raw"]
        assert "transparent" in call_kwargs["image_key_transparent"]

    def test_generates_sublimation_without_transparent(self) -> None:
        db = _mock_db_client(
            briefs=[
                {
                    "id": 10,
                    "concept": "Galaxy cat",
                    "product_type": "sublimation",
                    "audience": "cat lovers",
                    "visual_style": "cosmic watercolor",
                }
            ]
        )

        fake_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _mock_bedrock_response(fake_raw)

        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, bedrock_client=bedrock, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        assert s3.put_object.call_count == 1
        call_kwargs = db.update_brief_image_keys.call_args.kwargs
        assert call_kwargs["image_key_transparent"] is None

    def test_text_layout_passes_raw_image_through(self) -> None:
        db = _mock_db_client(
            briefs=[
                {
                    "id": 50,
                    "concept": "Born To Be Spooky",
                    "product_type": "dtf_apparel",
                    "audience": "October birthday people",
                    "visual_style": "retro halloween cake",
                    "layout_type": "text_top",
                    "headline_text": "Born To Be Spooky",
                    "tagline_text": None,
                    "font_color": "#FF6600",
                }
            ]
        )

        real_raw = _real_png(color="white")
        real_transparent = _real_png(color="red")
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = [
            _mock_bedrock_response(real_raw),
            _mock_bedrock_response(real_transparent),
        ]

        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, bedrock_client=bedrock, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1

    def test_full_bleed_passes_raw_image_through(self) -> None:
        db = _mock_db_client(
            briefs=[
                {
                    "id": 51,
                    "concept": "Abstract Pattern",
                    "product_type": "sublimation",
                    "audience": "art lovers",
                    "visual_style": "geometric",
                    "layout_type": "full_bleed",
                    "headline_text": None,
                    "tagline_text": None,
                    "font_color": None,
                }
            ]
        )

        real_raw = _real_png()
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _mock_bedrock_response(real_raw)

        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, bedrock_client=bedrock, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        raw_call = s3.put_object.call_args_list[0]
        raw_body = raw_call.kwargs.get("Body") or raw_call[1].get("Body")
        assert raw_body == real_raw

    def test_continues_on_generation_failure(self) -> None:
        db = _mock_db_client(
            briefs=[
                {
                    "id": 1,
                    "concept": "Failing design",
                    "product_type": "dtf_apparel",
                    "audience": "test",
                    "visual_style": "test",
                },
                {
                    "id": 2,
                    "concept": "Working design",
                    "product_type": "sublimation",
                    "audience": "test",
                    "visual_style": "test",
                },
            ]
        )

        fake_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = [
            Exception("Bedrock error"),
            _mock_bedrock_response(fake_raw),
        ]

        s3 = MagicMock()

        result = generate_images_for_briefs(
            db_client=db, bedrock_client=bedrock, s3_client=s3, bucket="test-bucket"
        )

        assert result["images_generated"] == 1
        assert result["errors"] == 1


class TestGenerateImageForBrief:
    def test_returns_error_when_brief_not_found(self) -> None:
        db = MagicMock()
        db.get_brief_by_id.return_value = None

        result = generate_image_for_brief(
            brief_id=999,
            db_client=db,
            bedrock_client=MagicMock(),
            s3_client=MagicMock(),
        )

        assert result["status"] == "error"
        assert "999" in result["error"]

    def test_generates_image_for_specific_brief(self) -> None:
        brief_row = {
            "id": 7,
            "concept": "Yoga Cat",
            "product_type": "dtf_apparel",
            "audience": "cat lovers",
            "visual_style": "retro vintage",
            "layout_type": "full_bleed",
            "headline_text": None,
            "tagline_text": None,
            "font_color": None,
            "regeneration_feedback": "make it more cartoonish",
        }
        db = MagicMock()
        db.get_brief_by_id.return_value = brief_row

        fake_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        fake_transparent = b"\x89PNG\r\n\x1a\n" + b"\x01" * 50
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = [
            _mock_bedrock_response(fake_raw),
            _mock_bedrock_response(fake_transparent),
        ]

        s3 = MagicMock()

        result = generate_image_for_brief(
            brief_id=7,
            db_client=db,
            bedrock_client=bedrock,
            s3_client=s3,
            bucket="test-bucket",
        )

        assert result["status"] == "success"
        assert result["brief_id"] == 7
        db.update_brief_image_keys.assert_called_once()

    def test_text_layout_uses_ideogram_model(self) -> None:
        from unittest.mock import AsyncMock

        from store_listing.orchestration.leonardo_client import LeonardoGenerationResult

        brief_row = {
            "id": 8,
            "concept": "This Meeting Could've Been An Email",
            "product_type": "dtf_apparel",
            "audience": "office workers",
            "visual_style": "minimalist line-art",
            "layout_type": "text_top_bottom",
            "headline_text": "STILL IN THIS MEETING",
            "tagline_text": "SEND AN EMAIL",
            "font_color": "#FF8C42",
            "scene_description": "a skeleton at an office desk holding coffee",
        }
        db = MagicMock()
        db.get_brief_by_id.return_value = brief_row

        fake_png = _real_png()
        fake_transparent = _real_png(color="red")

        leo_client = MagicMock()
        leo_client.generate = AsyncMock(
            return_value=LeonardoGenerationResult(
                image_bytes=fake_png, prompt="test", generation_id="gen-1"
            )
        )

        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _mock_bedrock_response(fake_transparent)

        s3 = MagicMock()

        result = generate_image_for_brief(
            brief_id=8,
            db_client=db,
            bedrock_client=bedrock,
            s3_client=s3,
            leonardo_client=leo_client,
        )

        assert result["status"] == "success"
        gen_call = leo_client.generate.call_args
        assert gen_call.kwargs.get("model") == "ideogram-v3.0"
        assert gen_call.kwargs.get("quality") == "QUALITY"

    def test_passes_regeneration_feedback_to_prompt(self) -> None:
        brief_row = {
            "id": 7,
            "concept": "Yoga Cat",
            "product_type": "sublimation",
            "audience": "cat lovers",
            "visual_style": "retro vintage",
            "layout_type": "full_bleed",
            "headline_text": None,
            "tagline_text": None,
            "font_color": None,
            "regeneration_feedback": "more vibrant colors please",
        }
        db = MagicMock()
        db.get_brief_by_id.return_value = brief_row

        fake_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _mock_bedrock_response(fake_raw)

        s3 = MagicMock()

        generate_image_for_brief(
            brief_id=7,
            db_client=db,
            bedrock_client=bedrock,
            s3_client=s3,
            bucket="test-bucket",
        )

        call_kwargs = bedrock.invoke_model.call_args
        body_str = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", ""))
        assert "more vibrant colors please" in body_str
