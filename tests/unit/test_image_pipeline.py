"""Tests for the design image generation pipeline."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

from PIL import Image

from store_listing.orchestration.image_pipeline import generate_images_for_briefs


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

    def test_composites_text_onto_image(self) -> None:
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
        raw_call = s3.put_object.call_args_list[0]
        raw_body = raw_call.kwargs.get("Body") or raw_call[1].get("Body")
        assert raw_body != real_raw

    def test_skips_compositing_for_full_bleed(self) -> None:
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
