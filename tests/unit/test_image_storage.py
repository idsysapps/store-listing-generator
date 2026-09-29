"""Tests for S3-compatible image storage."""

from __future__ import annotations

from unittest.mock import MagicMock

from store_listing.orchestration.image_storage import (
    ImageStorageClient,
    build_s3_key,
)


def _mock_s3_client() -> MagicMock:
    return MagicMock()


class TestBuildS3Key:
    def test_includes_brief_id_and_extension(self) -> None:
        key = build_s3_key(brief_id=42, product_type="dtf_apparel")
        assert "42" in key
        assert key.endswith(".png")

    def test_includes_product_type(self) -> None:
        key = build_s3_key(brief_id=1, product_type="dtf_apparel")
        assert "dtf_apparel" in key

    def test_includes_date_prefix(self) -> None:
        key = build_s3_key(brief_id=1, product_type="sublimation")
        parts = key.split("/")
        assert len(parts) >= 3

    def test_different_briefs_get_different_keys(self) -> None:
        key1 = build_s3_key(brief_id=1, product_type="dtf_apparel")
        key2 = build_s3_key(brief_id=2, product_type="dtf_apparel")
        assert key1 != key2


class TestImageStorageClient:
    def test_upload_calls_put_object(self) -> None:
        mock_s3 = _mock_s3_client()
        client = ImageStorageClient(s3_client=mock_s3, bucket="test-bucket")

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        key = client.upload(image_bytes, brief_id=1, product_type="dtf_apparel")

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Body"] == image_bytes
        assert call_kwargs["ContentType"] == "image/png"
        assert key.endswith(".png")

    def test_upload_returns_s3_key(self) -> None:
        mock_s3 = _mock_s3_client()
        client = ImageStorageClient(s3_client=mock_s3, bucket="my-bucket")

        image_bytes = b"\x89PNG\r\n\x1a\n"
        key = client.upload(image_bytes, brief_id=99, product_type="sublimation")

        assert "99" in key
        assert "sublimation" in key

    def test_get_url_with_endpoint(self) -> None:
        mock_s3 = _mock_s3_client()
        client = ImageStorageClient(
            s3_client=mock_s3,
            bucket="designs",
            endpoint_url="https://minio.local:9000",
        )

        url = client.get_url("designs/2026/09/dtf_apparel/42.png")
        assert "minio.local:9000" in url
        assert "designs" in url

    def test_get_url_without_endpoint_uses_aws(self) -> None:
        mock_s3 = _mock_s3_client()
        client = ImageStorageClient(s3_client=mock_s3, bucket="designs")

        url = client.get_url("designs/2026/09/dtf_apparel/42.png")
        assert "s3" in url or "amazonaws" in url

    def test_upload_both_raw_and_transparent(self) -> None:
        mock_s3 = _mock_s3_client()
        client = ImageStorageClient(s3_client=mock_s3, bucket="test-bucket")

        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        transparent_bytes = b"\x89PNG\r\n\x1a\n" + b"\x01" * 50

        keys = client.upload_design(
            raw_bytes=raw_bytes,
            transparent_bytes=transparent_bytes,
            brief_id=7,
            product_type="dtf_apparel",
        )

        assert mock_s3.put_object.call_count == 2
        assert keys["raw"] is not None and "raw" in keys["raw"]
        assert keys["transparent"] is not None and "transparent" in keys["transparent"]

    def test_upload_design_without_transparent(self) -> None:
        mock_s3 = _mock_s3_client()
        client = ImageStorageClient(s3_client=mock_s3, bucket="test-bucket")

        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

        keys = client.upload_design(
            raw_bytes=raw_bytes,
            transparent_bytes=None,
            brief_id=7,
            product_type="sublimation",
        )

        assert mock_s3.put_object.call_count == 1
        assert keys["raw"] is not None and "raw" in keys["raw"]
        assert keys.get("transparent") is None
