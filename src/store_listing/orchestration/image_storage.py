"""S3-compatible image storage for generated designs.

Supports both AWS S3 and S3-compatible endpoints (MinIO, local storage
arrays) via configurable endpoint URL.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

S3_BUCKET: str = os.environ.get("DESIGN_S3_BUCKET", "store-listing-designs")
S3_ENDPOINT_URL: str | None = os.environ.get("S3_ENDPOINT_URL")


def build_s3_key(brief_id: int, product_type: str, *, suffix: str = "") -> str:
    now = datetime.now(tz=UTC)
    date_prefix = now.strftime("%Y/%m")
    filename = f"{brief_id}{suffix}.png"
    return f"designs/{date_prefix}/{product_type}/{filename}"


class ImageStorageClient:
    def __init__(
        self,
        *,
        s3_client: Any,
        bucket: str,
        endpoint_url: str | None = None,
    ) -> None:
        self._s3 = s3_client
        self._bucket = bucket
        self._endpoint_url = endpoint_url

    def upload(
        self, image_bytes: bytes, brief_id: int, product_type: str, *, suffix: str = ""
    ) -> str:
        key = build_s3_key(brief_id, product_type, suffix=suffix)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=image_bytes,
            ContentType="image/png",
        )
        logger.info("Uploaded design image: s3://%s/%s", self._bucket, key)
        return key

    def upload_design(
        self,
        *,
        raw_bytes: bytes,
        transparent_bytes: bytes | None,
        brief_id: int,
        product_type: str,
    ) -> dict[str, str | None]:
        raw_key = self.upload(raw_bytes, brief_id, product_type, suffix="_raw")
        transparent_key = None
        if transparent_bytes is not None:
            transparent_key = self.upload(
                transparent_bytes, brief_id, product_type, suffix="_transparent"
            )
        return {"raw": raw_key, "transparent": transparent_key}

    def get_url(self, key: str) -> str:
        if self._endpoint_url:
            return f"{self._endpoint_url.rstrip('/')}/{self._bucket}/{key}"
        return f"https://{self._bucket}.s3.amazonaws.com/{key}"
