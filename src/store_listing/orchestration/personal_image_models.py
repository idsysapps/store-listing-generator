"""Image models available for a personal custom brief.

Sellable briefs stay on Flux Schnell. A personal brief may pick Qwen-Image 2.1
(the default) or FLUX.2 dev (issue #339).
"""

from __future__ import annotations

QWEN_IMAGE_21 = "qwen_image_2.1"
FLUX2_DEV = "flux2_dev"
DEFAULT_PERSONAL_IMAGE_MODEL = QWEN_IMAGE_21
PERSONAL_IMAGE_MODELS = frozenset({QWEN_IMAGE_21, FLUX2_DEV})


def normalize_personal_image_model(value: str | None) -> str:
    """Return a known personal image model, or the default."""
    if isinstance(value, str) and value.strip() in PERSONAL_IMAGE_MODELS:
        return value.strip()
    return DEFAULT_PERSONAL_IMAGE_MODEL
