"""Personal custom briefs pick Qwen-Image 2.1 or FLUX.2 dev (issue #339)."""

from store_listing.orchestration.personal_image_models import (
    DEFAULT_PERSONAL_IMAGE_MODEL,
    FLUX2_DEV,
    QWEN_IMAGE_21,
    normalize_personal_image_model,
)


def test_default_is_qwen_image_21() -> None:
    assert DEFAULT_PERSONAL_IMAGE_MODEL == QWEN_IMAGE_21
    assert normalize_personal_image_model(None) == QWEN_IMAGE_21
    assert normalize_personal_image_model("") == QWEN_IMAGE_21
    assert normalize_personal_image_model("unknown") == QWEN_IMAGE_21


def test_known_models_are_kept() -> None:
    assert normalize_personal_image_model(QWEN_IMAGE_21) == QWEN_IMAGE_21
    assert normalize_personal_image_model(FLUX2_DEV) == FLUX2_DEV
