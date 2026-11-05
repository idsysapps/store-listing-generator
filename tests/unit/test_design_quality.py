"""The deterministic QA gate for a rendered DTF design.

These are the checks the current pipeline did not run: contrast on the shirt,
text over the subject, chroma residue and a solid sticker fill.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from store_listing.orchestration.design_quality import (
    Candidate,
    QualityMetrics,
    choose_best,
    contrast_ratio,
    garment_hex,
    image_metrics,
    text_overlap,
)


def _png(values: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(values.astype(np.uint8), "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def test_contrast_ratio_black_on_white_is_the_maximum() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == 21.0


def test_garment_hex_uses_the_catalogue_name() -> None:
    assert garment_hex("Jet Black") == "#0B0B0B"
    assert garment_hex("Dark Heather") == "#3A3A3A"
    assert garment_hex("Mystery Fuchsia") == "#FFFFFF"


def test_image_metrics_measure_coverage_and_key_residue() -> None:
    values = np.zeros((10, 10, 4), dtype=np.uint8)
    values[0:5, 0:5] = (200, 30, 30, 255)

    metrics = image_metrics(_png(values))

    assert metrics.coverage == 0.25
    assert metrics.magenta_fraction == 0.0


def test_image_metrics_flag_magenta_residue_and_a_sticker_fill() -> None:
    values = np.zeros((10, 10, 4), dtype=np.uint8)
    values[:, :] = (232, 4, 74, 255)

    residue = image_metrics(_png(values))
    assert residue.magenta_fraction == 1.0

    fill = np.zeros((20, 20, 4), dtype=np.uint8)
    fill[2:18, 2:18] = (253, 243, 208, 255)
    sticker = image_metrics(_png(fill))
    assert sticker.light_fill > 0.9


def test_text_overlap_is_the_share_of_text_on_the_subject() -> None:
    text = np.zeros((10, 10, 4), dtype=np.uint8)
    text[0:5, 0:5] = (255, 255, 255, 255)
    subject = np.zeros((10, 10, 4), dtype=np.uint8)
    subject[2:7, 2:7] = (200, 50, 50, 255)

    assert text_overlap(_png(text), _png(subject)) == 0.36


def test_choose_best_prefers_less_text_on_the_subject() -> None:
    clean = Candidate(
        name="clean",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(
            contrast=10.0,
            text_overlap=0.0,
            magenta_fraction=0.0,
            coverage=0.2,
            light_fill=0.1,
            near_white_border=0.0,
        ),
    )
    overlapping = Candidate(
        name="overlapping",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(
            contrast=10.0,
            text_overlap=0.8,
            magenta_fraction=0.0,
            coverage=0.2,
            light_fill=0.1,
            near_white_border=0.0,
        ),
    )

    assert choose_best([overlapping, clean]).name == "clean"


def test_choose_best_prefers_a_clean_key_over_residue() -> None:
    clean = Candidate(
        name="clean",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(10.0, 0.0, 0.0, 0.2, 0.1, 0.0),
    )
    residue = Candidate(
        name="residue",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(10.0, 0.0, 0.2, 0.2, 0.1, 0.0),
    )

    assert choose_best([residue, clean]).name == "clean"


def test_choose_best_does_not_pick_an_empty_render() -> None:
    empty = Candidate(
        name="empty",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    rendered = Candidate(
        name="rendered",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(10.0, 0.0, 0.0, 0.22, 0.1, 0.0),
    )

    assert choose_best([empty, rendered]).name == "rendered"
    assert "the render is empty" in empty.metrics.issues


def test_choose_best_prefers_a_clean_cutout_over_a_solid_light_fill() -> None:
    clean = Candidate(
        name="clean",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(10.0, 0.0, 0.0, 0.22, 0.1, 0.0),
    )
    sticker = Candidate(
        name="sticker",
        raw_bytes=b"",
        transparent_bytes=b"",
        metrics=QualityMetrics(10.0, 0.0, 0.0, 0.22, 0.8, 0.0),
    )

    assert choose_best([sticker, clean]).name == "clean"
