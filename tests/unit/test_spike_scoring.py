"""Tests for the pipeline bake-off scoring maths.

The scoring is the part of the spike that has to be trustworthy: a wrong
contrast or residue number would make the architecture decision for us.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from spike.bakeoff.score_spike import (
    contrast_ratio,
    garment_hex,
    print_metrics,
    text_overlap,
)


def _png(values: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(values.astype(np.uint8), "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _path(tmp_path: Path, name: str, values: np.ndarray) -> Path:
    path = tmp_path / name
    path.write_bytes(_png(values))
    return path


def test_contrast_ratio_of_black_and_white_is_the_maximum() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == 21.0


def test_garment_hex_uses_the_catalogue_name_and_falls_back_to_white() -> None:
    assert garment_hex("Jet Black") == "#0B0B0B"
    assert garment_hex("  dark heather ") == "#3A3A3A"
    assert garment_hex("Mystery Fuchsia") == "#FFFFFF"
    assert garment_hex(None) == "#FFFFFF"


def test_print_metrics_counts_coverage_and_ignores_transparency(tmp_path: Path) -> None:
    values = np.zeros((10, 10, 4), dtype=np.uint8)
    values[0:5, 0:5] = (200, 30, 30, 255)

    metrics = print_metrics(_path(tmp_path, "solid.png", values))

    assert metrics["coverage"] == 0.25
    assert metrics["magenta_pct"] == 0.0


def test_print_metrics_flags_magenta_key_residue(tmp_path: Path) -> None:
    values = np.zeros((10, 10, 4), dtype=np.uint8)
    values[:, :] = (232, 4, 74, 255)

    metrics = print_metrics(_path(tmp_path, "magenta.png", values))

    assert metrics["magenta_pct"] == 1.0


def test_print_metrics_flags_a_solid_light_sticker_fill(tmp_path: Path) -> None:
    values = np.zeros((20, 20, 4), dtype=np.uint8)
    values[2:18, 2:18] = (242, 232, 213, 255)

    metrics = print_metrics(_path(tmp_path, "cream_fill.png", values))

    assert metrics["solid_light_pct"] > 0.9


def test_print_metrics_flags_a_near_white_border(tmp_path: Path) -> None:
    values = np.zeros((20, 20, 4), dtype=np.uint8)
    values[2:18, 2:18] = (30, 30, 30, 255)
    values[2:4, 2:18] = (255, 255, 255, 255)
    values[16:18, 2:18] = (255, 255, 255, 255)
    values[2:18, 2:4] = (255, 255, 255, 255)
    values[2:18, 16:18] = (255, 255, 255, 255)

    metrics = print_metrics(_path(tmp_path, "border.png", values))

    assert metrics["near_white_border"] > 0.5


def test_text_overlap_is_the_share_of_the_text_on_the_subject(tmp_path: Path) -> None:
    text = np.zeros((10, 10, 4), dtype=np.uint8)
    text[0:5, 0:5] = (255, 255, 255, 255)
    subject = np.zeros((10, 10, 4), dtype=np.uint8)
    subject[2:7, 2:7] = (200, 50, 50, 255)

    overlap = text_overlap(
        _path(tmp_path, "text.png", text),
        _path(tmp_path, "subject.png", subject),
    )

    assert overlap == 0.36
