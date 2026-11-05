"""Deterministic quality checks for a rendered DTF candidate.

The image pipeline used to upload the first render that passed the text gate.
That gate read the text on the raw (magenta) render, so it could not see that
the ink disappears on the shirt, that the type sits on the subject, or that
the key left a magenta fringe. These checks are the missing post-render gate.

They are deliberately simple and deterministic: no model call, no prompt. A
candidate is ranked by the things a printer would reject.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

__all__ = [
    "Candidate",
    "QualityMetrics",
    "choose_best",
    "contrast_ratio",
    "garment_hex",
    "image_metrics",
    "text_overlap",
]

# The catalogue colours the brief can recommend. The first recommendation is
# the garment the design is printed on (see docs/dtf-design-guidelines.md).
GARMENT_HEX: dict[str, str] = {
    "black": "#111111",
    "jet black": "#0B0B0B",
    "dark heather": "#3A3A3A",
    "white": "#FFFFFF",
    "event white": "#FAFAFA",
    "carolina blue mist": "#A8C6E0",
    "cherry blossom": "#F2C6D4",
    "true celadon": "#8FBFA8",
    "charcoal": "#36454F",
    "navy": "#1F2A44",
}

# A WCAG contrast below this is not legible on a shirt at thumbnail size.
MIN_TEXT_CONTRAST = 3.0
# More than a fifth of the type on the subject reads as a caption laid over
# the art rather than a designed lockup.
MAX_TEXT_OVERLAP = 0.20
# A keyed DTF print should not keep visible magenta.
MAX_MAGENTA_FRACTION = 0.005
# A design that covers most of its placement prints like a solid sheet.
MAX_COVERAGE = 0.65
# An empty key is a failed render, not a clean one.
MIN_COVERAGE = 0.03
# A large light plate inside the artwork bbox is the sticker fill the DTF
# guidelines reject.
MAX_LIGHT_FILL = 0.55
# A near-white band on opposite sides is a die-cut border.
MAX_NEAR_WHITE_BORDER = 0.35


def _rgb(value: str) -> tuple[int, int, int]:
    digits = value.lstrip("#")
    return tuple(int(digits[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _luminance(rgb: tuple[int, int, int]) -> float:
    channels = [value / 255 for value in rgb]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    """The WCAG contrast between two hex colours."""
    lighter = max(_luminance(_rgb(first)), _luminance(_rgb(second)))
    darker = min(_luminance(_rgb(first)), _luminance(_rgb(second)))
    return (lighter + 0.05) / (darker + 0.05)


def garment_hex(name: str | None) -> str:
    """The preview colour for a catalogue garment name, or white when unknown."""
    if not name:
        return "#FFFFFF"
    key = " ".join(name.lower().split())
    return GARMENT_HEX.get(key, "#FFFFFF")


def _load_rgba(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as opened:
        return np.asarray(opened.convert("RGBA")).astype(np.int16)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _band(shape: tuple[int, ...], side: str, width: int) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=bool)
    if side == "top":
        mask[:width, :] = True
    elif side == "bottom":
        mask[-width:, :] = True
    elif side == "left":
        mask[:, :width] = True
    else:
        mask[:, -width:] = True
    return mask


def _near_white_border(values: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """The strongest white band on opposite sides of the artwork bbox.

    A die-cut border runs around the artwork: top and bottom, or left and
    right, are both white. A single white word on one edge is not a border.
    """
    left, top, right, bottom = bbox
    crop = values[top:bottom, left:right]
    rgb = crop[..., :3]
    alpha = crop[..., 3]
    near_white = (
        (rgb.min(axis=2) > 220) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 30) & (alpha > 200)
    )
    opaque = alpha > 200
    band = max(2, int(min(crop.shape[:2]) * 0.03))

    def side(mask: np.ndarray) -> float:
        total = int((mask & opaque).sum())
        if total == 0:
            return 0.0
        return float((mask & near_white).sum() / total)

    top_score = side(_band(crop.shape, "top", band))
    bottom_score = side(_band(crop.shape, "bottom", band))
    left_score = side(_band(crop.shape, "left", band))
    right_score = side(_band(crop.shape, "right", band))
    return max(min(top_score, bottom_score), min(left_score, right_score))


@dataclass(frozen=True)
class QualityMetrics:
    """What the gate measured on one candidate."""

    contrast: float | None
    text_overlap: float
    magenta_fraction: float
    coverage: float
    light_fill: float
    near_white_border: float

    @property
    def rank(self) -> tuple[float, float, float, float, float, float]:
        """Lower is better: placement first, then print defects."""
        coverage_penalty = abs(self.coverage - 0.22) + max(0.0, self.coverage - 0.55)
        return (
            round(self.text_overlap, 2),
            self.magenta_fraction,
            coverage_penalty,
            self.light_fill,
            self.near_white_border,
            -(self.contrast or 0.0),
        )

    @property
    def issues(self) -> list[str]:
        issues: list[str] = []
        if self.contrast is not None and self.contrast < MIN_TEXT_CONTRAST:
            issues.append(f"text contrast {self.contrast:.2f} is below {MIN_TEXT_CONTRAST:.1f}")
        if self.text_overlap > MAX_TEXT_OVERLAP:
            issues.append(f"{self.text_overlap:.0%} of the text is on the subject")
        if self.magenta_fraction > MAX_MAGENTA_FRACTION:
            issues.append(f"{self.magenta_fraction:.1%} of the ink is magenta")
        if self.coverage < MIN_COVERAGE:
            issues.append("the render is empty")
        if self.coverage > MAX_COVERAGE:
            issues.append(f"{self.coverage:.0%} of the canvas is opaque")
        if self.light_fill > MAX_LIGHT_FILL:
            issues.append(f"{self.light_fill:.0%} of the artwork is a light fill")
        if self.near_white_border > MAX_NEAR_WHITE_BORDER:
            issues.append(f"{self.near_white_border:.0%} of the edge is a white border")
        return issues

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class Candidate:
    """One render the pipeline can choose from."""

    name: str
    raw_bytes: bytes
    transparent_bytes: bytes
    metrics: QualityMetrics
    layout: Any = None
    subject_bytes: bytes | None = None


def image_metrics(data: bytes) -> QualityMetrics:
    """The print metrics of a transparent render, without the contrast check."""
    values = _load_rgba(data)
    rgb = values[..., :3]
    alpha = values[..., 3]
    opaque = alpha > 10
    ink = max(1, int(opaque.sum()))
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    magenta = ((red > 180) & (green < 90) & (blue > 60) & (blue < 200) & opaque).sum()
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    light = (luminance > 0.72) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 60) & (alpha > 20)
    bbox = _bbox(opaque)
    if bbox is None:
        return QualityMetrics(None, 0.0, 0.0, 0.0, 0.0, 0.0)
    left, top, right, bottom = bbox
    crop_light = light[top:bottom, left:right]
    crop_opaque = opaque[top:bottom, left:right]
    return QualityMetrics(
        contrast=None,
        text_overlap=0.0,
        magenta_fraction=float(magenta / ink),
        coverage=float(opaque.sum() / opaque.size),
        light_fill=float(crop_light.sum() / max(1, int(crop_opaque.sum()))),
        near_white_border=_near_white_border(values, bbox),
    )


def text_overlap(text_bytes: bytes, subject_bytes: bytes) -> float:
    """The share of the text mask that sits on the subject mask."""
    text = _load_rgba(text_bytes)
    subject = _load_rgba(subject_bytes)
    if text.shape != subject.shape:
        return 0.0
    text_mask = text[..., 3] > 20
    subject_mask = subject[..., 3] > 20
    if text_mask.sum() == 0:
        return 0.0
    return float((text_mask & subject_mask).sum() / text_mask.sum())


def with_contrast(metrics: QualityMetrics, contrast: float | None) -> QualityMetrics:
    """The same metrics with the garment contrast filled in."""
    return replace(metrics, contrast=contrast)


def choose_best(candidates: list[Candidate]) -> Candidate:
    """The candidate with the best print/placement rank."""
    if not candidates:
        raise ValueError("choose_best needs at least one candidate")
    return min(candidates, key=lambda candidate: candidate.metrics.rank)
