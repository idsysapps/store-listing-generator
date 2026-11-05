"""Score the pipeline bake-off outputs.

The deterministic metrics are the ones the current pipeline does not measure:
contrast on the recommended garment, text over the subject, chroma residue and
a solid sticker fill. OCR is optional and runs on the DGX vision model.

    python spike/bakeoff/score_spike.py \
        --briefs briefs.jsonl \
        --manifest /tmp/spike/out/manifest.jsonl \
        --out /tmp/spike/out
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from store_listing.orchestration.llm_client import get_vision_client, get_vision_model
from store_listing.orchestration.text_validation import TextValidator
from store_listing.orchestration.typography import accent_for

logger = logging.getLogger("spike.score")

GARMENT_HEX = {
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


def _rgb(value: str) -> tuple[int, int, int]:
    digits = value.lstrip("#")
    return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _luminance(rgb: tuple[int, int, int]) -> float:
    channels = [value / 255 for value in rgb]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    lighter = max(_luminance(_rgb(first)), _luminance(_rgb(second)))
    darker = min(_luminance(_rgb(first)), _luminance(_rgb(second)))
    return (lighter + 0.05) / (darker + 0.05)


def garment_hex(name: str | None) -> str:
    if not name:
        return "#FFFFFF"
    return GARMENT_HEX.get(" ".join(name.lower().split()), "#FFFFFF")


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGBA")).astype(np.int16)


def print_metrics(path: Path) -> dict[str, float]:
    values = load_rgba(path)
    rgb = values[..., :3]
    alpha = values[..., 3]
    opaque = alpha > 10
    total = opaque.size
    ink = max(1, int(opaque.sum()))
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    magenta = ((red > 180) & (green < 90) & (blue > 60) & (blue < 200) & opaque).sum()
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    light = (luminance > 0.72) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 40) & (alpha > 200)
    bbox = _bbox(opaque)
    if bbox is None:
        return {
            "coverage": 0.0,
            "magenta_pct": 0.0,
            "solid_light_pct": 0.0,
            "near_white_border": 0.0,
            "text_overlap": 0.0,
        }
    left, top, right, bottom = bbox
    crop_light = light[top:bottom, left:right]
    crop_opaque = opaque[top:bottom, left:right]
    solid_light_pct = crop_light.sum() / max(1, int(crop_opaque.sum()))
    near_white_border = _near_white_border(values, bbox)
    return {
        "coverage": float(opaque.sum() / total),
        "magenta_pct": float(magenta / ink),
        "solid_light_pct": float(solid_light_pct),
        "near_white_border": float(near_white_border),
        "text_overlap": 0.0,
    }


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _near_white_border(values: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = bbox
    crop = values[top:bottom, left:right]
    rgb = crop[..., :3]
    alpha = crop[..., 3]
    near_white = (
        (rgb.min(axis=2) > 220) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 30) & (alpha > 200)
    )
    band = max(2, int(min(crop.shape[:2]) * 0.03))
    ring = np.zeros_like(near_white)
    ring[:band, :] = True
    ring[-band:, :] = True
    ring[:, :band] = True
    ring[:, -band:] = True
    opaque_ring = ring & (alpha > 200)
    if opaque_ring.sum() == 0:
        return 0.0
    return float((near_white & ring).sum() / opaque_ring.sum())


def text_overlap(text_path: Path | None, subject_path: Path | None) -> float:
    if (
        text_path is None
        or subject_path is None
        or not text_path.exists()
        or not subject_path.exists()
    ):
        return 0.0
    text = load_rgba(text_path)
    subject = load_rgba(subject_path)
    if text.shape != subject.shape:
        return 0.0
    text_mask = text[..., 3] > 20
    subject_mask = subject[..., 3] > 20
    if text_mask.sum() == 0:
        return 0.0
    return float((text_mask & subject_mask).sum() / text_mask.sum())


def flatten_on_colour(path: Path, colour: str) -> bytes:
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    background = Image.new("RGBA", image.size, (*_rgb(colour), 255))
    background.alpha_composite(image)
    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def current_entries(briefs: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in briefs:
        brief_id = int(row["id"])
        path = out_dir / "current" / f"{brief_id}_transparent.png"
        if not path.exists():
            continue
        plan = row.get("text_plan") or {}
        ink = str(plan.get("ink") or "dark")
        font_color = str(row.get("font_color") or "#000000")
        if ink == "light":
            resolved = "#F7F3E8"
        elif ink == "dark":
            resolved = "#111111"
        elif ink == "accent":
            resolved = accent_for(font_color)
        else:
            resolved = ink
        accent = (
            accent_for(resolved)
            if row.get("voice") == "loud" and row.get("copy_mode") != "no_text"
            else None
        )
        entries.append(
            {
                "brief_id": brief_id,
                "arm": "current",
                "seed": 0,
                "template": row.get("layout_type"),
                "garment": _garment_name(row),
                "headline": row.get("headline_text"),
                "tagline": row.get("tagline_text"),
                "expected": [
                    block for block in (row.get("headline_text"), row.get("tagline_text")) if block
                ],
                "raw_path": None,
                "transparent_path": str(path),
                "subject_path": None,
                "text_path": None,
                "text_ink": resolved,
                "accent_ink": accent,
                "prompt": None,
            }
        )
    return entries


def _garment_name(row: dict[str, Any]) -> str:
    materials = row.get("recommended_materials") or []
    if materials and isinstance(materials, list):
        return str(materials[0].get("color") or "White")
    return "White"


def run_ocr(
    path: Path, garment: str, expected: list[str], threshold: float = 0.0
) -> dict[str, Any]:
    if not expected:
        return {"ocr_pass": True, "ocr_observed": "", "ocr_error": ""}
    try:
        validator = TextValidator(client=get_vision_client(), model=get_vision_model())
        result = validator.validate(
            flatten_on_colour(path, garment_hex(garment)),
            headline=expected[0],
            tagline=expected[1] if len(expected) > 1 else None,
        )
    except Exception as error:  # noqa: BLE001 - report, do not stop the spike
        return {"ocr_pass": None, "ocr_observed": "", "ocr_error": str(error)}
    return {
        "ocr_pass": bool(result.matched),
        "ocr_observed": result.observed,
        "ocr_error": result.error or "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ocr-arms", default="current,text_first")
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    briefs = [
        json.loads(line)
        for line in args.briefs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries = current_entries(briefs, args.out) + generated
    ocr_arms = {arm.strip() for arm in args.ocr_arms.split(",") if arm.strip()}

    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = Path(entry["transparent_path"])
        if not path.exists():
            continue
        metrics: dict[str, Any] = dict(print_metrics(path))
        garment = entry.get("garment") or "White"
        metrics["text_overlap"] = text_overlap(
            Path(entry["text_path"]) if entry.get("text_path") else None,
            Path(entry["subject_path"]) if entry.get("subject_path") else None,
        )
        text_ink = entry.get("text_ink")
        accent = entry.get("accent_ink")
        metrics["contrast"] = contrast_ratio(text_ink, garment_hex(garment)) if text_ink else None
        metrics["accent_contrast"] = (
            contrast_ratio(accent, garment_hex(garment)) if accent else None
        )
        if not args.no_ocr and entry["arm"] in ocr_arms:
            metrics.update(run_ocr(path, garment, entry.get("expected") or []))
        else:
            metrics.update({"ocr_pass": None, "ocr_observed": "", "ocr_error": ""})
        row = {
            "brief_id": entry["brief_id"],
            "arm": entry["arm"],
            "seed": entry["seed"],
            "template": entry.get("template"),
            "garment": garment,
            "headline": entry.get("headline"),
            "tagline": entry.get("tagline"),
            **metrics,
        }
        rows.append(row)
        print(
            f"{row['brief_id']:>3} {row['arm']:<10} coverage={row['coverage']:.1%} "
            f"magenta={row['magenta_pct']:.2%} contrast={row['contrast']} "
            f"overlap={row['text_overlap']:.0%} ocr={row['ocr_pass']}"
        )

    fieldnames = [
        "brief_id",
        "arm",
        "seed",
        "template",
        "garment",
        "headline",
        "tagline",
        "coverage",
        "magenta_pct",
        "solid_light_pct",
        "near_white_border",
        "text_overlap",
        "contrast",
        "accent_contrast",
        "ocr_pass",
        "ocr_observed",
        "ocr_error",
    ]
    csv_path = args.out / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, Any]] = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "count": len(arm_rows),
            "mean_coverage": _mean(row["coverage"] for row in arm_rows),
            "mean_magenta_pct": _mean(row["magenta_pct"] for row in arm_rows),
            "mean_solid_light_pct": _mean(row["solid_light_pct"] for row in arm_rows),
            "mean_near_white_border": _mean(row["near_white_border"] for row in arm_rows),
            "mean_text_overlap": _mean(row["text_overlap"] for row in arm_rows),
            "min_contrast": _min(row["contrast"] for row in arm_rows),
            "min_accent_contrast": _min(row["accent_contrast"] for row in arm_rows),
            "ocr_pass_rate": _rate(row["ocr_pass"] for row in arm_rows),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def _min(values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return min(numbers) if numbers else None


def _rate(values: Any) -> float | None:
    booleans = [bool(value) for value in values if value is not None]
    return sum(booleans) / len(booleans) if booleans else None


if __name__ == "__main__":
    sys.exit(main())
