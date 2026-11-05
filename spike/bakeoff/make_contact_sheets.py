"""Build blind-ish contact sheets for the pipeline bake-off.

Each sheet puts the current render next to every generated variant for one
brief, composited on the garment colour. The labels say the arm and seed so
the reviewer can compare, but not the prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

GARMENT_HEX = {
    "black": "#111111",
    "jet black": "#0B0B0B",
    "dark heather": "#3A3A3A",
    "white": "#FFFFFF",
    "event white": "#FAFAFA",
    "carolina blue mist": "#A8C6E0",
    "cherry blossom": "#F2C6D4",
    "true celadon": "#8FBFA8",
}

THUMB = (520, 420)
LABEL_HEIGHT = 42
TILE_HEIGHT = THUMB[1] + LABEL_HEIGHT


def _rgb(value: str) -> tuple[int, int, int]:
    digits = value.lstrip("#")
    return tuple(int(digits[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _garment_hex(name: str | None) -> str:
    if not name:
        return "#FFFFFF"
    return GARMENT_HEX.get(" ".join(name.lower().split()), "#FFFFFF")


def _thumbnail(path: Path, garment: str) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    canvas = Image.new("RGB", image.size, _rgb(_garment_hex(garment)))
    canvas.paste(image, mask=image.getchannel("A"))
    canvas.thumbnail(THUMB, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (THUMB[0], TILE_HEIGHT), (250, 250, 250))
    tile.paste(
        canvas,
        ((THUMB[0] - canvas.width) // 2, LABEL_HEIGHT + (THUMB[1] - canvas.height) // 2),
    )
    return tile


def _sheet(entries: list[dict[str, Any]], output: Path) -> None:
    tiles: list[Image.Image] = []
    for entry in entries:
        path = Path(entry["transparent_path"])
        if not path.exists():
            continue
        tile = _thumbnail(path, entry.get("garment") or "White")
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, THUMB[0], LABEL_HEIGHT], fill=(20, 20, 20))
        label = f"{entry['arm']} seed={entry.get('seed', 0)}"
        draw.text((10, 12), label, fill=(245, 245, 245))
        tiles.append(tile)

    if not tiles:
        return
    width = min(3, len(tiles)) * THUMB[0]
    rows = (len(tiles) + 2) // 3
    sheet = Image.new("RGB", (width, rows * TILE_HEIGHT), (235, 235, 235))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * THUMB[0], (index // 3) * TILE_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

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
    by_brief: dict[int, list[dict[str, Any]]] = {}
    for row in briefs:
        brief_id = int(row["id"])
        entries = []
        current = args.out / "current" / f"{brief_id}_transparent.png"
        if current.exists():
            entries.append(
                {
                    "arm": "current",
                    "seed": 0,
                    "garment": _garment(row),
                    "transparent_path": str(current),
                }
            )
        entries.extend(entry for entry in generated if entry["brief_id"] == brief_id)
        by_brief[brief_id] = entries

    for brief_id, entries in sorted(by_brief.items()):
        _sheet(entries, args.out / "contact_sheets" / f"{brief_id}.png")
        print(f"wrote contact sheet for {brief_id} ({len(entries)} tiles)")
    return 0


def _garment(row: dict[str, Any]) -> str:
    materials = row.get("recommended_materials") or []
    if materials and isinstance(materials, list):
        return str(materials[0].get("color") or "White")
    return "White"


if __name__ == "__main__":
    sys.exit(main())
