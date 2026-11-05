"""Turn the bake-off CSV into markdown tables for the report."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

FOCUS_BRIEFS = (205, 206, 207, 209, 210, 211, 213)
CONTRAST_MIN = 3.0
MAGENTA_MAX = 0.005
SOLID_LIGHT_MAX = 0.30
BORDER_MAX = 0.30
OVERLAP_MAX = 0.20


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.1%}" if percent else f"{value:.2f}"
    return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.scores.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_brief: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        row["brief_id"] = int(row["brief_id"])
        for key in (
            "coverage",
            "magenta_pct",
            "solid_light_pct",
            "near_white_border",
            "text_overlap",
            "contrast",
            "accent_contrast",
        ):
            row[key] = float(row[key]) if row.get(key) not in (None, "") else None
        row["ocr_pass"] = _bool(row.get("ocr_pass"))
        by_arm[row["arm"]].append(row)
        by_brief[row["brief_id"]][row["arm"]].append(row)

    summary_rows = []
    for arm in ("current", "hybrid", "text_first"):
        arm_rows = by_arm.get(arm, [])
        if not arm_rows:
            continue
        summary_rows.append(
            [
                arm,
                str(len(arm_rows)),
                _mean(arm_rows, "coverage", percent=True),
                _mean(arm_rows, "magenta_pct", percent=True),
                _mean(arm_rows, "solid_light_pct", percent=True),
                _mean(arm_rows, "near_white_border", percent=True),
                _mean(arm_rows, "text_overlap", percent=True),
                _min(arm_rows, "contrast"),
                _min(arm_rows, "accent_contrast"),
                _rate(arm_rows, "ocr_pass"),
                str(_flagged(arm_rows)),
            ]
        )

    sections = [
        "## Arm summary",
        _table(
            [
                "arm",
                "n",
                "coverage",
                "magenta",
                "light fill",
                "white border",
                "text overlap",
                "min contrast",
                "min accent",
                "OCR",
                "flagged",
            ],
            summary_rows,
        ),
        "",
        "## Focus briefs",
    ]

    for brief_id in FOCUS_BRIEFS:
        if brief_id not in by_brief:
            continue
        focus_rows = []
        for arm in ("current", "hybrid", "text_first"):
            arm_rows = by_brief[brief_id].get(arm, [])
            for row in arm_rows:
                focus_rows.append(
                    [
                        arm,
                        str(row["seed"]),
                        _fmt(row["contrast"]),
                        _fmt(row["accent_contrast"]),
                        _fmt(row["text_overlap"], percent=True),
                        _fmt(row["magenta_pct"], percent=True),
                        _fmt(row["solid_light_pct"], percent=True),
                        _fmt(row["near_white_border"], percent=True),
                        _fmt(row["ocr_pass"]),
                    ]
                )
        sections.append(f"### {brief_id}")
        sections.append(
            _table(
                [
                    "arm",
                    "seed",
                    "contrast",
                    "accent",
                    "overlap",
                    "magenta",
                    "light fill",
                    "border",
                    "OCR",
                ],
                focus_rows,
            )
        )
        sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print("\n".join(sections))
    return 0


def _bool(value: Any) -> bool | None:
    if value in (None, "", "None"):
        return None
    return str(value).strip().lower() in ("true", "1", "yes")


def _mean(rows: list[dict[str, Any]], key: str, *, percent: bool = False) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return "-"
    mean = sum(values) / len(values)
    return f"{mean:.1%}" if percent else f"{mean:.2f}"


def _min(rows: list[dict[str, Any]], key: str) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    return f"{min(values):.2f}" if values else "-"


def _rate(rows: list[dict[str, Any]], key: str) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    return f"{sum(values) / len(values):.0%}" if values else "-"


def _flagged(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        contrast = row.get("contrast")
        accent = row.get("accent_contrast")
        if (
            (contrast is not None and contrast < CONTRAST_MIN)
            or (accent is not None and accent < CONTRAST_MIN)
            or (row.get("magenta_pct") or 0) > MAGENTA_MAX
            or (row.get("solid_light_pct") or 0) > SOLID_LIGHT_MAX
            or (row.get("near_white_border") or 0) > BORDER_MAX
            or (row.get("text_overlap") or 0) > OVERLAP_MAX
        ):
            count += 1
    return count


if __name__ == "__main__":
    sys.exit(main())
