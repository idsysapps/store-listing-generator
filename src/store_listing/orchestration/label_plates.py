"""Short labels drawn onto the blank controls of a picture.

The brief declares the strings. The image model draws the controls blank,
because it cannot place a readable mark. After the lettering gate, the boxes
are measured from the blank shapes that were actually drawn, and a font fills
each one. A phone keypad and a remote are the same input: an ordered list of
strings and the same number of shapes.

The count has to match. A missing or extra shape is not a guess, and a vision
model is not asked to invent a digit.
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from store_listing.orchestration.text_integration import REGIONS
from store_listing.orchestration.typography import font_path

logger = logging.getLogger(__name__)

__all__ = [
    "LabelPlacement",
    "LabelPlate",
    "draw_label_plates",
    "plates_from_value",
    "plates_to_value",
]

# A mark is a label, not a headline. Four characters covers a key group such
# as WXYZ and rejects a slogan that slipped into the plate.
MAX_MARK_LENGTH: Final[int] = 4
_MARK = re.compile(r"^[A-Za-z0-9*#+-]{1,4}$")

# The nine zones the brief already uses for carriers. A plate only uses the
# zone to limit the search; the box itself is measured from the picture.
_REGION_RECTS: Final[dict[str, tuple[float, float, float, float]]] = {
    "top_left": (0.0, 0.0, 0.5, 0.5),
    "top_center": (0.25, 0.0, 0.5, 0.5),
    "top_right": (0.5, 0.0, 0.5, 0.5),
    "middle_left": (0.0, 0.25, 0.5, 0.5),
    "center": (0.25, 0.25, 0.5, 0.5),
    "middle_right": (0.5, 0.25, 0.5, 0.5),
    "bottom_left": (0.0, 0.5, 0.5, 0.5),
    "bottom_center": (0.25, 0.5, 0.5, 0.5),
    "bottom_right": (0.5, 0.5, 0.5, 0.5),
    "any": (0.0, 0.0, 1.0, 1.0),
}

_LABEL_FONT: Final[str] = "dejavu"
_MIN_CONTROL_AREA: Final[int] = 40


@dataclass(frozen=True)
class LabelPlate:
    """One ordered list of labels, searched inside a named zone."""

    marks: tuple[str, ...]
    region: str = "any"


@dataclass(frozen=True)
class LabelPlacement:
    """Where one declared string was drawn."""

    text: str
    box: tuple[int, int, int, int]
    rotation: float


@dataclass(frozen=True)
class _Shape:
    area: int
    cx: float
    cy: float
    box: tuple[int, int, int, int]


def plates_from_value(value: Any) -> tuple[LabelPlate, ...]:
    """Plates from a JSONB value, a JSON string, or a list of plate dicts."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return ()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return ()

    plates: list[LabelPlate] = []
    for entry in value:
        plate = _plate_from_entry(entry)
        if plate is not None:
            plates.append(plate)
    return tuple(plates)


def plates_to_value(plates: tuple[LabelPlate, ...]) -> list[dict[str, Any]] | None:
    """The JSON stored on the brief, or None when the brief has no plates."""
    if not plates:
        return None
    return [{"region": plate.region, "marks": list(plate.marks)} for plate in plates]


def draw_label_plates(
    image_bytes: bytes,
    plates: tuple[LabelPlate, ...],
) -> tuple[bytes, tuple[LabelPlacement, ...]] | None:
    """Draw every plate, or None when a plate's shapes do not match its marks.

    An empty plate list returns the picture unchanged. Measurement uses the
    picture before any label is drawn, so one plate's ink cannot split the
    next plate's shapes.
    """
    if not plates:
        return image_bytes, ()

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except OSError:
        logger.warning("Label plates could not read the picture")
        return None

    measured: list[tuple[LabelPlate, list[_Shape], float]] = []
    for plate in plates:
        found = _measure(image, plate)
        if found is None:
            return None
        measured.append((plate, found[0], found[1]))

    placements: list[LabelPlacement] = []
    for plate, shapes, rotation in measured:
        for mark, shape in zip(plate.marks, shapes, strict=True):
            _draw_mark(image, mark, shape, rotation)
            placements.append(LabelPlacement(text=mark, box=shape.box, rotation=rotation))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), tuple(placements)


def _plate_from_entry(entry: Any) -> LabelPlate | None:
    if not isinstance(entry, dict):
        return None
    raw_marks = entry.get("marks")
    if not isinstance(raw_marks, list):
        return None
    marks = tuple(_clean_mark(mark) for mark in raw_marks)
    kept = tuple(mark for mark in marks if mark is not None)
    if not kept:
        return None
    region = str(entry.get("region") or "any").strip().lower()
    if region not in REGIONS:
        region = "any"
    return LabelPlate(marks=kept, region=region)


def _clean_mark(mark: Any) -> str | None:
    if not isinstance(mark, str):
        return None
    text = mark.strip()
    if not _MARK.fullmatch(text):
        return None
    if len(text) > MAX_MARK_LENGTH:
        return None
    return text


def _measure(image: Image.Image, plate: LabelPlate) -> tuple[list[_Shape], float] | None:
    shapes = _blank_shapes(image, plate.region)
    if len(shapes) != len(plate.marks):
        logger.warning(
            "Label plate expected %d blank controls and found %d",
            len(plate.marks),
            len(shapes),
        )
        return None
    ordered, rotation = _reading_order(shapes)
    return ordered, rotation


def _blank_shapes(image: Image.Image, region: str) -> list[_Shape]:
    """The similar pale controls inside the zone, without the object's body."""
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    blue = arr[:, :, 2].astype(np.int16)
    alpha = arr[:, :, 3]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    pale = (alpha > 180) & (luminance > 175)
    rx, ry, rw, rh = _REGION_RECTS.get(region, _REGION_RECTS["any"])
    x0, y0 = int(rx * width), int(ry * height)
    x1, y1 = int((rx + rw) * width), int((ry + rh) * height)
    zone = np.zeros((height, width), dtype=bool)
    zone[y0:y1, x0:x1] = True
    pale &= zone

    components = _components(pale)
    controls = [shape for shape in components if _is_control(shape)]
    if len(controls) < 2:
        return controls
    areas = sorted(shape.area for shape in controls)
    median = areas[len(areas) // 2]
    return [shape for shape in controls if median * 0.45 <= shape.area <= median * 3]


def _is_control(shape: _Shape) -> bool:
    if shape.area < _MIN_CONTROL_AREA:
        return False
    _x, _y, width, height = shape.box
    short = max(min(width, height), 1)
    return max(width, height) / short <= 3


def _components(mask: np.ndarray) -> list[_Shape]:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=np.uint8)
    shapes: list[_Shape] = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist(), strict=True):
        if visited[y0, x0]:
            continue
        queue: deque[tuple[int, int]] = deque([(y0, x0)])
        visited[y0, x0] = 1
        count = 0
        sum_x = 0
        sum_y = 0
        min_x, min_y = x0, y0
        max_x, max_y = x0, y0
        while queue:
            y, x = queue.popleft()
            count += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = 1
                    queue.append((ny, nx))
        shapes.append(
            _Shape(
                area=count,
                cx=sum_x / count,
                cy=sum_y / count,
                box=(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
            )
        )
    return shapes


def _reading_order(shapes: list[_Shape]) -> tuple[list[_Shape], float]:
    """Rows along the arrangement, left to right within a row.

    The axes come from the shapes themselves. The across-axis points to the
    right of the picture and the down-axis points downward, so a rotated grid
    is still read in its own row order.
    """
    if len(shapes) == 1:
        return shapes, 0.0

    points = np.array([(shape.cx, shape.cy) for shape in shapes], dtype=np.float64)
    across, down = _axes(points)
    center = points.mean(axis=0)
    centered = points - center
    across_c = centered @ across
    down_c = centered @ down
    rotation = float(np.degrees(np.arctan2(float(across[1]), float(across[0]))))
    order_idx = np.argsort(down_c, kind="mergesort")
    pitch = _pitch(points)
    rows: list[list[int]] = [[int(order_idx[0])]]
    for previous, current in zip(order_idx.tolist(), order_idx[1:].tolist(), strict=False):
        gap = float(down_c[current] - down_c[previous])
        if gap > pitch * 0.55:
            rows.append([int(current)])
        else:
            rows[-1].append(int(current))

    ordered: list[_Shape] = []
    for row in rows:
        row.sort(key=lambda index: float(across_c[index]))
        ordered.extend(shapes[index] for index in row)
    return ordered, rotation


def _axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The across-axis and the down-axis of a set of controls.

    Neighbouring controls in a row are closer than the next row, so the
    nearest-neighbour direction is the row. A square grid's diagonal is the
    long axis of the points, and that diagonal is not a row.
    """
    vectors: list[np.ndarray] = []
    for index, point in enumerate(points):
        others = np.delete(points, index, axis=0)
        delta = others - point
        distance = np.sqrt((delta**2).sum(axis=1))
        nearest = delta[int(distance.argmin())]
        if nearest[0] < 0:
            nearest = -nearest
        vectors.append(nearest)
    across = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(across))
    if norm < 1e-6:
        across = np.array([1.0, 0.0])
    else:
        across = across / norm
    down = np.array([-across[1], across[0]])
    if down[1] < 0:
        down = -down
    return across, down


def _pitch(points: np.ndarray) -> float:
    """The typical gap between neighbouring controls."""
    distances: list[float] = []
    for index, point in enumerate(points):
        others = np.delete(points, index, axis=0)
        gap = np.sqrt(((others - point) ** 2).sum(axis=1)).min()
        distances.append(float(gap))
    return max(float(np.median(distances)), 1.0)


def _draw_mark(image: Image.Image, text: str, shape: _Shape, rotation: float) -> None:
    _x, _y, width, height = shape.box
    size = max(8, int(min(width, height) * 0.62))
    font = ImageFont.truetype(str(font_path(_LABEL_FONT)), size)
    draw = ImageDraw.Draw(image)
    while size > 8:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= width * 0.8 and bottom - top <= height * 0.8:
            break
        size -= 1
        font = ImageFont.truetype(str(font_path(_LABEL_FONT)), size)
    left, top, right, bottom = (int(value) for value in draw.textbbox((0, 0), text, font=font))
    glyph = Image.new("RGBA", (max(right - left, 1), max(bottom - top, 1)), (0, 0, 0, 0))
    ImageDraw.Draw(glyph).text((-left, -top), text, font=font, fill=_ink(image, shape))
    rotated = glyph.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
    paste_x = int(shape.cx - rotated.width / 2)
    paste_y = int(shape.cy - rotated.height / 2)
    image.alpha_composite(rotated, (paste_x, paste_y))


def _ink(image: Image.Image, shape: _Shape) -> tuple[int, int, int, int]:
    """The dark outline around the control, or near-black when it has none."""
    x, y, width, height = shape.box
    pad = 4
    crop = image.crop(
        (
            max(x - pad, 0),
            max(y - pad, 0),
            min(x + width + pad, image.width),
            min(y + height + pad, image.height),
        )
    )
    arr = np.asarray(crop)
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    dark = (arr[:, :, 3] > 180) & (luminance < 90)
    if not dark.any():
        return (20, 20, 20, 255)
    red, green, blue = (int(channel) for channel in arr[dark].mean(axis=0)[:3])
    return (red, green, blue, 255)
