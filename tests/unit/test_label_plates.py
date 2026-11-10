"""A plate is an ordered list of short labels drawn onto blank controls."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from store_listing.orchestration.label_plates import (
    LabelPlate,
    draw_label_plates,
    plates_from_value,
)


def _ovals(
    centers: list[tuple[int, int]],
    *,
    size: tuple[int, int] = (240, 220),
    radius: tuple[int, int] = (18, 14),
    body: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if body is not None:
        draw.rectangle(body, fill=(236, 228, 206, 255))
    rx, ry = radius
    for x, y in centers:
        draw.ellipse(
            (x - rx, y - ry, x + rx, y + ry),
            fill=(250, 244, 224, 255),
            outline=(15, 15, 15, 255),
            width=3,
        )
    return image


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, width, height = box
    return x + width / 2, y + height / 2


def _nearest(point: tuple[float, float], centers: list[tuple[int, int]]) -> tuple[int, int]:
    return min(
        centers, key=lambda center: (center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2
    )


def _dark_inside(image: Image.Image, center: tuple[int, int], radius: int = 10) -> int:
    pixels = image.load()
    assert pixels is not None
    total = 0
    cx, cy = center
    for y in range(cy - radius, cy + radius):
        for x in range(cx - radius, cx + radius):
            raw = pixels[x, y]
            if not isinstance(raw, tuple) or len(raw) < 4:
                continue
            red, green, blue, alpha = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
            if alpha > 200 and (0.299 * red + 0.587 * green + 0.114 * blue) < 80:
                total += 1
    return total


class TestPlatesFromValue:
    def test_a_plate_keeps_short_marks_and_a_known_region(self) -> None:
        plates = plates_from_value(
            [{"region": "bottom_left", "marks": ["1", "QX", "HEADLINE", "", "*"]}]
        )

        assert len(plates) == 1
        assert plates[0].region == "bottom_left"
        assert plates[0].marks == ("1", "QX", "*")

    def test_an_unknown_region_searches_the_whole_picture(self) -> None:
        plates = plates_from_value([{"region": "nope", "marks": ["2"]}])

        assert plates[0].region == "any"

    def test_nothing_is_an_empty_plate_list(self) -> None:
        assert plates_from_value(None) == ()
        assert plates_from_value([]) == ()


class TestDrawLabelPlates:
    def test_strings_land_in_reading_order_on_an_upright_grid(self) -> None:
        centers = [(50, 40), (120, 40), (50, 140), (120, 140)]
        image = _ovals(centers)
        plate = LabelPlate(marks=("1", "2", "3", "4"), region="any")

        drawn = draw_label_plates(_png(image), (plate,))

        assert drawn is not None
        png, placements = drawn
        assert [placement.text for placement in placements] == ["1", "2", "3", "4"]
        landed = [_nearest(_box_center(placement.box), centers) for placement in placements]
        assert landed == centers
        assert all(abs(placement.rotation) < 15 for placement in placements)
        painted = Image.open(BytesIO(png)).convert("RGBA")
        assert _dark_inside(painted, centers[0]) > _dark_inside(image, centers[0])

    def test_a_rotated_grid_is_read_along_its_own_rows(self) -> None:
        centers = [(70, 90), (130, 60), (115, 179), (175, 149)]
        image = _ovals(centers, radius=(14, 12))
        plate = LabelPlate(marks=("1", "2", "3", "4"), region="any")

        drawn = draw_label_plates(_png(image), (plate,))

        assert drawn is not None
        png, placements = drawn
        landed = [_nearest(_box_center(placement.box), centers) for placement in placements]
        assert landed == centers
        assert all(20 < abs(placement.rotation) < 50 for placement in placements)
        painted = Image.open(BytesIO(png)).convert("RGBA")
        assert _dark_inside(painted, centers[0]) > 0

    def test_a_large_body_is_not_a_control(self) -> None:
        centers = [(70, 80), (150, 80)]
        image = _ovals(centers, body=(8, 8, 230, 200))
        plate = LabelPlate(marks=("A", "B"), region="any")

        drawn = draw_label_plates(_png(image), (plate,))

        assert drawn is not None
        _, placements = drawn
        landed = [_nearest(_box_center(placement.box), centers) for placement in placements]
        assert landed == centers

    def test_the_wrong_number_of_shapes_is_not_drawn(self) -> None:
        image = _ovals([(50, 60), (130, 60), (90, 140)])
        plate = LabelPlate(marks=("1", "2"), region="any")

        assert draw_label_plates(_png(image), (plate,)) is None

    def test_a_region_ignores_shapes_outside_it(self) -> None:
        image = _ovals([(40, 40), (90, 40)])
        plate = LabelPlate(marks=("1", "2"), region="bottom_right")

        assert draw_label_plates(_png(image), (plate,)) is None

    def test_no_plate_leaves_the_picture_alone(self) -> None:
        image = _ovals([(50, 60)])
        raw = _png(image)

        drawn = draw_label_plates(raw, ())

        assert drawn is not None
        png, placements = drawn
        assert png == raw
        assert placements == ()
