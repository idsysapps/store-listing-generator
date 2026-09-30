"""Tests for text compositing onto generated design images."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from store_listing.orchestration.text_compositor import (
    LayoutSpec,
    composite_text,
)


def _make_image(width: int = 1024, height: int = 1024, color: str = "white") -> bytes:
    img = Image.new("RGBA", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestLayoutSpec:
    def test_full_bleed_has_no_text(self) -> None:
        layout = LayoutSpec(layout_type="full_bleed")
        assert layout.headline_text is None
        assert layout.tagline_text is None

    def test_text_top_has_headline_only(self) -> None:
        layout = LayoutSpec(layout_type="text_top", headline_text="Born To Be Spooky")
        assert layout.headline_text == "Born To Be Spooky"
        assert layout.tagline_text is None

    def test_text_top_bottom_has_both(self) -> None:
        layout = LayoutSpec(
            layout_type="text_top_bottom",
            headline_text="Born To Be",
            tagline_text="Spooky",
        )
        assert layout.headline_text == "Born To Be"
        assert layout.tagline_text == "Spooky"

    def test_needs_text_returns_false_for_full_bleed(self) -> None:
        layout = LayoutSpec(layout_type="full_bleed")
        assert layout.needs_text is False

    def test_needs_text_returns_true_when_headline_present(self) -> None:
        layout = LayoutSpec(layout_type="text_top", headline_text="Hello")
        assert layout.needs_text is True

    def test_needs_text_returns_true_when_tagline_present(self) -> None:
        layout = LayoutSpec(layout_type="text_top_bottom", tagline_text="World")
        assert layout.needs_text is True


class TestCompositeText:
    def test_returns_original_image_for_full_bleed(self) -> None:
        image_bytes = _make_image()
        layout = LayoutSpec(layout_type="full_bleed")

        result = composite_text(image_bytes, layout)

        assert result == image_bytes

    def test_returns_original_when_no_text_fields(self) -> None:
        image_bytes = _make_image()
        layout = LayoutSpec(layout_type="text_top")

        result = composite_text(image_bytes, layout)

        assert result == image_bytes

    def test_adds_headline_text_to_top(self) -> None:
        image_bytes = _make_image(color="white")
        layout = LayoutSpec(layout_type="text_top", headline_text="TEST HEADLINE")

        result = composite_text(image_bytes, layout)

        assert result != image_bytes
        result_img = Image.open(BytesIO(result))
        assert result_img.size == (1024, 1024)

    def test_adds_headline_and_tagline(self) -> None:
        image_bytes = _make_image(color="white")
        layout = LayoutSpec(
            layout_type="text_top_bottom",
            headline_text="TOP TEXT",
            tagline_text="BOTTOM TEXT",
        )

        result = composite_text(image_bytes, layout)

        assert result != image_bytes
        result_img = Image.open(BytesIO(result))
        assert result_img.size == (1024, 1024)

    def test_output_is_valid_png(self) -> None:
        image_bytes = _make_image()
        layout = LayoutSpec(layout_type="text_top", headline_text="Hello World")

        result = composite_text(image_bytes, layout)

        img = Image.open(BytesIO(result))
        assert img.format == "PNG"

    def test_preserves_transparency(self) -> None:
        img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        transparent_bytes = buf.getvalue()

        layout = LayoutSpec(layout_type="text_top", headline_text="Over Transparent")

        result = composite_text(transparent_bytes, layout)

        result_img = Image.open(BytesIO(result))
        assert result_img.mode == "RGBA"

    def test_respects_custom_font_color(self) -> None:
        image_bytes = _make_image(color="black")
        layout = LayoutSpec(
            layout_type="text_top",
            headline_text="White Text",
            font_color="#FFFFFF",
        )

        result = composite_text(image_bytes, layout)

        assert result != image_bytes

    def test_handles_long_text_without_overflow(self) -> None:
        image_bytes = _make_image()
        layout = LayoutSpec(
            layout_type="text_top",
            headline_text="This Is A Very Long Headline That Should Still Fit",
        )

        result = composite_text(image_bytes, layout)

        result_img = Image.open(BytesIO(result))
        assert result_img.size == (1024, 1024)
