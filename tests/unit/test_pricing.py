"""Tests for the DTF price and print-size recommendation.

The price is a deterministic cost-plus calculation. The LLM may choose the
positioning, but it never supplies the number: the code applies the blank
cost, the per-square-inch print cost, the Amazon fee, the positioning margin
and the price floor/ceiling.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from store_listing.orchestration.pricing import (
    PricingPolicy,
    allowed_positionings,
    build_pricing,
    fallback_positioning,
    normalise_positioning,
    print_spec_for,
    quote_for_material,
    suggest_print_size,
)


def _material(
    *,
    substrate: str = "t-shirt",
    unit_cost: float | None = 4.50,
    sizes: list[str] | None = None,
    external_id: str = "bc3001-white",
) -> dict[str, object]:
    return {
        "external_id": external_id,
        "name": "Bella Canvas 3001",
        "color": "White",
        "substrate_type": substrate,
        "unit_cost": unit_cost,
        "sizes": sizes if sizes is not None else ["S", "M", "L", "XL", "2XL", "3XL"],
    }


def _quote(
    *,
    material: dict[str, object] | None = None,
    canvas: tuple[int, int] = (1024, 1024),
    positioning: str = "standard",
    confidence: int = 80,
    policy: PricingPolicy | None = None,
):
    return quote_for_material(
        material or _material(),
        "dtf_apparel",
        [],
        canvas,
        positioning,
        confidence,
        policy or PricingPolicy(),
    )


class TestPositioning:
    def test_low_confidence_is_budget_only(self) -> None:
        assert allowed_positionings(30) == ("budget",)

    def test_medium_confidence_is_budget_or_standard(self) -> None:
        assert allowed_positionings(55) == ("budget", "standard")

    def test_high_confidence_can_be_premium(self) -> None:
        assert allowed_positionings(80) == ("budget", "standard", "premium")

    def test_unknown_positioning_falls_back_to_standard(self) -> None:
        assert normalise_positioning("luxury") == "standard"

    def test_the_fallback_is_standard_for_a_medium_design(self) -> None:
        assert fallback_positioning(60) == "standard"

    def test_the_fallback_is_budget_for_a_weak_design(self) -> None:
        assert fallback_positioning(20) == "budget"


class TestPrintSpecs:
    def test_a_t_shirt_gets_a_full_front_area(self) -> None:
        spec = print_spec_for("dtf_apparel", _material())

        assert spec is not None
        assert spec.max_width_in == Decimal(12)
        assert spec.max_height_in == Decimal(16)
        assert spec.placement == "full_front"
        assert spec.max_price == Decimal("29.99")

    def test_a_hoodie_gets_a_smaller_area(self) -> None:
        spec = print_spec_for("dtf_apparel", _material(substrate="hoodie"))

        assert spec is not None
        assert spec.max_width_in == Decimal(12)
        assert spec.max_height_in == Decimal(12)
        assert spec.max_price == Decimal("49.99")

    def test_an_onesie_gets_a_small_area(self) -> None:
        spec = print_spec_for("dtf_apparel", _material(substrate="onesie"))

        assert spec is not None
        assert spec.max_width_in == Decimal(8)
        assert spec.max_height_in == Decimal(8)
        assert spec.max_price == Decimal("19.99")

    def test_a_hat_gets_a_front_panel(self) -> None:
        spec = print_spec_for("dtf_apparel", _material(substrate="cap"))

        assert spec is not None
        assert spec.max_width_in == Decimal(4)
        assert spec.max_height_in == Decimal("2.5")
        assert spec.placement == "front_panel"
        assert spec.max_price == Decimal("24.99")

    def test_a_tote_gets_its_own_area(self) -> None:
        spec = print_spec_for("dtf_apparel", _material(substrate="tote"))

        assert spec is not None
        assert spec.max_width_in == Decimal(10)
        assert spec.max_height_in == Decimal(10)
        assert spec.max_price == Decimal("34.99")

    def test_a_generic_substrate_uses_the_product_name(self) -> None:
        material = _material(substrate="dtf-apparel")

        spec = print_spec_for("dtf_apparel", material, ["t-shirt"])

        assert spec is not None
        assert spec.max_height_in == Decimal(16)

    def test_an_unknown_substrate_uses_the_estimated_default(self) -> None:
        spec = print_spec_for("dtf_apparel", _material(substrate="mystery"))

        assert spec is not None
        assert spec.estimated is True
        assert spec.max_width_in == Decimal(12)
        assert spec.max_height_in == Decimal(12)

    def test_explicit_material_attributes_override_the_table(self) -> None:
        material = _material()
        material["attributes"] = {
            "print_width_in": 10.0,
            "print_height_in": 12.0,
            "print_placement": "left_chest",
        }

        spec = print_spec_for("dtf_apparel", material)

        assert spec is not None
        assert spec.max_width_in == Decimal("10.0")
        assert spec.max_height_in == Decimal("12.0")
        assert spec.placement == "left_chest"
        assert spec.estimated is False

    def test_malformed_attributes_leave_the_table_in_place(self) -> None:
        material = _material()
        material["attributes"] = {"print_width_in": "wide"}

        spec = print_spec_for("dtf_apparel", material)

        assert spec is not None
        assert spec.max_width_in == Decimal(12)
        assert spec.max_height_in == Decimal(16)

    def test_a_non_dtf_product_has_no_spec(self) -> None:
        assert print_spec_for("sublimation", _material()) is None


class TestPrintSize:
    def test_a_square_canvas_keeps_its_aspect(self) -> None:
        spec = print_spec_for("dtf_apparel", _material())
        assert spec is not None

        size = suggest_print_size((1024, 1024), spec)

        assert size.width_in == Decimal(12)
        assert size.height_in == Decimal(12)
        assert size.required_pixels == (3600, 3600)
        assert size.target_dpi == 300

    def test_a_wide_canvas_fits_the_t_shirt_area(self) -> None:
        spec = print_spec_for("dtf_apparel", _material())
        assert spec is not None

        size = suggest_print_size((1216, 832), spec)

        assert size.width_in == Decimal(12)
        assert size.height_in == Decimal("8.2")
        assert size.required_pixels == (3600, 2460)

    def test_a_tall_canvas_fits_the_t_shirt_area(self) -> None:
        spec = print_spec_for("dtf_apparel", _material())
        assert spec is not None

        size = suggest_print_size((832, 1216), spec)

        assert size.width_in == Decimal("10.9")
        assert size.height_in == Decimal(16)
        assert size.required_pixels == (3270, 4800)

    def test_the_upscale_target_is_reported(self) -> None:
        spec = print_spec_for("dtf_apparel", _material())
        assert spec is not None

        size = suggest_print_size((1024, 1024), spec)

        assert size.source_pixels == (1024, 1024)
        assert float(size.upscale_factor) == pytest.approx(3.52, abs=0.01)


class TestPrice:
    def test_the_standard_floor_holds_for_a_cheap_tee(self) -> None:
        quote = _quote()

        assert quote is not None
        assert quote.recommended_price == Decimal("19.99")
        assert quote.positioning == "standard"
        assert quote.status == "ok"

    def test_the_budget_floor_is_lower(self) -> None:
        quote = _quote(positioning="budget")

        assert quote is not None
        assert quote.recommended_price == Decimal("14.99")

    def test_the_premium_floor_is_higher(self) -> None:
        quote = _quote(positioning="premium")

        assert quote is not None
        assert quote.recommended_price == Decimal("24.99")

    def test_a_premium_hoodie_pays_for_its_blank(self) -> None:
        quote = _quote(
            material=_material(substrate="hoodie", unit_cost=12.0), positioning="premium"
        )

        assert quote is not None
        assert quote.recommended_price == Decimal("36.99")
        assert quote.status == "ok"

    def test_a_standard_tier_downgrades_when_the_ceiling_is_hit(self) -> None:
        quote = _quote(
            material=_material(unit_cost=13.0),
            positioning="standard",
            confidence=80,
        )

        assert quote is not None
        assert quote.status == "downgraded"
        assert quote.positioning == "budget"
        assert quote.recommended_price == Decimal("24.99")

    def test_a_material_too_expensive_for_the_ceiling_has_no_price(self) -> None:
        quote = _quote(material=_material(unit_cost=20.0), positioning="premium", confidence=80)

        assert quote is not None
        assert quote.status == "no_viable_price"
        assert quote.recommended_price is None

    def test_a_missing_cost_has_no_quote(self) -> None:
        assert _quote(material=_material(unit_cost=None)) is None

    def test_a_non_dtf_product_has_no_quote(self) -> None:
        material = _material()
        assert (
            quote_for_material(
                material,
                "sublimation",
                [],
                (1024, 1024),
                "standard",
                80,
                PricingPolicy(),
            )
            is None
        )

    def test_the_amazon_fee_is_in_the_breakdown(self) -> None:
        quote = _quote()

        assert quote is not None
        assert quote.breakdown.marketplace_fee == Decimal("3.00")
        assert quote.breakdown.material_cost == Decimal("4.50")
        assert quote.breakdown.print_cost == Decimal("0.72")
        assert quote.breakdown.target_margin == Decimal("0.40")

    def test_the_actual_margin_reflects_the_floor(self) -> None:
        quote = _quote()

        assert quote is not None
        assert quote.breakdown.actual_margin > Decimal("0.40")


class TestSizePrices:
    def test_larger_sizes_carry_their_upcharge(self) -> None:
        quote = _quote()

        assert quote is not None
        assert quote.size_prices["S"] == Decimal("19.99")
        assert quote.size_prices["2XL"] == Decimal("21.99")
        assert quote.size_prices["3XL"] == Decimal("22.99")

    def test_xxl_and_xxxl_are_normalised(self) -> None:
        quote = _quote(
            material=_material(sizes=["S", "XXL", "XXXL"]),
        )

        assert quote is not None
        assert quote.size_prices["2XL"] == Decimal("21.99")
        assert quote.size_prices["3XL"] == Decimal("22.99")
        assert "XXL" not in quote.size_prices

    def test_no_sizes_means_no_size_map(self) -> None:
        quote = _quote(material=_material(sizes=[]))

        assert quote is not None
        assert quote.size_prices == {}


class TestBuildPricing:
    def test_each_material_gets_its_own_quote(self) -> None:
        materials = [
            _material(external_id="tee", unit_cost=4.50),
            _material(external_id="hoodie", substrate="hoodie", unit_cost=12.0),
        ]

        pricing = build_pricing(
            materials,
            "dtf_apparel",
            [],
            (1024, 1024),
            80,
            "standard",
            "A broad evergreen design.",
        )

        assert pricing is not None
        assert len(pricing.quotes) == 2
        assert pricing.quotes[0].external_id == "tee"
        assert pricing.quotes[1].external_id == "hoodie"
        assert pricing.status == "ok"
        assert pricing.effective_positioning == "standard"

    def test_a_downgrade_is_visible_at_the_brief_level(self) -> None:
        pricing = build_pricing(
            [_material(unit_cost=13.0)],
            "dtf_apparel",
            [],
            (1024, 1024),
            80,
            "standard",
            "Strong concept.",
        )

        assert pricing is not None
        assert pricing.requested_positioning == "standard"
        assert pricing.effective_positioning == "budget"
        assert pricing.status == "downgraded"

    def test_a_partial_batch_keeps_the_viable_quotes(self) -> None:
        materials = [
            _material(external_id="tee", unit_cost=4.50),
            _material(external_id="expensive", unit_cost=20.0),
        ]

        pricing = build_pricing(
            materials,
            "dtf_apparel",
            [],
            (1024, 1024),
            80,
            "premium",
            "Strong concept.",
        )

        assert pricing is not None
        assert pricing.status == "partial"
        assert pricing.quotes[0].recommended_price is not None
        assert pricing.quotes[1].recommended_price is None

    def test_no_materials_means_no_pricing(self) -> None:
        assert (
            build_pricing([], "dtf_apparel", [], (1024, 1024), 80, "standard", "No material.")
            is None
        )

    def test_the_snapshot_is_json_ready(self) -> None:
        pricing = build_pricing(
            [_material()], "dtf_apparel", [], (1024, 1024), 80, "standard", "A strong design."
        )

        assert pricing is not None
        data = pricing.to_dict()

        assert data["effective_positioning"] == "standard"
        assert data["quotes"][0]["recommended_price"] == 19.99
        assert data["quotes"][0]["print_size"]["required_pixels"] == {
            "width": 3600,
            "height": 3600,
        }
        assert data["quotes"][0]["size_prices"][0] == {"size": "S", "price": 19.99}
        assert data["quotes"][0]["price_breakdown"]["print_rate_per_sq_in"] == 0.005
