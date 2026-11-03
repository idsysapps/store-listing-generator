"""Recommended selling price and DTF print size for a design brief.

The model may choose a price positioning (budget, standard or premium), but it
never supplies the price. The price is deterministic: the blank cost, the DTF
print cost at a per-square-inch rate, the Amazon referral fee, the target
margin and a positioning floor/ceiling are all applied here.

The print size is also deterministic. It fits the rendered canvas aspect ratio
inside the substrate's print area, then reports the pixels needed at 300 DPI.
The pipeline does not resize or upscale the image; the operator upscales to the
reported target before printing (issue #295).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Final

__all__ = [
    "DEFAULT_POLICY",
    "POSITIONING_ORDER",
    "MaterialQuote",
    "PriceBreakdown",
    "PricingPolicy",
    "PricingSnapshot",
    "PrintSize",
    "PrintSpec",
    "allowed_positionings",
    "build_pricing",
    "estimated_prices",
    "fallback_positioning",
    "normalise_positioning",
    "policy_from_env",
    "print_spec_for",
    "quote_for_material",
    "suggest_print_size",
]

POSITIONING_ORDER: Final[tuple[str, ...]] = ("budget", "standard", "premium")
POSITIONING_RANK: Final[dict[str, int]] = {
    positioning: index for index, positioning in enumerate(POSITIONING_ORDER)
}
DEFAULT_POSITIONING: Final[str] = "standard"

# The rate the DTF operation charges for ink, film and powder, per square inch.
DEFAULT_PRINT_RATE: Final[Decimal] = Decimal("0.005")
# Amazon's apparel referral fee. The chart can override it for another plan.
DEFAULT_MARKETPLACE_FEE: Final[Decimal] = Decimal("0.15")
# The print file is upscaled by the operator before it reaches the printer.
DEFAULT_TARGET_DPI: Final[int] = 300
# Print sizes are rounded down to this step so the order stays printable.
PRINT_SIZE_STEP: Final[Decimal] = Decimal("0.1")

DEFAULT_MARGINS: Final[dict[str, Decimal]] = {
    "budget": Decimal("0.30"),
    "standard": Decimal("0.40"),
    "premium": Decimal("0.50"),
}
DEFAULT_FLOORS: Final[dict[str, Decimal]] = {
    "budget": Decimal("14.99"),
    "standard": Decimal("19.99"),
    "premium": Decimal("24.99"),
}
DEFAULT_SIZE_UPCHARGES: Final[dict[str, Decimal]] = {
    "2XL": Decimal("2.00"),
    "3XL": Decimal("3.00"),
}


@dataclass(frozen=True)
class PrintSpec:
    """The printable area of one DTF substrate, and its price ceiling."""

    key: str
    max_width_in: Decimal
    max_height_in: Decimal
    placement: str
    max_price: Decimal
    keywords: tuple[str, ...] = ()
    estimated: bool = False


PRINT_SPECS: Final[tuple[PrintSpec, ...]] = (
    PrintSpec(
        key="onesie",
        max_width_in=Decimal(8),
        max_height_in=Decimal(8),
        placement="full_front",
        max_price=Decimal("19.99"),
        keywords=("onesie", "bodysuit", "baby"),
    ),
    PrintSpec(
        key="hoodie",
        max_width_in=Decimal(12),
        max_height_in=Decimal(12),
        placement="full_front",
        max_price=Decimal("49.99"),
        keywords=("hoodie", "sweatshirt", "pullover"),
    ),
    PrintSpec(
        key="hat",
        max_width_in=Decimal(4),
        max_height_in=Decimal("2.5"),
        placement="front_panel",
        max_price=Decimal("24.99"),
        keywords=("hat", "cap", "beanie"),
    ),
    PrintSpec(
        key="tote",
        max_width_in=Decimal(10),
        max_height_in=Decimal(10),
        placement="full_front",
        max_price=Decimal("34.99"),
        keywords=("tote", "bag"),
    ),
    PrintSpec(
        key="t-shirt",
        max_width_in=Decimal(12),
        max_height_in=Decimal(16),
        placement="full_front",
        max_price=Decimal("29.99"),
        keywords=("t-shirt", "t shirt", "tshirt", "tee", "long sleeve", "tank", "top"),
    ),
)

DEFAULT_PRINT_SPEC: Final[PrintSpec] = PrintSpec(
    key="default",
    max_width_in=Decimal(12),
    max_height_in=Decimal(12),
    placement="full_front",
    max_price=Decimal("29.99"),
    estimated=True,
)


@dataclass(frozen=True)
class PricingPolicy:
    """The rates, margins and floors the price calculation uses."""

    print_rate_per_sq_in: Decimal = DEFAULT_PRINT_RATE
    marketplace_fee_rate: Decimal = DEFAULT_MARKETPLACE_FEE
    target_dpi: int = DEFAULT_TARGET_DPI
    margins: Mapping[str, Decimal] = field(default_factory=lambda: dict(DEFAULT_MARGINS))
    floors: Mapping[str, Decimal] = field(default_factory=lambda: dict(DEFAULT_FLOORS))
    size_upcharges: Mapping[str, Decimal] = field(
        default_factory=lambda: dict(DEFAULT_SIZE_UPCHARGES)
    )
    currency: str = "USD"
    policy_version: int = 1


DEFAULT_POLICY: Final[PricingPolicy] = PricingPolicy()


@dataclass(frozen=True)
class PrintSize:
    """The suggested physical print and the upscale target for it."""

    width_in: Decimal
    height_in: Decimal
    placement: str
    target_dpi: int
    required_pixels: tuple[int, int]
    source_pixels: tuple[int, int]
    upscale_factor: Decimal
    estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_in": _money(self.width_in),
            "height_in": _money(self.height_in),
            "placement": self.placement,
            "target_dpi": self.target_dpi,
            "required_pixels": {
                "width": self.required_pixels[0],
                "height": self.required_pixels[1],
            },
            "source_pixels": {
                "width": self.source_pixels[0],
                "height": self.source_pixels[1],
            },
            "upscale_factor": _rate(self.upscale_factor),
            "estimated": self.estimated,
        }


@dataclass(frozen=True)
class PriceBreakdown:
    """The costs behind one recommended price."""

    material_cost: Decimal
    print_cost: Decimal
    print_rate_per_sq_in: Decimal
    marketplace_fee: Decimal
    target_margin: Decimal
    actual_margin: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_cost": _money(self.material_cost),
            "print_cost": _money(self.print_cost),
            "print_rate_per_sq_in": _rate(self.print_rate_per_sq_in),
            "marketplace_fee": _money(self.marketplace_fee),
            "target_margin": _rate(self.target_margin),
            "actual_margin": _rate(self.actual_margin),
        }


@dataclass(frozen=True)
class MaterialQuote:
    """The recommended price and print for one recommended material."""

    external_id: str
    requested_positioning: str
    positioning: str
    status: str
    recommended_price: Decimal | None
    size_prices: Mapping[str, Decimal]
    print_size: PrintSize
    breakdown: PriceBreakdown
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_id": self.external_id,
            "requested_positioning": self.requested_positioning,
            "positioning": self.positioning,
            "status": self.status,
            "recommended_price": (
                _money(self.recommended_price) if self.recommended_price is not None else None
            ),
            "currency": self.currency,
            "size_prices": [
                {"size": size, "price": _money(price)} for size, price in self.size_prices.items()
            ],
            "print_size": self.print_size.to_dict(),
            "price_breakdown": self.breakdown.to_dict(),
        }


@dataclass(frozen=True)
class PricingSnapshot:
    """The positioning decision and the per-material quotes stored with a brief."""

    requested_positioning: str
    effective_positioning: str
    positioning_reason: str
    confidence: int
    target_margin: Decimal
    status: str
    currency: str
    policy_version: int
    print_rate_per_sq_in: Decimal
    quotes: tuple[MaterialQuote, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_positioning": self.requested_positioning,
            "effective_positioning": self.effective_positioning,
            "positioning_reason": self.positioning_reason,
            "confidence": self.confidence,
            "target_margin": _rate(self.target_margin),
            "status": self.status,
            "currency": self.currency,
            "policy_version": self.policy_version,
            "print_rate_per_sq_in": _rate(self.print_rate_per_sq_in),
            "quotes": [quote.to_dict() for quote in self.quotes],
        }


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _rate(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _env_decimal(name: str, default: Decimal) -> Decimal:
    value = _decimal(os.environ.get(name))
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def policy_from_env() -> PricingPolicy:
    """The policy, with the settings the chart can tune."""
    margins = {
        positioning: _env_decimal(
            f"PRICING_{positioning.upper()}_MARGIN", DEFAULT_MARGINS[positioning]
        )
        for positioning in POSITIONING_ORDER
    }
    floors = {
        positioning: _env_decimal(
            f"PRICING_{positioning.upper()}_FLOOR", DEFAULT_FLOORS[positioning]
        )
        for positioning in POSITIONING_ORDER
    }
    upcharges = {
        size: _env_decimal(f"PRICING_{size}_UPCHARGE", charge)
        for size, charge in DEFAULT_SIZE_UPCHARGES.items()
    }
    return PricingPolicy(
        print_rate_per_sq_in=_env_decimal("DTF_PRINT_COST_PER_SQ_IN", DEFAULT_PRINT_RATE),
        marketplace_fee_rate=_env_decimal("AMAZON_REFERRAL_FEE_RATE", DEFAULT_MARKETPLACE_FEE),
        target_dpi=_env_int("PRICING_TARGET_DPI", DEFAULT_TARGET_DPI),
        margins=margins,
        floors=floors,
        size_upcharges=upcharges,
        currency=os.environ.get("PRICING_CURRENCY", "USD") or "USD",
    )


def normalise_positioning(value: Any) -> str:
    """The positioning we understand, or standard when the model invented one."""
    if not isinstance(value, str):
        return DEFAULT_POSITIONING
    positioning = value.strip().lower()
    return positioning if positioning in POSITIONING_RANK else DEFAULT_POSITIONING


def allowed_positionings(confidence: float | None) -> tuple[str, ...]:
    """The tiers a design's sellability may carry.

    Confidence is the model's own sellability score. A weak design may only be
    budget, a medium one budget/standard, and a strong one may reach premium.
    The code never uses a tier above this set, whatever the model asked for.
    """
    try:
        score = float(confidence or 0)
    except (TypeError, ValueError):
        score = 0
    if score < 40:
        return ("budget",)
    if score < 70:
        return ("budget", "standard")
    return POSITIONING_ORDER


def fallback_positioning(confidence: float | None) -> str:
    """The positioning used when the pricing model is unavailable.

    A strong design defaults to standard, not premium: the model can ask for
    premium, but a failed call must not silently put every design on the
    premium floor.
    """
    try:
        score = float(confidence or 0)
    except (TypeError, ValueError):
        score = 0
    return "budget" if score < 40 else "standard"


def _material_text(material: Mapping[str, Any], specific_products: Sequence[str]) -> str:
    parts: list[str] = [
        str(material.get("substrate_type") or ""),
        str(material.get("name") or ""),
        str(material.get("material_composition") or ""),
    ]
    parts.extend(str(product) for product in specific_products)
    return " ".join(parts).lower().replace("_", " ").replace("-", " ")


def print_spec_for(
    product_type: str | None,
    material: Mapping[str, Any],
    specific_products: Sequence[str] = (),
) -> PrintSpec | None:
    """The DTF print area for a material, or None for another product type."""
    if product_type != "dtf_apparel":
        return None
    text = _material_text(material, specific_products)
    spec = DEFAULT_PRINT_SPEC
    for candidate in PRINT_SPECS:
        if any(keyword.replace("-", " ") in text for keyword in candidate.keywords):
            spec = candidate
            break

    attributes = material.get("attributes")
    if isinstance(attributes, Mapping):
        width = _decimal(attributes.get("print_width_in"))
        height = _decimal(attributes.get("print_height_in"))
        if width is not None and height is not None and width > 0 and height > 0:
            placement = attributes.get("print_placement")
            return replace(
                spec,
                max_width_in=width,
                max_height_in=height,
                placement=(
                    placement.strip()
                    if isinstance(placement, str) and placement.strip()
                    else spec.placement
                ),
                estimated=False,
            )
    return spec


def _round_down(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def suggest_print_size(
    canvas_size: tuple[int, int],
    spec: PrintSpec,
    *,
    policy: PricingPolicy = DEFAULT_POLICY,
) -> PrintSize:
    """Fit the canvas aspect ratio into the substrate's print area."""
    pixel_width, pixel_height = canvas_size
    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError("The design canvas needs a positive width and height")

    scale = min(
        spec.max_width_in / Decimal(pixel_width),
        spec.max_height_in / Decimal(pixel_height),
    )
    width = _round_down(Decimal(pixel_width) * scale, PRINT_SIZE_STEP)
    height = _round_down(Decimal(pixel_height) * scale, PRINT_SIZE_STEP)
    width = max(width, PRINT_SIZE_STEP)
    height = max(height, PRINT_SIZE_STEP)

    required_width = math.ceil(width * policy.target_dpi)
    required_height = math.ceil(height * policy.target_dpi)
    upscale = max(
        Decimal(required_width) / Decimal(pixel_width),
        Decimal(required_height) / Decimal(pixel_height),
    )

    return PrintSize(
        width_in=width,
        height_in=height,
        placement=spec.placement,
        target_dpi=policy.target_dpi,
        required_pixels=(required_width, required_height),
        source_pixels=(pixel_width, pixel_height),
        upscale_factor=upscale,
        estimated=spec.estimated,
    )


def _round_up_to_99(value: Decimal) -> Decimal:
    whole = value.to_integral_value(rounding=ROUND_FLOOR)
    candidate = whole + Decimal("0.99")
    if candidate < value:
        candidate += Decimal("1.00")
    return candidate


def _price_for_tier(
    material_cost: Decimal,
    print_cost: Decimal,
    positioning: str,
    spec: PrintSpec,
    policy: PricingPolicy,
) -> Decimal | None:
    margin = policy.margins.get(positioning)
    floor = policy.floors.get(positioning)
    if margin is None or floor is None:
        return None
    divisor = Decimal(1) - policy.marketplace_fee_rate - margin
    if divisor <= 0:
        return None
    price = _round_up_to_99(max((material_cost + print_cost) / divisor, floor))
    if price > spec.max_price:
        return None
    return price


def _normalise_size(value: Any) -> str:
    size = str(value).strip().upper().replace(" ", "")
    return {"XXL": "2XL", "XXXL": "3XL", "XXXXL": "4XL"}.get(size, size)


def _size_prices(
    base_price: Decimal,
    sizes: Any,
    policy: PricingPolicy,
) -> dict[str, Decimal]:
    if not isinstance(sizes, (list, tuple)):
        return {}
    prices: dict[str, Decimal] = {}
    for raw_size in sizes:
        size = _normalise_size(raw_size)
        if not size:
            continue
        prices[size] = base_price + policy.size_upcharges.get(size, Decimal(0))
    return prices


def _breakdown(
    material_cost: Decimal,
    print_cost: Decimal,
    price: Decimal,
    positioning: str,
    policy: PricingPolicy,
) -> PriceBreakdown:
    fee = (price * policy.marketplace_fee_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    margin = (price - material_cost - print_cost - fee) / price if price else Decimal(0)
    return PriceBreakdown(
        material_cost=material_cost,
        print_cost=print_cost,
        print_rate_per_sq_in=policy.print_rate_per_sq_in,
        marketplace_fee=fee,
        target_margin=policy.margins[positioning],
        actual_margin=margin,
    )


def quote_for_material(
    material: Mapping[str, Any],
    product_type: str | None,
    specific_products: Sequence[str],
    canvas_size: tuple[int, int],
    requested_positioning: str,
    confidence: float | None,
    policy: PricingPolicy = DEFAULT_POLICY,
) -> MaterialQuote | None:
    """The recommended price and print for one material, or None when unsuitable."""
    spec = print_spec_for(product_type, material, specific_products)
    if spec is None:
        return None
    material_cost = _decimal(material.get("unit_cost"))
    if material_cost is None or material_cost < 0:
        return None

    requested = normalise_positioning(requested_positioning)
    allowed = allowed_positionings(confidence)
    start_rank = min(POSITIONING_RANK[requested], max(POSITIONING_RANK[t] for t in allowed))
    tiers = [tier for tier in reversed(POSITIONING_ORDER) if POSITIONING_RANK[tier] <= start_rank]

    print_size = suggest_print_size(canvas_size, spec, policy=policy)
    print_cost = policy.print_rate_per_sq_in * print_size.width_in * print_size.height_in
    external_id = str(material.get("external_id") or "")

    for tier in tiers:
        price = _price_for_tier(material_cost, print_cost, tier, spec, policy)
        if price is None:
            continue
        status = "ok" if tier == requested and tier in allowed else "downgraded"
        return MaterialQuote(
            external_id=external_id,
            requested_positioning=requested,
            positioning=tier,
            status=status,
            recommended_price=price,
            size_prices=_size_prices(price, material.get("sizes"), policy),
            print_size=print_size,
            breakdown=_breakdown(material_cost, print_cost, price, tier, policy),
            currency=policy.currency,
        )

    return MaterialQuote(
        external_id=external_id,
        requested_positioning=requested,
        positioning="budget",
        status="no_viable_price",
        recommended_price=None,
        size_prices={},
        print_size=print_size,
        breakdown=PriceBreakdown(
            material_cost=material_cost,
            print_cost=print_cost,
            print_rate_per_sq_in=policy.print_rate_per_sq_in,
            marketplace_fee=Decimal(0),
            target_margin=policy.margins["budget"],
            actual_margin=Decimal(0),
        ),
        currency=policy.currency,
    )


def estimated_prices(
    material: Mapping[str, Any],
    product_type: str | None,
    specific_products: Sequence[str],
    canvas_size: tuple[int, int],
    policy: PricingPolicy = DEFAULT_POLICY,
) -> dict[str, float | None]:
    """The price each positioning would produce, for the pricing prompt."""
    spec = print_spec_for(product_type, material, specific_products)
    if spec is None:
        return {}
    material_cost = _decimal(material.get("unit_cost"))
    if material_cost is None or material_cost < 0:
        return {}
    print_size = suggest_print_size(canvas_size, spec, policy=policy)
    print_cost = policy.print_rate_per_sq_in * print_size.width_in * print_size.height_in
    return {
        tier: (
            _money(price)
            if (price := _price_for_tier(material_cost, print_cost, tier, spec, policy)) is not None
            else None
        )
        for tier in POSITIONING_ORDER
    }


def build_pricing(
    materials: Sequence[Mapping[str, Any]],
    product_type: str | None,
    specific_products: Sequence[str],
    canvas_size: tuple[int, int],
    confidence: float | None,
    requested_positioning: str,
    positioning_reason: str,
    policy: PricingPolicy = DEFAULT_POLICY,
) -> PricingSnapshot | None:
    """Build the brief-level pricing snapshot from its recommended materials."""
    requested = normalise_positioning(requested_positioning)
    quotes = tuple(
        quote
        for material in materials
        if isinstance(material, Mapping)
        if (
            quote := quote_for_material(
                material,
                product_type,
                specific_products,
                canvas_size,
                requested,
                confidence,
                policy,
            )
        )
        is not None
    )
    if not quotes:
        return None

    primary = quotes[0]
    if primary.status == "no_viable_price":
        status = "no_viable_price"
    elif any(quote.status == "no_viable_price" for quote in quotes):
        status = "partial"
    elif any(quote.status == "downgraded" for quote in quotes):
        status = "downgraded"
    else:
        status = "ok"

    return PricingSnapshot(
        requested_positioning=requested,
        effective_positioning=primary.positioning,
        positioning_reason=positioning_reason,
        confidence=int(confidence or 0),
        target_margin=primary.breakdown.target_margin,
        status=status,
        currency=policy.currency,
        policy_version=policy.policy_version,
        print_rate_per_sq_in=policy.print_rate_per_sq_in,
        quotes=quotes,
    )
