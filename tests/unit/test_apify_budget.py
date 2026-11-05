"""Rough Apify cost of a harvest capped at 100 results."""

from decimal import Decimal

from store_listing.ingest.trends.apify_budget import (
    CHARGE_CAP_USD,
    HARVEST_RESULT_CAP,
    PAID_SEED_LIMIT,
    all_in_monthly_usd,
    format_all_in,
    format_budget,
    per_month_usd,
    per_run_usd,
    per_target_limit,
    select_paid_seeds,
)


def test_result_cap_and_charge_cap() -> None:
    assert HARVEST_RESULT_CAP == 100
    assert PAID_SEED_LIMIT == 20
    assert CHARGE_CAP_USD == Decimal("0.50")


def test_pinterest_run_is_about_thirty_five_cents() -> None:
    assert per_run_usd("pinterest") == Decimal("0.35")
    assert per_month_usd("pinterest") == Decimal("2.99")


def test_capped_sources_stay_under_the_monthly_credit() -> None:
    assert per_run_usd("x") == Decimal("0.015")
    assert per_month_usd("x") == Decimal("0.13")
    assert per_run_usd("tiktok") == Decimal("0.50")
    assert per_month_usd("tiktok") == Decimal("2.17")
    assert per_run_usd("instagram") == Decimal("0.26")
    assert per_month_usd("instagram") == Decimal("1.13")
    assert all_in_monthly_usd() == Decimal("6.42")


def test_budget_lines_the_scheduler_can_print() -> None:
    assert format_budget("pinterest") == (
        "Apify pinterest harvest: up to 100 results, about $0.35 this run, about $2.99 per month"
    )
    assert format_all_in() == "Apify all-in at 100 results per harvest: about $6.42 per month"


def test_paid_seed_slice_keeps_the_first_twenty_unique_seeds() -> None:
    seeds = [f"seed {index}" for index in range(25)]
    seeds.insert(3, "seed 1")

    assert select_paid_seeds(seeds) == [f"seed {index}" for index in range(20)]


def test_per_target_limit_splits_the_harvest_cap() -> None:
    assert per_target_limit(1) == 100
    assert per_target_limit(20) == 5
    assert per_target_limit(4) == 25


def test_per_target_env_can_only_lower_the_cap(monkeypatch) -> None:
    monkeypatch.setenv("TIKTOK_RESULTS_PER_PAGE", "5")

    assert per_target_limit(1, "TIKTOK_RESULTS_PER_PAGE") == 5
    assert per_target_limit(20, "TIKTOK_RESULTS_PER_PAGE") == 5
