"""Tests for the idea devices.

A design template decides the silhouette. The idea device decides what is
happening in the picture, so a batch is not eight still lifes of one object
(issue #240).
"""

from __future__ import annotations

import re

from store_listing.orchestration.brief_prompts import SEASONAL_WORDS
from store_listing.orchestration.idea_devices import (
    DEVICE_IDS,
    IDEA_DEVICES,
    device_for,
    idea_device_rules,
    plan_device_slots,
)


class TestTheVocabulary:
    def test_there_are_enough_devices_for_a_batch(self) -> None:
        assert len(DEVICE_IDS) >= 6

    def test_every_device_has_an_instruction_and_examples(self) -> None:
        for device_id, device in IDEA_DEVICES.items():
            assert device.instruction.strip(), device_id
            assert device.examples, device_id

    def test_the_devices_are_not_the_same_instruction(self) -> None:
        instructions = [device.instruction for device in IDEA_DEVICES.values()]

        assert len(set(instructions)) == len(instructions)

    def test_a_device_is_found_by_id(self) -> None:
        assert device_for("juxtaposition") is IDEA_DEVICES["juxtaposition"]

    def test_an_unknown_device_is_none(self) -> None:
        assert device_for("something_else") is None
        assert device_for(None) is None


class TestTheRules:
    def test_the_rules_name_every_device(self) -> None:
        rules = idea_device_rules()

        for device_id in DEVICE_IDS:
            assert device_id in rules

    def test_the_rules_say_the_scene_is_not_a_still_life(self) -> None:
        rules = idea_device_rules().lower()

        assert "still life" in rules
        assert "not a still life of one object" in rules

    def test_the_rules_carry_no_seasonal_anchor(self) -> None:
        # A seasonal example anchors every brief of the batch (issue #200).
        rules = idea_device_rules().lower()

        for word in SEASONAL_WORDS:
            assert not re.search(rf"\b{re.escape(word)}", rules), f"the rules mention {word}"


class TestTheSlots:
    def test_the_slots_are_deterministic(self) -> None:
        assert plan_device_slots(8, "2026-10-22") == plan_device_slots(8, "2026-10-22")

    def test_a_batch_of_eight_covers_the_vocabulary(self) -> None:
        slots = plan_device_slots(8, "2026-10-22")

        assert len(set(slots)) >= 6

    def test_no_device_runs_away_with_a_big_batch(self) -> None:
        slots = plan_device_slots(12, "2026-10-22")

        assert max(slots.count(device) for device in set(slots)) <= 2

    def test_one_brief_gets_one_device(self) -> None:
        slots = plan_device_slots(1, "custom")

        assert len(slots) == 1
        assert slots[0] in DEVICE_IDS

    def test_the_slots_are_a_rotation_and_not_a_constant(self) -> None:
        first = plan_device_slots(8, "2026-10-22")
        second = plan_device_slots(8, "2026-10-23")

        assert first != second
