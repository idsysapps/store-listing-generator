"""The idea device of a design.

A brief can be varied in its silhouette (the design templates) and still be
the same picture: one object, standing still, on a plain field. The brief
prompt asked for "ONE idea, not a list of objects", and the model wrote
"a bold minimalist illustration of a running figure with a finisher medal".
There is no idea in that, only a subject (issue #240).

An idea device is *how* the scene is built, not what it is about. The closed
vocabulary below carries an instruction and evergreen examples, so the model
does not fall back on the first noun it thought of. Each brief of a batch is
given a device slot, so the batch covers the vocabulary instead of the model
picking the same safe device eight times.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEVICE_IDS",
    "IDEA_DEVICES",
    "IdeaDevice",
    "device_for",
    "idea_device_rules",
    "plan_device_slots",
]


@dataclass(frozen=True)
class IdeaDevice:
    """One way a scene can carry an idea."""

    id: str
    label: str
    instruction: str
    examples: tuple[str, ...]


IDEA_DEVICES: Final[dict[str, IdeaDevice]] = {
    "juxtaposition": IdeaDevice(
        id="juxtaposition",
        label="Juxtaposition",
        instruction="two things that do not belong together, together, doing something ordinary",
        examples=(
            "a goldfish in a business suit at a tiny desk",
            "a garden gnome driving a night train through a kitchen",
        ),
    ),
    "transformation": IdeaDevice(
        id="transformation",
        label="Transformation",
        instruction="the subject caught halfway through becoming something else",
        examples=(
            "a paper crane unfolding into a real crane",
            "a raincloud knitting itself into a scarf",
        ),
    ),
    "scale": IdeaDevice(
        id="scale",
        label="Scale",
        instruction="something tiny made monumental, or something monumental made tiny",
        examples=(
            "a teacup the size of a cathedral with a gardener at the rim",
            "a mountain range growing out of a matchbox",
        ),
    ),
    "narrative": IdeaDevice(
        id="narrative",
        label="Narrative",
        instruction="a moment with a before and an after implied, caught mid-action",
        examples=(
            "a raccoon caught mid-heist, pockets full of snacks, a torch beam approaching",
            "a tortoise crossing a finish line that is still being painted",
        ),
    ),
    "collection": IdeaDevice(
        id="collection",
        label="Collection",
        instruction="a set of related things arranged with a visual system",
        examples=(
            "nine houseplants in mismatched pots, each with a face",
            "a chart of twelve invented moons, each with a tiny flag",
        ),
    ),
    "pattern": IdeaDevice(
        id="pattern",
        label="Pattern",
        instruction="the motif repeated as a decorative system that covers the design",
        examples=(
            "a field of lightning bolts and daisies woven into a repeat",
            "a wallpaper of tiny submarines and teacups",
        ),
    ),
}

DEVICE_IDS: Final[tuple[str, ...]] = tuple(IDEA_DEVICES)


def device_for(name: str | None) -> IdeaDevice | None:
    """The device a brief named, or None when it named none we use."""
    if not name:
        return None
    return IDEA_DEVICES.get(name.strip().lower())


def idea_device_rules() -> str:
    """The idea-device rules, for the prompts that write the scene of a design."""
    lines = [
        "IDEA DEVICE - how the scene is built, not what it is about:",
        (
            "- The scene is a moment, a transformation or a world, not a still life of "
            "one object on a plain field."
        ),
    ]
    for device in IDEA_DEVICES.values():
        examples = "; ".join(f"'{example}'" for example in device.examples)
        lines.append(f"- {device.id}: {device.instruction}. For example {examples}.")
    lines.append(
        "- Each brief uses the idea_device of its brief slot in the input. If the "
        "device is 'pattern', the motif fills the design; otherwise the scene shows "
        "the idea happening, not the object sitting still."
    )
    return "\n".join(lines) + "\n"


def plan_device_slots(count: int, batch_id: str = "") -> list[str]:
    """``count`` device ids for a batch, seeded by the batch.

    A pass over the vocabulary comes first, so a batch of eight covers six
    devices and repeats two of them; a device is never handed out a third
    time until every device has had two slots.
    """
    rng = random.Random(f"{batch_id}|devices|{count}")
    picks: list[str] = []
    while len(picks) < count:
        pass_ = list(DEVICE_IDS)
        rng.shuffle(pass_)
        picks.extend(pass_)
    return picks[:count]
