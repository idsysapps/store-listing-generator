"""Read-only report of the seasonal/evergreen mix of recent design briefs.

    python -m store_listing.orchestration.brief_mix_report --days 14

The report never writes: it reads the briefs of the last few days, works out
which calendar event each one belongs to, and prints the mix per batch. It is
the measurement that shows whether the pool is balanced or whether the
season has taken over the output.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from store_listing.ingest.trends.google_trends import DatabaseClient
from store_listing.orchestration.brief_mix import BriefMix, classify_themes, theme_of_queries

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 14


@dataclass
class BatchMix:
    """The mix of one batch of briefs."""

    batch_id: str
    total: int = 0
    seasonal: int = 0
    evergreen: int = 0
    by_theme: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.batch_id}: total={self.total} seasonal={self.seasonal} "
            f"evergreen={self.evergreen} themes={self.by_theme}"
        )


def group_brief_mix(
    links: list[tuple[str, str, str]],
) -> list[BatchMix]:
    """The mix of every batch in the brief/seed links.

    `links` holds (batch_id, concept, seed query) rows, one row per brief and
    the seed it was built from.
    """
    by_batch: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for batch_id, concept, seed_query in links:
        by_batch[batch_id][concept].append(seed_query)

    batches: list[BatchMix] = []
    for batch_id, briefs in by_batch.items():
        mix: BriefMix = classify_themes(theme_of_queries(queries) for queries in briefs.values())
        batches.append(
            BatchMix(
                batch_id=batch_id,
                total=mix.total,
                seasonal=mix.seasonal,
                evergreen=mix.evergreen,
                by_theme=mix.by_theme,
            )
        )
    return batches


def render(batches: list[BatchMix]) -> str:
    """The report as text."""
    lines = [batch.summary() for batch in batches]
    total = sum(batch.total for batch in batches)
    seasonal = sum(batch.seasonal for batch in batches)
    lines.append(f"ALL: total={total} seasonal={seasonal} evergreen={total - seasonal}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"how many days of briefs to look at (default {DEFAULT_DAYS})",
    )
    args = parser.parse_args(argv)

    links = DatabaseClient().list_recent_brief_seed_links(days=args.days)
    if not links:
        print(f"No briefs in the last {args.days} days.")
        return 0

    print(render(group_brief_mix(links)))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
