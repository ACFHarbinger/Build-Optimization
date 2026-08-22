"""Live wiki ingest: ``python -m pipeline.catalogue``.

Writes the gitignored app-data cache (or ``$STS2_CATALOGUE_DIR``). Does not
modify the git worktree.
"""

from __future__ import annotations

import sys

from .ingest import ingest_live
from .store import CatalogueStore, default_data_dir


def main() -> int:
    store = CatalogueStore()
    print(f"catalogue dir: {default_data_dir()}", file=sys.stderr)
    cards = ingest_live(store)
    print(f"ingested {len(cards)} catalogue rows -> {store.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
