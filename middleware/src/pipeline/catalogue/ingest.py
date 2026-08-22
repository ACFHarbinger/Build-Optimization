"""Fetch slaythespire.wiki.gg Lua card modules into the local cache.

Never writes into the git worktree by default. Tests inject a fetcher so
CI does not hit the network. A live ingest is ``python -m pipeline.catalogue``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from json import JSONDecodeError
from typing import Callable, Dict, Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .lua_cards import cards_from_lua
from .models import WIKI_ATTRIBUTION, CatalogueCard
from .store import CatalogueStore

WIKI_ORIGIN = "https://slaythespire.wiki.gg"
API_PATH = "/api.php"
USER_AGENT = (
    "Build-Optimization/0.1 (https://github.com/ACFHarbinger/Build-Optimization; "
    "STS2 catalogue ingest; +https://slaythespire.wiki.gg)"
)

# Structured Lua modules (not HTML infobox pages). Underscores are canonical
# MediaWiki titles; the API accepts spaces too.
DEFAULT_MODULES: Sequence[str] = (
    "Module:Cards/StS2 data/Ironclad",
    "Module:Cards/StS2 data/Silent",
    "Module:Cards/StS2 data/Defect",
    "Module:Cards/StS2 data/Necrobinder",
    "Module:Cards/StS2 data/Regent",
    "Module:Cards/StS2 data/Colorless",
)

Fetcher = Callable[[str], str]


class CatalogueFetchError(RuntimeError):
    """Raised when a wiki module cannot be retrieved or is empty."""


def module_api_url(title: str) -> str:
    query = urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
        }
    )
    return f"{WIKI_ORIGIN}{API_PATH}?{query}"


def _read_url(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_module(title: str, timeout: float = 30.0) -> str:
    """GET one Lua module via the MediaWiki API, falling back to ``action=raw``."""
    try:
        body = json.loads(_read_url(module_api_url(title), timeout))
        pages = body.get("query", {}).get("pages", [])
        if pages and not pages[0].get("missing"):
            revisions = pages[0].get("revisions") or []
            if revisions:
                slot = revisions[0].get("slots", {}).get("main", {})
                content = slot.get("content") or revisions[0].get("content")
                if content:
                    return str(content)
    except (HTTPError, URLError, JSONDecodeError, KeyError, IndexError, TypeError):
        pass
    raw_title = title.replace(" ", "_")
    raw_url = f"{WIKI_ORIGIN}/wiki/{raw_title}?action=raw"
    text = _read_url(raw_url, timeout)
    if not text.strip() or text.lstrip().startswith("<!DOCTYPE"):
        raise CatalogueFetchError(f"empty or HTML response for {title}")
    return text


def ingest_modules(
    store: CatalogueStore,
    modules: Sequence[str] = DEFAULT_MODULES,
    fetcher: Optional[Fetcher] = None,
    rate_limit_s: float = 1.0,
) -> List[CatalogueCard]:
    """Fetch + parse modules and write the gitignored wiki cache.

    Args:
        store: Cache destination (app-data dir by default).
        modules: MediaWiki module titles.
        fetcher: Injected ``title -> lua source`` (tests). Live ingest uses
            :func:`fetch_module` when this is omitted.
        rate_limit_s: Pause between live requests. Ignored for a custom fetcher.
    """
    get = fetcher or fetch_module
    cards: List[CatalogueCard] = []
    fetched: Dict[str, int] = {}
    for index, title in enumerate(modules):
        if fetcher is None and index > 0 and rate_limit_s > 0:
            time.sleep(rate_limit_s)
        lua = get(title)
        parsed = cards_from_lua(lua, default_character=_character_from_title(title))
        fetched[title] = len(parsed)
        if not parsed:
            raise CatalogueFetchError(f"parsed zero cards from {title}")
        cards.extend(parsed)
    source = {
        "wiki": WIKI_ORIGIN,
        "modules": list(modules),
        "cards_per_module": fetched,
        "attribution": WIKI_ATTRIBUTION,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    store.write_cache(cards, source=source)
    return store.load_cache()


def _character_from_title(title: str) -> str:
    tail = title.rsplit("/", 1)[-1]
    return tail.lower()


def ingest_live(store: Optional[CatalogueStore] = None, modules: Optional[Iterable[str]] = None) -> List[CatalogueCard]:
    """Networked ingest used by ``python -m pipeline.catalogue``."""
    return ingest_modules(
        store or CatalogueStore(),
        modules=tuple(modules) if modules is not None else DEFAULT_MODULES,
        fetcher=None,
    )
