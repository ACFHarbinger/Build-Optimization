"""
Pipeline: STS2 canonical card catalogue (SA2).

Ingests slaythespire.wiki.gg structured Lua modules into a *gitignored*
user-local cache, then merges a gitignored local overlay (add + override
by ``card_id``). The committed sample Ironclad JSON remains the test
fixture; this package is the runtime catalogue once the advisor is wired.

Never commits wiki dumps, card art, or screenshots.
"""

from .ids import aliases_for, card_ids, character_slug, display_name, name_slug
from .ingest import DEFAULT_MODULES, CatalogueFetchError, fetch_module, ingest_live, ingest_modules, module_api_url
from .lua_cards import cards_from_lua, parse_module_entries
from .models import WIKI_ATTRIBUTION, Catalogue, CatalogueCard
from .store import CatalogueStore, default_data_dir

__all__ = [
    "WIKI_ATTRIBUTION",
    "DEFAULT_MODULES",
    "Catalogue",
    "CatalogueCard",
    "CatalogueStore",
    "CatalogueFetchError",
    "aliases_for",
    "card_ids",
    "cards_from_lua",
    "character_slug",
    "default_data_dir",
    "display_name",
    "fetch_module",
    "ingest_live",
    "ingest_modules",
    "module_api_url",
    "name_slug",
    "parse_module_entries",
]
