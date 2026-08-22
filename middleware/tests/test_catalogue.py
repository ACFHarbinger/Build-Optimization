"""Tests for pipeline.catalogue (SA2) — wiki ingest, overlay, stable ids."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.item import Rarity, Slot
from pipeline.catalogue import (
    WIKI_ATTRIBUTION,
    CatalogueCard,
    CatalogueFetchError,
    CatalogueStore,
    card_ids,
    cards_from_lua,
    ingest_modules,
    module_api_url,
    name_slug,
)
from pipeline.catalogue.ids import character_slug, display_name

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wiki" / "sts2_lua_snippet.lua"


@pytest.fixture
def lua_source() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class TestIds:
    def test_strike_ids_are_character_tagged_and_plus_is_distinct(self) -> None:
        base, upgraded = card_ids("Strike (Ironclad)", "Ironclad")
        assert base == "ironclad:strike"
        assert upgraded == "ironclad:strike+"
        assert base != upgraded

    def test_display_name_strips_wiki_disambiguator(self) -> None:
        assert display_name("Strike (Ironclad)") == "Strike"
        assert display_name("Strike (Ironclad)", upgraded=True) == "Strike+"

    def test_regent_drops_leading_the(self) -> None:
        assert character_slug("The Regent") == "regent"

    def test_slug_handles_punctuation(self) -> None:
        assert name_slug("One-Two Punch") == "one_two_punch"
        assert name_slug("Pact's End") == "pacts_end"


class TestLuaParse:
    def test_base_and_upgrade_rows(self, lua_source: str) -> None:
        cards = cards_from_lua(lua_source)
        by_id = {c.card_id: c for c in cards}
        strike = by_id["ironclad:strike"]
        strike_up = by_id["ironclad:strike+"]
        assert strike.name == "Strike"
        assert strike.character == "ironclad"
        assert strike.slot == "attack"
        assert strike.stats["attack"] == pytest.approx(6.0)
        assert strike_up.stats["attack"] == pytest.approx(9.0)
        assert strike.upgraded is False
        assert strike_up.upgraded is True
        assert strike_up.base_id == "ironclad:strike"
        assert "Strike (Ironclad)" in strike.aliases

    def test_strength_and_multi_hit(self, lua_source: str) -> None:
        by_id = {c.card_id: c for c in cards_from_lua(lua_source)}
        assert by_id["ironclad:inflame"].stats["strength_gain"] == pytest.approx(2.0)
        assert by_id["ironclad:inflame+"].stats["strength_gain"] == pytest.approx(3.0)
        assert "strength" in by_id["ironclad:inflame"].tags
        assert by_id["ironclad:twin_strike"].stats["multi_hit"] == pytest.approx(2.0)

    def test_character_tag_from_color_not_module_default(self, lua_source: str) -> None:
        by_id = {c.card_id: c for c in cards_from_lua(lua_source)}
        shrug = by_id["silent:shrug_it_off"]
        assert shrug.character == "silent"
        assert shrug.stats["block"] == pytest.approx(8.0)

    def test_image_and_text_not_persisted(self, lua_source: str) -> None:
        strike = next(c for c in cards_from_lua(lua_source) if c.card_id == "ironclad:strike")
        dumped = strike.to_dict()
        assert "Image" not in dumped
        assert "text" not in dumped
        assert "wiki_text" not in dumped


class TestOverlayAndCache:
    def test_ingest_uses_fetcher_and_writes_gitignored_cache(self, lua_source: str, tmp_path: Path) -> None:
        store = CatalogueStore(tmp_path)
        cards = ingest_modules(
            store,
            modules=("Module:Cards/StS2_data/Ironclad",),
            fetcher=lambda _title: lua_source,
        )
        assert store.cache_path.is_file()
        assert store.cache_path.parent == tmp_path
        assert any(c.card_id == "ironclad:strike+" for c in cards)
        payload = store.cache_path.read_text(encoding="utf-8")
        assert "slaythespire.wiki.gg" in payload
        assert "StS2_Ironclad-Strike.png" not in payload

    def test_overlay_overrides_and_adds(self, lua_source: str, tmp_path: Path) -> None:
        store = CatalogueStore(tmp_path)
        ingest_modules(store, modules=("Ironclad",), fetcher=lambda _t: lua_source)
        store.upsert_overlay(
            CatalogueCard(
                card_id="ironclad:strike",
                name="Strike",
                character="ironclad",
                slot="attack",
                cost=1.0,
                rarity="basic",
                stats={"attack": 99.0},
                tags=["custom"],
            )
        )
        store.upsert_overlay(
            CatalogueCard(
                card_id="ironclad:custom_stub",
                name="Custom Stub",
                character="ironclad",
                slot="skill",
                cost=0.0,
                rarity="rare",
                stats={"block": 4.0},
                tags=["overlay"],
            )
        )
        catalogue = store.load()
        by_id = catalogue.index()
        assert by_id["ironclad:strike"].stats["attack"] == pytest.approx(99.0)
        assert "custom" in by_id["ironclad:strike"].tags
        assert "custom_stub" in by_id["ironclad:custom_stub"].card_id
        item = by_id["ironclad:strike"].to_item()
        assert item.item_id == "ironclad:strike"
        assert item.slot == Slot.ATTACK
        assert item.rarity == Rarity.COMMON

    def test_empty_parse_is_an_error(self, tmp_path: Path) -> None:
        store = CatalogueStore(tmp_path)
        with pytest.raises(CatalogueFetchError):
            ingest_modules(store, modules=("Empty",), fetcher=lambda _t: "return {}")


class TestApiUrlAndAttribution:
    def test_module_url_targets_wikigg_api(self) -> None:
        url = module_api_url("Module:Cards/StS2_data/Ironclad")
        assert url.startswith("https://slaythespire.wiki.gg/api.php?")
        assert "StS2" in url

    def test_attribution_mentions_wiki_and_forbids_art_claim(self) -> None:
        assert "slaythespire.wiki.gg" in WIKI_ATTRIBUTION
        assert "CC BY-SA" in WIKI_ATTRIBUTION
        assert "does not redistribute" in WIKI_ATTRIBUTION
