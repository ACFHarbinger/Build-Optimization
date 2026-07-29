"""
Warhammer 40,000: Darktide data sources.

Provides two sources that share a unified Darktide domain mapping:

DarktideFileSource
    Extends FileSource with Darktide-specific rarity names
    (profane → COMMON, redeemed → UNCOMMON, anointed → RARE,
    exalted → EPIC, transcendent → LEGENDARY) and a "ranged"
    slot alias that maps ranged weapons to Slot.RING_1.

    Darktide has five gear slots:
        melee  weapon  → Slot.WEAPON
        ranged weapon  → Slot.RING_1   (repurposed)
        curio  slot 1  → Slot.ACCESSORY_1
        curio  slot 2  → Slot.ACCESSORY_2
        curio  slot 3  → Slot.AMULET

DarktideScraper
    Subclasses WebScraperSource to scrape Darktide item data from
    community databases.  The default target is ratemyweapon.com,
    but the base_url is a constructor parameter so any mirror or
    alternative source can be substituted.

    When a network request fails (or requests/beautifulsoup4 are not
    installed), the scraper transparently falls back to the bundled
    sample JSON via DarktideFileSource so the pipeline never stalls.

Usage::

    # Local file (always available)
    from data.datasets.games.darktide_scraper import DarktideFileSource
    source = DarktideFileSource("src/data/sample/darktide.json")
    items     = source.fetch_items()
    synergies = source.fetch_synergies()

    # Web scraper with local fallback
    from data.datasets.games.darktide_scraper import DarktideScraper
    scraper = DarktideScraper(
        base_url="https://darktide.gameslantern.com",
        fallback_json="src/data/sample/darktide.json",
        cache_dir=".cache/darktide",
    )
    items = scraper.fetch_items()
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional

from core.item import Item, Rarity, Slot
from core.synergy import SynergyRule

from .file_source import FileSource
from .scraper import WebScraperSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Darktide domain mappings
# ---------------------------------------------------------------------------

#: Maps Darktide rarity tier names → Rarity enum values.
DARKTIDE_RARITY_MAP: Dict[str, Rarity] = {
    "profane": Rarity.COMMON,
    "redeemed": Rarity.UNCOMMON,
    "anointed": Rarity.RARE,
    "exalted": Rarity.EPIC,
    "transcendent": Rarity.LEGENDARY,
}

#: Maps Darktide slot names → Slot enum values.
#: "ranged" repurposes RING_1 since Darktide has a dedicated ranged slot
#: that the generic Slot enum does not model directly.
DARKTIDE_SLOT_MAP: Dict[str, Slot] = {
    "weapon": Slot.WEAPON,  # melee weapon
    "ranged": Slot.RING_1,  # ranged weapon (repurposed)
    "accessory_1": Slot.ACCESSORY_1,  # curio slot 1
    "accessory_2": Slot.ACCESSORY_2,  # curio slot 2
    "amulet": Slot.AMULET,  # curio slot 3
}


# ---------------------------------------------------------------------------
# DarktideFileSource
# ---------------------------------------------------------------------------


class DarktideFileSource(FileSource):
    """
    File-based source for Darktide item and synergy data.

    Extends FileSource by merging the Darktide rarity tier names and the
    ``"ranged"`` slot alias into the lookup maps used during JSON parsing.

    Args:
        items_path:     Path to the Darktide JSON or CSV data file.
        synergies_path: Optional separate synergies file; defaults to reading
                        synergies from the items JSON.
    """

    # Override class-level maps to include Darktide names alongside standards
    SLOT_MAP: Dict[str, Slot] = {**FileSource.SLOT_MAP, **DARKTIDE_SLOT_MAP}
    RARITY_MAP: Dict[str, Rarity] = {**FileSource.RARITY_MAP, **DARKTIDE_RARITY_MAP}


# ---------------------------------------------------------------------------
# DarktideScraper
# ---------------------------------------------------------------------------


class DarktideScraper(WebScraperSource):
    """
    Web scraper for Warhammer 40,000: Darktide community databases.

    Targets Darktide item databases (default: ratemyweapon.com layout) and
    parses weapon modifiers, curio traits, and Blessing tags into the
    standard Item / SynergyRule domain objects.

    When the network is unreachable or the required libraries (requests,
    beautifulsoup4) are not installed, the scraper falls back to a local
    JSON file transparently.

    Args:
        base_url:       Root URL of the community database to scrape.
        fallback_json:  Local JSON path used when the scrape fails.
                        Defaults to the bundled sample file.
        rate_limit:     Minimum seconds between HTTP requests (default 1.5).
        cache_dir:      Directory for caching raw HTML pages (optional).
    """

    #: Darktide-aware SLOT_MAP and RARITY_MAP used during parsing
    SLOT_MAP = DARKTIDE_SLOT_MAP
    RARITY_MAP = {**FileSource.RARITY_MAP, **DARKTIDE_RARITY_MAP}

    #: Stats expected per item category; unknown columns are still included
    #: but down-weighted in scoring.
    WEAPON_STATS = ("damage", "stopping_power", "sweep", "mobility", "penetration")
    RANGED_STATS = ("damage", "range", "mobility", "stopping_power", "penetration")
    CURIO_STATS = (
        "toughness_regeneration",
        "health_bonus",
        "stamina",
        "sprint_recovery",
        "damage_reduction",
        "coherency",
    )

    def __init__(
        self,
        base_url: str = "https://darktide.gameslantern.com",
        fallback_json: str = "src/data/sample/darktide.json",
        rate_limit: float = 1.5,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(base_url=base_url, rate_limit=rate_limit, cache_dir=cache_dir)
        self._fallback = DarktideFileSource(items_path=fallback_json)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_items(self) -> List[Item]:
        """
        Fetch Darktide items from the community database.

        Attempts to scrape melee weapons, ranged weapons, and curios from
        the configured base_url.  Falls back to the local JSON if the
        scrape fails for any reason.

        Returns:
            List of Item objects with Darktide stats, rarities, and tags.
        """
        try:
            melee = self._scrape_weapons(ranged=False)
            ranged = self._scrape_weapons(ranged=True)
            curios = self._scrape_curios()
            items = melee + ranged + curios
            if not items:
                raise ValueError("Scraper returned no items")
            logger.info("DarktideScraper: fetched %d items from %s", len(items), self.base_url)
            return items
        except Exception as exc:
            logger.warning("DarktideScraper: scrape failed (%s) — using fallback JSON.", exc)
            return self._fallback.fetch_items()

    def fetch_synergies(self) -> List[SynergyRule]:
        """
        Return Darktide Blessing synergy rules.

        Blessing interactions are derived from known game mechanics rather
        than scraped, since they are expressed as relational bonus tables
        that are not easily parsed from HTML.

        Returns:
            List of SynergyRule objects covering all tracked Blessings.
        """
        # Synergies are defined in the bundled JSON alongside items
        return self._fallback.fetch_synergies()

    # ------------------------------------------------------------------
    # Scraping implementation
    # ------------------------------------------------------------------

    def _scrape_weapons(self, ranged: bool = False) -> List[Item]:
        """
        Scrape weapon listing pages.

        Args:
            ranged: If True, targets ranged weapon pages; melee otherwise.

        Returns:
            List of partially-parsed Item objects.
        """
        category = "ranged" if ranged else "melee"
        slot_key = "ranged" if ranged else "weapon"
        slot = self.SLOT_MAP[slot_key]

        try:
            html = self._fetch_page(f"/weapons/{category}")
            soup = self._parse_html(html)
        except NotImplementedError:
            return []

        items: List[Item] = []
        # Each weapon card is typically a <div class="weapon-card"> element.
        # The exact selector depends on the target site's HTML structure.
        cards = soup.select(".weapon-card, .item-card, [data-item-type='weapon']")
        for card in cards:
            item = self._parse_weapon_card(card, slot, ranged=ranged)
            if item is not None:
                items.append(item)

        logger.debug("_scrape_weapons(ranged=%s): parsed %d items", ranged, len(items))
        return items

    def _scrape_curios(self) -> List[Item]:
        """
        Scrape the curio listing page.

        Returns:
            List of curio Item objects across ACCESSORY_1/2 and AMULET slots.
        """
        try:
            html = self._fetch_page("/curios")
            soup = self._parse_html(html)
        except NotImplementedError:
            return []

        items: List[Item] = []
        curio_slots = [Slot.ACCESSORY_1, Slot.ACCESSORY_2, Slot.AMULET]
        slot_idx = 0

        cards = soup.select(".curio-card, .item-card, [data-item-type='curio']")
        for card in cards:
            # Rotate through the three curio slots
            slot = curio_slots[slot_idx % len(curio_slots)]
            item = self._parse_curio_card(card, slot)
            if item is not None:
                items.append(item)
                slot_idx += 1

        logger.debug("_scrape_curios: parsed %d items", len(items))
        return items

    # ------------------------------------------------------------------
    # Card parsers
    # ------------------------------------------------------------------

    def _parse_weapon_card(
        self,
        card: "Any",  # BeautifulSoup Tag
        slot: Slot,
        ranged: bool,
    ) -> Optional[Item]:
        """
        Parse a single weapon card element into an Item.

        Expected HTML structure (generic, site-dependent)::

            <div class="weapon-card"
                 data-rarity="exalted"
                 data-level="20"
                 data-rating="380">
              <h3 class="weapon-name">Power Sword Mk IV</h3>
              <ul class="modifiers">
                <li data-stat="damage">74%</li>
                <li data-stat="stopping_power">58%</li>
                ...
              </ul>
              <ul class="blessings">
                <li>Power Cycler</li>
                <li>Heavy Hitter</li>
              </ul>
            </div>

        Args:
            card:   BeautifulSoup Tag for the weapon card.
            slot:   Pre-resolved Slot enum value.
            ranged: True if this is a ranged weapon card.

        Returns:
            Item if parsing succeeds; None otherwise.
        """
        try:
            name_el = card.select_one(".weapon-name, .item-name, h3")
            name = name_el.get_text(strip=True) if name_el else card.get("data-name", "")
            if not name:
                return None

            rarity_str = card.get("data-rarity", "anointed").lower()
            level_str = card.get("data-level", "1")
            rating_str = card.get("data-rating", "200")

            rarity = self.RARITY_MAP.get(rarity_str, Rarity.COMMON)
            level = int(level_str) if level_str.isdigit() else 1

            # Rating [1..400] → Ordo Dockets cost approximation
            rating = int(rating_str) if rating_str.isdigit() else 200
            cost = self._rating_to_cost(rating, rarity)

            # Parse modifier stats from <li data-stat="…">value%</li>
            stats: Dict[str, float] = {}
            stat_keys = self.RANGED_STATS if ranged else self.WEAPON_STATS
            for li in card.select("[data-stat]"):
                key = li.get("data-stat", "").lower().replace(" ", "_")
                if key in stat_keys:
                    val_text = li.get_text(strip=True).rstrip("%")
                    with contextlib.suppress(ValueError):
                        stats[key] = float(val_text)

            if not stats:
                # Fallback: assign mid-range defaults based on rarity tier
                stats = self._default_weapon_stats(rarity, ranged)

            # Build tags: weapon type + blessing names
            weapon_type = card.get("data-weapon-type", "weapon").lower().replace(" ", "_")
            tags = {("ranged" if ranged else "melee"), weapon_type}
            for blessing_el in card.select(".blessing, [data-blessing]"):
                blessing = blessing_el.get_text(strip=True).lower().replace(" ", "_")
                tags.add(blessing)

            return Item(
                name=name,
                slot=slot,
                stats=stats,
                cost=cost,
                rarity=rarity,
                level=min(level, 30),
                tags=frozenset(tags),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse weapon card: %s", exc)
            return None

    def _parse_curio_card(
        self,
        card: Any,  # BeautifulSoup Tag
        slot: Slot,
    ) -> Optional[Item]:
        """
        Parse a single curio card element into an Item.

        Args:
            card: BeautifulSoup Tag for the curio card.
            slot: Pre-resolved Slot enum value.

        Returns:
            Item if parsing succeeds; None otherwise.
        """
        try:
            name_el = card.select_one(".curio-name, .item-name, h3")
            name = name_el.get_text(strip=True) if name_el else card.get("data-name", "")
            if not name:
                return None

            rarity_str = card.get("data-rarity", "anointed").lower()
            level_str = card.get("data-level", "1")
            rating_str = card.get("data-rating", "200")

            rarity = self.RARITY_MAP.get(rarity_str, Rarity.COMMON)
            level = int(level_str) if level_str.isdigit() else 1
            rating = int(rating_str) if rating_str.isdigit() else 200
            cost = self._rating_to_cost(rating, rarity)

            stats: Dict[str, float] = {}
            for li in card.select("[data-stat]"):
                key = li.get("data-stat", "").lower().replace(" ", "_")
                if key in self.CURIO_STATS:
                    val_text = li.get_text(strip=True).rstrip("%")
                    with contextlib.suppress(ValueError):
                        stats[key] = float(val_text)

            if not stats:
                stats = self._default_curio_stats(rarity)

            tags: set = {"curio"}
            for trait_el in card.select(".trait, [data-trait]"):
                trait = trait_el.get_text(strip=True).lower().replace(" ", "_")
                tags.add(trait)

            return Item(
                name=name,
                slot=slot,
                stats=stats,
                cost=cost,
                rarity=rarity,
                level=min(level, 30),
                tags=frozenset(tags),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse curio card: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rating_to_cost(rating: int, rarity: Rarity) -> float:
        """
        Approximate Ordo Docket cost from item rating and rarity tier.

        Darktide's item shop prices scale with both base rating and rarity.
        This formula mirrors observed in-game pricing for guidance.

        Args:
            rating: Item rating (1–400).
            rarity: Rarity tier enum.

        Returns:
            Estimated cost in Ordo Dockets.
        """
        # Base cost grows quadratically with rating
        base = 500.0 + (rating / 400.0) ** 2 * 10_000.0
        # Rarity multiplier (profane → ×0.5, transcendent → ×1.5)
        rarity_mult = {
            Rarity.COMMON: 0.50,
            Rarity.UNCOMMON: 0.65,
            Rarity.RARE: 0.85,
            Rarity.EPIC: 1.10,
            Rarity.LEGENDARY: 1.50,
        }.get(rarity, 1.0)
        return round(base * rarity_mult, -2)  # round to nearest 100 Dockets

    @staticmethod
    def _default_weapon_stats(rarity: Rarity, ranged: bool) -> Dict[str, float]:
        """Fallback mid-range stats when the scraper finds no modifier data."""
        tier = {1: 30.0, 2: 42.0, 3: 55.0, 4: 68.0, 5: 82.0}[rarity.value]
        offset = tier * 0.15
        if ranged:
            return {
                "damage": tier,
                "range": tier + offset,
                "mobility": tier - offset,
                "stopping_power": tier - offset * 2,
                "penetration": tier,
            }
        return {
            "damage": tier,
            "stopping_power": tier,
            "sweep": tier - offset,
            "mobility": tier - offset,
            "penetration": tier - offset,
        }

    @staticmethod
    def _default_curio_stats(rarity: Rarity) -> Dict[str, float]:
        """Fallback mid-range stats when the scraper finds no curio trait data."""
        tier = {1: 8.0, 2: 15.0, 3: 25.0, 4: 38.0, 5: 55.0}[rarity.value]
        return {"toughness_regeneration": tier, "health_bonus": tier * 0.6, "stamina": tier * 0.5}
