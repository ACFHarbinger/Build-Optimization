"""
Game API data source skeleton for retrieving items from game REST APIs.

This is a skeleton implementation — subclass and implement for specific games.
"""

import logging
from typing import Any, Dict, List, Optional

from src.core.item import Item, Rarity, Slot
from src.core.synergy import SynergyRule

from .base import DataSource

logger = logging.getLogger(__name__)


class GameAPISource(DataSource):
    """Retrieve items and synergies from a game REST API.

    This is a skeleton — subclass for specific game APIs like:
    - Path of Exile (poe.ninja API)
    - Genshin Impact (Enka.Network API)
    - League of Legends (Riot Data Dragon)
    - Diablo IV (d4builds API)

    Attributes:
        base_url: API base URL.
        api_key: Optional authentication key.
        timeout: Request timeout in seconds.
        cache_ttl: Cache time-to-live in seconds (0 = no cache).
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        cache_ttl: int = 3600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}

    def fetch_items(self) -> List[Item]:
        """Fetch items from the API.

        Override this method for specific game API implementations.

        Returns:
            List of Item objects parsed from the API response.
        """
        raw = self._get("/items")
        return self._parse_items(raw)

    def fetch_synergies(self) -> List[SynergyRule]:
        """Fetch synergy rules from the API.

        Override this method for specific game API implementations.

        Returns:
            List of SynergyRule objects.
        """
        raw = self._get("/synergies")
        return self._parse_synergies(raw)

    # ------------------------------------------------------------------
    # HTTP helpers (skeleton — uses requests when available)
    # ------------------------------------------------------------------

    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request to the API.

        Args:
            endpoint: API endpoint path (e.g., "/items").

        Returns:
            Parsed JSON response.

        Raises:
            NotImplementedError: When requests library is not available
                or endpoint is not configured.
        """
        url = f"{self.base_url}{endpoint}"

        # Check cache
        if url in self._cache:
            logger.debug(f"Cache hit for {url}")
            return self._cache[url]

        try:
            import requests

            headers: Dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Cache result
            if self.cache_ttl > 0:
                self._cache[url] = data

            return data

        except ImportError:
            logger.warning("requests library not installed. Install with: pip install requests")
            raise NotImplementedError("Install 'requests' to use GameAPISource: pip install requests")
        except Exception as e:
            logger.error(f"API request failed for {url}: {e}")
            raise

    # ------------------------------------------------------------------
    # Parsers (override for specific game APIs)
    # ------------------------------------------------------------------

    def _parse_items(self, raw: Dict[str, Any]) -> List[Item]:
        """Parse raw API response into Item objects.

        Override this for game-specific response formats.
        Default assumes a standard format with 'items' key.
        """
        items: List[Item] = []
        for entry in raw.get("items", []):
            items.append(
                Item(
                    name=entry.get("name", "Unknown"),
                    slot=Slot[entry.get("slot", "WEAPON").upper()],
                    stats=entry.get("stats", {}),
                    cost=float(entry.get("cost", 0)),
                    rarity=Rarity[entry.get("rarity", "COMMON").upper()],
                    level=int(entry.get("level", 1)),
                    tags=frozenset(entry.get("tags", [])),
                )
            )
        return items

    def _parse_synergies(self, raw: Dict[str, Any]) -> List[SynergyRule]:
        """Parse raw API response into SynergyRule objects."""
        return [
            SynergyRule(
                name=s.get("name", ""),
                tag=s.get("tag", ""),
                threshold=s.get("threshold", 2),
                bonus_stats=s.get("bonus_stats", {}),
            )
            for s in raw.get("synergies", [])
        ]


# =============================================================================
# Example game-specific subclasses (skeletons for extension)
# =============================================================================


class PathOfExileAPI(GameAPISource):
    """Skeleton for Path of Exile items via poe.ninja API."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(base_url="https://poe.ninja/api/data", **kwargs)

    def fetch_items(self) -> List[Item]:
        # Override with PoE-specific parsing
        raise NotImplementedError("PoE API integration pending")


class GenshinImpactAPI(GameAPISource):
    """Skeleton for Genshin Impact artifacts via community API."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(base_url="https://api.genshin.dev", **kwargs)

    def fetch_items(self) -> List[Item]:
        raise NotImplementedError("Genshin API integration pending")


class RiotDataDragon(GameAPISource):
    """Skeleton for League of Legends items via Riot Data Dragon."""

    def __init__(self, patch: str = "14.1.1", **kwargs: Any) -> None:
        super().__init__(
            base_url=f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US",
            **kwargs,
        )

    def fetch_items(self) -> List[Item]:
        raise NotImplementedError("Riot Data Dragon integration pending")
