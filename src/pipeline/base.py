"""
Base data source abstract class for the data pipeline.

All data sources (file, API, scraper) inherit from DataSource
and implement the same interface for fetching items and synergies.
"""

from abc import ABC, abstractmethod
from typing import List

from src.core.item import Item
from src.core.synergy import SynergyRule


class DataSource(ABC):
    """Abstract base class for all data sources.

    Subclasses implement fetch_items() and fetch_synergies() to
    retrieve game data from different backends (files, APIs, web scraping).
    """

    @abstractmethod
    def fetch_items(self) -> List[Item]:
        """Retrieve all available items from the data source.

        Returns:
            List of Item objects parsed from the source.
        """
        ...

    @abstractmethod
    def fetch_synergies(self) -> List[SynergyRule]:
        """Retrieve synergy rules from the data source.

        Returns:
            List of SynergyRule objects.
        """
        ...

    def validate(self) -> bool:
        """Validate that the data source is accessible and well-formed.

        Returns:
            True if the source is valid and ready for use.
        """
        try:
            items = self.fetch_items()
            synergies = self.fetch_synergies()
            return len(items) > 0
        except Exception:
            return False

    def summary(self) -> str:
        """Return a brief summary of the data source contents."""
        try:
            items = self.fetch_items()
            synergies = self.fetch_synergies()
            slots = set(item.slot.name for item in items)
            return f"Items: {len(items)}, Synergies: {len(synergies)}, Slots: {', '.join(sorted(slots))}"
        except Exception as e:
            return f"Error: {e}"
