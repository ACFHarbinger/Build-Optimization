"""
Web scraper data source skeleton for retrieving items from game wikis.

This is a skeleton implementation — subclass for specific game wikis.
"""

import logging
import time
from typing import Any, List, Optional

from core.item import Item
from core.synergy import SynergyRule

from .base import DataSource

logger = logging.getLogger(__name__)


class WebScraperSource(DataSource):
    """Retrieve items by scraping game wiki pages.

    Skeleton for scraping sources like:
    - fandom.com game wikis
    - Game-specific community databases
    - builds.gg, mobafire.com, etc.

    Attributes:
        base_url: Wiki base URL.
        rate_limit: Minimum seconds between requests.
        cache_dir: Optional directory for cached pages.
    """

    def __init__(
        self,
        base_url: str,
        rate_limit: float = 1.0,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.cache_dir = cache_dir
        self._last_request_time: float = 0.0

    def fetch_items(self) -> List[Item]:
        """Scrape items from wiki pages.

        Override this method for specific wiki implementations.
        The base implementation raises NotImplementedError.

        Returns:
            List of Item objects parsed from scraped HTML.
        """
        raise NotImplementedError(
            f"Implement fetch_items() for {self.__class__.__name__}. "
            "Use _fetch_page() to retrieve HTML and parse with BeautifulSoup."
        )

    def fetch_synergies(self) -> List[SynergyRule]:
        """Scrape synergy rules from wiki pages.

        Override this method for specific wiki implementations.

        Returns:
            List of SynergyRule objects.
        """
        raise NotImplementedError(f"Implement fetch_synergies() for {self.__class__.__name__}.")

    # ------------------------------------------------------------------
    # Scraping helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, path: str) -> str:
        """Fetch a page with rate limiting and optional caching.

        Args:
            path: Page path relative to base_url.

        Returns:
            HTML content as string.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        # Check file cache
        if self.cache_dir:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        try:
            import requests

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            html = response.text
            self._last_request_time = time.time()

            # Save to cache
            if self.cache_dir:
                self._write_cache(url, html)

            return html

        except ImportError:
            raise NotImplementedError("Install 'requests' to use WebScraperSource: pip install requests") from None

    def _parse_html(self, html: str) -> Any:
        """Parse HTML with BeautifulSoup.

        Args:
            html: Raw HTML string.

        Returns:
            BeautifulSoup object.
        """
        try:
            from bs4 import BeautifulSoup

            return BeautifulSoup(html, "html.parser")
        except ImportError:
            raise NotImplementedError(
                "Install 'beautifulsoup4' to use WebScraperSource: pip install beautifulsoup4"
            ) from None

    def _read_cache(self, url: str) -> Optional[str]:
        """Read cached page if available."""
        if not self.cache_dir:
            return None
        import hashlib
        from pathlib import Path

        cache_path = Path(self.cache_dir) / hashlib.md5(url.encode()).hexdigest()
        if cache_path.exists():
            return cache_path.read_text()
        return None

    def _write_cache(self, url: str, content: str) -> None:
        """Write page content to cache."""
        if not self.cache_dir:
            return
        import hashlib
        from pathlib import Path

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / hashlib.md5(url.encode()).hexdigest()
        cache_path.write_text(content)
