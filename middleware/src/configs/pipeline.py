"""
Pipeline configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataSourceConfig:
    """Data source configuration.

    Attributes:
        type: Source type (file, api, scraper).
        items_path: Path to items file (for file source).
        synergies_path: Path to synergies file (for file source).
        base_url: Base API/scraper URL.
        api_key: Optional API key.
        timeout: Request timeout.
        cache_ttl: Cache time-to-live.
        rate_limit: Scraper rate limit.
        cache_dir: Scraper cache directory.
    """

    type: str = "file"
    items_path: Optional[str] = None
    synergies_path: Optional[str] = None
    base_url: str = ""
    api_key: Optional[str] = None
    timeout: int = 30
    cache_ttl: int = 3600
    rate_limit: float = 1.0
    cache_dir: str = ".cache"


@dataclass
class PipelineConfig:
    """Data pipeline configuration.

    Attributes:
        source: DataSource configuration.
        transforms: List of transforms to apply (e.g., 'filter_by_level', 'deduplicate').
    """

    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    transforms: List[str] = field(default_factory=lambda: ["filter_by_level", "deduplicate"])
