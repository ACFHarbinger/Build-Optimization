"""
Game datasets.
"""

from .file_source import FileSource
from .game_api import GameAPISource
from .scraper import WebScraperSource

__all__ = ["WebScraperSource", "FileSource", "GameAPISource"]
