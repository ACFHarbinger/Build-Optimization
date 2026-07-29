"""Sphinx configuration for the Build-Optimization Python middleware reference.

Run: sphinx-build -b html docs/sphinx docs/_build/api/python
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "middleware" / "src"))

project = "Build-Optimization — Python Middleware"
author = "ACFHarbinger"
release = "0.1.0"
html_title = "Build-Optimization Python Reference"
copyright = "2026, ACFHarbinger"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

autoapi_dirs = [str(REPO_ROOT / "middleware" / "src")]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_ignore = ["**/test_*.py", "**/__pycache__/**"]
autoapi_add_toctree_entry = True
autoapi_keep_files = False

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = False
