"""
Bridge to the compiled C++ backend extension module (``backend/``, built via
Pixi + CMake — see ``backend/pixi.toml``).

The extension isn't on ``sys.path`` by default: it's built by a separate
Pixi-managed toolchain from the uv-managed environment that runs this
process, and lands in ``backend/`` (``CMAKE_INSTALL_PREFIX=.`` in
``backend/pixi.toml``'s ``install`` task) rather than being pip-installed.
This module locates and imports it lazily, with a clear, actionable error
if it hasn't been built yet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_module: Any = None


def _backend_dir() -> Path:
    from constants import ROOT_DIR

    return Path(ROOT_DIR) / "backend"


def load_backend() -> Any:
    """Import and return the compiled ``build_optimizer_backend`` module.

    Cached after the first successful import.

    Raises:
        ImportError: if the extension hasn't been built for the running
            Python's ABI (build it via ``cd backend && pixi run build``,
            or ``just backend::build`` — see docs/DEVELOPMENT.md).
    """
    global _backend_module
    if _backend_module is not None:
        return _backend_module

    backend_dir = _backend_dir()
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    try:
        import build_optimizer_backend as backend_module
    except ImportError as exc:
        raise ImportError(
            f"C++ backend extension module not found in {backend_dir}. "
            "Build it first: `cd backend && pixi run build && pixi run install` "
            "(the compiled .so must match this process's Python ABI, e.g. "
            "cpython-311-x86_64-linux-gnu)."
        ) from exc

    _backend_module = backend_module
    return _backend_module
