"""
Path constants for the project.

This module provides platform-independent path resolution for the WSmart+ Route
project root and asset locations. Used by:
- GUI initialization (gui/src/windows/main_window.py) for icon loading
- All modules needing project-relative paths (configs, data, outputs)
- CLI entry points (main.py) for workspace detection

Root Directory Resolution
--------------------------
The ROOT_DIR is dynamically computed by searching upward from cwd until finding
the project root marker directory name. This supports:
- Running from any subdirectory (notebooks/, logic/test/, gui/)
- Multiple project clones (WSmart-Route, WSmartPlus-Route)
- Virtual environment isolation (paths work regardless of venv location)

Path Resolution Order:
1. Get current working directory
2. Search upward for "WSmart-Route" or "WSmartPlus-Route" in path parts
3. Set ROOT_DIR to that location
4. Derive asset paths relative to ROOT_DIR

Critical Files
--------------
- ICON_FILE: GUI application icon (used in window title bar, taskbar)
"""

import os
from pathlib import Path

# Dynamic root directory resolution
# Searches upward from cwd for project root marker ("WSmart-Route" or "WSmartPlus-Route")
path: Path = Path(os.getcwd())  # Current working directory
parts: tuple[str, ...] = path.parts  # Split path into components

try:
    # Primary project name (standard repository)
    root_dir = Path(*parts[: parts.index("WSmart-Route") + 1])
except ValueError:
    # Legacy project name (backward compatibility)
    root_dir = Path(*parts[: parts.index("WSmartPlus-Route") + 1])

# Project root directory (absolute path)
# Example: /home/user/Repositories/WSmart-Route
ROOT_DIR: Path = root_dir

# GUI application icon (PNG format, white logo on transparent background)
# Used in: PySide6 QMainWindow.setWindowIcon(), system tray, taskbar
# Dimensions: 512x512 px (scales down for UI)
ICON_FILE: str = os.path.join(ROOT_DIR, "assets", "images", "logo-wsmartroute-white.png")
