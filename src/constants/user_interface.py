"""
Constants related to the User Interface and Visualization.

This module defines visual styling for terminal output, matplotlib plots, and
GUI application themes. Used by:
- logic/src/utils/logging/ (progress bars, colored output)
- logic/src/utils/logging/plotting/ (matplotlib charts)
- gui/src/windows/main_window.py (PySide6 application style)

Visual Design Principles
------------------------
1. **Accessibility**: Colorblind-safe palettes, high contrast
2. **Consistency**: Same colors for same concepts across CLI/GUI/plots
3. **Terminal Safety**: ANSI color compatibility (tqdm, rich)
4. **Print Compatibility**: Patterns distinguishable in grayscale

Color Palette Rationale
-----------------------
TQDM_COLOURS and PLOT_COLOURS use the 6-color ANSI palette plus black.
This ensures:
- Terminal emulator compatibility (256-color terminals)
- Distinct colors for multi-route/multi-policy visualizations
- Colorblind-safe (avoid red-green alone for critical distinctions)
"""

from typing import List, Tuple, Union

# WSmart+ route simulation settings (Visuals)
# ---------------------------------------------

# Progress bar update interval (seconds)
# Controls tqdm refresh rate. Lower = smoother (higher CPU), higher = choppier (lower CPU).
# Default: 0.1s (10 Hz) is smooth for most terminals without CPU overhead.
PBAR_WAIT_TIME: float = 0.1  # seconds (100ms refresh interval)

# TQDM progress bar color cycling
# Used in: Parallel simulation workers (one color per worker)
# Cycles through colors when >6 workers run concurrently.
# Note: "red" is used for errors; avoid using for normal progress.
TQDM_COLOURS = [
    "red",  # Worker 0 (or error indicator)
    "blue",  # Worker 1
    "green",  # Worker 2 (success indicator)
    "yellow",  # Worker 3 (warning indicator)
    "magenta",  # Worker 4
    "cyan",  # Worker 5
]

# Plotting - Matplotlib Styling
# ------------------------------

# Marker shapes for scatter plots and line plots
# Cycles through markers for multiple series on same plot.
# Shapes chosen for visual distinction: Pentagon, Square, Triangle, Octagon, Star
MARKERS: List[str] = ["P", "s", "^", "8", "*"]

# Plot color palette (matplotlib color names)
# Same as TQDM_COLOURS plus black for reference lines/baselines.
# Use for: Multi-policy comparisons, Pareto fronts, time series
PLOT_COLOURS: List[str] = [
    "red",  # Policy 0 or worst-case
    "blue",  # Policy 1 or baseline
    "green",  # Policy 2 or best-case
    "yellow",  # Policy 3 (use with caution on white backgrounds)
    "magenta",  # Policy 4
    "cyan",  # Policy 5
    "black",  # Reference lines, grid, text
]

# Line styles for multi-series plots
# Combines standard matplotlib linestyles with custom dash patterns.
# Use when: >6 series need distinction (combine with colors: 6 colors × 5 linestyles = 30 unique combinations)
LINESTYLES: List[Union[str, Tuple[int, Tuple[int, ...]]]] = [
    "dotted",  # · · · · (frequent dots)
    "dashed",  # - - - - (medium dashes)
    "dashdot",  # -·-·-·-· (dash-dot alternating)
    (0, (3, 5, 1, 5, 1, 5)),  # Custom: dash-dot-dot (- · · - · ·)
    "solid",  # ________ (continuous line, default)
]

# GUI Settings - PySide6 Application
# -----------------------------------

# Ctrl+C graceful shutdown timeout (seconds)
# Time allowed for GUI to close cleanly before forced termination.
# Includes: Saving state, closing database connections, stopping background workers.
CTRL_C_TIMEOUT: float = 2.0  # seconds (2s is typical for most Qt apps)

# Available Qt application styles
# Platform-specific availability:
# - "fusion": Cross-platform, modern (recommended default)
# - "windows": Windows native look (Windows only)
# - "windowsxp": Legacy Windows XP theme (Windows only)
# - "macintosh": macOS native look (macOS only, deprecated in Qt6)
# Used in: main.py gui command: python main.py gui --style fusion
APP_STYLES: List[str] = ["fusion", "windows", "windowsxp", "macintosh"]
