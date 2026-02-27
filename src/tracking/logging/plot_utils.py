"""
Plotting utilities for the WSmart+ Route framework.

This file acts as a facade for the plotting sub-package.
"""

import matplotlib.pyplot as plt

from .plotting import (
    discrete_cmap,
    draw_graph,
    plot_attention_maps_wrapper,
    plot_linechart,
    plot_tsp,
    plot_vehicle_routes,
    visualize_interactive_plot,
)

__all__ = [
    "plot_linechart",
    "draw_graph",
    "plot_tsp",
    "plot_vehicle_routes",
    "discrete_cmap",
    "plot_attention_maps_wrapper",
    "visualize_interactive_plot",
    "plt",
]
