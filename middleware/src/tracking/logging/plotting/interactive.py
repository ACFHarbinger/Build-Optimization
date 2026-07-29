"""
Interactive plots.
"""

from __future__ import annotations

import plotly.express as px


def visualize_interactive_plot(**kwargs):
    """
    Execution function for interactive visualization using Plotly.

    Args:
        **kwargs: Keyword arguments containing 'plot_target', 'title', 'x_labels', 'y_labels', 'figsize'.
    """
    interactive_fig = px.imshow(
        kwargs["plot_target"],
        text_auto=".2f",
        color_continuous_scale="Viridis",
        title=kwargs["title"],
        labels={"x": kwargs["x_labels"], "y": kwargs["y_labels"]},
        width=kwargs["figsize"],
        height=kwargs["figsize"],
    )
    interactive_fig.update_xaxes(tickvals=list(range(len(kwargs["x_labels"]))), ticktext=kwargs["x_labels"])
    interactive_fig.update_yaxes(tickvals=list(range(len(kwargs["y_labels"]))), ticktext=kwargs["y_labels"])
    interactive_fig.show()
